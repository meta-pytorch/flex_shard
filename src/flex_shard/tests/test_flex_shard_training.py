# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from unittest import mock

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointWrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from .. import BucketSpec, flex_shard, MixedPrecisionPolicy
from ..custom_placements.block_shard import (
    BucketedBlockShard,
    make_bucketed_block_placement_fn,
)
from ..custom_placements.shard import per_param_placements, Shard
from .common import (
    check_flex_shard_parity,
    expected_shard,
    make_test_sgd,
    make_transformer_model,
    transformer_bucket_specs,
    transformer_inputs,
)


device_type = torch.device(get_devtype())


class _UnevenMLP(torch.nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.out_activation = torch.nn.PReLU(device=device)
        self.in_proj = torch.nn.Linear(3, 5, device=device)
        self.out_proj = torch.nn.Linear(5, 3, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_activation(self.out_proj(torch.tanh(self.in_proj(x))))


class _UnevenMLPStack(torch.nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                _UnevenMLP(device=device),
                _UnevenMLP(device=device),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _init_params_deterministically(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            values = torch.arange(
                param.numel(),
                dtype=param.dtype,
                device=param.device,
            ).view_as(param)
            param.copy_(values.div(max(param.numel(), 1)).add_(idx))


def _average_reference_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)


def _prefer_recompute_context_fn():
    def prefer_recompute_policy(ctx, func, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(prefer_recompute_policy)


def _checkpoint_transformer_execution_units(model: torch.nn.Module) -> None:
    model.tok_embeddings = checkpoint_wrapper(
        model.tok_embeddings,
        context_fn=_prefer_recompute_context_fn,
    )
    model.pos_embeddings = checkpoint_wrapper(
        model.pos_embeddings,
        context_fn=_prefer_recompute_context_fn,
    )
    for idx, layer in enumerate(model.layers):
        model.layers[idx] = checkpoint_wrapper(
            layer,
            context_fn=_prefer_recompute_context_fn,
        )
    model.norm = checkpoint_wrapper(
        model.norm,
        context_fn=_prefer_recompute_context_fn,
    )
    model.output = checkpoint_wrapper(
        model.output,
        context_fn=_prefer_recompute_context_fn,
    )


def _layer_buckets_with_grouped_root_rest(
    num_layers: int,
    mesh,
    *,
    reshard_after_forward: bool,
) -> list[BucketSpec]:
    return [
        *[
            BucketSpec(
                [f"layers.{idx}.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=reshard_after_forward,
            )
            for idx in range(num_layers)
        ],
        BucketSpec(
            ["tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*"],
            placement_fn=per_param_placements,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        ),
    ]


class TestFlexShardTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_input_lifetime(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        width = 1024
        model = torch.nn.ModuleList(
            [
                torch.nn.Linear(width, width, bias=False, device=device_type.type)
                for _ in range(2)
            ]
        )
        # Give packed FP32 reductions a distinct allocation size from BF16 grads.
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [f"{index}.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=False,
                )
                for index in range(2)
            ],
        )
        model.set_max_pending_reduce_grads(1)

        x = torch.ones((2, width), dtype=torch.bfloat16, device=device_type.type)
        output = sum(layer((index + 1) * x) for index, layer in enumerate(model))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        packed_ptrs: list[int] = []
        original_pack = Shard._pack_reduce_scatter_grad
        original_reduce_scatter = dist.reduce_scatter_tensor

        def record_packed_buffer(placement, tensors, infos, world_size):
            packed, layout = original_pack(placement, tensors, infos, world_size)
            packed_ptrs.append(packed.data_ptr())
            if len(packed_ptrs) == 2:
                # The fix orders this producer stream after the first consumer.
                torch.cuda.current_stream().synchronize()
            return packed, layout

        def delayed_reduce_scatter(output, input, *args, **kwargs):
            if len(packed_ptrs) == 1:
                torch.cuda._sleep(500_000_000)
            # Stage before the real collective so the original packed buffer's
            # cross-stream lifetime remains FlexShard's responsibility.
            staged_input = input.clone()
            return original_reduce_scatter(output, staged_input, *args, **kwargs)

        with (
            mock.patch.object(
                Shard,
                "_pack_reduce_scatter_grad",
                record_packed_buffer,
            ),
            mock.patch.object(
                dist,
                "reduce_scatter_tensor",
                delayed_reduce_scatter,
            ),
        ):
            output.float().sum().backward()

        for index, param in enumerate(model.parameters()):
            grad = param.grad
            assert grad is not None
            self.assertEqual(grad, torch.full_like(grad, 2 * (index + 1)))
        self.assertEqual(len(packed_ptrs), len(model))
        self.assertEqual(packed_ptrs[0], packed_ptrs[1])

    def _check_unshard_buffer_lifetime(
        self,
        mesh,
        placement_fn,
        placement_type,
        run_backward,
        expect_reuse_before_forward,
        expect_reuse_after_forward,
    ):
        width = 1024
        x = torch.ones(
            (2, width),
            dtype=torch.bfloat16,
            device=device_type.type,
            requires_grad=run_backward,
        )
        model = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    width,
                    width,
                    bias=False,
                    device=device_type.type,
                    dtype=torch.bfloat16,
                )
            ]
        )
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].weight.diagonal().fill_(1)
        model[0].weight.requires_grad_(False)

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["0.*"],
                    placement_fn=placement_fn,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ],
        )
        comm_context = next(iter(model._flex_shard_eager_comm_contexts.values()))
        torch.cuda.empty_cache()

        original_finish_prepared_unshard = placement_type.finish_prepared_unshard
        gathered_buffer_metadata = None
        reused_before_forward = None
        scratch_keepalive: list[torch.Tensor] = []

        def record_gathered_buffer(placement, prepared):
            nonlocal gathered_buffer_metadata
            gathered_buffer = prepared.buffers[1]
            gathered_buffer_metadata = (
                tuple(gathered_buffer.shape),
                gathered_buffer.dtype,
                gathered_buffer.device,
                gathered_buffer.data_ptr(),
            )
            if not run_backward:
                torch.cuda._sleep(500_000_000)
            return original_finish_prepared_unshard(placement, prepared)

        def try_reuse_gathered_buffer():
            buffer_metadata = gathered_buffer_metadata
            if buffer_metadata is None:
                raise AssertionError("Expected the all-gather output buffer.")
            shape, dtype, device, data_ptr = buffer_metadata
            with torch.cuda.stream(comm_context.unshard_stream):
                scratch_buffer = torch.zeros(
                    shape,
                    dtype=dtype,
                    device=device,
                )
            comm_context.unshard_stream.synchronize()
            scratch_keepalive.append(scratch_buffer)
            return scratch_buffer.data_ptr() == data_ptr

        def probe_before_forward(module, args):
            nonlocal reused_before_forward
            reused_before_forward = try_reuse_gathered_buffer()

        before_forward_hook = (
            model[0].register_forward_pre_hook(probe_before_forward)
            if expect_reuse_before_forward is not None
            else None
        )
        with mock.patch.object(
            placement_type,
            "finish_prepared_unshard",
            record_gathered_buffer,
        ):
            if run_backward:
                output = model[0](x)
            else:
                with torch.no_grad():
                    output = model[0](x)
        if before_forward_hook is not None:
            before_forward_hook.remove()

        if expect_reuse_before_forward is not None:
            self.assertEqual(reused_before_forward, expect_reuse_before_forward)
        if expect_reuse_after_forward is not None:
            self.assertEqual(
                try_reuse_gathered_buffer(),
                expect_reuse_after_forward,
            )
        if not run_backward:
            self.assertEqual(output, x)
            return

        torch.cuda._sleep(500_000_000)
        backward_delay_done = torch.cuda.Event()
        backward_delay_done.record()
        output.sum().backward()
        self.assertTrue(try_reuse_gathered_buffer())
        self.assertTrue(backward_delay_done.query())
        self.assertEqual(x.grad, torch.ones_like(x))

    @skip_if_lt_x_gpu(2)
    def test_unshard_buffer_lifetime(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        bucketed_block_placements = make_bucketed_block_placement_fn(
            dims=(0,),
            blocks_per_rank=(1, 1),
        )
        cases = (
            ("shard", per_param_placements, Shard, False, True, None),
            (
                "bucketed_block",
                bucketed_block_placements,
                BucketedBlockShard,
                False,
                False,
                True,
            ),
            (
                "bucketed_block_backward",
                bucketed_block_placements,
                BucketedBlockShard,
                True,
                None,
                False,
            ),
        )
        for case in cases:
            with self.subTest(placement=case[0]):
                self._check_unshard_buffer_lifetime(mesh, *case[1:])

    @skip_if_lt_x_gpu(2)
    def test_uneven_shard_bucket_matches_unsharded_reference(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        torch.manual_seed(42)
        model = _UnevenMLPStack(device=device_type)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [f"layers.{layer_idx}.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=True,
                )
                for layer_idx in range(len(model.layers))
            ],
        )

        storage = model.sharded_bucket_storages[0]
        infos = list(storage.param_infos.values())
        element_size = infos[0].dtype.itemsize
        self.assertEqual(
            {
                fqn: info.storage_nbytes // element_size
                for fqn, info in storage.param_infos.items()
            },
            {
                "layers.0.out_activation.weight": 1,
                "layers.0.in_proj.weight": 9,
                "layers.0.in_proj.bias": 3,
                "layers.0.out_proj.weight": 10,
                "layers.0.out_proj.bias": 2,
            },
        )
        self.assertEqual(storage.total_bytes, 25 * element_size)
        self.assertEqual(
            sum(info.local_numel for info in infos),
            25 if self.rank == 0 else 14,
        )
        self.assertEqual(infos[0].fqn, "layers.0.out_activation.weight")
        self.assertEqual(infos[0].local_numel, 1 if self.rank == 0 else 0)
        local_params = [storage.get_local_view(info.fqn) for info in infos]
        prepared_unshard = infos[0].placement.prepare_unshard_bucket(
            local_params,
            infos,
            mesh,
            None,
        )
        send_buf = prepared_unshard.buffers[0]
        self.assertEqual(send_buf.numel(), 25)
        self.assertEqual(
            send_buf.untyped_storage().data_ptr(),
            storage.byte_storage.untyped_storage().data_ptr(),
        )
        del prepared_unshard

        optimizer = make_test_sgd(model.parameters(), lr=0.1)
        reference_optimizer = make_test_sgd(reference.parameters(), lr=0.1)
        torch.manual_seed(43 + self.rank)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            inputs = torch.randn(4, 3, device=device_type)
            output = model(inputs)
            reference_output = reference(inputs)
            self.assertEqual(output, reference_output)

            output.square().sum().backward()
            reference_output.square().sum().backward()
            _average_reference_grads(reference)
            check_flex_shard_parity(
                self,
                reference,
                model,
                self.rank,
                self.world_size,
            )

            optimizer.step()
            reference_optimizer.step()
            check_flex_shard_parity(
                self,
                reference,
                model,
                self.rank,
                self.world_size,
            )

    @skip_if_lt_x_gpu(2)
    def test_gradient_accumulation(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(
            device=device_type.type,
            vocab_size=15,
        )
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            buckets=transformer_bucket_specs(
                args.n_layers,
                mesh,
                reshard_after_forward=False,
            ),
        )

        torch.manual_seed(42 + self.rank + 1)
        inputs = [
            transformer_inputs(args, batch_size=3, device=device_type),
            transformer_inputs(args, batch_size=2, device=device_type),
        ]
        optim = make_test_sgd(model.parameters(), lr=0.1)
        ref_optim = make_test_sgd(reference.parameters(), lr=0.1)

        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        for x in inputs:
            loss = model(x).sum()
            ref_loss = reference(x).sum()
            self.assertEqual(loss, ref_loss)
            loss.backward()
            ref_loss.backward()

        _average_reference_grads(reference)
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

        optim.step()
        ref_optim.step()

        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

    @skip_if_lt_x_gpu(2)
    def test_mixed_precision_policy(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(
            device=device_type.type,
            vocab_size=15,
        )
        _init_params_deterministically(model)
        reference = copy.deepcopy(model).to(torch.bfloat16)

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["tok_embeddings.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32,
                    ),
                    reshard_after_forward=False,
                ),
                *transformer_bucket_specs(
                    args.n_layers, mesh, reshard_after_forward=False
                )[1:],
            ],
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=2, device=device_type)
        output = model.tok_embeddings(x)
        ref_output = reference.tok_embeddings(x)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output, ref_output)

        output.float().sum().backward()
        ref_output.float().sum().backward()

        grad = model.tok_embeddings._parameters["weight"].grad
        self.assertIsNotNone(grad)
        self.assertEqual(grad.dtype, torch.float32)

        ref_grad = reference.tok_embeddings.weight.grad.to(torch.float32)
        dist.all_reduce(ref_grad, op=dist.ReduceOp.AVG)
        expected_grad = expected_shard(
            ref_grad,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.assertEqual(grad, expected_grad)

    @skip_if_lt_x_gpu(2)
    def test_reshard_after_forward_with_activation_checkpointing(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(
            device=device_type.type,
            vocab_size=15,
        )
        _init_params_deterministically(model)
        reference = copy.deepcopy(model)
        _checkpoint_transformer_execution_units(model)
        _checkpoint_transformer_execution_units(reference)

        flex_shard(
            model,
            buckets=transformer_bucket_specs(
                args.n_layers,
                mesh,
                reshard_after_forward=True,
            ),
        )

        self.assertIsInstance(model.layers[0], CheckpointWrapper)
        composed_context_fn = model.layers[0].checkpoint_fn.keywords["context_fn"]
        self.assertIsNot(composed_context_fn, _prefer_recompute_context_fn)
        forward_ctx, _ = composed_context_fn()
        from ..flex_shard.unshard_op import UNSHARD_BUCKET_OP

        self.assertEqual(
            forward_ctx.policy_fn(
                None,
                UNSHARD_BUCKET_OP,
            ),
            CheckpointPolicy.MUST_RECOMPUTE,
        )
        self.assertEqual(
            forward_ctx.policy_fn(None, torch.ops.aten.mm.default),
            CheckpointPolicy.PREFER_RECOMPUTE,
        )
        self.assertEqual(
            forward_ctx.policy_fn(
                None,
                torch.ops._c10d_functional.all_to_all_single.default,
            ),
            CheckpointPolicy.PREFER_RECOMPUTE,
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=3, device=device_type)
        optim = make_test_sgd(model.parameters(), lr=0.1)
        ref_optim = make_test_sgd(reference.parameters(), lr=0.1)

        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        loss = model(x).sum()
        ref_loss = reference(x).sum()
        self.assertEqual(loss, ref_loss)
        loss.backward()
        ref_loss.backward()

        _average_reference_grads(reference)
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

        optim.step()
        ref_optim.step()
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

    @skip_if_lt_x_gpu(2)
    def test_reshard_after_forward_grouped_root_rest_bucket_unsupported(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, model = make_transformer_model(device=device_type.type, n_layers=2)
        _checkpoint_transformer_execution_units(model)

        with self.assertRaisesRegex(RuntimeError, "recomputation-safe"):
            flex_shard(
                model,
                buckets=_layer_buckets_with_grouped_root_rest(
                    args.n_layers,
                    mesh,
                    reshard_after_forward=True,
                ),
            )


if __name__ == "__main__":
    run_tests()
