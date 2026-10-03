# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
from collections import Counter
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
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..custom_placements.shard import per_param_placements, Shard
from ..flex_shard import bucket_runtime
from .common import (
    check_flex_shard_parity,
    expected_shard,
    flex_shard_cuda,
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


class _MixedPrecisionWeights(torch.nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.low_weight = torch.nn.Parameter(torch.randn(8, 8, device=device))
        # Interleaved so the bf16 params are not adjacent in declaration order.
        self.full_weight = torch.nn.Parameter(torch.randn(8, 8, device=device))
        self.low_weight2 = torch.nn.Parameter(torch.randn(8, 8, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # matmul rejects mismatched dtypes, so this checks each unsharded dtype.
        low_x = x.bfloat16()
        low = low_x @ self.low_weight + 2 * (low_x @ self.low_weight2)
        return low.float() + x @ self.full_weight


class _ParallelLinears(torch.nn.Module):
    def __init__(self, width: int, *, device: torch.device) -> None:
        super().__init__()
        self.first = torch.nn.Linear(width, width, bias=False, device=device)
        self.second = torch.nn.Linear(width, width, bias=False, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.first(x) + self.second(x)


class _AccumulatingWeight(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8, 8, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Grads of 256 and then 1 accumulate to 257 in fp32 but 256 in bf16."""
        return (x @ self.weight).float().sum()


class _AccumulatingWeights(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.a = _AccumulatingWeight(device=device, dtype=dtype)
        self.b = _AccumulatingWeight(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.b(x)


def _init_params_deterministically(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            values = torch.arange(
                param.numel(),
                dtype=param.dtype,
                device=param.device,
            ).view_as(param)
            param.copy_(values.div(max(param.numel(), 1)).add_(idx))


class _RankRoutedExperts(torch.nn.Module):
    """Each rank routes to one expert, like experts that got tokens on one rank.

    The side branch runs on every rank, but only rank 0's output uses it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(2))
        self.side = torch.nn.Linear(8, 8)

    def forward(
        self, x: torch.Tensor, use_side: bool = True, shift: int = 0
    ) -> torch.Tensor:
        out = self.experts[(dist.get_rank() + shift) % 2](x)
        if use_side:
            side = self.side(x)
            if dist.get_rank() == 0:
                out = out + side
        return out


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
        width = 512
        model = torch.nn.ModuleList(
            [_ParallelLinears(width, device=device_type) for _ in range(2)]
        )
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        # 1 MiB FP32 grads stay in the caching allocator's small pool, so the
        # 2 MiB packed buckets are the only large-pool blocks and the second
        # pack reuses the first pack's freed block.
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

        # Grad-requiring inputs let each layer's input-grad trigger reduce
        # mid-backward, one bucket after the other.
        x = torch.ones(
            (2, width),
            dtype=torch.bfloat16,
            device=device_type.type,
            requires_grad=True,
        )
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

        for index, layer in enumerate(model):
            for param in layer.parameters():
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

        before_forward_hook = model[0].register_forward_pre_hook(probe_before_forward)
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
        before_forward_hook.remove()

        # The unshard copied the gathered bucket into the persistent buffers, so
        # the all-gather output is reusable before module compute and backward.
        self.assertTrue(reused_before_forward)
        if not run_backward:
            self.assertEqual(output, x)
            return

        torch.cuda._sleep(500_000_000)
        backward_delay_done = torch.cuda.Event()
        backward_delay_done.record()
        output.sum().backward()
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
            ("shard", per_param_placements, Shard, False),
            ("bucketed_block", bucketed_block_placements, BucketedBlockShard, False),
            (
                "bucketed_block_backward",
                bucketed_block_placements,
                BucketedBlockShard,
                True,
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
        # Three microbatches, in three modes:
        # - "sync": every microbatch reduce-scatters.
        # - "per_bucket": sync is off for every bucket but the layer's on the
        #   first two microbatches, set through the per-bucket setters.
        # - "keep": sync off for the model on the first two microbatches,
        #   keeping every bucket but the norm's unsharded between them.
        # Only the layer reshards after forward. The last microbatch skips the
        # layer and the positional embeddings, so their kept grads are reduced
        # in the end-of-backward callback.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        args, base_model = make_transformer_model(
            device=device_type.type,
            vocab_size=15,
        )
        _init_params_deterministically(base_model)
        torch.manual_seed(42 + self.rank + 1)
        inputs = [
            transformer_inputs(args, batch_size=batch_size, device=device_type)
            for batch_size in (3, 2, 2)
        ]

        def microbatch_loss(module, idx):
            x = inputs[idx]
            if idx < len(inputs) - 1:
                return module(x).sum()
            return module.output(module.norm(module.tok_embeddings(x))).sum()

        for mode in ("sync", "per_bucket", "keep"):
            model = copy.deepcopy(base_model)
            reference = copy.deepcopy(base_model)
            buckets = transformer_bucket_specs(args.n_layers, mesh)
            buckets[2] = dataclasses.replace(buckets[2], reshard_after_forward=True)
            flex_shard(model, buckets=buckets)
            storages = model.sharded_bucket_storages
            layer, norm = storages[2], storages[3]
            if mode == "keep":
                model.set_reshard_after_backward(False)
                norm.set_reshard_after_backward(True)
            optim = make_test_sgd(model.parameters(), lr=0.1)
            ref_optim = make_test_sgd(reference.parameters(), lr=0.1)

            runtime = bucket_runtime.BucketRuntime
            with (
                mock.patch.object(
                    runtime,
                    "begin_unshard",
                    autospec=True,
                    side_effect=runtime.begin_unshard,
                ) as unshards,
                mock.patch.object(
                    runtime,
                    "reduce_grads",
                    autospec=True,
                    side_effect=runtime.reduce_grads,
                ) as reduces,
            ):
                for idx in range(len(inputs)):
                    last = idx == len(inputs) - 1
                    if mode == "per_bucket":
                        if last:
                            model.reshard()  # keeps the accumulated grads
                        for storage in storages:
                            storage.set_requires_gradient_sync(last or storage is layer)
                    if mode == "keep":
                        model.set_requires_gradient_sync(last)
                    loss = microbatch_loss(model, idx)
                    ref_loss = microbatch_loss(reference, idx)
                    self.assertEqual(loss, ref_loss)
                    loss.backward()
                    ref_loss.backward()

            reduced = Counter(call.args[0].debug_fqn for call in reduces.call_args_list)
            gathered = Counter(
                call.args[0].debug_fqn for call in unshards.call_args_list
            )
            once = dict.fromkeys(
                ["tok_embeddings", "pos_embeddings", "layers.0", "norm", "output"], 1
            )
            if mode == "per_bucket":
                # The layer reduce-scatters in both microbatches that use it,
                # the other buckets once per step.
                self.assertEqual(reduced, {**once, "layers.0": 2})
            if mode == "keep":
                # One reduce-scatter per bucket per step. Kept buckets
                # all-gather once per step; the layer also re-gathers in each
                # backward and the norm in each forward.
                self.assertEqual(reduced, once)
                self.assertEqual(gathered, {**once, "layers.0": 3, "norm": 3})
            _average_reference_grads(reference)
            check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

            optim.step()
            ref_optim.step()
            check_flex_shard_parity(self, reference, model, self.rank, self.world_size)
            # The syncing backward resharded, so forward sees the stepped shards.
            self.assertEqual(model(inputs[0]).sum(), reference(inputs[0]).sum())

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
        # A microbatch without sync first: the kept bf16 param accumulates its
        # grads in the fp32 reduce dtype until the syncing backward.
        model.set_reshard_after_backward(False)
        model.set_requires_gradient_sync(False)
        model.tok_embeddings(x).float().sum().backward()
        model.set_requires_gradient_sync(True)
        reference.tok_embeddings(x).float().sum().backward()
        unsharded_param = model.tok_embeddings.weight
        self.assertEqual(unsharded_param.dtype, torch.bfloat16)
        self.assertEqual(unsharded_param.grad.dtype, torch.float32)

        output = model.tok_embeddings(x)
        ref_output = reference.tok_embeddings(x)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output, ref_output)

        output.float().sum().backward()
        ref_output.float().sum().backward()
        self.assertIsNone(unsharded_param.grad)

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

        # A syncing backward without a kept grad defers the upcast: the bf16
        # grad reaches the reduce-scatter, whose copy-in widens it, and
        # grad_dtype is restored afterwards.
        runtime = bucket_runtime.BucketRuntime
        with mock.patch.object(
            runtime, "reduce_grads", autospec=True, side_effect=runtime.reduce_grads
        ) as reduces:
            model.tok_embeddings(x).float().sum().backward()
        self.assertEqual(
            [grad.dtype for grad in reduces.call_args.args[1]], [torch.bfloat16]
        )
        self.assertEqual(unsharded_param.grad_dtype, torch.float32)

    @skip_if_lt_x_gpu(2)
    def test_rank_dependent_unused_params(self):
        # Ranks leave different params of a bucket, or a whole bucket's
        # outputs, without grads; every rank still reduce-scatters every
        # bucket, with zeros for its unused params. Three microbatches: the
        # first without sync (keeping fp32 grads on rank 0 only), the second
        # skipping the side bucket, so the pending sync makes it reduce on
        # every rank, and routing each rank to the other expert, and the third
        # syncing with the side bucket's outputs unused on rank 1.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        torch.manual_seed(0)
        model = _RankRoutedExperts().to(device_type, torch.bfloat16)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    [pattern],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
                    reshard_after_forward=False,
                )
                for pattern in ("experts.*", "side.*")
            ],
        )
        x = torch.randn(4, 8, device=device_type, dtype=torch.bfloat16)
        for idx, (use_side, shift) in enumerate(((True, 0), (False, 1), (True, 0))):
            model.set_requires_gradient_sync(idx > 0)
            model(x, use_side=use_side, shift=shift).sum().backward()
            reference(x, use_side=use_side, shift=shift).sum().backward()
        for param in reference.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
        _average_reference_grads(reference)
        check_flex_shard_parity(self, reference, model, self.rank, self.world_size)

    @skip_if_lt_x_gpu(2)
    def test_per_param_mixed_precision_policy(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        torch.manual_seed(42)
        model = _MixedPrecisionWeights(device=device_type)
        reference = copy.deepcopy(model)
        reference.low_weight = torch.nn.Parameter(reference.low_weight.bfloat16())
        reference.low_weight2 = torch.nn.Parameter(reference.low_weight2.bfloat16())
        mixed_placement = MixedBucketPlacement({})

        def mixed_placement_fn(named_params, mesh):
            del mesh
            return {fqn: (mixed_placement.shard0,) for fqn, _ in named_params}

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=mixed_placement_fn,
                    mesh=mesh,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=torch.bfloat16,
                        param_dtype_overrides={"full_weight": torch.float32},
                    ),
                    reshard_after_forward=False,
                )
            ],
        )

        torch.manual_seed(43 + self.rank)
        x = torch.randn(4, 8, device=device_type)
        with (
            mock.patch.object(
                dist,
                "all_gather_into_tensor",
                wraps=dist.all_gather_into_tensor,
            ) as all_gather,
            mock.patch.object(
                dist,
                "reduce_scatter_tensor",
                wraps=dist.reduce_scatter_tensor,
            ) as reduce_scatter,
        ):
            output = model(x)
            output.sum().backward()

        reference_output = reference(x)
        reference_output.sum().backward()
        self.assertEqual(output, reference_output)
        self.assertEqual(all_gather.call_count, 1)
        self.assertEqual(reduce_scatter.call_count, 1)

        model_params = dict(model.named_parameters())
        for fqn, reference_param in reference.named_parameters():
            grad = model_params[fqn].grad
            expected_grad = reference_param.grad.float()
            dist.all_reduce(expected_grad, op=dist.ReduceOp.AVG)
            self.assertEqual(
                grad,
                expected_shard(
                    expected_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                ),
            )

    @skip_if_lt_x_gpu(2)
    def test_grads_accumulate_in_unsharded_grad_dtype(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        bf16, fp32 = torch.bfloat16, torch.float32
        # Each case exercises one source of the unsharded accumulation dtype.
        # Grads of 256, from a backward without sync, and then 1, from a
        # syncing one, accumulate to 257 only in fp32. As in FSDP2, the upcast
        # is deferred: the backward without sync still holds the grad in the
        # compute dtype when it reshards and widens it after.
        cases = (
            ("storage_dtype", fp32, None, bf16, None, fp32, 257.0),
            ("explicit_grad_dtype", bf16, fp32, None, None, fp32, 257.0),
            ("reduce_dtype", bf16, fp32, None, bf16, bf16, 256.0),
        )
        for case in cases:
            (
                name,
                dtype,
                grad_dtype,
                param_dtype,
                reduce_dtype,
                accumulation_dtype,
                grad_value,
            ) = case
            compute_dtype = param_dtype or dtype
            with self.subTest(name=name):
                model = _AccumulatingWeight(device=device_type, dtype=dtype)
                if grad_dtype is not None:
                    model.weight.grad_dtype = grad_dtype
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            mp_policy=MixedPrecisionPolicy(
                                param_dtype=param_dtype,
                                reduce_dtype=reduce_dtype,
                            ),
                            reshard_after_forward=False,
                        )
                    ],
                )

                x = torch.ones(1, 8, dtype=compute_dtype, device=device_type)
                runtime = bucket_runtime.BucketRuntime
                at_reshard = []

                def record_reshard(bucket, reshard=runtime.reshard):
                    at_reshard.append(
                        [
                            p.grad.dtype
                            for p in bucket.unsharded_params
                            if p.grad is not None
                        ]
                    )
                    reshard(bucket)

                with mock.patch.object(
                    runtime, "reshard", autospec=True, side_effect=record_reshard
                ) as reshards:
                    model.set_requires_gradient_sync(False)
                    model(256 * x).backward()
                unsharded_param = reshards.call_args.args[0].unsharded_params[0]
                self.assertEqual(at_reshard, [[compute_dtype]])
                self.assertEqual(unsharded_param.grad.dtype, accumulation_dtype)
                model.set_requires_gradient_sync(True)
                model(x).backward()
                grad = model._parameters["weight"].grad
                self.assertIsNotNone(grad)
                self.assertEqual(grad.dtype, fp32)
                self.assertEqual(
                    grad,
                    expected_shard(
                        torch.full(
                            (8, 8),
                            grad_value,
                            dtype=fp32,
                            device=device_type,
                        ),
                        rank=self.rank,
                        world_size=self.world_size,
                    ),
                )

    @skip_if_lt_x_gpu(2)
    def test_bucket_reduces_in_promoted_grad_dtype(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        model = _AccumulatingWeights(device=device_type, dtype=torch.bfloat16)
        model.b.weight.grad_dtype = torch.float32
        flex_shard_cuda(model, mesh)

        x = torch.ones(1, 8, dtype=torch.bfloat16, device=device_type)
        # Grads of 256 and then 1: b accumulates them in fp32, a in bf16.
        model.set_requires_gradient_sync(False)
        model(256 * x).backward()
        model.set_requires_gradient_sync(True)
        model(x).backward()

        # b's fp32 sum of 257 survives only if the shared reduction promotes
        # a's bf16 and b's fp32 gradients to fp32.
        for module, grad_value, grad_dtype in (
            (model.a, 256.0, torch.bfloat16),
            (model.b, 257.0, torch.float32),
        ):
            grad = module._parameters["weight"].grad
            self.assertIsNotNone(grad)
            self.assertEqual(grad.dtype, grad_dtype)
            self.assertEqual(grad, torch.full_like(grad, grad_value))

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

        # FlexShard leaves the user's activation-checkpoint wrapper untouched.
        self.assertIsInstance(model.layers[0], CheckpointWrapper)
        self.assertIs(
            model.layers[0].checkpoint_fn.keywords["context_fn"],
            _prefer_recompute_context_fn,
        )

        torch.manual_seed(42 + self.rank + 1)
        x = transformer_inputs(args, batch_size=3, device=device_type)
        optim = make_test_sgd(model.parameters(), lr=0.1)
        ref_optim = make_test_sgd(reference.parameters(), lr=0.1)

        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        # A microbatch without sync first, keeping the params unsharded after
        # it: the next forward skips the all-gather and still reshards after
        # forward, and the recompute in backward re-gathers.
        model.set_reshard_after_backward(False)
        model.set_requires_gradient_sync(False)
        model(x).sum().backward()
        model.set_requires_gradient_sync(True)
        reference(x).sum().backward()
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


if __name__ == "__main__":
    run_tests()
