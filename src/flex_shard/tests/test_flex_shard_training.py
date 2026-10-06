# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import warnings
import weakref
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
from ..flex_shard.checkpoint import get_flex_shard_global_layouts
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
        self.weight = torch.nn.Parameter(
            torch.zeros(8, 8, device=device, dtype=dtype)
        )

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


class _ReusedWeight(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.zeros(8, 8, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-use grads 256 and 1 sum to 257 in fp32 but round to 256 in bf16."""
        return ((256 * x) @ self.weight).float().sum() + (x @ self.weight).float().sum()


class _ReusedWeights(torch.nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.a = _ReusedWeight(device=device, dtype=dtype)
        self.b = _ReusedWeight(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x) + self.b(x)


class _FusedWgradLinearFn(torch.autograd.Function):
    """``x @ weight.t()`` whose backward adds the weight grad into
    ``weight.main_grad`` in place and gives autograd none, like
    TransformerEngine's ``fuse_wgrad_accumulation``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, weight)
        # The parameter object carrying main_grad, as TransformerEngine keeps it.
        ctx.weight_ref = weakref.ref(weight)
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        main_grad = ctx.weight_ref().main_grad
        main_grad.add_(grad_output.t().to(main_grad.dtype) @ x.to(main_grad.dtype))
        return grad_output @ weight, None


class _FusedWgradLinear(torch.nn.Module):
    """A linear layer with fused weight-grad accumulation and an ordinary bias."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8, device=device, dtype=dtype))
        self.bias = torch.nn.Parameter(torch.randn(8, device=device, dtype=dtype))
        self.weights_seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weights_seen.append(self.weight)
        return _FusedWgradLinearFn.apply(x, self.weight) + self.bias


class _DelayedWgradLinearFn(torch.autograd.Function):
    """``x @ weight.t()`` whose backward leaves the weight grad to
    ``backward_dw()``, like TransformerEngine's ``delay_wgrad_compute``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, pending: list):
        ctx.save_for_backward(x, weight)
        ctx.weight_ref = weakref.ref(weight)
        ctx.pending = pending
        return x @ weight.t()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        ctx.pending.append((ctx.weight_ref, x, grad_output))
        return grad_output @ weight, None, None


class _RunDelayedWgrad(torch.autograd.Function):
    """Identity whose backward runs after the next layer's backward and calls
    ``callback``, as Megatron-LM's MoE layer runs the experts' delayed weight
    grads in the token dispatch's backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, callback) -> torch.Tensor:
        ctx.callback = callback
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        ctx.callback()
        return grad_output, None


class _DelayedWgradLinear(torch.nn.Module):
    """A linear layer whose weight grad waits for ``backward_dw()``, which adds
    it to the weight's grad, and an ordinary bias."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8, device=device))
        self.bias = torch.nn.Parameter(torch.randn(8, device=device))
        self.pending: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DelayedWgradLinearFn.apply(x, self.weight, self.pending) + self.bias

    def backward_dw(self) -> None:
        for weight_ref, x, grad_output in self.pending:
            weight, wgrad = weight_ref(), grad_output.t() @ x
            weight.grad = wgrad if weight.grad is None else weight.grad + wgrad
        self.pending.clear()


class _DelayedWgradModel(torch.nn.Module):
    """``out(relu(layer(inp(x))))``: ``layer``'s delayed weight grad is computed
    after ``layer``'s backward, which then calls ``on_wgrad``."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.inp = torch.nn.Linear(8, 8, device=device)
        self.layer = _DelayedWgradLinear(device=device)
        self.out = torch.nn.Linear(8, 6, device=device)
        self.on_wgrad = lambda: None

    def _backward_dw(self) -> None:
        self.layer.backward_dw()
        self.on_wgrad()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _RunDelayedWgrad.apply(self.inp(x), self._backward_dw)
        return self.out(torch.relu(self.layer(x)))


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
    def test_bucketed_block_copy_free_matches_reference(self):
        # Refills gather straight into the persistent buckets, and with
        # gradient_bucket=True the reduce-scatter reads the grads' bucket as
        # is; training matches an unsharded reference, with and without
        # reshard-after-forward and no-sync.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        placement_fn = make_bucketed_block_placement_fn(
            dims=(0,), blocks_per_rank=(1,) * self.world_size
        )
        begin_bucket_unshard = bucket_runtime.begin_bucket_unshard
        gradient_bucket_of = BucketedBlockShard._gradient_bucket_of
        for reshard_after_forward, gradient_bucket in (
            (False, True),
            (True, True),
            (True, False),
        ):
            with self.subTest(
                reshard_after_forward=reshard_after_forward,
                gradient_bucket=gradient_bucket,
            ):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(16, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 8),
                ).to(device_type)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            [f"{idx}.*"],
                            placement_fn=placement_fn,
                            mesh=mesh,
                            reshard_after_forward=reshard_after_forward,
                            gradient_bucket=gradient_bucket,
                        )
                        for idx in (0, 2, 4)
                    ],
                )
                optimizer = make_test_sgd(model.parameters(), lr=0.1)
                reference_optimizer = make_test_sgd(reference.parameters(), lr=0.1)
                in_place: list[bool] = []
                reduced_bucket: list[bool] = []

                def recording_begin(*args, **kwargs):
                    in_place.append(kwargs.get("persistent_buffers") is not None)
                    return begin_bucket_unshard(*args, **kwargs)

                def recording_gradient_bucket_of(placement, *args):
                    bucket = gradient_bucket_of(placement, *args)
                    reduced_bucket.append(bucket is not None)
                    return bucket

                torch.manual_seed(1 + self.rank)
                with (
                    mock.patch.object(
                        bucket_runtime, "begin_bucket_unshard", recording_begin
                    ),
                    mock.patch.object(
                        BucketedBlockShard,
                        "_gradient_bucket_of",
                        recording_gradient_bucket_of,
                    ),
                ):
                    for step in range(4):
                        optimizer.zero_grad(set_to_none=True)
                        reference_optimizer.zero_grad(set_to_none=True)
                        # The last step accumulates two microbatches without
                        # sync into the same grads.
                        microbatches = 2 if step == 3 else 1
                        for microbatch in range(microbatches):
                            model.set_requires_gradient_sync(
                                microbatch == microbatches - 1
                            )
                            x = torch.randn(4, 16, device=device_type)
                            output = model(x)
                            reference_output = reference(x)
                            self.assertEqual(output, reference_output)
                            output.square().sum().backward()
                            reference_output.square().sum().backward()
                        _average_reference_grads(reference)
                        optimizer.step()
                        reference_optimizer.step()
                # The first unshard of each bucket copies; refills are in place.
                self.assertEqual(in_place[:3], [False] * 3)
                self.assertTrue(all(in_place[3:]))
                self.assertGreater(len(in_place), 3)
                self.assertTrue(reduced_bucket)
                self.assertTrue(
                    all(found == gradient_bucket for found in reduced_bucket)
                )

    @skip_if_lt_x_gpu(2)
    def test_gradient_divide_factor(self):
        # The reduced gradient is the sum over the mesh divided by the factor,
        # e.g. the dense data-parallel size for an expert bucket on an expert
        # data-parallel mesh.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        cases = (
            # (dtype, factor in BucketSpec, factor set later, no-sync)
            (torch.float32, 3.0, None, True),  # PREMUL_SUM
            (torch.bfloat16, None, 8, False),
            (torch.float16, 6, None, False),  # divides before and after the sum
        )
        for dtype, factor, later_factor, no_sync in cases:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
                ).to(device_type, dtype)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                            gradient_divide_factor=factor,
                        )
                    ],
                )
                if later_factor is not None:
                    model.set_gradient_divide_factor(later_factor)
                torch.manual_seed(1 + self.rank)
                for step in range(2 if no_sync else 1):
                    model.set_requires_gradient_sync(step == int(no_sync))
                    x = torch.randn(4, 8, device=device_type, dtype=dtype)
                    model(x).sum().backward()
                    reference(x).sum().backward()
                tolerance = {} if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
                for param, ref_param in zip(model.parameters(), reference.parameters()):
                    expected = ref_param.grad.float()
                    dist.all_reduce(expected)
                    expected = expected_shard(
                        expected / (later_factor or factor),
                        rank=self.rank,
                        world_size=self.world_size,
                    )
                    self.assertEqual(param.grad.float(), expected, **tolerance)

    @skip_if_lt_x_gpu(2)
    def test_finalize_backward(self):
        # Grads accumulated without sync reduce-scatter outside backward, as a
        # pipeline stage does after its last backward: after automatic
        # finalization at each backward's end, and in manual mode, where
        # backward finishes nothing itself.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        for manual, async_op in ((False, True), (True, False)):
            with self.subTest(manual=manual, async_op=async_op):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
                ).to(device_type)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )
                model.set_manual_backward_finalization(manual)
                model.set_requires_gradient_sync(False)
                model.set_reshard_after_backward(False)
                torch.manual_seed(1 + self.rank)
                for _ in range(2):
                    x = torch.randn(4, 8, device=device_type)
                    model(x).sum().backward()
                    reference(x).sum().backward()
                model.set_requires_gradient_sync(True)
                handle = model.finalize_backward(async_op=async_op)
                if async_op:
                    with self.assertRaisesRegex(RuntimeError, "wait on the previous"):
                        model.finalize_backward()
                    handle.wait()
                else:
                    self.assertIsNone(handle)
                for param, ref_param in zip(model.parameters(), reference.parameters()):
                    expected = ref_param.grad.clone()
                    dist.all_reduce(expected, op=dist.ReduceOp.AVG)
                    self.assertEqual(
                        param.grad,
                        expected_shard(
                            expected, rank=self.rank, world_size=self.world_size
                        ),
                    )

    @skip_if_lt_x_gpu(2)
    def test_defer_post_backward(self):
        # A weight grad computed after its layer's backward, as with
        # TransformerEngine's delay_wgrad_compute: the layer's bucket defers its
        # post-backward, which would otherwise reduce-scatter before that grad
        # exists, until finish_deferred_backward. Once with sync and
        # reshard-after-forward, once with a backward without sync first.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        for reshard_after_forward, no_sync in ((True, False), (False, True)):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                torch.manual_seed(0)
                model = _DelayedWgradModel(device=device_type)
                reference = copy.deepcopy(model)
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            [f"{name}.*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            reshard_after_forward=reshard_after_forward,
                            defer_post_backward=name == "layer",
                        )
                        for name in ("inp", "layer", "out")
                    ],
                )
                model.on_wgrad = lambda model=model: model.finish_deferred_backward(
                    model.layer.weight
                )
                with self.assertRaisesRegex(ValueError, "does not defer"):
                    model.finish_deferred_backward(model.out.weight)
                torch.manual_seed(1 + self.rank)
                for step in range(2 if no_sync else 1):
                    model.set_requires_gradient_sync(step == int(no_sync))
                    x = torch.randn(4, 8, device=device_type)
                    model(x).sum().backward()
                    reference(x).sum().backward()
                for param, ref_param in zip(model.parameters(), reference.parameters()):
                    expected = ref_param.grad.clone()
                    dist.all_reduce(expected, op=dist.ReduceOp.AVG)
                    self.assertEqual(
                        param.grad,
                        expected_shard(
                            expected, rank=self.rank, world_size=self.world_size
                        ),
                    )
        # A syncing backward that ends with the bucket unfinished raises instead
        # of reduce-scattering without the late weight grad.
        model.on_wgrad = lambda: None
        with self.assertRaisesRegex(RuntimeError, "finish_deferred_backward"):
            model(torch.randn(4, 8, device=device_type)).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_unshard(self):
        # A schedule that runs a layer's computation directly, bypassing the
        # forward hooks that gather its bucket, unshards every bucket first, as
        # Megatron-LM's EP overlap schedule does. It also splits each backward
        # into two calls, so backward finalization is manual: the buckets keep
        # their grads until finalize_backward after the last microbatch. Two
        # steps, so the second unshard gathers the updated shards, which its
        # outputs check. The first layer's bucket has a gradient bucket, which
        # unshard() sets up in place of its bypassed pre-backward hook, so it
        # reduces without copying.
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 6)
        ).to(device_type)
        reference = copy.deepcopy(model)
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["0.*"],
                    placement_fn=make_bucketed_block_placement_fn(
                        dims=(0,), blocks_per_rank=(1,) * self.world_size
                    ),
                    mesh=mesh,
                    reshard_after_forward=False,
                    gradient_bucket=True,
                ),
                BucketSpec(
                    ["2.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                ),
            ],
        )
        model.set_manual_backward_finalization(True)
        optimizer = make_test_sgd(model.parameters(), lr=0.1)
        reference_optimizer = make_test_sgd(reference.parameters(), lr=0.1)
        torch.manual_seed(1 + self.rank)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", "FlexShard bucket .* gradient_bucket=True")
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
                model.unshard()
                for microbatch in range(2):
                    model.set_requires_gradient_sync(microbatch == 1)
                    x = torch.randn(4, 8, device=device_type)
                    # The first layer's forward hooks never run.
                    hidden = torch.relu(
                        torch.nn.functional.linear(x, model[0].weight, model[0].bias)
                    )
                    detached = hidden.detach().requires_grad_()
                    output = model[2](detached)
                    reference_output = reference(x)
                    self.assertEqual(output, reference_output)
                    output.sum().backward()
                    hidden.backward(detached.grad)
                    reference_output.sum().backward()
                model.finalize_backward()
                _average_reference_grads(reference)
                optimizer.step()
                reference_optimizer.step()

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
        # every rank, and routing each rank to the other expert, so kept and
        # fresh grads add up, and the third syncing with the side
        # bucket's outputs unused on rank 1.
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
    def test_kept_grads_accumulate_in_param_grad_dtype(self):
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        model = _ReusedWeights(device=device_type, dtype=torch.bfloat16)
        model.b.weight.grad_dtype = torch.float32
        flex_shard_cuda(model, mesh)
        model.set_reshard_after_backward(False)
        x = torch.ones(1, 8, dtype=torch.bfloat16, device=device_type)

        model.set_requires_gradient_sync(False)
        model(x).backward()
        # Like FSDP2, a's kept grad stays in its bf16 grad dtype, not the
        # bucket's fp32 reduce dtype promoted by b's grad_dtype.
        self.assertEqual(model.a._parameters["weight"].grad.dtype, torch.bfloat16)
        model.set_requires_gradient_sync(True)
        model(x).backward()
        # b's two uses sum in its bf16 compute dtype while its upcast is
        # deferred, as in FSDP2, so each backward adds 256 rather than 257.
        for module, grad_value, grad_dtype in (
            (model.a, 512.0, torch.bfloat16),
            (model.b, 512.0, torch.float32),
        ):
            grad = module._parameters["weight"].grad
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

    @skip_if_lt_x_gpu(2)
    def test_pre_backward_and_post_reduce_hooks(self):
        # Kernels that add weight grads into main_grad and give autograd none
        # (fused gradient accumulation) need the grad allocated before their
        # backward. The hooks do what a trainer would: the pre-backward hook
        # allocates each unsharded param's grad and aliases it as main_grad,
        # the fused weight grads and the bias's autograd grads accumulate there
        # over two backwards without sync, and the post-reduce hook drops the
        # alias once the third reduce-scatters them. The first bucket reshards
        # after forward, so its backward re-gathers before the hook runs.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        for dtype, reduce_dtype in (
            (torch.float32, None),
            (torch.bfloat16, torch.float32),
        ):
            grad_dtype = reduce_dtype or dtype
            calls = []

            def alias_main_grads(named_params, grad_dtype=grad_dtype):
                calls.append(("pre_backward", [fqn for fqn, _ in named_params]))
                for _, param in named_params:
                    if param.grad is None:
                        param.grad = torch.zeros(
                            param.shape, dtype=grad_dtype, device=param.device
                        )
                    param.main_grad = param.grad

            def drop_main_grads(named_params):
                calls.append(("post_reduce", [fqn for fqn, _ in named_params]))
                for _, param in named_params:
                    self.assertIsNone(param.grad)
                    del param.main_grad

            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                model = torch.nn.Sequential(
                    _FusedWgradLinear(device=device_type, dtype=dtype),
                    _FusedWgradLinear(device=device_type, dtype=dtype),
                )
                reference = copy.deepcopy(model)
                # As Megatron-LM's wrapper does, so the local shards keep
                # grads in the accumulation dtype too.
                for param in (*model.parameters(), *reference.parameters()):
                    param.grad_dtype = grad_dtype
                for layer in reference:
                    layer.weight.main_grad = torch.zeros_like(
                        layer.weight, dtype=grad_dtype
                    )
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            [f"{idx}.*"],
                            placement_fn=per_param_placements,
                            mesh=mesh,
                            mp_policy=MixedPrecisionPolicy(reduce_dtype=reduce_dtype),
                            reshard_after_forward=idx == 0,
                            pre_backward_hook=alias_main_grads,
                            post_reduce_hook=drop_main_grads,
                        )
                        for idx in range(2)
                    ],
                )
                torch.manual_seed(1 + self.rank)
                for idx in range(3):
                    x = torch.randn(4, 8, device=device_type, dtype=dtype)
                    model.set_requires_gradient_sync(idx == 2)
                    model(x).float().sum().backward()
                    reference(x).float().sum().backward()
                    for layer in model:
                        weight = layer.weights_seen[-1]
                        if idx < 2:
                            self.assertIs(weight.main_grad, weight.grad)
                            self.assertEqual(weight.grad.dtype, grad_dtype)
                        else:
                            self.assertIsNone(weight.grad)
                            self.assertFalse(hasattr(weight, "main_grad"))
                fqns = [["1.weight", "1.bias"], ["0.weight", "0.bias"]]
                self.assertEqual(
                    calls,
                    [("pre_backward", fqns[0]), ("pre_backward", fqns[1])] * 2
                    + [
                        ("pre_backward", fqns[0]),
                        ("post_reduce", fqns[0]),
                        ("pre_backward", fqns[1]),
                        ("post_reduce", fqns[1]),
                    ],
                )
                for layer in reference:
                    layer.weight.grad = layer.weight.main_grad
                _average_reference_grads(reference)
                check_flex_shard_parity(
                    self, reference, model, self.rank, self.world_size
                )

    @skip_if_lt_x_gpu(2)
    def test_tied_weights_in_one_bucket(self):
        # Registered tying (output.weight is tok_embeddings.weight), which FSDP2
        # supports within one FSDP group. One bucket holds every name of the
        # shared parameter, so FlexShard hooks it on the root, around the
        # per-layer buckets: both names swap together, both uses read the
        # gathered weight, and the grads of both reduce once, with
        # reshard-after-forward on or off. Each step runs a backward without
        # sync first.
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        for reshard_after_forward in (False, True):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                torch.manual_seed(0)
                args, model = make_transformer_model(
                    device=device_type.type, n_layers=2, weight_tying=True
                )
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                reference = copy.deepcopy(model)
                spec = dict(
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=reshard_after_forward,
                )
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            [
                                "tok_embeddings.*",
                                "pos_embeddings.*",
                                "norm.*",
                                "output.*",
                            ],
                            **spec,
                        ),
                        *(
                            BucketSpec([f"layers.{idx}.*"], **spec)
                            for idx in range(args.n_layers)
                        ),
                    ],
                )
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                optim = make_test_sgd(model.parameters(), lr=0.1)
                ref_optim = make_test_sgd(reference.parameters(), lr=0.1)
                torch.manual_seed(1 + self.rank)
                for _ in range(2):
                    x = transformer_inputs(args, batch_size=2, device=device_type)
                    for sync in (False, True):
                        model.set_requires_gradient_sync(sync)
                        model(x).sum().backward()
                        reference(x).sum().backward()
                    _average_reference_grads(reference)
                    check_flex_shard_parity(
                        self, reference, model, self.rank, self.world_size
                    )
                    optim.step()
                    ref_optim.step()
                    optim.zero_grad(set_to_none=True)
                    ref_optim.zero_grad(set_to_none=True)
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                layouts = get_flex_shard_global_layouts(model)
                self.assertIs(
                    layouts["output.weight"], layouts["tok_embeddings.weight"]
                )


if __name__ == "__main__":
    run_tests()
