# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``fsdp2_compatible`` buckets against FSDP2's ``fully_shard``, bit for bit."""

import contextlib
import copy
import itertools
import math
from collections import defaultdict
from collections.abc import Callable, Iterator
from unittest import mock

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FSDPModule,
    fully_shard,
    MixedPrecisionPolicy as FSDPMixedPrecisionPolicy,
)
from torch.distributed.tensor import Shard as DTensorShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests

from .. import BucketSpec, flex_shard, MixedPrecisionPolicy
from ..custom_placements import shard as shard_module
from ..custom_placements.shard import Shard

device_type = torch.device(get_devtype())

_VOCAB = 11
# Uneven over four ranks: every Shard(0) parameter below pads its last chunks.
_DIM = 13
# Even over four ranks, as FSDP2 requires for Shard(1).
_HIDDEN = 8
_BATCH = 6


class _StackedLinear(nn.Module):
    """Two projections stacked as torchtitan stores them: weight ``[N, F, D]``
    and bias ``[N, F]``, sharded on their matrix rows, ``Shard(1)``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(2, _HIDDEN, _DIM))
        self.bias = nn.Parameter(torch.randn(2, _HIDDEN))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.einsum("bd,nfd->bnf", x, self.weight) + self.bias).sum(1)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Registered before the children, but FSDP2's post-order walk takes it
        # after their parameters.
        self.scale = nn.Parameter(torch.randn(_DIM))
        self.stacked = _StackedLinear()
        self.proj = nn.Linear(_HIDDEN, _DIM)
        # Never gets a grad, so FSDP2 leaves it out of the reduce-scatter. Its
        # three rows leave the last rank an empty shard.
        self.unused = nn.Linear(_DIM, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(torch.relu(self.stacked(x * self.scale)))


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(_VOCAB, _DIM)
        self.layers = nn.ModuleList(_Block() for _ in range(2))
        self.norm = nn.LayerNorm(_DIM)
        self.lm_head = nn.Linear(_DIM, _VOCAB, bias=False)
        self.norm.bias.requires_grad_(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm(h))


def _shard_dim(fqn: str) -> int:
    return 1 if ".stacked." in fqn else 0


def _apply_fsdp2(
    model: _Decoder,
    mesh,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    divide_factor: float | None,
) -> None:
    """Group the decoder as torchtitan's ``apply_fsdp_to_decoder`` does."""
    dims = {param: _shard_dim(fqn) for fqn, param in model.named_parameters()}
    fsdp_config = {
        "mesh": mesh,
        "mp_policy": FSDPMixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=False,
        ),
    }
    fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=True)
    fully_shard([model.norm, model.lm_head], **fsdp_config, reshard_after_forward=False)
    for layer in model.layers:
        fully_shard(
            layer,
            **fsdp_config,
            reshard_after_forward=True,
            shard_placement_fn=lambda param: DTensorShard(dims[param]),
        )
    fully_shard(model, **fsdp_config)
    if divide_factor is not None:
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.set_gradient_divide_factor(divide_factor)


def _apply_flex_shard(
    model: _Decoder,
    mesh,
    param_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    divide_factor: float | None,
) -> None:
    def placement_fn(named_params, mesh):
        del mesh
        return {fqn: (Shard(_shard_dim(fqn)),) for fqn, _ in named_params}

    def bucket(patterns: list[str], reshard_after_forward: bool) -> BucketSpec:
        return BucketSpec(
            patterns,
            placement_fn=placement_fn,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype
            ),
            gradient_divide_factor=divide_factor,
            reshard_after_forward=reshard_after_forward,
            fsdp2_compatible=True,
        )

    flex_shard(
        model,
        buckets=[
            bucket(["tok_embeddings"], True),
            *(bucket([f"layers.{i}"], True) for i in range(len(model.layers))),
            bucket(["norm", "lm_head"], False),
        ],
    )


@contextlib.contextmanager
def _record_collective_inputs() -> Iterator[dict[str, list[torch.Tensor]]]:
    """Record every all-gather and reduce-scatter input, by collective.

    FSDP2 calls ``dist.reduce_scatter_single``; FlexShard calls
    ``dist.reduce_scatter_tensor``, which forwards to c10d's module global.
    """
    recorded: dict[str, list[torch.Tensor]] = defaultdict(list)
    all_gather_single = dist.all_gather_single
    reduce_scatter_single = c10d.reduce_scatter_single

    def record_all_gather(output, input, *args, **kwargs):
        recorded["all_gather"].append(input.detach().clone())
        return all_gather_single(output, input, *args, **kwargs)

    def record_reduce_scatter(output, input, *args, **kwargs):
        recorded["reduce_scatter"].append(input.detach().clone())
        return reduce_scatter_single(output, input, *args, **kwargs)

    patches = [
        (dist, "all_gather_single", record_all_gather),
        (dist, "reduce_scatter_single", record_reduce_scatter),
        (c10d, "reduce_scatter_single", record_reduce_scatter),
    ]
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in patches]
    try:
        for owner, name, patched in patches:
            setattr(owner, name, patched)
        yield recorded
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _train(
    model: nn.Module,
    *,
    rank: int,
    steps: int,
    microbatches: int,
    no_sync: bool,
) -> list[dict[str, object]]:
    """Train with AdamW; return per-step losses, local grads and params."""
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=True)
    history: list[dict[str, object]] = []
    for step in range(steps):
        losses = []
        for microbatch in range(microbatches):
            if no_sync:
                model.set_requires_gradient_sync(microbatch == microbatches - 1)
            generator = torch.Generator().manual_seed(
                10_000 * step + 100 * microbatch + rank
            )
            tokens, targets = (
                torch.randint(_VOCAB, (_BATCH,), generator=generator).to(device_type)
                for _ in range(2)
            )
            loss = F.cross_entropy(model(tokens).float(), targets)
            loss.backward()
            losses.append(loss.detach())
        grads = {
            fqn: None if param.grad is None else _local(param.grad).detach().clone()
            for fqn, param in model.named_parameters()
        }
        optim.step()
        optim.zero_grad()
        params = {
            fqn: _local(param).detach().clone()
            for fqn, param in model.named_parameters()
        }
        history.append({"losses": losses, "grads": grads, "params": params})
    return history


def _all_gather_copy_out_reference(
    out: list[torch.Tensor],
    src: torch.Tensor,
    split_sizes: list[int],
    outer_sizes: list[int],
    num_chunks: int,
) -> None:
    """The composite kernel of ``fsdp::_all_gather_copy_out_`` from
    pytorch/pytorch#197701, for PyTorch builds without it."""
    chunks = src.view(num_chunks, -1)
    offset = 0
    for output, split_size, outer_size in zip(
        out, split_sizes, outer_sizes, strict=True
    ):
        inner_size = split_size // outer_size
        output.view(outer_size, num_chunks, inner_size).copy_(
            chunks.narrow(1, offset, split_size)
            .view(num_chunks, outer_size, inner_size)
            .transpose(0, 1)
        )
        offset += split_size


def _reduce_scatter_copy_in_reference(
    out: torch.Tensor,
    tensors: list[torch.Tensor],
    num_leading_dims: list[int],
    num_chunks: int,
) -> torch.Tensor:
    """The composite kernel of ``fsdp::_reduce_scatter_copy_in_`` from
    pytorch/pytorch#197701, for PyTorch builds without it."""
    chunks = out.view(num_chunks, -1)
    if all(dim == 0 for dim in num_leading_dims):
        torch._chunk_cat(tensors, 0, num_chunks, out=chunks)
        return out
    offset = 0
    for tensor, dim in zip(tensors, num_leading_dims, strict=True):
        outer_size = math.prod(tensor.shape[:dim])
        inner_size = -(-tensor.shape[dim] // num_chunks) * math.prod(
            tensor.shape[dim + 1 :]
        )
        chunk = chunks.narrow(1, offset, outer_size * inner_size)
        if dim == 0:
            chunk.copy_(torch._chunk_cat([tensor], 0, num_chunks))
        else:
            chunk.view(num_chunks, outer_size, inner_size).copy_(
                tensor.view(outer_size, num_chunks, inner_size).transpose(0, 1)
            )
        offset += outer_size * inner_size
    return out


class TestFlexShardFSDP2Parity(FSDPTest):
    @property
    def world_size(self) -> int:
        # Sums of four values make the reduction order observable.
        return 4

    def _assert_same_inputs(
        self,
        expected: list[torch.Tensor],
        actual: list[torch.Tensor],
        context: str,
    ) -> None:
        # Pair the collectives by size, in issue order, in case the backends
        # prefetch in different orders.
        def by_numel(inputs):
            grouped = defaultdict(list)
            for tensor in inputs:
                grouped[tensor.numel()].append(tensor)
            return grouped

        expected_by_numel, actual_by_numel = by_numel(expected), by_numel(actual)
        self.assertEqual(
            sorted(expected_by_numel), sorted(actual_by_numel), msg=context
        )
        for numel, tensors in expected_by_numel.items():
            self.assertEqual(len(tensors), len(actual_by_numel[numel]), msg=context)
            for expected_tensor, actual_tensor in zip(
                tensors, actual_by_numel[numel], strict=True
            ):
                self.assertEqual(
                    expected_tensor.dtype, actual_tensor.dtype, msg=context
                )
                self.assertTrue(
                    torch.equal(expected_tensor, actual_tensor),
                    msg=f"{context}: {numel}-element inputs differ",
                )

    def _assert_bitwise_equal(
        self, expected: torch.Tensor | None, actual: torch.Tensor | None, context: str
    ) -> None:
        if expected is None or actual is None:
            self.assertIs(expected, actual, msg=context)
            return
        self.assertEqual(expected.dtype, actual.dtype, msg=context)
        self.assertEqual(expected.shape, actual.shape, msg=context)
        self.assertTrue(torch.equal(expected, actual), msg=context)

    def _run_backend(
        self,
        apply: Callable[..., None],
        reference: _Decoder,
        mesh,
        config: tuple,
        *,
        microbatches: int,
        no_sync: bool,
    ):
        param_dtype, reduce_dtype, divide_factor = config
        model = copy.deepcopy(reference).to(device_type)
        apply(model, mesh, param_dtype, reduce_dtype, divide_factor)
        with _record_collective_inputs() as recorded:
            history = _train(
                model,
                rank=self.rank,
                steps=3,
                microbatches=microbatches,
                no_sync=no_sync,
            )
        return history, recorded

    @skip_if_lt_x_gpu(4)
    def test_matches_fully_shard(self):
        with mock.patch.object(
            shard_module, "_native_collective_copy_ops", return_value=None
        ):
            self._check_matches_fully_shard()

    @skip_if_lt_x_gpu(4)
    def test_matches_fully_shard_with_native_copies(self):
        """Shard(1) params copied directly between their layout and the
        collective buffers fill the same buffers as FSDP2's default copies."""
        native_ops = shard_module._native_collective_copy_ops() or (
            _all_gather_copy_out_reference,
            _reduce_scatter_copy_in_reference,
        )
        counted_ops = tuple(mock.Mock(wraps=op) for op in native_ops)
        with mock.patch.object(
            shard_module, "_native_collective_copy_ops", return_value=counted_ops
        ):
            self._check_matches_fully_shard()
        for op in counted_ops:
            self.assertGreater(op.call_count, 0)

    def _check_matches_fully_shard(self):
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        reference = _Decoder()
        dtype_configs = [
            (None, None),
            (torch.bfloat16, torch.float32),
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, None),
        ]
        # None averages over the mesh; torchtitan sets 1.0 and scales the loss.
        divide_factors = [None, 1.0]
        accumulations = [(1, False), (2, False), (2, True)]
        for (param_dtype, reduce_dtype), divide_factor, (
            microbatches,
            no_sync,
        ) in itertools.product(dtype_configs, divide_factors, accumulations):
            config = (param_dtype, reduce_dtype, divide_factor)
            context = (
                f"param_dtype={param_dtype} reduce_dtype={reduce_dtype} "
                f"divide_factor={divide_factor} microbatches={microbatches} "
                f"no_sync={no_sync}"
            )
            fsdp2_history, fsdp2_recorded = self._run_backend(
                _apply_fsdp2,
                reference,
                mesh,
                config,
                microbatches=microbatches,
                no_sync=no_sync,
            )
            flex_history, flex_recorded = self._run_backend(
                _apply_flex_shard,
                reference,
                mesh,
                config,
                microbatches=microbatches,
                no_sync=no_sync,
            )
            for collective in ("all_gather", "reduce_scatter"):
                self.assertTrue(fsdp2_recorded[collective], msg=context)
                self._assert_same_inputs(
                    fsdp2_recorded[collective],
                    flex_recorded[collective],
                    f"{context} {collective}",
                )
            for step, (expected, actual) in enumerate(
                zip(fsdp2_history, flex_history, strict=True)
            ):
                for expected_loss, actual_loss in zip(
                    expected["losses"], actual["losses"], strict=True
                ):
                    self._assert_bitwise_equal(
                        expected_loss, actual_loss, f"{context} step {step} loss"
                    )
                for kind in ("grads", "params"):
                    self.assertEqual(list(expected[kind]), list(actual[kind]))
                    for fqn in expected[kind]:
                        self._assert_bitwise_equal(
                            expected[kind][fqn],
                            actual[kind][fqn],
                            f"{context} step {step} {kind} {fqn}",
                        )


if __name__ == "__main__":
    run_tests()
