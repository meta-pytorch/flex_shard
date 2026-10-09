# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard ``Shard`` buckets against FSDP2's ``fully_shard``, bit for bit."""

import copy
import itertools
from collections.abc import Callable, Iterator

import torch
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
from ..custom_placements.shard import Shard

device_type = torch.device(get_devtype())

_VOCAB = 11
# Odd, so every Shard(0) parameter below pads its last chunks.
_DIM = 13
# Divisible by the world size, as FSDP2 requires for Shard(1).
_HIDDEN = 8
_BATCH = 6
# Param/reduce dtypes, gradient divide factors (None averages over the mesh;
# torchtitan sets 1.0 and scales the loss), and (microbatches, no_sync).
_DTYPE_CONFIGS = [
    (None, None),
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, None),
]
_DIVIDE_FACTORS = [None, 1.0]
_ACCUMULATIONS = [(1, False), (2, False), (2, True)]


def _configs() -> Iterator[tuple[tuple, str]]:
    """Yield every combination of the above, with a description."""
    for (param_dtype, reduce_dtype), divide_factor, (
        microbatches,
        no_sync,
    ) in itertools.product(_DTYPE_CONFIGS, _DIVIDE_FACTORS, _ACCUMULATIONS):
        config = (param_dtype, reduce_dtype, divide_factor, microbatches, no_sync)
        context = (
            f"param_dtype={param_dtype} reduce_dtype={reduce_dtype} "
            f"divide_factor={divide_factor} microbatches={microbatches} "
            f"no_sync={no_sync}"
        )
        yield config, context


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
        )

    flex_shard(
        model,
        buckets=[
            bucket(["tok_embeddings"], True),
            *(bucket([f"layers.{i}"], True) for i in range(len(model.layers))),
            bucket(["norm", "lm_head"], False),
        ],
    )


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


class _FullyShardParity(FSDPTest):
    def _assert_bitwise_equal(
        self, expected: torch.Tensor | None, actual: torch.Tensor | None, context: str
    ) -> None:
        if expected is None or actual is None:
            self.assertIs(expected, actual, msg=context)
            return
        self.assertEqual(expected.dtype, actual.dtype, msg=context)
        self.assertEqual(expected.shape, actual.shape, msg=context)
        self.assertTrue(torch.equal(expected, actual), msg=context)

    def _assert_same_history(
        self,
        expected_history: list[dict[str, object]],
        actual_history: list[dict[str, object]],
        context: str,
    ) -> None:
        for step, (expected, actual) in enumerate(
            zip(expected_history, actual_history, strict=True)
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

    def _run_backend(
        self,
        apply: Callable[..., None],
        reference: _Decoder,
        mesh,
        config: tuple,
    ) -> list[dict[str, object]]:
        param_dtype, reduce_dtype, divide_factor, microbatches, no_sync = config
        model = copy.deepcopy(reference).to(device_type)
        apply(model, mesh, param_dtype, reduce_dtype, divide_factor)
        return _train(
            model,
            rank=self.rank,
            steps=3,
            microbatches=microbatches,
            no_sync=no_sync,
        )


class TestFlexShardMatchesFullyShard(_FullyShardParity):
    @property
    def world_size(self) -> int:
        # A sum of two values does not depend on their order, so buckets match
        # fully_shard bit for bit at two ranks whatever their parameter order.
        return 2

    @skip_if_lt_x_gpu(2)
    def test_matches_fully_shard(self):
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        torch.manual_seed(0)
        reference = _Decoder()
        for config, context in _configs():
            fsdp2_history = self._run_backend(_apply_fsdp2, reference, mesh, config)
            flex_history = self._run_backend(_apply_flex_shard, reference, mesh, config)
            self._assert_same_history(fsdp2_history, flex_history, context)


if __name__ == "__main__":
    run_tests()
