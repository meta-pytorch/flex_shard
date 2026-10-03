#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradient divide factors and expert parallelism.

With expert parallelism, an expert's gradient already includes the tokens its
expert-parallel peers routed to it, so expert buckets on the expert
data-parallel mesh divide by the dense data-parallel size, as Megatron and
torchtitan do with FSDP2's gradient divide factor.
"""

import copy

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard as DTensorShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import BucketSpec, flex_shard
from ..custom_placements.shard import per_param_placements
from ..custom_placements.utils import _gradient_reduce_scatter_op
from ..flex_shard.checkpoint import get_flex_shard_global_layouts
from ..flex_shard.placement_contract import GlobalLayout
from ..layout_adapters.dtensor import dtensor_to_global_layout
from .common import expected_shard, single_rank_cuda_mesh

device_type = torch.device(get_devtype())

_DIM, _HIDDEN, _NUM_EXPERTS, _TOKENS = 8, 16, 4, 16


class TestGradientReduceScatterOp(TestCase):
    def test_sum_never_divides(self) -> None:
        self.assertEqual(
            _gradient_reduce_scatter_op(dist.ReduceOp.SUM, None, 4, torch.float32),
            (dist.ReduceOp.SUM, None, None),
        )

    def test_avg_defaults_to_the_group_size(self) -> None:
        for factor in (None, 4):
            self.assertEqual(
                _gradient_reduce_scatter_op(
                    dist.ReduceOp.AVG, factor, 4, torch.bfloat16
                ),
                (dist.ReduceOp.AVG, None, None),
            )

    def test_other_factors_premultiply_in_fp32_and_bf16(self) -> None:
        for dtype in (torch.float32, torch.bfloat16):
            op, pre_factor, post_factor = _gradient_reduce_scatter_op(
                dist.ReduceOp.AVG, 8, 4, dtype
            )
            self.assertEqual(op, dist.ReduceOp.PREMUL_SUM)
            self.assertIsNone(pre_factor)
            self.assertIsNone(post_factor)

    def test_fp16_divides_before_and_after_the_sum(self) -> None:
        self.assertEqual(
            _gradient_reduce_scatter_op(dist.ReduceOp.AVG, 8, 4, torch.float16),
            (dist.ReduceOp.SUM, 4, 2.0),
        )
        self.assertEqual(
            _gradient_reduce_scatter_op(dist.ReduceOp.AVG, 2.5, 4, torch.float16),
            (dist.ReduceOp.SUM, None, 2.5),
        )

    def test_one_rank_group_sums(self) -> None:
        self.assertEqual(
            _gradient_reduce_scatter_op(dist.ReduceOp.AVG, None, 1, torch.float32),
            (dist.ReduceOp.SUM, None, None),
        )
        self.assertEqual(
            _gradient_reduce_scatter_op(dist.ReduceOp.AVG, 4, 1, torch.float32),
            (dist.ReduceOp.SUM, None, 4),
        )


def _mlp(dtype: torch.dtype = torch.float32) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 6)).to(
        device_type, dtype
    )


def _bucket(mesh, **kwargs) -> BucketSpec:
    return BucketSpec(
        ["*"],
        placement_fn=per_param_placements,
        mesh=mesh,
        reshard_after_forward=False,
        **kwargs,
    )


class TestGradientDivideFactorSingleRank(TestCase):
    def test_rejects_invalid_factors(self) -> None:
        with single_rank_cuda_mesh() as mesh:
            for factor in (0, -2.0, float("inf")):
                with self.assertRaisesRegex(ValueError, "positive number"):
                    flex_shard(
                        _mlp(), buckets=[_bucket(mesh, gradient_divide_factor=factor)]
                    )
            with self.assertRaisesRegex(ValueError, "requires gradient_reduce_op=AVG"):
                flex_shard(
                    _mlp(),
                    buckets=[
                        _bucket(
                            mesh,
                            gradient_reduce_op=dist.ReduceOp.SUM,
                            gradient_divide_factor=2,
                        )
                    ],
                )
            model = flex_shard(
                _mlp(), buckets=[_bucket(mesh, gradient_reduce_op=dist.ReduceOp.SUM)]
            )
            with self.assertRaisesRegex(ValueError, "requires gradient_reduce_op=AVG"):
                model.set_gradient_divide_factor(2)
            model.set_gradient_reduce_op(dist.ReduceOp.AVG)
            model.set_gradient_divide_factor(2)
            with self.assertRaisesRegex(ValueError, "requires gradient_reduce_op=AVG"):
                model.set_gradient_reduce_op(dist.ReduceOp.SUM)

    def test_one_rank_mesh_divides_by_the_factor(self) -> None:
        # A 1-rank mesh, e.g. expert data parallelism of size 1, still applies
        # the factor (as a division after a SUM, not NCCL's AVG).
        with single_rank_cuda_mesh() as mesh:
            model = _mlp()
            reference = copy.deepcopy(model)
            flex_shard(model, buckets=[_bucket(mesh, gradient_divide_factor=4)])
            x = torch.randn(4, 8, device=device_type)
            model(x).sum().backward()
            reference(x).sum().backward()
            for param, ref_param in zip(model.parameters(), reference.parameters()):
                self.assertEqual(param.grad, ref_param.grad / 4)


class TestGradientDivideFactor(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_reduced_gradient_is_the_sum_over_the_factor(self) -> None:
        mesh = init_device_mesh(device_type.type, (self.world_size,))
        cases = [
            # (param dtype, factor in BucketSpec, factor set later, no-sync)
            (torch.float32, None, None, False),
            (torch.float32, 2, None, False),
            (torch.float32, 8, None, False),
            (torch.float32, 3.0, None, True),
            (torch.float32, None, 4, False),
            (torch.bfloat16, 8, None, False),
            (torch.float16, 6, None, False),
        ]
        for dtype, factor, later_factor, no_sync in cases:
            with self.subTest(
                dtype=dtype, factor=factor, later_factor=later_factor, no_sync=no_sync
            ):
                model = _mlp(dtype)
                reference = copy.deepcopy(model)
                flex_shard(
                    model, buckets=[_bucket(mesh, gradient_divide_factor=factor)]
                )
                if later_factor is not None:
                    model.set_gradient_divide_factor(later_factor)
                torch.manual_seed(1 + self.rank)
                inputs = [
                    torch.randn(4, 8, device=device_type, dtype=dtype)
                    for _ in range(2 if no_sync else 1)
                ]
                for idx, x in enumerate(inputs):
                    model.set_requires_gradient_sync(idx == len(inputs) - 1)
                    model(x).sum().backward()
                    reference(x).sum().backward()
                divisor = later_factor or factor or self.world_size
                tolerance = {} if dtype == torch.float32 else dict(atol=2e-2, rtol=2e-2)
                for param, ref_param in zip(model.parameters(), reference.parameters()):
                    expected = ref_param.grad.float()
                    dist.all_reduce(expected)
                    expected = expected_shard(
                        expected / divisor, rank=self.rank, world_size=self.world_size
                    )
                    self.assertEqual(param.grad.float(), expected, **tolerance)


class _Experts(nn.Module):
    """Stacked experts; a token's expert id selects its weights."""

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        super().__init__()
        self.w1 = nn.Parameter(w1.clone())
        self.w2 = nn.Parameter(w2.clone())

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum("thd,td->th", self.w1[expert_ids], x).relu()
        return torch.einsum("tdh,th->td", self.w2[expert_ids], hidden)


def _dispatch_to_experts(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    experts: nn.Module,
    num_local_experts: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    """Send each token to the expert-parallel rank owning its expert and back."""
    owners = expert_ids // num_local_experts
    order = torch.argsort(owners, stable=True)
    send_counts = torch.bincount(owners, minlength=ep_group.size())
    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=ep_group)
    send_splits, recv_splits = send_counts.tolist(), recv_counts.tolist()
    recv_ids = expert_ids.new_empty(sum(recv_splits))
    dist.all_to_all_single(
        recv_ids, expert_ids[order], recv_splits, send_splits, group=ep_group
    )
    recv_x = dist_nn.all_to_all_single(
        x.new_empty(sum(recv_splits), x.shape[1]),
        x[order],
        recv_splits,
        send_splits,
        group=ep_group,
    )
    # This rank owns experts [ep_rank * L, (ep_rank + 1) * L).
    out = experts(recv_x, recv_ids % num_local_experts)
    back = dist_nn.all_to_all_single(
        out.new_empty(x.shape), out, send_splits, recv_splits, group=ep_group
    )
    return back[torch.argsort(order)]


class _ToyMoE(nn.Module):
    """Top-1 MoE; with ``ep_group``, ``experts`` holds this rank's experts only."""

    def __init__(
        self,
        experts: _Experts,
        num_local_experts: int = _NUM_EXPERTS,
        ep_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(_DIM, _DIM)
        self.router = nn.Linear(_DIM, _NUM_EXPERTS, bias=False)
        self.experts = experts
        self.proj_out = nn.Linear(_DIM, _DIM)
        self.num_local_experts = num_local_experts
        self.ep_group = ep_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.proj_in(x).relu()
        probs = self.router(hidden).softmax(dim=-1)
        expert_ids = probs.argmax(dim=-1)
        if self.ep_group is None:
            out = self.experts(hidden, expert_ids)
        else:
            out = _dispatch_to_experts(
                hidden, expert_ids, self.experts, self.num_local_experts, self.ep_group
            )
        return self.proj_out(out * probs.gather(1, expert_ids[:, None]))


def _rank_inputs(step: int, rank: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1000 * step + rank)
    return torch.randn(_TOKENS, _DIM, generator=generator).to(device_type)


class TestExpertParallelTraining(FSDPTest):
    """Dense params shard over all 4 ranks; experts are split over ep (2) and
    each rank's experts shard over efsdp (2), with a real all-to-all dispatch
    and different tokens on every rank. Each step matches a single-process
    reference over the global batch."""

    @property
    def world_size(self) -> int:
        return 4

    def _check_training(self, *, global_token_loss: bool, dtensor_experts: bool):
        sparse_mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("efsdp", "ep")
        )
        dp_mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("dp",)
        )
        efsdp_mesh, ep_mesh = sparse_mesh["efsdp"], sparse_mesh["ep"]
        num_local_experts = _NUM_EXPERTS // ep_mesh.size()
        ep_rank, efsdp_rank = ep_mesh.get_local_rank(), efsdp_mesh.get_local_rank()
        local_experts = slice(
            ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts
        )

        torch.manual_seed(0)
        w1 = torch.randn(_NUM_EXPERTS, _HIDDEN, _DIM, device=device_type) * 0.3
        w2 = torch.randn(_NUM_EXPERTS, _DIM, _HIDDEN, device=device_type) * 0.3
        reference = _ToyMoE(_Experts(w1, w2)).to(device_type)
        if dtensor_experts:
            # Experts start as EP-sharded DTensors, as torchtitan's are.
            model = _ToyMoE(_Experts(w1, w2), num_local_experts, ep_mesh.get_group())
            for name in ("w1", "w2"):
                full = getattr(model.experts, name).detach()
                setattr(
                    model.experts,
                    name,
                    nn.Parameter(distribute_tensor(full, ep_mesh, [DTensorShard(0)])),
                )
            dtensor_to_global_layout(model)
        else:
            model = _ToyMoE(
                _Experts(w1[local_experts], w2[local_experts]),
                num_local_experts,
                ep_mesh.get_group(),
            )
        model.to(device_type)
        for name, param in model.named_parameters():
            if not name.startswith("experts."):
                param.data.copy_(reference.get_parameter(name).detach())

        if global_token_loss:
            # torchtitan: sum everywhere; the loss divides by the global token count.
            dense_kwargs = expert_kwargs = dict(gradient_reduce_op=dist.ReduceOp.SUM)
        else:
            # Megatron: average; experts divide by the dense data-parallel size.
            dense_kwargs = {}
            expert_kwargs = dict(gradient_divide_factor=dp_mesh.size())
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["proj_in.*", "router.*", "proj_out.*"],
                    placement_fn=per_param_placements,
                    mesh=dp_mesh,
                    reshard_after_forward=False,
                    **dense_kwargs,
                ),
                BucketSpec(
                    ["experts.*"],
                    placement_fn=per_param_placements,
                    mesh=efsdp_mesh,
                    reshard_after_forward=True,
                    **expert_kwargs,
                ),
            ],
        )
        if dtensor_experts:
            # The declared EP layout composes with FlexShard's efsdp shard.
            expected = GlobalLayout(
                (_NUM_EXPERTS, _HIDDEN, _DIM),
                ((ep_rank * num_local_experts + efsdp_rank, 0, 0),),
                ((0, 0, 0),),
                ((1, _HIDDEN, _DIM),),
            )
            self.assertEqual(
                get_flex_shard_global_layouts(model)["experts.w1"], expected
            )

        optim = torch.optim.SGD(model.parameters(), lr=0.5)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.5)
        num_tokens = _TOKENS * self.world_size
        for step in range(3):
            out = model(_rank_inputs(step, self.rank))
            if global_token_loss:
                loss = out.square().mean(dim=-1).sum() / num_tokens
            else:
                loss = out.square().mean()
            loss.backward()
            ref_loss = sum(
                reference(_rank_inputs(step, rank)).square().mean()
                for rank in range(self.world_size)
            )
            (ref_loss / self.world_size).backward()

            ref_params = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                ref_param = ref_params[name]
                if name.startswith("experts."):
                    rank, world_size = efsdp_rank, efsdp_mesh.size()
                    want, want_grad = (
                        ref_param.detach()[local_experts],
                        ref_param.grad[local_experts],
                    )
                else:
                    rank, world_size = self.rank, dp_mesh.size()
                    want, want_grad = ref_param.detach(), ref_param.grad
                msg = f"{name} at step {step}"
                self.assertEqual(
                    param.detach(),
                    expected_shard(want, rank=rank, world_size=world_size),
                    msg=msg,
                )
                self.assertEqual(
                    param.grad,
                    expected_shard(want_grad, rank=rank, world_size=world_size),
                    msg=msg,
                )
            optim.step()
            ref_optim.step()
            optim.zero_grad()
            ref_optim.zero_grad()

    @skip_if_lt_x_gpu(4)
    def test_expert_buckets_divide_by_dense_data_parallel_size(self) -> None:
        self._check_training(global_token_loss=False, dtensor_experts=False)

    @skip_if_lt_x_gpu(4)
    def test_global_token_loss_with_dtensor_experts(self) -> None:
        self._check_training(global_token_loss=True, dtensor_experts=True)


if __name__ == "__main__":
    run_tests()
