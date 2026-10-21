# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expert parallelism for the routed experts: torchtitan's all-to-all token
dispatch, in plain PyTorch.

With expert parallelism of degree ``ep``, each group of ``ep`` ranks splits the
routed experts: the group's rank ``r`` holds experts ``[r * e, (r + 1) * e)``,
``e = E / ep``. ``dispatch`` sends the routed copies of each token to the ranks
holding their experts and ``combine`` brings the expert outputs back, each with
one all-to-all over the group. ``train.py`` then FSDP-shards each rank's experts
over the ranks holding the same experts (expert FSDP, EFSDP).

Shape suffixes as in model.py, plus R tokens that this rank's experts process,
e this rank's experts and G the ranks of its expert-parallel group.
"""

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from flex_shard import GlobalLayout, set_global_layout
from model import RoutedExperts


class _AllToAll(torch.autograd.Function):
    """``all_to_all_single`` along dim 0; backward sends the grads back."""

    @staticmethod
    def forward(ctx, x, output_splits, input_splits, group):
        ctx.output_splits, ctx.input_splits, ctx.group = (
            output_splits,
            input_splits,
            group,
        )
        out = x.new_empty(sum(output_splits), *x.shape[1:])
        dist.all_to_all_single(
            out, x.contiguous(), output_splits, input_splits, group=group
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        grad_x = grad.new_empty(sum(ctx.input_splits), *grad.shape[1:])
        dist.all_to_all_single(
            grad_x,
            grad.contiguous(),
            ctx.input_splits,
            ctx.output_splits,
            group=ctx.group,
        )
        return grad_x, None, None, None


class AllToAllDispatcher:
    """torchtitan's ``AllToAllTokenDispatcher``, op for op."""

    def __init__(self, group: dist.ProcessGroup):
        self.group = group
        self.size = dist.get_world_size(group)

    def dispatch(
        self, routed_ND: torch.Tensor, num_tokens_per_expert_E: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Send expert-sorted routed tokens to the ranks holding their experts.

        Returns the tokens this rank's experts process, grouped by expert, the
        count per expert, and what ``combine`` needs.
        """
        with torch.no_grad():
            # Row r: the tokens rank r sends to each of this rank's experts.
            received_Ge = torch.empty_like(num_tokens_per_expert_E).view(self.size, -1)
            dist.all_to_all_single(
                received_Ge,
                num_tokens_per_expert_E.view(self.size, -1),
                group=self.group,
            )
            input_splits = (
                num_tokens_per_expert_E.view(self.size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # The device-to-host copy waits for the counts.
            output_splits = received_Ge.sum(dim=1).to(torch.device("cpu"))
        input_splits, output_splits = input_splits.tolist(), output_splits.tolist()
        routed_RD = _AllToAll.apply(routed_ND, output_splits, input_splits, self.group)
        routed_RD, permuted_indices, num_tokens_per_local_expert_e = self._permute(
            routed_RD, received_Ge
        )
        return (
            routed_RD,
            num_tokens_per_local_expert_e,
            (permuted_indices, input_splits, output_splits),
        )

    def combine(self, routed_RD: torch.Tensor, metadata: tuple) -> torch.Tensor:
        """Return the expert outputs to the ranks of their tokens, in the order
        ``dispatch`` received the tokens in."""
        permuted_indices, input_splits, output_splits = metadata
        unpermuted_RD = routed_RD.new_empty(routed_RD.shape)
        unpermuted_RD[permuted_indices, :] = routed_RD
        return _AllToAll.apply(unpermuted_RD, input_splits, output_splits, self.group)

    def _permute(
        self, routed_RD: torch.Tensor, received_Ge: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Regroup the received tokens from (rank, expert) order to (expert,
        rank) order, so each expert's tokens are contiguous."""
        counts = received_Ge.reshape(-1)
        device = counts.device
        # Where each (rank, expert) segment starts in the received tokens.
        input_starts = (counts.cumsum(0) - counts).view_as(received_Ge)
        segment_lens = received_Ge.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)
        segment_ids = torch.arange(
            segment_lens.shape[0], device=device
        ).repeat_interleave(segment_lens, output_size=routed_RD.shape[0])
        output_starts = segment_lens.cumsum(0) - segment_lens
        permuted_indices = (
            input_starts[segment_ids]
            + torch.arange(segment_ids.shape[0], device=device)
            - output_starts[segment_ids]
        )
        return routed_RD[permuted_indices, :], permuted_indices, received_Ge.sum(0)


def apply_expert_parallel(model: nn.Module, ep_mesh: DeviceMesh) -> None:
    """Keep this rank's slice of each routed-expert weight and route tokens to
    the ranks holding their experts.

    Call on the meta-device model, before ``flex_shard``. Each slice declares
    where it sits in the full weight, so that FlexShard's initialization and
    checkpoints still see the full weights.
    """
    size, rank = ep_mesh.size(), ep_mesh.get_local_rank()
    dispatcher = AllToAllDispatcher(ep_mesh.get_group())
    for module in model.modules():
        if not isinstance(module, RoutedExperts):
            continue
        for linear in (module.w13, module.w2):
            weight = linear.weight
            if weight.shape[0] % size:
                raise ValueError(
                    f"{weight.shape[0]} experts don't split over {size} ranks"
                )
            local = nn.Parameter(
                weight.new_empty(weight.shape[0] // size, *weight.shape[1:])
            )
            set_global_layout(
                local,
                GlobalLayout(
                    global_shape=tuple(weight.shape),
                    global_offsets=(
                        (rank * local.shape[0],) + (0,) * (weight.ndim - 1),
                    ),
                    local_offsets=((0,) * weight.ndim,),
                    local_sizes=(tuple(local.shape),),
                ),
            )
            linear.weight = local
        module.dispatcher = dispatcher
