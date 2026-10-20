# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muon for parameters that hold batches of matrices, as DeepSeek V3's do.

Muon orthogonalizes each matrix's momentum with a Newton-Schulz iteration, so
it needs every matrix whole. ``train.py --optimizer muon`` stores each one
whole on one rank: every rank then updates its own matrices, without
communication.
"""

import math
from collections.abc import Callable

import torch
from torch import nn

Matrices = Callable[[torch.Tensor], list[torch.Tensor]]


def newton_schulz(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """Approximately orthogonalize each matrix of ``g [batch, rows, cols]``,
    in BF16, with Keller Jordan's quintic iteration."""
    a, b, c = coefficients
    x = g.to(torch.bfloat16, copy=True)
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.mT
    # Bring the spectral norm to at most 1.
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.mT
        x = torch.baddbmm(
            x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a
        )
    return x.mT if transposed else x


class Muon(torch.optim.Optimizer):
    """Muon with Nesterov momentum and decoupled weight decay.

    ``matrices`` maps each parameter to a function that splits a tensor laid
    out like it into ``[batch, rows, cols]`` views, the matrices Newton-Schulz
    orthogonalizes one by one. As Kimi K2 does, each matrix's step is scaled by
    ``0.2 * sqrt(max(rows, cols))`` to match AdamW's update size, so one
    learning rate fits both optimizers.
    """

    def __init__(
        self,
        params,
        matrices: dict[torch.Tensor, Matrices],
        *,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps
        )
        super().__init__(params, defaults)
        self.matrices = matrices

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for group in self.param_groups:
            lr, momentum = group["lr"], group["momentum"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buf = state["momentum_buffer"]
                buf.lerp_(param.grad, 1 - momentum)
                update = param.grad.lerp(buf, momentum)
                param.mul_(1 - lr * group["weight_decay"])
                if param.numel() == 0:
                    continue
                matrices = self.matrices[param]
                for param_view, update_view in zip(
                    matrices(param), matrices(update), strict=True
                ):
                    rows, cols = update_view.shape[-2:]
                    param_view.add_(
                        newton_schulz(update_view, group["ns_steps"]),
                        alpha=-lr * 0.2 * math.sqrt(max(rows, cols)),
                    )


def kimi_matrices(fqn: str, config) -> Matrices | None:
    """Kimi K2's Muon matrices for a DeepSeek V3 parameter, or ``None`` for
    parameters AdamW trains: embeddings, norms and the output projection.

    Query, key and value projections split per head and per role (no-rotary
    and rotary queries, latent and rotary keys, keys and values); routed
    experts split per expert; the gate and up projections are two matrices.
    """
    heads, nope, rope = config.n_heads, config.qk_nope_head_dim, config.qk_rope_head_dim
    if fqn.endswith("attention.wq.weight"):  # [heads * (nope + rope), dim]

        def per_head(t):
            t = t.view(heads, nope + rope, -1)
            return [t[:, :nope], t[:, nope:]]

        return per_head
    if fqn.endswith("attention.wkv_a.weight"):  # [kv_lora_rank + rope, dim]
        rank = config.kv_lora_rank
        return lambda t: [t[:rank].unsqueeze(0), t[rank:].unsqueeze(0)]
    if fqn.endswith("attention.wkv_b.weight"):  # [heads * (nope + v), kv_lora_rank]
        assert nope == config.v_head_dim
        return lambda t: [t.view(2 * heads, nope, -1)]
    if fqn.endswith("routed_experts.w13.weight"):  # [experts, 2, hidden, dim]
        return lambda t: [t.flatten(0, 1)]
    if fqn.endswith("routed_experts.w2.weight"):  # [experts, dim, hidden]
        return lambda t: [t]
    if fqn.endswith("w13.weight"):  # [2, hidden, dim]
        return lambda t: [t]
    if fqn.endswith(("attention.wo.weight", "w2.weight", "router.gate.weight")):
        return lambda t: [t.unsqueeze(0)]
    return None


def muon_parameters(model: nn.Module) -> dict[torch.Tensor, Matrices]:
    """The model's Muon parameters, each with its matrices."""
    return {
        param: matrices
        for fqn, param in model.named_parameters()
        if (matrices := kimi_matrices(fqn, model.config)) is not None
    }
