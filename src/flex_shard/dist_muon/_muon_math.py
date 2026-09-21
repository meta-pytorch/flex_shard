# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-only numerical kernels shared by native and DTensor DistMuon."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _adjust_muon_learning_rate(
    lr: float,
    adjust_lr_fn: str | None,
    compute_matrix_shape: torch.Size | tuple[int, ...],
) -> float:
    """Adjust Muon's learning rate for the matrix aspect ratio."""
    rows, columns = compute_matrix_shape[-2:]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    elif adjust_lr_fn == "spectral_unclamped":
        ratio = math.sqrt(rows / columns)
    else:
        raise ValueError(f"unsupported adjust_lr_fn {adjust_lr_fn!r}")
    return lr * ratio


def _prepare_muon_input(
    gradient: Tensor,
    momentum_buffer: Tensor,
    *,
    momentum: float,
    nesterov: bool,
    out: Tensor,
) -> Tensor:
    """Update momentum and prepare the Tensor passed to Muon computation."""
    momentum_buffer.lerp_(gradient, 1 - momentum)
    if nesterov:
        torch.lerp(
            gradient,
            momentum_buffer,
            momentum,
            out=out,
        )
    else:
        out.copy_(momentum_buffer)
    return out


def _compute_muon_direction(
    prepared: Tensor,
    *,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    out: Tensor,
) -> Tensor:
    """Compute Muon's approximate orthogonal update direction."""
    direction = _zeropower_via_newtonschulz(
        prepared,
        ns_coefficients=ns_coefficients,
        ns_steps=ns_steps,
        eps=eps,
    )
    out.copy_(direction)
    return out


def _apply_muon_update(
    parameter: Tensor,
    direction: Tensor,
    *,
    lr: float,
    weight_decay: float,
    adjust_lr_fn: str | None,
    compute_matrix_shape: torch.Size | tuple[int, ...],
) -> Tensor:
    """Apply decoupled weight decay and a computed Muon direction."""
    adjusted_lr = _adjust_muon_learning_rate(
        lr,
        adjust_lr_fn,
        compute_matrix_shape,
    )
    parameter.mul_(1 - lr * weight_decay)
    parameter.add_(direction, alpha=-adjusted_lr)
    return parameter


def _zeropower_via_newtonschulz(
    update: Tensor,
    *,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> Tensor:
    """Compute Muon's approximate polar factor without optimizer state."""
    a, b, c = ns_coefficients
    result = update.to(dtype=torch.bfloat16, copy=True)
    transposed = result.shape[-2] > result.shape[-1]
    if transposed:
        result = result.transpose(-2, -1)
    result.div_(result.norm(dim=(-2, -1), keepdim=True).clamp_min(eps))

    if result.ndim == 2:
        for _ in range(ns_steps):
            gram = result @ result.T
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            result = torch.addmm(result, gram_update, result, beta=a)
    else:
        original_shape = result.shape
        matrices = result.reshape(-1, *original_shape[-2:])
        # Batched kernels and independent matrix calls can use different BF16
        # reduction orders.
        for _ in range(ns_steps):
            gram = matrices @ matrices.transpose(-2, -1)
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            matrices = torch.baddbmm(matrices, gram_update, matrices, beta=a)
        result = matrices.reshape(original_shape)

    return result.transpose(-2, -1) if transposed else result
