# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise FP8 grouped GEMM for routed experts, on TorchAO's DeepGEMM backend.

FP8 forward and input gradient with 1x128 activation and 128x128 weight blocks,
and a BF16 weight gradient. The weights are either a ``BlockwiseFp8Weight`` that
FlexShard's FP8 all-gather produced, used as gathered, or high-precision weights
quantized on every call, as FSDP2's all-gathered ones are. Both quantize the
same blocks, so given the same weights both compute the same bytes.
"""

from __future__ import annotations

from typing import Any

import torch
from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.deepgemm_grouped_kernels import (
    can_use_deepgemm_grouped_training,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    build_deepgemm_grouped_offset_plan,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    _scaling_type_value,
    BLOCKWISE_128X128_SCALING_TYPE,
    BLOCKWISE_1X128_SCALING_TYPE,
    triton_fp8_blockwise_act_quant_lhs,
)
from torchao.prototype.moe_training.blockwise_fp8.grouped_mm_backend import (
    _DeepGemmGroupedMMBackend,
    _GroupedMMBackend,
    _select_fp8_blockwise_grouped_mm_backend,
)
from torchao.prototype.moe_training.utils import (
    pad_token_groups,
    unpad_token_groups,
)
from torchao.quantization.quantize_.common import KernelPreference

from .blockwise_fp8_weight import BlockwiseFp8Weight

_BLOCK_SIZE = 128


def blockwise_fp8_grouped_mm(
    input_RK: torch.Tensor,
    weight: torch.Tensor,
    offsets_E: torch.Tensor,
    *,
    num_groups: int,
    groups_aligned: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
    kernel_preference: KernelPreference = KernelPreference.AUTO,
) -> torch.Tensor:
    """``input_RK @ weight[e].T`` for the rows of each group ``e``.

    Args:
        input_RK: ``[R, K]`` rows, grouped by ``offsets_E``.
        weight: the groups' ``[N, K]`` weights stacked on dimension 0,
            ``[E * N, K]``: a gathered ``BlockwiseFp8Weight``, or a
            high-precision tensor quantized on this call.
        offsets_E: each group's exclusive cumulative row end, int32.
        num_groups: ``E``.
        groups_aligned: every group's row count is a multiple of 128. DeepGEMM
            then runs without padding, and without reading the group sizes on
            the host. Otherwise groups are padded (TorchAO's padding).
        out_dtype: output and gradient dtype.
        kernel_preference: TorchAO's; ``EMULATED`` for its reference backend.

    Returns:
        ``[R, N]``.
    """
    return _BlockwiseFp8GroupedMM.apply(
        input_RK,
        weight,
        offsets_E,
        num_groups,
        groups_aligned,
        out_dtype,
        kernel_preference,
    )


def _grouped_backend(
    a: torch.Tensor,
    group_end_offsets: torch.Tensor,
    *,
    groups_aligned: bool,
    out_dtype: torch.dtype,
    kernel_preference: KernelPreference,
    original_group_end_offsets: torch.Tensor | None = None,
    padded_group_start_offsets: torch.Tensor | None = None,
) -> _GroupedMMBackend:
    if (
        groups_aligned
        and kernel_preference == KernelPreference.AUTO
        and can_use_deepgemm_grouped_training(a, out_dtype, _BLOCK_SIZE)
    ):
        # TorchAO's selection reads the group sizes on the host to check that
        # they're aligned; the caller vouches for it instead.
        return _DeepGemmGroupedMMBackend(
            offset_plan=build_deepgemm_grouped_offset_plan(
                group_end_offsets,
                num_rows=a.shape[0],
                groups_block_aligned_by_construction=True,
            )
        )
    return _select_fp8_blockwise_grouped_mm_backend(
        kernel_preference,
        a,
        out_dtype,
        _BLOCK_SIZE,
        group_end_offsets,
        original_group_end_offsets=original_group_end_offsets,
        padded_group_start_offsets=padded_group_start_offsets,
        num_rows=a.shape[0],
    )


def _gathered_rhs_ENK(
    weight: BlockwiseFp8Weight, num_groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """A gathered weight's groups: ``(E, N, K)`` FP8 with K contiguous, and
    ``(E, N / 128, K / 128)`` scales."""
    data, scale = weight.fp8_data, weight.recip_scale
    return (
        data.view(num_groups, -1, data.shape[-1]),
        scale.view(num_groups, -1, scale.shape[-1]),
    )


def _forward_rhs(
    weight: torch.Tensor,
    num_groups: int,
    backend: _GroupedMMBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The forward's RHS in the backend's layout, scales likewise: DeepGEMM's
    ``(E, N, K)`` with K contiguous, or the emulated backend's ``(E, K, N)``
    with K contiguous."""
    if isinstance(weight, BlockwiseFp8Weight):
        data, scale = _gathered_rhs_ENK(weight, num_groups)
        if isinstance(backend, _DeepGemmGroupedMMBackend):
            return data, scale
        return data.mT, scale.mT
    w_ENK = weight.view(num_groups, -1, weight.shape[-1])
    return backend.quantize_forward_rhs(w_ENK.mT, _BLOCK_SIZE, e4m3_dtype)


def _dgrad_rhs(
    weight: torch.Tensor,
    num_groups: int,
    backend: _GroupedMMBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The input gradient's RHS, scales likewise: DeepGEMM's ``(E, K, N)``
    with N contiguous, or the emulated backend's ``(E, N, K)`` with N
    contiguous. A 128x128 block's transpose has the same scale, so a gathered
    weight is transposed, not quantized again."""
    if isinstance(weight, BlockwiseFp8Weight):
        data, scale = _gathered_rhs_ENK(weight, num_groups)
        data, scale = data.mT.contiguous(), scale.mT.contiguous()
        if isinstance(backend, _DeepGemmGroupedMMBackend):
            return data, scale
        return data.mT, scale.mT
    w_ENK = weight.view(num_groups, -1, weight.shape[-1])
    return backend.quantize_dgrad_rhs(w_ENK.mT, _BLOCK_SIZE, e4m3_dtype)


class _BlockwiseFp8GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        input_RK: torch.Tensor,
        weight: torch.Tensor,
        offsets_E: torch.Tensor,
        num_groups: int,
        groups_aligned: bool,
        out_dtype: torch.dtype,
        kernel_preference: KernelPreference,
    ) -> torch.Tensor:
        if input_RK.ndim != 2 or weight.ndim != 2:
            raise ValueError(
                "blockwise_fp8_grouped_mm takes a 2-D input and a 2-D weight, got "
                f"{tuple(input_RK.shape)} and {tuple(weight.shape)}."
            )
        if offsets_E.dtype != torch.int32 or offsets_E.numel() != num_groups:
            raise ValueError("offsets_E must hold num_groups int32 offsets.")
        rows_N, in_K = weight.shape[0] // num_groups, weight.shape[1]
        if (
            rows_N * num_groups != weight.shape[0]
            or input_RK.shape[1] != in_K
            or rows_N % _BLOCK_SIZE
            or in_K % _BLOCK_SIZE
        ):
            raise ValueError(
                f"Can't group a {tuple(weight.shape)} weight into {num_groups} "
                f"128-aligned [N, K] blocks for a {tuple(input_RK.shape)} input."
            )
        num_rows = input_RK.shape[0]
        start_offsets = end_offsets = offsets_E
        padded = not groups_aligned
        if padded:
            padded_input, start_offsets, end_offsets = pad_token_groups(
                input_RK, offsets_E, alignment_size=_BLOCK_SIZE
            )
        else:
            padded_input = input_RK
        backend = _grouped_backend(
            padded_input,
            end_offsets,
            groups_aligned=groups_aligned,
            out_dtype=out_dtype,
            kernel_preference=kernel_preference,
            original_group_end_offsets=offsets_E if padded else None,
            padded_group_start_offsets=start_offsets if padded else None,
        )
        input_fp8, input_scale = triton_fp8_blockwise_act_quant_lhs(
            padded_input.contiguous(), block_size=_BLOCK_SIZE, dtype=e4m3_dtype
        )
        rhs, rhs_scale = _forward_rhs(weight, num_groups, backend)
        out = backend.grouped_mm(
            input_fp8,
            rhs,
            input_scale,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            rhs_scale,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            end_offsets,
            out_dtype,
            _BLOCK_SIZE,
        )
        if padded:
            out = unpad_token_groups(
                out, offsets_E, start_offsets, num_rows, alignment_size=_BLOCK_SIZE
            )
        # The input gradient needs only the weight; the weight gradient, the
        # unpadded input.
        ctx.save_for_backward(input_RK, weight, offsets_E, start_offsets, end_offsets)
        ctx.num_groups = num_groups
        ctx.padded = padded
        ctx.num_rows = num_rows
        ctx.out_dtype = out_dtype
        ctx.backend = backend
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):
        input_RK, weight, offsets_E, start_offsets, end_offsets = ctx.saved_tensors
        backend = ctx.backend
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            padded_grad_out = grad_out
            if ctx.padded:
                padded_grad_out, _, _ = pad_token_groups(
                    grad_out, offsets_E, alignment_size=_BLOCK_SIZE
                )
            grad_fp8, grad_scale = triton_fp8_blockwise_act_quant_lhs(
                padded_grad_out.contiguous(), block_size=_BLOCK_SIZE, dtype=e4m3_dtype
            )
            rhs, rhs_scale = _dgrad_rhs(weight, ctx.num_groups, backend)
            grad_input = backend.grouped_mm(
                grad_fp8,
                rhs,
                grad_scale,
                _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
                rhs_scale,
                _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
                end_offsets,
                ctx.out_dtype,
                _BLOCK_SIZE,
            )
            if ctx.padded:
                grad_input = unpad_token_groups(
                    grad_input,
                    offsets_E,
                    start_offsets,
                    ctx.num_rows,
                    alignment_size=_BLOCK_SIZE,
                )
        if ctx.needs_input_grad[1]:
            # BF16: for group e, grad_out[e].T @ input[e], on the unpadded rows.
            grad_ENK = torch._grouped_mm(
                grad_out.t(), input_RK, offs=offsets_E, out_dtype=ctx.out_dtype
            )
            grad_weight = grad_ENK.reshape(weight.shape).to(weight.dtype)
        return grad_input, grad_weight, None, None, None, None, None


__all__ = ["blockwise_fp8_grouped_mm"]
