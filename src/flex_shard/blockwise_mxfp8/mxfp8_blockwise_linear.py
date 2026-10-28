# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 linear consumer for 32x32-block prequantized weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import ScalingType, SwizzleType
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mxfp8_quantize_2d_1x32_cutedsl as _torchao_mxfp8_quantize_1x32,
    mxfp8_quantize_2d_32x1_cutedsl as _torchao_mxfp8_quantize_32x1,
)

from ..blockwise_quant.linear import (
    convert_to_flex_shard_prequantized_linear,
    FlexShardPrequantizedLinear,
)
from ..custom_placements.blockwise_quant_bucketed_block_shard import (
    _pad_2d_to_block_shape,
)
from .blockwise_mxfp8_weight import BlockwiseMxfp8Weight

# torchao's 2D MXFP8 quantizers require both dims to be multiples of 128; the
# gathered weight's blocked scales already cover 128-padded dims.
_TILE = 128


def _mx_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``mat_a @ mat_b`` with 1x32 e8m0 scales in the tcgen05 blocked layout."""
    return F.scaled_mm(
        mat_a,
        mat_b,
        scale_a=scale_a.view(torch.float8_e8m0fnu),
        scale_recipe_a=ScalingType.BlockWise1x32,
        scale_b=scale_b.view(torch.float8_e8m0fnu),
        scale_recipe_b=ScalingType.BlockWise1x32,
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        output_dtype=out_dtype,
    )


@dataclass(frozen=True)
class Mxfp8BlockwiseGemms:
    """MXFP8 GEMMs: 1x32 activations/gradients against a 32x32-block weight."""

    out_dtype: torch.dtype
    wgrad_dtype: torch.dtype
    scaling_mode: str = "rceil"

    def forward(
        self,
        x_2d: torch.Tensor,
        weight: BlockwiseMxfp8Weight,
    ) -> torch.Tensor:
        tokens = x_2d.shape[0]
        weight_matrix = weight.fp8_data.flatten(0, -2)
        out_dim, _ = weight_matrix.shape
        weight_fp8 = _pad_2d_to_block_shape(weight_matrix, _TILE)
        x_fp8, x_scale = _torchao_mxfp8_quantize_1x32(
            _pad_2d_to_block_shape(x_2d, _TILE),
            scaling_mode=self.scaling_mode,
        )
        out = _mx_scaled_mm(
            x_fp8,
            weight_fp8.t(),
            x_scale,
            weight.fwd_scale,
            self.out_dtype,
        )
        return out[:tokens, :out_dim]

    def backward(
        self,
        grad_output_2d: torch.Tensor,
        x_2d: torch.Tensor,
        weight: BlockwiseMxfp8Weight,
        needs_x_grad: bool,
        needs_weight_grad: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        tokens = grad_output_2d.shape[0]
        weight_matrix = weight.fp8_data.flatten(0, -2)
        out_dim, in_dim = weight_matrix.shape
        grad_output_padded = _pad_2d_to_block_shape(grad_output_2d, _TILE)

        grad_x = grad_weight = None
        if needs_x_grad:
            grad_output_fp8, grad_output_scale = _torchao_mxfp8_quantize_1x32(
                grad_output_padded,
                scaling_mode=self.scaling_mode,
            )
            # Blackwell scaled_mm accepts the row-major weight directly; square
            # tiles make the same fp8_data valid for DGRAD.
            grad_x = _mx_scaled_mm(
                grad_output_fp8,
                _pad_2d_to_block_shape(weight_matrix, _TILE),
                grad_output_scale,
                weight.dgrad_scale,
                self.out_dtype,
            )[:tokens, :in_dim]
        if needs_weight_grad:
            # 32x1 casts scale along tokens and return column-major data.
            grad_output_t_fp8, grad_output_t_scale = _torchao_mxfp8_quantize_32x1(
                grad_output_padded,
                scaling_mode=self.scaling_mode,
            )
            x_fp8, x_t_scale = _torchao_mxfp8_quantize_32x1(
                _pad_2d_to_block_shape(x_2d, _TILE),
                scaling_mode=self.scaling_mode,
            )
            grad_weight = _mx_scaled_mm(
                grad_output_t_fp8.t(),
                x_fp8,
                grad_output_t_scale,
                x_t_scale,
                self.wgrad_dtype,
            )
            grad_weight = grad_weight[:out_dim, :in_dim].contiguous().view(weight.shape)
        return grad_x, grad_weight


class FlexShardMxfp8BlockwiseLinear(FlexShardPrequantizedLinear, nn.Linear):
    """nn.Linear that consumes FlexShard 32x32-block MXFP8 all-gather output."""

    weight_type = BlockwiseMxfp8Weight

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        scaling_mode: str = "rceil",
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.scaling_mode = scaling_mode

    @classmethod
    def from_float(
        cls,
        mod: nn.Linear,
        scaling_mode: str = "rceil",
    ) -> FlexShardMxfp8BlockwiseLinear:
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                scaling_mode=scaling_mode,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.train(mod.training)
        return new_mod

    def _prequantized_gemms(self) -> Mxfp8BlockwiseGemms:
        # Called in forward. Emit WGRAD in the weight's grad_dtype, as
        # torchtitan's MXFP8 linear does. FlexShard sets it to the reduce dtype
        # but clears it to None from pre-backward until post-backward, so an
        # activation checkpoint recompute falls back to the weight's dtype.
        return Mxfp8BlockwiseGemms(
            out_dtype=self.weight.dtype,
            wgrad_dtype=self.weight.grad_dtype or self.weight.dtype,
            scaling_mode=self.scaling_mode,
        )


def convert_to_flex_shard_mxfp8_blockwise_linear(
    module: nn.Module,
    filter_fn: Callable[[str, nn.Linear], bool] | None = None,
    scaling_mode: str = "rceil",
    linear_cls: type[FlexShardMxfp8BlockwiseLinear] = FlexShardMxfp8BlockwiseLinear,
) -> nn.Module:
    """Replace matching nn.Linear children with the FlexShard MXFP8 linear."""
    return convert_to_flex_shard_prequantized_linear(
        module,
        linear_cls,
        filter_fn,
        scaling_mode=scaling_mode,
    )


__all__ = [
    "convert_to_flex_shard_mxfp8_blockwise_linear",
    "FlexShardMxfp8BlockwiseLinear",
    "Mxfp8BlockwiseGemms",
]
