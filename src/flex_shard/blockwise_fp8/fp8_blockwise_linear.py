# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blockwise FP8 linear consumer for prequantized weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_128X128_SCALING_TYPE as _TORCHAO_BLOCKWISE_128X128_SCALING_TYPE,
    BLOCKWISE_1X128_SCALING_TYPE as _TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
    triton_fp8_blockwise_act_quant_lhs as _torchao_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs as _torchao_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs as _torchao_act_quant_transposed_lhs,
    triton_fp8_gemm_1x128_128x1 as _torchao_gemm_1x128_128x1,
    triton_fp8_gemm_1x128_128x128 as _torchao_gemm_1x128_128x128,
)
from torchao.prototype.blockwise_fp8_training.linear import (
    _run_blockwise_mm as _torchao_run_blockwise_mm,
    Float8BlockwiseLinear as TorchAOFloat8BlockwiseLinear,
)

from ..blockwise_quant.linear import (
    _as_column_major_2d,
    _pad_last_dim_to,
    convert_to_flex_shard_prequantized_linear,
    FlexShardPrequantizedLinear,
)
from ..custom_placements.blockwise_quant_bucketed_block_shard import (
    _pad_2d_to_block_shape,
)
from .blockwise_fp8_weight import BlockwiseFp8Weight


@dataclass(frozen=True)
class Fp8BlockwiseGemms:
    """torchao 1x128 activation / 128x128 weight blockwise FP8 GEMMs."""

    block_size: int
    out_dtype: torch.dtype
    use_triton: bool

    def forward(
        self,
        x_2d: torch.Tensor,
        weight: BlockwiseFp8Weight,
    ) -> torch.Tensor:
        if self.block_size != weight.block_size:
            raise ValueError(
                f"linear block_size {self.block_size} does not match gathered weight "
                f"block_size {weight.block_size}"
            )
        if self.block_size != 128:
            raise AssertionError("torchao Float8BlockwiseLinear only supports 128")

        weight_matrix = weight.fp8_data.flatten(0, -2)
        logical_out_dim, _ = weight_matrix.shape
        weight_fp8 = _pad_2d_to_block_shape(weight_matrix, self.block_size)
        _, padded_in_dim = weight_fp8.shape
        x_2d = _pad_last_dim_to(x_2d, padded_in_dim)
        x_fp8, x_scale = _torchao_act_quant_lhs(
            x_2d,
            self.block_size,
            dtype=weight.fp8_data.dtype,
        )
        weight_t_fp8 = weight_fp8.t()
        weight_t_scale = weight.recip_scale.t()
        out = _torchao_run_blockwise_mm(
            use_triton=self.use_triton,
            triton_kernel=_torchao_gemm_1x128_128x128,
            mat_a=x_fp8,
            mat_b=weight_t_fp8,
            scale_a=x_scale,
            scale_recipe_a=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
            scale_b=weight_t_scale,
            scale_recipe_b=_TORCHAO_BLOCKWISE_128X128_SCALING_TYPE,
            out_dtype=self.out_dtype,
        )
        return out[:, :logical_out_dim]

    def backward(
        self,
        grad_output_2d: torch.Tensor,
        x_2d: torch.Tensor,
        weight: BlockwiseFp8Weight,
        needs_x_grad: bool,
        needs_weight_grad: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        fp8_dtype = weight.fp8_data.dtype
        weight_matrix = weight.fp8_data.flatten(0, -2)
        logical_out_dim, logical_in_dim = weight_matrix.shape
        weight_fp8 = _pad_2d_to_block_shape(weight_matrix, self.block_size)
        padded_out_dim = weight_fp8.shape[0]
        grad_output_2d_padded = _pad_last_dim_to(grad_output_2d, padded_out_dim)

        grad_x = grad_weight = None
        if needs_x_grad:
            grad_output_fp8, grad_output_scale = _torchao_act_quant_lhs(
                grad_output_2d_padded,
                self.block_size,
                dtype=fp8_dtype,
            )
            weight_fp8 = _as_column_major_2d(weight_fp8)
            weight_scale = _as_column_major_2d(weight.recip_scale)
            grad_x = _torchao_run_blockwise_mm(
                use_triton=self.use_triton,
                triton_kernel=_torchao_gemm_1x128_128x128,
                mat_a=grad_output_fp8,
                mat_b=weight_fp8,
                scale_a=grad_output_scale,
                scale_recipe_a=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
                scale_b=weight_scale,
                scale_recipe_b=_TORCHAO_BLOCKWISE_128X128_SCALING_TYPE,
                out_dtype=self.out_dtype,
            )
            grad_x = grad_x[:, :logical_in_dim]
        if needs_weight_grad:
            # Both wgrad operands must share a block-aligned token dimension.
            grad_output_2d_wgrad = _pad_2d_to_block_shape(
                grad_output_2d_padded,
                self.block_size,
            )
            x_2d = _pad_2d_to_block_shape(x_2d, self.block_size)
            grad_output_t_fp8, grad_output_t_scale = _torchao_act_quant_transposed_lhs(
                grad_output_2d_wgrad,
                self.block_size,
                dtype=fp8_dtype,
            )
            x_fp8, x_scale = _torchao_act_quant_rhs(
                x_2d,
                self.block_size,
                dtype=fp8_dtype,
            )
            grad_weight = _torchao_run_blockwise_mm(
                use_triton=self.use_triton,
                triton_kernel=_torchao_gemm_1x128_128x1,
                mat_a=grad_output_t_fp8,
                mat_b=x_fp8,
                scale_a=grad_output_t_scale,
                scale_recipe_a=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
                scale_b=x_scale.transpose(-1, -2),
                scale_recipe_b=_TORCHAO_BLOCKWISE_1X128_SCALING_TYPE,
                triton_scale_b=x_scale,
                out_dtype=self.out_dtype,
            )
            # Slicing padded columns (e.g. w2 in_dim=10944 padded to 11008) yields a
            # non-contiguous view; the FlexShard grad reduce path requires contiguous
            # grads, so materialize before returning.
            grad_weight = (
                grad_weight[:logical_out_dim, :logical_in_dim]
                .to(weight.dtype)
                .contiguous()
                .view(weight.shape)
            )
        return grad_x, grad_weight


class FlexShardFloat8BlockwiseLinear(
    FlexShardPrequantizedLinear,
    TorchAOFloat8BlockwiseLinear,
):
    """torchao Float8BlockwiseLinear that consumes FlexShard FP8 all-gather output."""

    weight_type = BlockwiseFp8Weight

    @classmethod
    def from_float(
        cls,
        mod,
        use_triton: bool = False,
    ):
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=mod.bias is not None,
                use_triton=use_triton,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        new_mod.train(mod.training)
        return new_mod

    def _prequantized_gemms(self) -> Fp8BlockwiseGemms:
        return Fp8BlockwiseGemms(
            block_size=self.block_size,
            out_dtype=self.dtype,
            use_triton=self.use_triton,
        )


def convert_to_flex_shard_float8_blockwise_linear(
    module: nn.Module,
    filter_fn: Callable[[str, nn.Linear], bool] | None = None,
    use_triton: bool = False,
    linear_cls: type[FlexShardFloat8BlockwiseLinear] = FlexShardFloat8BlockwiseLinear,
) -> nn.Module:
    """Replace matching nn.Linear children with the FlexShard torchao wrapper."""
    return convert_to_flex_shard_prequantized_linear(
        module,
        linear_cls,
        filter_fn,
        use_triton=use_triton,
    )


__all__ = [
    "convert_to_flex_shard_float8_blockwise_linear",
    "Fp8BlockwiseGemms",
    "FlexShardFloat8BlockwiseLinear",
    "TorchAOFloat8BlockwiseLinear",
]
