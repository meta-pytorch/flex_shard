# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""32x32-block MXFP8 weight representation for FlexShard all-gather output."""

from __future__ import annotations

from typing import Any

import torch
from torchao.prototype.mx_formats.utils import from_blocked

from ..blockwise_quant.weight import BlockwiseQuantWeight
from ..custom_placements.blockwise_quant_bucketed_block_shard import _matrix_shape
from ..custom_placements.mxfp8_bucketed_block_shard import (
    _blocked_scale_shape,
    MX_BLOCK_SIZE,
)

_E8M0_EXPONENT_BIAS = 127


def blockwise_mxfp8_dequant_weight(
    fp8_data: torch.Tensor,
    fwd_scale: torch.Tensor,
) -> torch.Tensor:
    """Dequantize row-major e4m3 data with its blocked forward-orientation scale."""
    out_dim, in_dim = fp8_data.shape
    scale_cols = (in_dim + MX_BLOCK_SIZE - 1) // MX_BLOCK_SIZE
    logical_scale = from_blocked(
        fwd_scale.reshape(-1).view(torch.uint8), out_dim, scale_cols
    )
    scale = torch.exp2(logical_scale.to(torch.float32) - _E8M0_EXPONENT_BIAS)
    scale = scale.repeat_interleave(MX_BLOCK_SIZE, dim=1)[:, :in_dim]
    return fp8_data.to(torch.float32) * scale


class BlockwiseMxfp8Weight(BlockwiseQuantWeight):
    """32x32-block MXFP8 weight returned by the FlexShard all-gather.

    ``fp8_data`` is the row-major ``(out_dim, in_dim)`` e4m3 weight.
    ``fwd_scale`` is the tcgen05 blocked scale for that data (forward GEMM
    operand ``weight.T``), and ``dgrad_scale`` is the blocked scale for the
    transposed ``(in_dim, out_dim)`` data (input-gradient GEMM operand).

    Stacked weights keep the parameter shape, e.g. ``[num_linears, out, in]``.
    Their scales cover the flattened ``[num_linears * out, in]`` matrix.
    """

    __slots__ = ["_data", "_fwd_scale", "_dgrad_scale", "_orig_dtype"]
    _inner_tensor_names = ("_data", "_fwd_scale", "_dgrad_scale")

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        fwd_scale: torch.Tensor,
        dgrad_scale: torch.Tensor,
        orig_dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = False,
    ):
        return cls._make_wrapper(data, orig_dtype, requires_grad)

    def __init__(
        self,
        data: torch.Tensor,
        fwd_scale: torch.Tensor,
        dgrad_scale: torch.Tensor,
        orig_dtype: torch.dtype = torch.bfloat16,
        requires_grad: bool = False,
    ) -> None:
        _ = requires_grad
        if data.ndim < 2:
            raise ValueError(
                f"BlockwiseMxfp8Weight expects 2D or stacked 2D data, got {data.shape}"
            )
        out_dim, in_dim = _matrix_shape(data.shape)
        for name, scale, expected in (
            ("fwd_scale", fwd_scale, _blocked_scale_shape(out_dim, in_dim)),
            ("dgrad_scale", dgrad_scale, _blocked_scale_shape(in_dim, out_dim)),
        ):
            if tuple(scale.shape) != expected:
                raise ValueError(
                    f"{name} shape must be {expected} for data {tuple(data.shape)}, "
                    f"got {tuple(scale.shape)}"
                )
            if scale.device != data.device:
                raise ValueError(
                    f"data and {name} must be on the same device, got "
                    f"{data.device} and {scale.device}"
                )
        self._data = data
        self._fwd_scale = fwd_scale
        self._dgrad_scale = dgrad_scale
        self._orig_dtype = orig_dtype

    @property
    def fp8_data(self) -> torch.Tensor:
        return self._data

    @property
    def fwd_scale(self) -> torch.Tensor:
        return self._fwd_scale

    @property
    def dgrad_scale(self) -> torch.Tensor:
        return self._dgrad_scale

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        result = blockwise_mxfp8_dequant_weight(
            self._data.flatten(0, -2),
            self._fwd_scale,
        )
        return result.reshape(self._data.shape).to(dtype or self._orig_dtype)

    def _metadata(self) -> dict[str, Any]:
        return {"orig_dtype": self._orig_dtype}

    @classmethod
    def _from_inner(
        cls,
        inner_tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        requires_grad: bool,
    ) -> BlockwiseMxfp8Weight:
        return cls(
            inner_tensors["_data"],
            inner_tensors["_fwd_scale"],
            inner_tensors["_dgrad_scale"],
            metadata["orig_dtype"],
            requires_grad=requires_grad,
        )


__all__ = [
    "BlockwiseMxfp8Weight",
    "blockwise_mxfp8_dequant_weight",
]
