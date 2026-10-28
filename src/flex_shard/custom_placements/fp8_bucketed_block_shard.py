# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 all-gather on bucketed block sharding with blockwise quantization.

``Fp8BucketedBlockShard`` uses ``block_size`` as both the shard row unit and
the quantization block, so the all-gather moves **fp8 + tiny fp32 scales**
buffers. ``blockwise_quant_bucketed_block_shard`` describes the shared row
sharding, stacked weights, and compacted unshard storage.
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

import torch
import torch.nn as nn
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_weight_quant_transposed_rhs as _torchao_weight_quant_t_rhs,
)

from .blockwise_quant_bucketed_block_shard import (
    _callable_name,
    _ceil_div,
    _pad_2d_to_block_shape,
    _scale_shape,
    _validate_2d_non_empty,
    _validate_block_size,
    BlockwiseQuantBucketedBlockShard,
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import PlacementFn
    from ..flex_shard.placement_contract import Placement


class BlockwiseFp8WeightFactory(Protocol):
    """Construct a consumer-specific tensor from gathered FP8 operands.

    ``fp8_data`` has the parameter's shape. ``recip_scale`` holds one scale per
    block of the ``(prod(shape[:-1]), shape[-1])`` matrix.

    The result must view ``fp8_data`` and ``recip_scale`` (a tensor subclass
    may hold them as inner tensors) rather than copy them: eager unshards
    refill those buffers in place, and the weight built on the first unshard is
    reused afterwards.
    """

    def __call__(
        self,
        fp8_data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor: ...


def _quantize_dense_weight_to_blockwise_fp8(
    weight: torch.Tensor,
    block_size: int,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_dim, in_dim = _validate_2d_non_empty(
        weight.shape,
        "dense blockwise fp8 weight",
    )
    padded_weight = _pad_2d_to_block_shape(weight, block_size)
    fp8_data_t, recip_scale_t = _torchao_weight_quant_t_rhs(
        padded_weight,
        block_size,
        fp8_dtype,
    )
    # Transposing the column-major transposed-RHS outputs produces row-major
    # views in the original weight orientation without a layout copy.
    fp8_data = fp8_data_t.t()[:out_dim, :in_dim]
    recip_scale = recip_scale_t.t()
    # Cropping padded columns leaves gaps between rows and requires compaction.
    if not fp8_data.is_contiguous():
        fp8_data = fp8_data.contiguous()
    return fp8_data, recip_scale


def make_fp8_bucketed_block_placement_fn(
    *,
    weight_factory: BlockwiseFp8WeightFactory,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    """Assign one FP8 bucketed block placement using the mesh rank count."""

    def fp8_bucketed_block_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        placement = Fp8BucketedBlockShard(
            world_size=mesh.size(),
            weight_factory=weight_factory,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return fp8_bucketed_block_placements


class Fp8BucketedBlockShard(BlockwiseQuantBucketedBlockShard):
    """Bucketed block shard that all-gathers block-wise fp8 instead of bf16.

    Shard rows and quantization blocks are both ``block_size``. Each chunk
    carries one fp32 reciprocal-scale tensor of shape
    ``(ceil(rows / block_size), ceil(in_dim / block_size))``, so rank chunks
    concatenate flat into each parameter's global scale. ``finish_prepared_unshard``
    passes the gathered FP8 data and scales to the configured ``weight_factory``.
    """

    scale_dtype = torch.float32

    def __init__(
        self,
        *,
        world_size: int,
        weight_factory: BlockwiseFp8WeightFactory,
        block_size: int = 128,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        _validate_block_size(block_size)
        if not callable(weight_factory):
            raise TypeError("weight_factory must be callable.")
        self.block_size = block_size
        self.weight_factory = weight_factory
        super().__init__(
            world_size=world_size,
            shard_block_rows=block_size,
            qdata_dtype=fp8_dtype,
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, Fp8BucketedBlockShard)
        return (
            self.world_size == other.world_size
            and self.block_size == other.block_size
            and self.qdata_dtype == other.qdata_dtype
            and self.weight_factory is other.weight_factory
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.world_size,
                self.block_size,
                self.qdata_dtype,
                id(self.weight_factory),
            )
        )

    def __repr__(self) -> str:
        return (
            "Fp8BucketedBlockShard("
            f"world_size={self.world_size}, "
            f"block_size={self.block_size}, "
            f"weight_factory={_callable_name(self.weight_factory)})"
        )

    def _chunk_scale_numels(self, rows: int, in_dim: int) -> tuple[int, ...]:
        return (_ceil_div(rows, self.block_size) * _ceil_div(in_dim, self.block_size),)

    def _quantize_local_weight(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _quantize_dense_weight_to_blockwise_fp8(
            tensor,
            self.block_size,
            self.qdata_dtype,
        )

    def _quantize_local_chunk(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        quant, recip_scale = self._quantize_local_weight(tensor)
        return quant, (recip_scale,)

    def _param_scale_view(
        self,
        scale_index: int,
        scale: torch.Tensor,
        matrix_shape: torch.Size,
    ) -> torch.Tensor:
        return scale.view(_scale_shape(matrix_shape, self.block_size))

    def _make_quantized_weight(
        self,
        qdata: torch.Tensor,
        scales: tuple[torch.Tensor, ...],
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        (recip_scale,) = scales
        return self.weight_factory(
            qdata,
            recip_scale,
            self.block_size,
            orig_dtype=orig_dtype,
            requires_grad=requires_grad,
        )


__all__ = [
    "BlockwiseFp8WeightFactory",
    "Fp8BucketedBlockShard",
    "make_fp8_bucketed_block_placement_fn",
]
