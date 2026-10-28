# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MXFP8 all-gather on bucketed block sharding with 32x32 block scales.

``Mxfp8BucketedBlockShard`` quantizes each rank's dense rows to e4m3 data with
one e8m0 scale per 32x32 tile, so one gathered payload serves both the forward
GEMM (``x @ W.T``, scales along ``in_dim``) and the input-gradient GEMM
(``grad @ W``, scales along ``out_dim``). Both scale orientations are sent in
the tcgen05 blocked (128x4-tile swizzled) layout that the MX GEMM consumes.

Shard units stay 128 rows while quantization blocks are 32 wide. A blocked
scale is a row-block-major sequence of 512-byte tiles covering 128 scale rows by
4 scale columns, so with 128-row shard units:

- forward scales, logical ``(out_dim, in_dim / 32)``: each rank owns whole
  128-row tile rows, and a parameter's global blocked scale is the flat
  concatenation of its rank chunks.
- input-gradient scales, logical ``(in_dim, out_dim / 32)``: each rank owns
  whole tile columns, and the global blocked scale concatenates rank chunks
  along the tile-column dimension. Compaction copies those tile columns into
  place, so both global scales view the compacted buffer.

Only a parameter's last chunk can be a partial 128-row unit, and its padding
matches the global padding. Stacked weights are quantized as their flattened
matrix, the operand their consumer multiplies by, so a 32-row tile spans two
stacked matrices when their ``out_dim`` is not a multiple of 32.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, TYPE_CHECKING

import torch
import torch.nn as nn

from .blockwise_quant_bucketed_block_shard import (
    _callable_name,
    _ceil_div,
    _pad_2d_to_block_shape,
    _ScaleRegion,
    _validate_2d_non_empty,
    BlockwiseQuantBucketedBlockShard,
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import PlacementFn
    from ..flex_shard.placement_contract import Placement

MX_BLOCK_SIZE = 32
# One blocked scale tile covers 128 data rows by 4 * 32 data columns and holds
# its scales as a 32x16 block of TorchAO's 2D blocked shape.
_SCALE_TILE_ROWS = 128
_SCALE_TILE_DATA_COLS = 4 * MX_BLOCK_SIZE
_SCALE_TILE_SHAPE = (32, 16)
_SCALE_TILE_NUMEL = _SCALE_TILE_SHAPE[0] * _SCALE_TILE_SHAPE[1]
# Scale tensor indices.
_FWD_SCALE = 0
_DGRAD_SCALE = 1


class BlockwiseMxfp8WeightFactory(Protocol):
    """Construct a consumer-specific tensor from gathered MXFP8 operands.

    ``fp8_data`` has the parameter's shape. ``fwd_scale`` and ``dgrad_scale``
    are the blocked scales of its ``(prod(shape[:-1]), shape[-1])`` matrix and
    of that matrix's transpose.

    The result must view the operands rather than copy them: eager unshards
    refill those buffers in place, and the weight built on the first unshard is
    reused afterwards.
    """

    def __call__(
        self,
        fp8_data: torch.Tensor,
        fwd_scale: torch.Tensor,
        dgrad_scale: torch.Tensor,
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor: ...


class _DgradScaleChunk(NamedTuple):
    """Where one rank chunk's input-gradient scale sits in the gathered payloads."""

    rank: int
    # Byte offset of the chunk's input-gradient scale in the rank's payload.
    offset: int
    tile_cols: int
    in_dim: int


def _blocked_scale_shape(data_rows: int, data_cols: int) -> tuple[int, int]:
    """TorchAO's 2D shape for the blocked scale of ``(data_rows, data_cols)``."""
    tile_rows, tile_cols = _SCALE_TILE_SHAPE
    return (
        _ceil_div(data_rows, _SCALE_TILE_ROWS) * tile_rows,
        _ceil_div(data_cols, _SCALE_TILE_DATA_COLS) * tile_cols,
    )


def _blocked_scale_numel(data_rows: int, data_cols: int) -> int:
    rows, cols = _blocked_scale_shape(data_rows, data_cols)
    return rows * cols


def _transpose_32x32_blocked_scale(
    scale: torch.Tensor,
    data_rows: int,
    data_cols: int,
) -> torch.Tensor:
    """Re-block 32x32 tile scales of ``(rows, cols)`` data for ``(cols, rows)``.

    ``data_rows`` and ``data_cols`` must be multiples of 128.
    """
    # Imported on use: importing it initializes CUDA.
    from torchao.prototype.mx_formats.utils import from_blocked, to_blocked

    logical = from_blocked(
        scale.reshape(-1).view(torch.uint8),
        data_rows,
        data_cols // MX_BLOCK_SIZE,
    )
    tile_scales = logical[::MX_BLOCK_SIZE]
    transposed = tile_scales.t().repeat_interleave(MX_BLOCK_SIZE, dim=0)
    return (
        to_blocked(transposed)
        .view(scale.dtype)
        .view(_blocked_scale_shape(data_cols, data_rows))
    )


def _quantize_dense_weight_to_mxfp8_32x32(
    weight: torch.Tensor,
    scaling_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a dense weight with 32x32 tile scales.

    Returns row-major e4m3 data cropped to ``weight.shape``, the blocked forward
    scale for ``(out_dim, in_dim)`` data, and the blocked input-gradient scale
    for the transposed ``(in_dim, out_dim)`` data.
    """
    # Imported on use: importing it initializes CUDA.
    from torchao.prototype.moe_training.kernels.mxfp8.quant import (
        mxfp8_quantize_cuda_3d,
    )

    out_dim, in_dim = _validate_2d_non_empty(
        weight.shape,
        "dense mxfp8 weight",
    )
    # Pad to whole 128x128 scale tiles: the kernel allocates blocked scales
    # uninitialized, so any padding tile left for it would hold arbitrary bytes
    # (possibly e8m0 NaN) that the GEMM multiplies with zero-padded data.
    padded_weight = _pad_2d_to_block_shape(weight, _SCALE_TILE_ROWS)
    padded_out, padded_in = padded_weight.shape
    # The transposed expert view makes the kernel's column-major output the
    # row-major weight and its blocked scale the forward orientation.
    data_t, fwd_scale = mxfp8_quantize_cuda_3d(
        padded_weight.t().unsqueeze(0),
        block_size=MX_BLOCK_SIZE,
        scale_block_dim1=MX_BLOCK_SIZE,
        scale_block_dim2=MX_BLOCK_SIZE,
        scaling_mode=scaling_mode,
        blocked_scale_output=True,
    )
    fp8_data = data_t[0].t()[:out_dim, :in_dim].contiguous()
    fwd_scale = fwd_scale.view(_blocked_scale_shape(padded_out, padded_in))
    dgrad_scale = _transpose_32x32_blocked_scale(fwd_scale, padded_out, padded_in)
    return fp8_data, fwd_scale, dgrad_scale


def make_mxfp8_bucketed_block_placement_fn(
    *,
    weight_factory: BlockwiseMxfp8WeightFactory,
    scaling_mode: str = "rceil",
) -> PlacementFn:
    """Assign one MXFP8 bucketed block placement using the mesh rank count."""

    def mxfp8_bucketed_block_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        placement = Mxfp8BucketedBlockShard(
            world_size=mesh.size(),
            weight_factory=weight_factory,
            scaling_mode=scaling_mode,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return mxfp8_bucketed_block_placements


class Mxfp8BucketedBlockShard(BlockwiseQuantBucketedBlockShard):
    """Bucketed block shard that all-gathers 32x32-block MXFP8 weights.

    Each chunk carries two e8m0 scale tensors in the blocked layout: the
    forward scale for ``(rows, in_dim)`` data and the input-gradient scale for
    the transposed ``(in_dim, rows)`` data. ``finish_prepared_unshard`` passes
    the gathered data and both global scales, all views of the compacted
    buffer, to the configured ``weight_factory``.
    """

    scale_dtype = torch.float8_e8m0fnu
    num_scale_tensors = 2

    def __init__(
        self,
        *,
        world_size: int,
        weight_factory: BlockwiseMxfp8WeightFactory,
        scaling_mode: str = "rceil",
    ) -> None:
        if not callable(weight_factory):
            raise TypeError("weight_factory must be callable.")
        self.weight_factory = weight_factory
        # TorchAO lowercases the mode, so equal placements must too.
        self.scaling_mode = scaling_mode.lower()
        super().__init__(
            world_size=world_size,
            shard_block_rows=_SCALE_TILE_ROWS,
            qdata_dtype=torch.float8_e4m3fn,
        )

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, Mxfp8BucketedBlockShard)
        return (
            self.world_size == other.world_size
            and self.scaling_mode == other.scaling_mode
            and self.weight_factory is other.weight_factory
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.world_size,
                self.scaling_mode,
                id(self.weight_factory),
            )
        )

    def __repr__(self) -> str:
        return (
            "Mxfp8BucketedBlockShard("
            f"world_size={self.world_size}, "
            f"scaling_mode={self.scaling_mode!r}, "
            f"weight_factory={_callable_name(self.weight_factory)})"
        )

    def _chunk_scale_numels(self, rows: int, in_dim: int) -> tuple[int, ...]:
        return (
            _blocked_scale_numel(rows, in_dim),
            _blocked_scale_numel(in_dim, rows),
        )

    def _quantize_local_chunk(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        fp8_data, fwd_scale, dgrad_scale = _quantize_dense_weight_to_mxfp8_32x32(
            tensor,
            self.scaling_mode,
        )
        return fp8_data, (fwd_scale, dgrad_scale)

    def _scale_regions(
        self,
        scale_index: int,
        rank_pack_plans: tuple[
            BlockwiseQuantBucketedBlockShard._QuantRankPackPlan, ...
        ],
    ) -> tuple[_ScaleRegion, ...]:
        """Order input-gradient scale tiles as each parameter's global tile grid.

        A parameter's input-gradient scale is a grid of 512-byte tiles,
        ``in_dim / 128`` tile rows by ``out_dim / 128`` tile columns, and each
        rank chunk holds a band of whole tile columns. Compacting tile row by tile
        row, chunk by chunk, writes the grid directly.
        """
        if scale_index == _FWD_SCALE:
            return super()._scale_regions(scale_index, rank_pack_plans)
        itemsize = self._scale_itemsize
        # Each parameter's chunks, in rank order.
        chunks_by_param: dict[int, list[_DgradScaleChunk]] = {}
        for rank, plan in enumerate(rank_pack_plans):
            offset = plan.scale_byte_offset + itemsize * sum(
                chunk.scale_numels[_FWD_SCALE] for chunk in plan.chunks
            )
            for chunk in plan.chunks:
                rows, in_dim = chunk.local_shape
                chunks_by_param.setdefault(chunk.param_index, []).append(
                    _DgradScaleChunk(
                        rank, offset, _ceil_div(rows, _SCALE_TILE_ROWS), in_dim
                    )
                )
                offset += itemsize * chunk.scale_numels[_DGRAD_SCALE]
        regions: list[_ScaleRegion] = []
        for param_index in sorted(chunks_by_param):
            param_chunks = chunks_by_param[param_index]
            in_tiles = _ceil_div(param_chunks[0].in_dim, _SCALE_TILE_ROWS)
            for tile_row in range(in_tiles):
                for chunk in param_chunks:
                    tile_row_nbytes = itemsize * chunk.tile_cols * _SCALE_TILE_NUMEL
                    regions.append(
                        _ScaleRegion(
                            chunk.rank,
                            chunk.offset + tile_row * tile_row_nbytes,
                            tile_row_nbytes,
                        )
                    )
        return tuple(regions)

    def _param_scale_view(
        self,
        scale_index: int,
        scale: torch.Tensor,
        matrix_shape: torch.Size,
    ) -> torch.Tensor:
        out_dim, in_dim = matrix_shape
        if scale_index == _FWD_SCALE:
            return scale.view(_blocked_scale_shape(out_dim, in_dim))
        return scale.view(_blocked_scale_shape(in_dim, out_dim))

    def _make_quantized_weight(
        self,
        qdata: torch.Tensor,
        scales: tuple[torch.Tensor, ...],
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        fwd_scale, dgrad_scale = scales
        return self.weight_factory(
            qdata,
            fwd_scale,
            dgrad_scale,
            orig_dtype=orig_dtype,
            requires_grad=requires_grad,
        )


__all__ = [
    "BlockwiseMxfp8WeightFactory",
    "make_mxfp8_bucketed_block_placement_fn",
    "Mxfp8BucketedBlockShard",
]
