# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP8 all-gather on bucketed block sharding with blockwise quantization.

``Fp8BucketedBlockShard`` cuts a bucket into contiguous ``block_size``
row units over the original dense parameters. Every rank owns complete block
rows, so it can quantize its dense shard to fp8 locally -- no cross-rank
``amax``. The all-gather moves temporary **fp8 + tiny scales** buffers, and the
gathered fp8 weight is **bit-identical** to "all-gather bf16, then
block-quantize".

The master weight stays dense bf16/fp32-sharded with no fp8 padding in
persistent storage. FP8 data, scales, and tail padding exist only in the
unshard collective buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_weight_quant_transposed_rhs as _torchao_weight_quant_t_rhs,
)
from typing_extensions import override

from ..flex_shard.placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from ..flex_shard.utils import (
    _record_comm_if_eager,
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
)
from .block_shard import BucketedBlockShard
from .utils import foreach_copy_


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo, PlacementFn
    from ..flex_shard.placement_contract import Placement

_VEC_COPY_DTYPE = torch.int32
_VEC_COPY_NBYTES = _VEC_COPY_DTYPE.itemsize
# CUDA's vectorized copy paths -- ``CatArrayBatchedCopy_vectorized`` for
# ``torch.cat`` and ``TensorIterator``'s vectorized elementwise kernels for
# ``copy_`` -- require 16-byte-aligned source and destination pointers. Cat also
# requires each copied slice to contain a multiple of 16 bytes. Padding rank
# segments and scale starts satisfies the pointer requirement and lets common
# model shapes with naturally aligned payload lengths use the vectorized path.
_VEC_ALIGN_BYTES = 16


class BlockwiseFp8WeightFactory(Protocol):
    """Construct a consumer-specific tensor from gathered FP8 operands."""

    def __call__(
        self,
        fp8_data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor: ...


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _validate_block_size(block_size: int) -> None:
    if not isinstance(block_size, int):
        raise TypeError(f"block_size must be an int, got {type(block_size).__name__}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")


def _validate_2d_non_empty(shape: torch.Size, what: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{what} expects 2D, got {tuple(shape)}")
    out_dim, in_dim = shape
    if out_dim <= 0 or in_dim <= 0:
        raise ValueError(f"{what} requires non-empty 2D shape, got {tuple(shape)}")
    return out_dim, in_dim


def _scale_shape(shape: torch.Size, block_size: int) -> tuple[int, int]:
    _validate_block_size(block_size)
    out_dim, in_dim = _validate_2d_non_empty(shape, "blockwise scale")
    return (_ceil_div(out_dim, block_size), _ceil_div(in_dim, block_size))


def _padded_shape(shape: torch.Size, block_size: int) -> tuple[int, int]:
    scale_rows, scale_cols = _scale_shape(shape, block_size)
    return scale_rows * block_size, scale_cols * block_size


def _pad_2d_to_block_shape(
    tensor: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    _validate_block_size(block_size)
    out_dim, in_dim = _validate_2d_non_empty(tensor.shape, "blockwise tensor")
    padded_out, padded_in = _padded_shape(tensor.shape, block_size)
    if out_dim == padded_out and in_dim == padded_in:
        return tensor.contiguous()
    padded = torch.zeros(
        (padded_out, padded_in),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:out_dim, :in_dim].copy_(tensor)
    return padded


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


def _callable_name(value: Any) -> str:
    return getattr(value, "__qualname__", type(value).__qualname__)


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


class Fp8BucketedBlockShard(BucketedBlockShard):
    """Bucketed block shard that all-gathers block-wise fp8 instead of bf16.

    The bucket planner cuts the original dense parameters into contiguous
    ``block_size``-row units. Persistent shards stay dense and contain only real
    parameter rows; temporary unshard buffers hold fp8 data, scales, and tail
    padding for the all-gather. ``finish_prepared_unshard`` passes the gathered
    FP8 data and scales to the configured ``weight_factory``.
    """

    @dataclass(frozen=True)
    class _BlockRowParam:
        """Per-parameter block-row span used by the bucket partition planner."""

        fqn: str
        global_shape: torch.Size
        param_offset: int

    @dataclass(frozen=True)
    class _BlockRowUnit:
        """One indivisible row-block unit, including a possible partial tail."""

        fqn: str
        row_start: int
        row_end: int
        dense_numel: int
        cost_bytes: int

    @dataclass(frozen=True)
    class _Fp8Chunk:
        """One rank-owned contiguous row range for a single parameter."""

        rank: int
        fqn: str
        row_start: int
        row_end: int
        scale_row_start: int
        scale_row_end: int
        global_dense_offset: int
        rank_dense_offset: int
        fp8_offset: int
        scale_offset: int

    @dataclass(frozen=True)
    class _Fp8PackChunk:
        """Static copy-in plan for one rank-local parameter chunk."""

        param_index: int
        fqn: str
        local_shape: torch.Size
        fp8_numel: int
        scale_numel: int

    @dataclass(frozen=True)
    class _Fp8RankPackPlan:
        """Static copy-in plan for one rank."""

        chunks: tuple[Fp8BucketedBlockShard._Fp8PackChunk, ...]
        fp8_numel: int
        scale_byte_offset: int
        packed_nbytes: int
        fp8_chunk_numels: tuple[int, ...]
        scale_chunk_nbytes: tuple[int, ...]

    @dataclass(frozen=True)
    class _Fp8BucketMetadata:
        """FP8 communication layout derived from dense rank intervals."""

        chunks: tuple[Fp8BucketedBlockShard._Fp8Chunk, ...]
        rank_pack_plans: tuple[Fp8BucketedBlockShard._Fp8RankPackPlan, ...]
        rank_fp8_numels: tuple[int, ...]
        rank_scale_numels: tuple[int, ...]
        rank_scale_byte_offsets: tuple[int, ...]
        rank_packed_nbytes: tuple[int, ...]
        packed_send_nbytes_per_rank: int

    @dataclass(frozen=True)
    class _Fp8MetadataPlan:
        """Immutable metadata retaining its identity-keyed bucket layout."""

        bucket_layout: Any
        metadata: Fp8BucketedBlockShard._Fp8BucketMetadata

    @dataclass(frozen=True)
    class _Fp8UnshardState:
        """Prepared-unshard state needed to all-gather and compact rank rows."""

        infos: list[ParamInfo]
        pg: Any
        debug_fqn: str | None
        metadata: Fp8BucketedBlockShard._Fp8BucketMetadata

    @dataclass(frozen=True)
    class _LocalPackedSourceState:
        """Source identity and view state for one cached local shard."""

        storage: Any
        version: int
        storage_offset: int
        shape: torch.Size
        stride: tuple[int, ...]
        dtype: torch.dtype
        is_conj: bool
        is_neg: bool

    @dataclass(frozen=True)
    class _LocalPackedInfoState:
        """ParamInfo state that can change the quantized payload."""

        info: ParamInfo
        unsharded_dtype: torch.dtype

    @dataclass(frozen=True)
    class _LocalPackedCacheEntry:
        """Versioned local FP8 send payload for one bucket and rank."""

        info_states: tuple[Fp8BucketedBlockShard._LocalPackedInfoState, ...]
        source_states: tuple[Fp8BucketedBlockShard._LocalPackedSourceState, ...]
        metadata: Fp8BucketedBlockShard._Fp8BucketMetadata
        packed: torch.Tensor

    def __init__(
        self,
        *,
        world_size: int,
        weight_factory: BlockwiseFp8WeightFactory,
        block_size: int = 128,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    ) -> None:
        if world_size <= 0:
            raise ValueError(
                f"Fp8BucketedBlockShard world_size must be positive, got {world_size}."
            )
        super().__init__(blocks_per_rank=(1,) * world_size)
        _validate_block_size(block_size)
        if not callable(weight_factory):
            raise TypeError("weight_factory must be callable.")
        if torch.empty((), dtype=fp8_dtype).element_size() != 1:
            raise ValueError(
                f"Fp8BucketedBlockShard requires a one-byte fp8_dtype, got {fp8_dtype}."
            )
        self.block_size = block_size
        self.fp8_dtype = fp8_dtype
        self.weight_factory = weight_factory
        self._metadata_plans: dict[
            int,
            Fp8BucketedBlockShard._Fp8MetadataPlan,
        ] = {}
        self._local_packed_cache: dict[
            tuple[int, int, torch.device, Any],
            Fp8BucketedBlockShard._LocalPackedCacheEntry,
        ] = {}

    @property
    def world_size(self) -> int:
        return len(self.blocks_per_rank)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, Fp8BucketedBlockShard)
        return (
            self.world_size == other.world_size
            and self.block_size == other.block_size
            and self.fp8_dtype == other.fp8_dtype
            and self.weight_factory is other.weight_factory
        )

    def __hash__(self) -> int:
        return hash(
            (
                type(self),
                self.world_size,
                self.block_size,
                self.fp8_dtype,
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

    @staticmethod
    def _can_partition_units(
        unit_costs: list[int],
        num_partitions: int,
        capacity: int,
    ) -> bool:
        partitions = 1
        current = 0
        for unit_cost in unit_costs:
            if unit_cost > capacity:
                return False
            if current + unit_cost <= capacity:
                current += unit_cost
            else:
                partitions += 1
                current = unit_cost
        return partitions <= num_partitions

    @staticmethod
    def _min_partition_capacity(
        unit_costs: list[int],
        num_partitions: int,
    ) -> int:
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}.")
        if not unit_costs:
            return 0
        low = max(
            max(unit_costs),
            (sum(unit_costs) + num_partitions - 1) // num_partitions,
        )
        high = sum(unit_costs)
        while low < high:
            mid = (low + high) // 2
            if Fp8BucketedBlockShard._can_partition_units(
                unit_costs,
                num_partitions,
                mid,
            ):
                high = mid
            else:
                low = mid + 1
        return low

    @staticmethod
    def _rank_unit_cuts(
        unit_costs: list[int],
        world_size: int,
        capacity: int,
    ) -> tuple[int, ...]:
        cuts = [0]
        current = 0
        for unit_idx, unit_cost in enumerate(unit_costs):
            if current > 0 and current + unit_cost > capacity:
                cuts.append(unit_idx)
                current = 0
            current += unit_cost
        cuts.append(len(unit_costs))

        while len(cuts) - 1 < world_size:
            split_at = None
            for idx, (start, end) in enumerate(zip(cuts, cuts[1:])):
                if end - start > 1:
                    split_at = idx + 1
                    cuts.insert(split_at, start + 1)
                    break
            if split_at is None:
                cuts.append(cuts[-1])
        if len(cuts) - 1 > world_size:
            raise AssertionError(
                "FP8 block-row planner produced more rank intervals than ranks."
            )
        return tuple(cuts)

    def _validate_param_shape(self, fqn: str, shape: torch.Size) -> None:
        if len(shape) != 2:
            raise ValueError(
                "Fp8BucketedBlockShard only supports 2D weights, "
                f"but {fqn!r} has shape {tuple(shape)}."
            )
        out_dim, in_dim = shape
        if out_dim <= 0 or in_dim <= 0:
            raise ValueError(
                "Fp8BucketedBlockShard requires non-empty 2D weights, "
                f"but {fqn!r} has shape {tuple(shape)}."
            )

    def _build_block_row_units(
        self,
        params: list[_BlockRowParam],
    ) -> list[_BlockRowUnit]:
        fp8_itemsize = torch.empty((), dtype=self.fp8_dtype).element_size()
        scale_itemsize = torch.float32.itemsize
        units: list[Fp8BucketedBlockShard._BlockRowUnit] = []
        for param in params:
            out_dim, in_dim = param.global_shape
            num_col_blocks = _ceil_div(in_dim, self.block_size)
            for row_start in range(0, out_dim, self.block_size):
                row_end = min(row_start + self.block_size, out_dim)
                dense_numel = (row_end - row_start) * in_dim
                units.append(
                    Fp8BucketedBlockShard._BlockRowUnit(
                        fqn=param.fqn,
                        row_start=row_start,
                        row_end=row_end,
                        dense_numel=dense_numel,
                        cost_bytes=dense_numel * fp8_itemsize
                        + num_col_blocks * scale_itemsize,
                    )
                )
        return units

    def _block_row_params_from_named(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
    ) -> tuple[list[_BlockRowParam], list[_BlockRowUnit]]:
        dtype = named_params[0][1].dtype
        params: list[Fp8BucketedBlockShard._BlockRowParam] = []
        param_offset = 0
        for fqn, param in named_params:
            if param.dtype != dtype:
                raise ValueError(
                    "Fp8BucketedBlockShard requires one dtype per bucket: "
                    f"{named_params[0][0]!r} uses {dtype} but {fqn!r} uses "
                    f"{param.dtype}."
                )
            placements = param_placements[fqn]
            if placements != (self,):
                raise ValueError(
                    "Fp8BucketedBlockShard requires the same placement "
                    f"instance for every parameter in a bucket; {fqn!r} uses "
                    f"{placements!r}."
                )
            self._validate_param_shape(fqn, param.shape)
            params.append(
                Fp8BucketedBlockShard._BlockRowParam(
                    fqn=fqn,
                    global_shape=param.shape,
                    param_offset=param_offset,
                )
            )
            param_offset += param.numel()
        return params, self._build_block_row_units(params)

    def _block_row_params_from_infos(
        self,
        infos: list[ParamInfo],
    ) -> tuple[list[_BlockRowParam], list[_BlockRowUnit]]:
        bucket_layout = self._bucket_layout(infos[0])
        params: list[Fp8BucketedBlockShard._BlockRowParam] = []
        for info in infos:
            self._validate_param_shape(info.fqn, info.global_shape)
            param_layout = bucket_layout.param_layouts[info.fqn]
            params.append(
                Fp8BucketedBlockShard._BlockRowParam(
                    fqn=info.fqn,
                    global_shape=info.global_shape,
                    param_offset=param_layout.param_offset,
                )
            )
        return params, self._build_block_row_units(params)

    def _build_fp8_metadata(
        self,
        params: list[_BlockRowParam],
        units: list[_BlockRowUnit],
        unit_cuts: tuple[int, ...],
    ) -> _Fp8BucketMetadata:
        params_by_fqn = {param.fqn: param for param in params}
        param_indices_by_fqn = {param.fqn: index for index, param in enumerate(params)}
        chunks: list[Fp8BucketedBlockShard._Fp8Chunk] = []
        rank_pack_plans: list[Fp8BucketedBlockShard._Fp8RankPackPlan] = []
        rank_fp8_numels: list[int] = []
        rank_scale_numels: list[int] = []
        for rank, (unit_start, unit_end) in enumerate(
            zip(unit_cuts[:-1], unit_cuts[1:], strict=True)
        ):
            rank_dense_offset = 0
            fp8_offset = 0
            scale_offset = 0
            rank_units = units[unit_start:unit_end]
            rank_pack_chunks: list[Fp8BucketedBlockShard._Fp8PackChunk] = []
            idx = 0
            while idx < len(rank_units):
                first = rank_units[idx]
                last = first
                idx += 1
                while idx < len(rank_units) and rank_units[idx].fqn == first.fqn:
                    last = rank_units[idx]
                    idx += 1
                param = params_by_fqn[first.fqn]
                _, in_dim = param.global_shape
                row_start = first.row_start
                row_end = last.row_end
                scale_row_start = row_start // self.block_size
                scale_row_end = _ceil_div(row_end, self.block_size)
                chunk_dense_numel = (row_end - row_start) * in_dim
                scale_numel = (scale_row_end - scale_row_start) * _ceil_div(
                    in_dim,
                    self.block_size,
                )
                chunks.append(
                    Fp8BucketedBlockShard._Fp8Chunk(
                        rank=rank,
                        fqn=param.fqn,
                        row_start=row_start,
                        row_end=row_end,
                        scale_row_start=scale_row_start,
                        scale_row_end=scale_row_end,
                        global_dense_offset=param.param_offset + row_start * in_dim,
                        rank_dense_offset=rank_dense_offset,
                        fp8_offset=fp8_offset,
                        scale_offset=scale_offset,
                    )
                )
                rank_pack_chunks.append(
                    Fp8BucketedBlockShard._Fp8PackChunk(
                        param_index=param_indices_by_fqn[param.fqn],
                        fqn=param.fqn,
                        local_shape=torch.Size((row_end - row_start, in_dim)),
                        fp8_numel=chunk_dense_numel,
                        scale_numel=scale_numel,
                    )
                )
                rank_dense_offset += chunk_dense_numel
                fp8_offset += chunk_dense_numel
                scale_offset += scale_numel
            rank_fp8_numels.append(fp8_offset)
            rank_scale_numels.append(scale_offset)
            rank_pack_plans.append(
                Fp8BucketedBlockShard._Fp8RankPackPlan(
                    chunks=tuple(rank_pack_chunks),
                    fp8_numel=fp8_offset,
                    scale_byte_offset=_align_up(fp8_offset, _VEC_ALIGN_BYTES),
                    packed_nbytes=_align_up(fp8_offset, _VEC_ALIGN_BYTES)
                    + scale_offset * torch.float32.itemsize,
                    fp8_chunk_numels=tuple(
                        chunk.fp8_numel for chunk in rank_pack_chunks
                    ),
                    scale_chunk_nbytes=tuple(
                        chunk.scale_numel * torch.float32.itemsize
                        for chunk in rank_pack_chunks
                    ),
                )
            )
        scale_itemsize = torch.float32.itemsize
        rank_scale_byte_offsets = tuple(
            _align_up(fp8_numel, _VEC_ALIGN_BYTES) for fp8_numel in rank_fp8_numels
        )
        rank_packed_nbytes = tuple(
            scale_byte_offset + scale_numel * scale_itemsize
            for scale_byte_offset, scale_numel in zip(
                rank_scale_byte_offsets,
                rank_scale_numels,
                strict=True,
            )
        )
        return Fp8BucketedBlockShard._Fp8BucketMetadata(
            chunks=tuple(chunks),
            rank_pack_plans=tuple(rank_pack_plans),
            rank_fp8_numels=tuple(rank_fp8_numels),
            rank_scale_numels=tuple(rank_scale_numels),
            rank_scale_byte_offsets=rank_scale_byte_offsets,
            rank_packed_nbytes=rank_packed_nbytes,
            packed_send_nbytes_per_rank=_align_up(
                max(rank_packed_nbytes, default=0),
                _VEC_ALIGN_BYTES,
            ),
        )

    def _fp8_metadata_from_infos(
        self,
        infos: list[ParamInfo],
    ) -> _Fp8BucketMetadata:
        bucket_layout = self._bucket_layout(infos[0])
        plan = self._metadata_plans.get(id(bucket_layout))
        if plan is not None and plan.bucket_layout is bucket_layout:
            return plan.metadata

        params, units = self._block_row_params_from_infos(infos)
        unit_cuts = self._unit_cuts_from_rank_layout(
            units,
            bucket_layout.rank_offsets,
            bucket_layout.rank_numels,
        )
        metadata = self._build_fp8_metadata(params, units, unit_cuts)
        self._metadata_plans[id(bucket_layout)] = self._Fp8MetadataPlan(
            bucket_layout=bucket_layout,
            metadata=metadata,
        )
        return metadata

    def _unit_cuts_from_rank_layout(
        self,
        units: list[_BlockRowUnit],
        rank_offsets: tuple[int, ...],
        rank_numels: tuple[int, ...],
    ) -> tuple[int, ...]:
        unit_dense_offsets = [0]
        for unit in units:
            unit_dense_offsets.append(unit_dense_offsets[-1] + unit.dense_numel)
        offset_to_unit = {offset: idx for idx, offset in enumerate(unit_dense_offsets)}
        rank_ends = tuple(
            start + numel
            for start, numel in zip(rank_offsets, rank_numels, strict=True)
        )
        try:
            return tuple(
                [offset_to_unit[offset] for offset in rank_offsets]
                + [offset_to_unit[rank_ends[-1]]]
            )
        except KeyError as error:
            raise AssertionError(
                "FP8 block-row rank layout is not aligned to row-block units."
            ) from error

    def _chunk_numel(
        self,
        chunk: _Fp8Chunk,
        in_dim: int,
    ) -> int:
        return (chunk.row_end - chunk.row_start) * in_dim

    def _quantize_local_weight(
        self,
        tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _quantize_dense_weight_to_blockwise_fp8(
            tensor,
            self.block_size,
            self.fp8_dtype,
        )

    def _make_blockwise_fp8_weight(
        self,
        fp8_data: torch.Tensor,
        recip_scale: torch.Tensor,
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        weight = self.weight_factory(
            fp8_data,
            recip_scale,
            self.block_size,
            orig_dtype=orig_dtype,
            requires_grad=requires_grad,
        )
        if not isinstance(weight, torch.Tensor):
            raise TypeError(
                "weight_factory must return a torch.Tensor, got "
                f"{type(weight).__name__}."
            )
        return weight

    @staticmethod
    def _local_packed_source_state(
        tensor: torch.Tensor,
    ) -> _LocalPackedSourceState:
        return Fp8BucketedBlockShard._LocalPackedSourceState(
            # Keep the storage alive so allocator address reuse cannot turn a
            # replacement tensor into a false cache hit.
            storage=tensor.untyped_storage(),
            version=tensor._version,
            storage_offset=tensor.storage_offset(),
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
            is_conj=tensor.is_conj(),
            is_neg=tensor.is_neg(),
        )

    @staticmethod
    def _local_packed_source_matches(
        tensor: torch.Tensor,
        state: _LocalPackedSourceState,
    ) -> bool:
        return (
            tensor._version == state.version
            and tensor.untyped_storage() is state.storage
            and tensor.storage_offset() == state.storage_offset
            and tensor.shape == state.shape
            and tensor.stride() == state.stride
            and tensor.dtype == state.dtype
            and tensor.is_conj() == state.is_conj
            and tensor.is_neg() == state.is_neg
        )

    @staticmethod
    def _local_packed_info_state(info: ParamInfo) -> _LocalPackedInfoState:
        return Fp8BucketedBlockShard._LocalPackedInfoState(
            info=info,
            unsharded_dtype=info.unsharded_dtype,
        )

    @staticmethod
    def _local_packed_info_matches(
        info: ParamInfo,
        state: _LocalPackedInfoState,
    ) -> bool:
        return info is state.info and info.unsharded_dtype == state.unsharded_dtype

    @staticmethod
    def _local_packed_cache_context(
        tensors: list[torch.Tensor],
    ) -> tuple[torch.device, Any]:
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        stream = torch.cuda.current_stream(device)
        return device, stream

    def _pack_local_quantized_chunks(  # noqa: C901
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        rank: int,
        metadata: _Fp8BucketMetadata,
    ) -> torch.Tensor:
        if len(tensors) != len(infos):
            raise ValueError(
                "FP8 copy-in requires one tensor per ParamInfo, got "
                f"{len(tensors)} tensors and {len(infos)} infos."
            )
        pack_plan = metadata.rank_pack_plans[rank]
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        local_packed = torch.empty(
            metadata.packed_send_nbytes_per_rank,
            dtype=torch.uint8,
            device=device,
        )
        fp8_end = pack_plan.fp8_numel
        scale_start = pack_plan.scale_byte_offset
        payload_end = pack_plan.packed_nbytes
        if fp8_end < scale_start:
            local_packed[fp8_end:scale_start].zero_()
        if payload_end < local_packed.numel():
            local_packed[payload_end:].zero_()
        local_fp8_bytes = local_packed[:fp8_end]
        local_scale_bytes = local_packed[scale_start:payload_end]
        local_chunks: list[
            tuple[Fp8BucketedBlockShard._Fp8PackChunk, torch.Tensor]
        ] = []
        staging_dsts: list[torch.Tensor] = []
        staging_srcs: list[torch.Tensor] = []
        for chunk in pack_plan.chunks:
            info = infos[chunk.param_index]
            if info.fqn != chunk.fqn:
                raise AssertionError(
                    "FP8 pack plan parameter order changed: expected "
                    f"{chunk.fqn!r} at index {chunk.param_index}, got {info.fqn!r}."
                )
            tensor = tensors[chunk.param_index].reshape(chunk.local_shape)
            if tensor.numel() != chunk.fp8_numel:
                raise AssertionError(
                    "FP8 local dense shard does not match planned chunk for "
                    f"{chunk.fqn!r}: expected {chunk.fp8_numel} elements, got "
                    f"{tensor.numel()}."
                )
            quant_dtype = info.unsharded_dtype
            if tensor.is_contiguous() and tensor.dtype == quant_dtype:
                quant_input = tensor
            else:
                quant_input = torch.empty(
                    tensor.shape,
                    dtype=quant_dtype,
                    device=tensor.device,
                )
                if tensor.numel() > 0:
                    staging_dsts.append(quant_input)
                    staging_srcs.append(tensor)
            local_chunks.append((chunk, quant_input))

        foreach_copy_(staging_dsts, staging_srcs)

        fp8_srcs: list[torch.Tensor] = []
        scale_srcs: list[torch.Tensor] = []
        for chunk, quant_input in local_chunks:
            quant, recip_scale = self._quantize_local_weight(quant_input)
            if quant.numel() != chunk.fp8_numel:
                raise AssertionError(
                    "FP8 quantizer returned the wrong data size for "
                    f"{chunk.fqn!r}: expected {chunk.fp8_numel}, got "
                    f"{quant.numel()}."
                )
            if recip_scale.numel() != chunk.scale_numel:
                raise AssertionError(
                    "FP8 quantizer returned the wrong scale size for "
                    f"{chunk.fqn!r}: expected {chunk.scale_numel}, got "
                    f"{recip_scale.numel()}."
                )
            fp8_srcs.append(quant.reshape(-1).view(torch.uint8))
            scale_srcs.append(recip_scale.reshape(-1).view(torch.uint8))

        if len(fp8_srcs) == 1:
            packed_dsts = [local_fp8_bytes, local_scale_bytes]
        elif fp8_srcs:
            packed_dsts = [
                *torch.split(local_fp8_bytes, pack_plan.fp8_chunk_numels),
                *torch.split(local_scale_bytes, pack_plan.scale_chunk_nbytes),
            ]
        else:
            packed_dsts = []
        foreach_copy_(packed_dsts, [*fp8_srcs, *scale_srcs])
        return local_packed

    def _cached_local_quantized_chunks(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        rank: int,
    ) -> tuple[torch.Tensor, _Fp8BucketMetadata]:
        """Cache one forward payload for its backward recomputation.

        Some fused optimizers mutate parameters without advancing ``_version``.
        Forward therefore always refreshes the entry; backward may consume that
        entry once when every secondary source and execution-context check agrees.
        """
        metadata = self._fp8_metadata_from_infos(infos)
        if torch.compiler.is_compiling():
            return (
                self._pack_local_quantized_chunks(tensors, infos, rank, metadata),
                metadata,
            )

        device, stream = self._local_packed_cache_context(tensors)
        cache_prefix = (rank, id(metadata))
        cache_key = (*cache_prefix, device, stream)
        if not _is_backward_graph_task():
            packed = self._pack_local_quantized_chunks(
                tensors,
                infos,
                rank,
                metadata,
            )
            for existing_key in tuple(self._local_packed_cache):
                if existing_key[:2] == cache_prefix:
                    del self._local_packed_cache[existing_key]
            self._local_packed_cache[cache_key] = self._LocalPackedCacheEntry(
                info_states=tuple(
                    self._local_packed_info_state(info) for info in infos
                ),
                source_states=tuple(
                    self._local_packed_source_state(tensor) for tensor in tensors
                ),
                metadata=metadata,
                packed=packed,
            )
            return packed, metadata

        cached = self._local_packed_cache.pop(cache_key, None)
        info_matches = (
            cached is not None
            and cached.metadata is metadata
            and len(cached.info_states) == len(infos)
            and all(
                self._local_packed_info_matches(info, state)
                for info, state in zip(infos, cached.info_states, strict=True)
            )
        )
        if (
            cached is not None
            and info_matches
            and len(cached.source_states) == len(tensors)
            and all(
                self._local_packed_source_matches(tensor, state)
                for tensor, state in zip(
                    tensors,
                    cached.source_states,
                    strict=True,
                )
            )
        ):
            return cached.packed, metadata

        return (
            self._pack_local_quantized_chunks(tensors, infos, rank, metadata),
            metadata,
        )

    @override
    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout:
        from ..flex_shard.bucket_storage import BucketLayout, BucketParamLayout

        if not named_params:
            return BucketStorageLayout(param_layouts={}, total_bytes=0)
        world_size = mesh.size()
        self._validate_world_size(world_size)

        rank = mesh.get_local_rank()
        dtype = named_params[0][1].dtype
        params, units = self._block_row_params_from_named(
            named_params,
            param_placements,
        )
        unit_costs = [unit.cost_bytes for unit in units]
        unit_dense_numels = [unit.dense_numel for unit in units]

        max_rank_cost_nbytes = self._min_partition_capacity(unit_costs, world_size)
        unit_cuts = self._rank_unit_cuts(
            unit_costs,
            world_size,
            max_rank_cost_nbytes,
        )
        unit_dense_offsets = [0]
        for unit_dense_numel in unit_dense_numels:
            unit_dense_offsets.append(unit_dense_offsets[-1] + unit_dense_numel)

        rank_offsets = tuple(unit_dense_offsets[cut] for cut in unit_cuts[:-1])
        rank_ends = tuple(unit_dense_offsets[cut] for cut in unit_cuts[1:])
        rank_numels = tuple(
            end - start for start, end in zip(rank_offsets, rank_ends, strict=True)
        )
        metadata = self._build_fp8_metadata(params, units, unit_cuts)
        local_chunks = {
            chunk.fqn: chunk for chunk in metadata.chunks if chunk.rank == rank
        }

        bucket_param_layouts = {}
        storage_layouts: dict[str, BucketParamStorageLayout] = {}
        for param in params:
            chunk = local_chunks.get(param.fqn)
            if chunk is None:
                local_shape = torch.Size([0, param.global_shape[1]])
                local_numel = 0
                byte_offset = 0
                local_global_offset = param.param_offset
            else:
                local_shape = torch.Size(
                    [chunk.row_end - chunk.row_start, param.global_shape[1]]
                )
                local_numel = self._chunk_numel(chunk, param.global_shape[1])
                byte_offset = chunk.rank_dense_offset * dtype.itemsize
                local_global_offset = chunk.global_dense_offset
            bucket_param_layouts[param.fqn] = BucketParamLayout(
                param_offset=param.param_offset,
                local_global_offset=local_global_offset,
            )
            storage_layouts[param.fqn] = BucketParamStorageLayout(
                local_shape=local_shape,
                local_numel=local_numel,
                byte_offset=byte_offset,
                storage_nbytes=local_numel * dtype.itemsize,
            )

        bucket_layout = BucketLayout(
            global_numel=unit_dense_offsets[-1],
            local_numel=rank_numels[rank],
            rank_offsets=rank_offsets,
            rank_numels=rank_numels,
            param_layouts=bucket_param_layouts,
        )
        self._metadata_plans[id(bucket_layout)] = self._Fp8MetadataPlan(
            bucket_layout=bucket_layout,
            metadata=metadata,
        )
        for fqn, layout in storage_layouts.items():
            storage_layouts[fqn] = BucketParamStorageLayout(
                local_shape=layout.local_shape,
                local_numel=layout.local_numel,
                byte_offset=layout.byte_offset,
                storage_nbytes=layout.storage_nbytes,
                bucket_layout=bucket_layout,
            )

        return BucketStorageLayout(
            param_layouts=storage_layouts,
            total_bytes=rank_numels[rank] * dtype.itemsize,
        )

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Quantize dense local shards into temporary fp8 + scale send buffers."""
        prepared = self._prepare_local_unshard_payload(
            tensors,
            infos,
            mesh,
            debug_fqn,
        )
        state = prepared.placement_state
        if not isinstance(state, Fp8BucketedBlockShard._Fp8UnshardState):
            raise AssertionError(
                "Expected Fp8BucketedBlockShard._Fp8UnshardState, "
                f"got {type(state).__name__}"
            )
        local = prepared.buffers[0]
        gathered = torch.empty(
            mesh.size() * local.numel(),
            dtype=local.dtype,
            device=local.device,
        )
        prepared.buffers.append(gathered)
        return prepared

    def _prepare_local_unshard_payload(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Prepare the local send payload without allocating gather output."""
        rank = mesh.get_local_rank()
        with _record_copy_in_if_eager():
            local_packed, metadata = self._cached_local_quantized_chunks(
                tensors,
                infos,
                rank,
            )

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_packed],
            placement_state=Fp8BucketedBlockShard._Fp8UnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                metadata=metadata,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """All-gather packed fp8 data and fp32 scale bytes in one collective.

        Each rank's payload is padded to the same size, so the collective writes
        directly into one flat output tensor. Reshard-after-forward tags the
        semantic unshard op ``MUST_RECOMPUTE`` -- so the gathered fp8 weight is
        freed after forward and re-gathered in backward. The local packed payload
        is reused while its dense source shards remain unchanged. Both fp8 data
        and fp32 scale bit patterns are transported as uint8, making the
        collective dtype-agnostic.
        """
        state = prepared.placement_state
        if not isinstance(state, Fp8BucketedBlockShard._Fp8UnshardState):
            raise AssertionError(
                "Expected Fp8BucketedBlockShard._Fp8UnshardState, "
                f"got {type(state).__name__}"
            )
        local, gathered = prepared.buffers
        with _record_comm_if_eager("FlexShard::all_gather", state.debug_fqn):
            dist.all_gather_into_tensor(
                output_tensor=gathered,
                input_tensor=local,
                group=state.pg,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Slice each param from this placement's contiguous gathered buffer."""
        if not isinstance(
            prepared.placement_state, Fp8BucketedBlockShard._Fp8UnshardState
        ):
            raise AssertionError(
                "Expected Fp8BucketedBlockShard._Fp8UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        rank_rows = prepared.buffers[1].view(
            len(state.metadata.rank_fp8_numels),
            prepared.buffers[0].numel(),
        )
        return self._finish_unshard_from_rank_rows(
            prepared,
            rank_rows,
        )

    def _finish_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
    ) -> PlacementUnshardResult:
        """Unpack directly from rank-major rows."""
        state = prepared.placement_state
        if not isinstance(state, Fp8BucketedBlockShard._Fp8UnshardState):
            raise AssertionError(
                "Expected Fp8BucketedBlockShard._Fp8UnshardState, "
                f"got {type(state).__name__}"
            )
        expected_shape = (
            len(state.metadata.rank_fp8_numels),
            state.metadata.packed_send_nbytes_per_rank,
        )
        if tuple(rank_rows.shape) != expected_shape:
            raise ValueError(
                f"Expected gathered rank rows with shape {expected_shape}, "
                f"got {tuple(rank_rows.shape)}."
            )
        if rank_rows.dtype != torch.uint8:
            raise ValueError(
                "Packed FP8 gathered rank rows must use uint8 transport, got "
                f"{rank_rows.dtype}."
            )
        return self._finish_fp8_unshard_from_rank_rows(
            prepared,
            rank_rows,
            state,
        )

    def _finish_fp8_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
        state: _Fp8UnshardState,
    ) -> PlacementUnshardResult:
        metadata = state.metadata
        fp8_nbytes = sum(metadata.rank_fp8_numels)
        scale_nbytes = sum(metadata.rank_scale_numels) * torch.float32.itemsize
        scale_byte_offset = _align_up(fp8_nbytes, _VEC_ALIGN_BYTES)
        compact = torch.empty(
            scale_byte_offset + scale_nbytes,
            dtype=torch.uint8,
            device=rank_rows.device,
        )
        fp8_bytes = compact[:fp8_nbytes]
        scale_bytes = compact[scale_byte_offset:]
        row_stride_bytes = rank_rows.stride(0)
        fp8_storage_regions = tuple(
            (rank * row_stride_bytes, rank_fp8_numel)
            for rank, rank_fp8_numel in enumerate(metadata.rank_fp8_numels)
        )
        scale_storage_regions = tuple(
            (
                rank * row_stride_bytes + scale_byte_start,
                rank_scale_numel * torch.float32.itemsize,
            )
            for rank, (rank_scale_numel, scale_byte_start) in enumerate(
                zip(
                    metadata.rank_scale_numels,
                    metadata.rank_scale_byte_offsets,
                    strict=True,
                )
            )
        )

        with _record_copy_out_if_eager():
            # Rank cuts partition one globally ordered block-row stream. Joining
            # each rank's valid regions therefore restores parameter order without
            # scattering every chunk separately. View the physical row storage once,
            # including any mixed-bucket padding between rows, so cat inputs do not
            # need intermediate per-rank views. Retain byte cats for unaligned or
            # odd inputs.
            use_vec_copies = _is_vec_copy_viewable(rank_rows) and _is_vec_copy_viewable(
                compact
            )
            if use_vec_copies:
                rank_storage_bytes = rank_rows.as_strided(
                    ((rank_rows.size(0) - 1) * row_stride_bytes + rank_rows.size(1),),
                    (1,),
                )
                rank_storage_copy_units = rank_storage_bytes.view(_VEC_COPY_DTYPE)
                compact_copy_units = compact.view(_VEC_COPY_DTYPE)
                if all(
                    rank_fp8_numel % _VEC_COPY_NBYTES == 0
                    for rank_fp8_numel in metadata.rank_fp8_numels
                ):
                    _cat_byte_regions_from_flat_storage(
                        fp8_storage_regions,
                        rank_storage_copy_units,
                        compact_copy_units[: fp8_nbytes // _VEC_COPY_NBYTES],
                        copy_unit_nbytes=_VEC_COPY_NBYTES,
                    )
                else:
                    _cat_byte_regions_from_flat_storage(
                        fp8_storage_regions,
                        rank_storage_bytes,
                        fp8_bytes,
                    )
                _cat_byte_regions_from_flat_storage(
                    scale_storage_regions,
                    rank_storage_copy_units,
                    compact_copy_units[scale_byte_offset // _VEC_COPY_NBYTES :],
                    copy_unit_nbytes=_VEC_COPY_NBYTES,
                )
            else:
                rank_byte_rows = rank_rows.unbind(0)
                _cat_byte_regions_from_rank_rows(
                    fp8_storage_regions,
                    rank_byte_rows,
                    fp8_bytes,
                    row_stride_bytes=row_stride_bytes,
                )
                _cat_byte_regions_from_rank_rows(
                    scale_storage_regions,
                    rank_byte_rows,
                    scale_bytes,
                    row_stride_bytes=row_stride_bytes,
                )

        fp8_flat = fp8_bytes.view(self.fp8_dtype)
        scale_flat = scale_bytes.view(torch.float32)
        fp8_offset = 0
        scale_offset = 0
        full_params: list[torch.Tensor] = []
        for info in state.infos:
            fp8_numel = info.global_numel
            scale_shape = _scale_shape(info.global_shape, self.block_size)
            scale_numel = scale_shape[0] * scale_shape[1]
            fp8_data = fp8_flat[fp8_offset : fp8_offset + fp8_numel].view(
                info.global_shape
            )
            recip_scale = scale_flat[scale_offset : scale_offset + scale_numel].view(
                scale_shape
            )
            full_params.append(
                self._make_blockwise_fp8_weight(
                    fp8_data,
                    recip_scale,
                    orig_dtype=info.unsharded_dtype,
                    requires_grad=info.requires_grad,
                )
            )
            fp8_offset += fp8_numel
            scale_offset += scale_numel
        if fp8_offset != fp8_flat.numel() or scale_offset != scale_flat.numel():
            raise AssertionError("Compacted FP8 bucket metadata is inconsistent.")
        return PlacementUnshardResult(
            full_params=full_params,
            consumer_buffers=[compact],
        )


def _cat_byte_regions_from_flat_storage(
    byte_regions: tuple[tuple[int, int], ...],
    source: torch.Tensor,
    out: torch.Tensor,
    *,
    copy_unit_nbytes: int = 1,
) -> None:
    torch.cat(
        [
            source.narrow(
                0,
                byte_offset // copy_unit_nbytes,
                byte_length // copy_unit_nbytes,
            )
            for byte_offset, byte_length in byte_regions
            if byte_length > 0
        ],
        out=out,
    )


def _cat_byte_regions_from_rank_rows(
    byte_regions: tuple[tuple[int, int], ...],
    rows: tuple[torch.Tensor, ...],
    out: torch.Tensor,
    *,
    row_stride_bytes: int,
) -> None:
    torch.cat(
        [
            row.narrow(
                0,
                byte_offset - rank * row_stride_bytes,
                byte_length,
            )
            for rank, (row, (byte_offset, byte_length)) in enumerate(
                zip(rows, byte_regions, strict=True)
            )
            if byte_length > 0
        ],
        out=out,
    )


def _align_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def _is_vec_copy_viewable(tensor: torch.Tensor) -> bool:
    """Return whether a uint8 tensor satisfies PyTorch's dtype-view contract."""
    if torch.compiler.is_compiling():
        return False
    return (
        tensor.dtype == torch.uint8
        and tensor.ndim > 0
        and tensor.stride(-1) == 1
        and tensor.size(-1) % _VEC_COPY_NBYTES == 0
        and tensor.storage_offset() % _VEC_COPY_NBYTES == 0
        and all(stride % _VEC_COPY_NBYTES == 0 for stride in tensor.stride()[:-1])
    )


def _is_backward_graph_task() -> bool:
    return torch._C._current_graph_task_id() >= 0


__all__ = [
    "BlockwiseFp8WeightFactory",
    "Fp8BucketedBlockShard",
    "make_fp8_bucketed_block_placement_fn",
]
