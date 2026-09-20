# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, TypeAlias

import torch
import torch.distributed as dist
import torch.nn as nn
from typing_extensions import override

from ..flex_shard.placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..flex_shard.utils import _record_comm_if_eager, _record_function_if_eager
from .utils import (
    _to_dist_reduce_op,
    copy_tensor_to_dtype,
    foreach_copy_,
    pack_tensors_into_flat_buffer_with_scratch,
)

try:
    from ..flex_shard.bucket_storage import (
        gradient_reduce_op_from_infos,
        GradientReduceOp,
    )
except ImportError:
    GradientReduceOp: TypeAlias = str

    def gradient_reduce_op_from_infos(infos: list[ParamInfo]) -> GradientReduceOp:
        _ = infos
        return "avg"


try:
    from ..flex_shard.utils import _record_copy_in_if_eager, _record_copy_out_if_eager
except ImportError:

    def _record_copy_in_if_eager():
        return _record_function_if_eager("FlexShard::copy_in", None)

    def _record_copy_out_if_eager():
        return _record_function_if_eager("FlexShard::copy_out", None)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import (
        BucketLayout,
        BucketParamLayout,
        ParamInfo,
        PlacementFn,
    )


def _unsharded_dtype(info: ParamInfo) -> torch.dtype:
    return getattr(info, "unsharded_dtype", info.dtype)


def _grad_reduce_dtype(info: ParamInfo) -> torch.dtype:
    return getattr(info, "grad_reduce_dtype", info.dtype)


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"Expected positive alignment, got {alignment}.")
    return ((value + alignment - 1) // alignment) * alignment


class BlockShard(Placement):
    """Per-parameter block sharding in complete blocks along one dimension.

    ``blocks_per_rank`` assigns each rank a number of contiguous blocks.
    Collective buffers reserve a fixed-size slot for each parameter and zero-pad
    ranks with smaller local shards; persistent parameter storage contains only
    the compact true local shard and preserves the parameter's rank.
    """

    @dataclass(frozen=True)
    class _PaddedBucketLayout:
        param_offsets: tuple[int, ...]
        block_numels: tuple[int, ...]
        padded_segment_numel: int

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        layout: BlockShard._PaddedBucketLayout
        pg: Any
        debug_fqn: str | None

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        layout: BlockShard._PaddedBucketLayout
        pg: Any
        debug_fqn: str | None
        gradient_reduce_op: GradientReduceOp

    def __init__(self, blocks_per_rank: tuple[int, ...], dim: int = 0) -> None:
        # Type annotations are not enforced at runtime, and bool subclasses int.
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"BlockShard dim must be an integer, got {dim!r}.")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in blocks_per_rank
        ):
            raise ValueError(
                "BlockShard blocks_per_rank must contain non-negative integers."
            )
        if sum(blocks_per_rank) == 0:
            raise ValueError(
                "BlockShard blocks_per_rank must contain at least one positive count."
            )
        self.dim = dim
        self.blocks_per_rank = blocks_per_rank

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlockShard)
            and self.dim == other.dim
            and self.blocks_per_rank == other.blocks_per_rank
        )

    def __hash__(self) -> int:
        return hash((type(self), self.dim, self.blocks_per_rank))

    def __repr__(self) -> str:
        if self.dim == 0:
            return f"BlockShard(blocks_per_rank={self.blocks_per_rank})"
        return f"BlockShard(blocks_per_rank={self.blocks_per_rank}, dim={self.dim})"

    def _validate_world_size(self, world_size: int) -> None:
        if len(self.blocks_per_rank) != world_size:
            raise ValueError(
                "BlockShard blocks_per_rank length must match world size: "
                f"got {len(self.blocks_per_rank)} counts for world size {world_size}."
            )

    def _normalize_dim(self, global_shape: torch.Size) -> int:
        if not global_shape:
            raise ValueError(
                "BlockShard requires a parameter with at least one dimension."
            )
        dim = self.dim if self.dim >= 0 else self.dim + len(global_shape)
        if dim < 0 or dim >= len(global_shape):
            raise ValueError(
                f"BlockShard dim {self.dim} is invalid for parameter shape "
                f"{tuple(global_shape)}."
            )
        return dim

    def _block_dim_size(self, global_shape: torch.Size) -> int:
        dim = self._normalize_dim(global_shape)
        total_blocks = sum(self.blocks_per_rank)
        dim_size = global_shape[dim]
        if dim_size % total_blocks != 0:
            raise ValueError(
                f"BlockShard requires global_shape[{dim}] to be divisible by "
                f"sum(blocks_per_rank): got size {dim_size} and blocks_per_rank "
                f"{self.blocks_per_rank}."
            )
        return dim_size // total_blocks

    def _block_numel(self, global_shape: torch.Size) -> int:
        dim = self._normalize_dim(global_shape)
        return (
            math.prod(global_shape[:dim])
            * self._block_dim_size(global_shape)
            * math.prod(global_shape[dim + 1 :])
        )

    def _shard_bounds_for_rank(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> tuple[int, int]:
        self._validate_world_size(world_size)
        if rank < 0 or rank >= world_size:
            raise ValueError(
                f"BlockShard rank must be in [0, {world_size}), got {rank}."
            )
        block_dim_size = self._block_dim_size(global_shape)
        start = sum(self.blocks_per_rank[:rank]) * block_dim_size
        end = start + self.blocks_per_rank[rank] * block_dim_size
        return start, end

    def _padded_bucket_layout(
        self,
        infos: list[ParamInfo],
    ) -> BlockShard._PaddedBucketLayout:
        param_offsets: list[int] = []
        block_numels: list[int] = []
        max_blocks_per_rank = max(self.blocks_per_rank)
        offset = 0
        for info in infos:
            param_offsets.append(offset)
            block_numel = self._block_numel(info.global_shape)
            block_numels.append(block_numel)
            offset += max_blocks_per_rank * block_numel
        return BlockShard._PaddedBucketLayout(
            param_offsets=tuple(param_offsets),
            block_numels=tuple(block_numels),
            padded_segment_numel=offset,
        )

    @staticmethod
    def _validate_bucket_inputs(
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
    ) -> None:
        if not tensors or not infos:
            raise ValueError("BlockShard requires at least one bucket parameter.")
        if len(tensors) != len(infos):
            raise ValueError(
                "BlockShard tensor and ParamInfo counts must match: "
                f"got {len(tensors)} tensors and {len(infos)} infos."
            )
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("BlockShard requires one device per bucket.")

    @override
    def compute_local_shape(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> torch.Size:
        dim = self._normalize_dim(global_shape)
        start, end = self._shard_bounds_for_rank(
            global_shape,
            rank,
            world_size,
        )
        local_shape = list(global_shape)
        local_shape[dim] = end - start
        return torch.Size(local_shape)

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        dim = self._normalize_dim(param.shape)
        start, end = self._shard_bounds_for_rank(param.shape, rank, world_size)
        return param.narrow(dim, start, end - start)

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Pack true local shards into fixed-size, per-parameter padded slots."""
        self._validate_bucket_inputs(tensors, infos)
        world_size = mesh.size()
        self._validate_world_size(world_size)
        layout = self._padded_bucket_layout(infos)
        dtype = _unsharded_dtype(infos[0])
        device = tensors[0].device
        if any(_unsharded_dtype(info) != dtype for info in infos):
            raise ValueError("BlockShard requires one unsharded dtype per bucket.")

        with _record_copy_in_if_eager():
            send_buf = torch.zeros(
                layout.padded_segment_numel,
                dtype=dtype,
                device=device,
            )
            copy_dsts: list[torch.Tensor] = []
            copy_srcs: list[torch.Tensor] = []
            for index, (tensor, info) in enumerate(zip(tensors, infos, strict=True)):
                expected_numel = info.local_numel
                if tensor.numel() != expected_numel:
                    raise ValueError(
                        f"BlockShard local tensor {info.fqn!r} has "
                        f"{tensor.numel()} elements, expected {expected_numel}."
                    )
                if expected_numel > 0:
                    offset = layout.param_offsets[index]
                    copy_dsts.append(send_buf[offset : offset + expected_numel])
                    copy_srcs.append(tensor.reshape(-1))
            foreach_copy_(copy_dsts, copy_srcs)
            gathered = torch.empty(
                world_size * layout.padded_segment_numel,
                dtype=dtype,
                device=device,
            )

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send_buf, gathered],
            placement_state=BlockShard._UnshardState(
                infos=infos,
                layout=layout,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        if not isinstance(prepared.placement_state, BlockShard._UnshardState):
            raise AssertionError(
                "Expected BlockShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        with _record_comm_if_eager(
            "FlexShard::all_gather",
            prepared.placement_state.debug_fqn,
        ):
            dist.all_gather_into_tensor(
                output_tensor=prepared.buffers[1],
                input_tensor=prepared.buffers[0],
                group=prepared.placement_state.pg,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        if not isinstance(prepared.placement_state, BlockShard._UnshardState):
            raise AssertionError(
                "Expected BlockShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        world_size = len(self.blocks_per_rank)
        gathered = prepared.buffers[1].view(
            world_size,
            state.layout.padded_segment_numel,
        )
        full_params: list[torch.Tensor] = []
        with _record_copy_out_if_eager():
            for index, info in enumerate(state.infos):
                offset = state.layout.param_offsets[index]
                rank_shards = [
                    gathered[
                        rank,
                        offset : offset
                        + self.blocks_per_rank[rank] * state.layout.block_numels[index],
                    ].view(
                        self.compute_local_shape(
                            info.global_shape,
                            rank,
                            world_size,
                        )
                    )
                    for rank in range(world_size)
                ]
                full_params.append(
                    torch.cat(
                        rank_shards,
                        dim=self._normalize_dim(info.global_shape),
                    )
                )
        return PlacementUnshardResult(full_params=full_params)

    def _pack_reduce_scatter_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        world_size: int,
    ) -> tuple[torch.Tensor, BlockShard._PaddedBucketLayout]:
        self._validate_world_size(world_size)
        layout = self._padded_bucket_layout(infos)
        dtype = _grad_reduce_dtype(infos[0])
        device = tensors[0].device
        if any(_grad_reduce_dtype(info) != dtype for info in infos):
            raise ValueError("BlockShard requires one reduce dtype per bucket.")
        send_buf = torch.zeros(
            world_size * layout.padded_segment_numel,
            dtype=dtype,
            device=device,
        )
        send_rows = send_buf.view(world_size, layout.padded_segment_numel)
        copy_dsts: list[torch.Tensor] = []
        copy_srcs: list[torch.Tensor] = []
        for tensor, info in zip(tensors, infos, strict=True):
            if tensor.shape != info.global_shape:
                raise ValueError(
                    f"BlockShard full gradient {info.fqn!r} has shape "
                    f"{tuple(tensor.shape)}, expected {tuple(info.global_shape)}."
                )

        for rank in range(world_size):
            for index, (tensor, block_numel) in enumerate(
                zip(tensors, layout.block_numels, strict=True)
            ):
                numel = self.blocks_per_rank[rank] * block_numel
                if numel > 0:
                    offset = layout.param_offsets[index]
                    local_shard = self.extract_local_shard(
                        tensor,
                        rank,
                        world_size,
                    )
                    copy_dsts.append(
                        send_rows[rank, offset : offset + numel].view(local_shard.shape)
                    )
                    copy_srcs.append(local_shard)
        foreach_copy_(copy_dsts, copy_srcs)
        return send_buf, layout

    def _unpack_reduce_scatter_grad(
        self,
        recv_buf: torch.Tensor,
        state: BlockShard._ReduceGradState,
    ) -> list[torch.Tensor]:
        sharded_grads: list[torch.Tensor] = []
        for index, info in enumerate(state.infos):
            offset = state.layout.param_offsets[index]
            sharded_grads.append(
                recv_buf[offset : offset + info.local_numel].view(info.local_shape)
            )
        return sharded_grads

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        self._validate_bucket_inputs(tensors, infos)
        world_size = mesh.size()
        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_in",
            debug_fqn,
        ):
            send_buf, layout = self._pack_reduce_scatter_grad(
                tensors,
                infos,
                world_size,
            )
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send_buf],
            placement_state=BlockShard._ReduceGradState(
                infos=infos,
                layout=layout,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                gradient_reduce_op=gradient_reduce_op_from_infos(infos),
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        if not isinstance(prepared.placement_state, BlockShard._ReduceGradState):
            raise AssertionError(
                "Expected BlockShard._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        send_buf = prepared.buffers[0]
        recv_buf = torch.empty(
            state.layout.padded_segment_numel,
            dtype=send_buf.dtype,
            device=send_buf.device,
        )
        with _record_comm_if_eager(
            "FlexShard::post_backward_reduce",
            state.debug_fqn,
        ):
            dist.reduce_scatter_tensor(
                output=recv_buf,
                input=send_buf,
                op=_to_dist_reduce_op(state.gradient_reduce_op),
                group=state.pg,
            )
        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out",
            state.debug_fqn,
        ):
            sharded_grads = self._unpack_reduce_scatter_grad(
                recv_buf,
                state,
            )
        return PlacementReduceGradResult(sharded_grads, [recv_buf])


class BucketedBlockShard(Placement):
    """Block-shard one bucket-global DBuffer-style param-major layout.

    ``dims`` selects the prefix dimensions that must remain row-aligned; currently
    they must be ``(0,)``, ``(0, 1)``, etc. ``blocks_per_rank`` assigns each rank
    a number of contiguous bucket-global blocks. The placement plans the whole
    bucket as one param-major logical buffer, then shards that buffer into
    rank-local ranges so the all-gather output is directly viewable as full
    parameters.
    """

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        pg: Any
        debug_fqn: str | None
        num_gathered_views: int

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        rank: int
        pg: Any
        debug_fqn: str | None
        padded_segment_numel: int
        gradient_reduce_op: GradientReduceOp

    def __init__(
        self,
        dims: tuple[int, ...] = (0,),
        blocks_per_rank: tuple[int, ...] = (1,),
    ) -> None:
        if not dims:
            raise ValueError("BucketedBlockShard dims must be non-empty.")
        expected_prefix_dims = tuple(range(len(dims)))
        if dims != expected_prefix_dims:
            raise ValueError(
                "BucketedBlockShard currently requires prefix dims "
                f"{expected_prefix_dims}, but got {dims}."
            )
        if any(block_count < 0 for block_count in blocks_per_rank):
            raise ValueError("BucketedBlockShard blocks_per_rank must be non-negative.")
        if sum(blocks_per_rank) == 0:
            raise ValueError(
                "BucketedBlockShard blocks_per_rank must contain at least one "
                "positive count."
            )
        self.dims = dims
        self.blocks_per_rank = blocks_per_rank

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BucketedBlockShard):
            return False
        return self.dims == other.dims and self.blocks_per_rank == other.blocks_per_rank

    def __hash__(self) -> int:
        return hash((type(self), self.dims, self.blocks_per_rank))

    def __repr__(self) -> str:
        return (
            "BucketedBlockShard("
            f"dims={self.dims}, blocks_per_rank={self.blocks_per_rank})"
        )

    def _validate_world_size(self, world_size: int) -> None:
        if len(self.blocks_per_rank) != world_size:
            raise ValueError(
                "BucketedBlockShard blocks_per_rank length must match world size: "
                f"got {len(self.blocks_per_rank)} block counts for world size "
                f"{world_size}."
            )

    def _prefix_numel(self, global_shape: torch.Size) -> int:
        if len(global_shape) < len(self.dims):
            raise ValueError(
                f"BucketedBlockShard dims {self.dims} are invalid for parameter shape "
                f"{tuple(global_shape)}."
            )
        return math.prod(global_shape[: len(self.dims)])

    def _suffix_shape(self, global_shape: torch.Size) -> torch.Size:
        return torch.Size(global_shape[len(self.dims) :])

    def _suffix_numel(self, global_shape: torch.Size) -> int:
        return math.prod(self._suffix_shape(global_shape))

    def _bucket_layout(self, info: ParamInfo) -> BucketLayout:
        layout = info.bucket_layout
        if layout is None:
            raise AssertionError(
                "Expected BucketedBlockShard ParamInfo to carry a bucket layout."
            )
        return layout

    def _param_layout(self, info: ParamInfo) -> BucketParamLayout:
        layout = self._bucket_layout(info)
        return layout.param_layouts[info.fqn]

    @override
    def compute_local_shape(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> torch.Size:
        raise NotImplementedError(
            "BucketedBlockShard computes local shapes from the whole bucket. "
            "Use bucket_storage_layout() instead."
        )

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "BucketedBlockShard extracts local shards from bucket-global ranges. "
            "Use bucket storage views instead."
        )

    def _param_alignment_numel(
        self,
        named_params: list[tuple[str, nn.Parameter]],
    ) -> int:
        alignment = 1
        for _, param in named_params:
            self._prefix_numel(param.shape)
            suffix_numel = self._suffix_numel(param.shape)
            alignment = math.lcm(alignment, suffix_numel)
        return alignment

    def _bucket_param_offsets(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        alignment_numel: int,
    ) -> tuple[dict[str, int], int]:
        offsets: dict[str, int] = {}
        current_offset = 0
        for fqn, param in named_params:
            current_offset = _align_up(current_offset, alignment_numel)
            offsets[fqn] = current_offset
            current_offset += param.numel()
        return offsets, current_offset

    def _bucket_rank_layout(
        self,
        unpadded_global_numel: int,
        alignment_numel: int,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        total_blocks = sum(self.blocks_per_rank)
        padded_global_numel = _align_up(
            unpadded_global_numel,
            alignment_numel * total_blocks,
        )
        block_numel = padded_global_numel // total_blocks
        offsets: list[int] = []
        numels: list[int] = []
        offset = 0
        for block_count in self.blocks_per_rank:
            offsets.append(offset)
            numel = block_count * block_numel
            numels.append(numel)
            offset += numel
        return padded_global_numel, tuple(offsets), tuple(numels)

    def _local_shape_from_numel(
        self,
        global_shape: torch.Size,
        local_numel: int,
    ) -> torch.Size:
        suffix_shape = self._suffix_shape(global_shape)
        suffix_numel = math.prod(suffix_shape)
        if local_numel % suffix_numel != 0:
            raise ValueError(
                "BucketedBlockShard bucket split produced a shard that is not "
                f"aligned to flattened prefix rows for shape {tuple(global_shape)}."
            )
        return torch.Size([local_numel // suffix_numel, *suffix_shape])

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
        self._validate_world_size(mesh.size())

        rank = mesh.get_local_rank()
        dtype = named_params[0][1].dtype
        alignment_numel = self._param_alignment_numel(named_params)
        param_offsets, unpadded_global_numel = self._bucket_param_offsets(
            named_params,
            alignment_numel,
        )
        (
            padded_global_numel,
            rank_offsets,
            rank_numels,
        ) = self._bucket_rank_layout(unpadded_global_numel, alignment_numel)
        rank_start = rank_offsets[rank]
        rank_end = rank_start + rank_numels[rank]

        local_metadata: dict[str, tuple[torch.Size, int, int]] = {}
        param_layouts: dict[str, BucketParamLayout] = {}
        has_local_param_data = False
        for fqn, param in named_params:
            if param.dtype != dtype:
                raise ValueError(
                    "BucketedBlockShard requires one dtype per bucket: "
                    f"{named_params[0][0]!r} uses {dtype} but {fqn!r} uses "
                    f"{param.dtype}."
                )
            placements = param_placements[fqn]
            if placements != (self,):
                raise ValueError(
                    "BucketedBlockShard requires the same placement instance "
                    f"for every parameter in a bucket; {fqn!r} uses {placements!r}."
                )
            param_start = param_offsets[fqn]
            param_end = param_start + param.numel()
            local_start = max(rank_start, param_start)
            local_end = min(rank_end, param_end)
            local_numel = max(0, local_end - local_start)
            if local_numel > 0:
                has_local_param_data = True
            local_shape = self._local_shape_from_numel(param.shape, local_numel)
            byte_offset = (local_start - rank_start) * dtype.itemsize
            local_metadata[fqn] = (local_shape, local_numel, byte_offset)
            param_layouts[fqn] = BucketParamLayout(
                param_offset=param_start,
                local_global_offset=local_start,
            )

        bucket_layout = BucketLayout(
            global_numel=padded_global_numel,
            local_numel=rank_numels[rank],
            rank_offsets=rank_offsets,
            rank_numels=rank_numels,
            param_layouts=param_layouts,
        )
        storage_layouts: dict[str, BucketParamStorageLayout] = {}
        for fqn, _ in named_params:
            local_shape, local_numel, byte_offset = local_metadata[fqn]
            storage_layouts[fqn] = BucketParamStorageLayout(
                local_shape=local_shape,
                local_numel=local_numel,
                byte_offset=byte_offset,
                storage_nbytes=local_numel * dtype.itemsize,
                bucket_layout=bucket_layout,
            )

        if rank_numels[rank] > 0 and not has_local_param_data:
            raise ValueError(
                "BucketedBlockShard planned a rank-local bucket range that "
                "contains only padding. Split the bucket or choose blocks_per_rank "
                "that assign parameter data to every non-empty rank."
            )

        return BucketStorageLayout(
            param_layouts=storage_layouts,
            total_bytes=rank_numels[rank] * dtype.itemsize,
        )

    @override
    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        param_data = param.detach()
        if param_data.device.type == "meta" or info.local_numel == 0:
            return
        param_layout = self._param_layout(info)
        param_offset = param_layout.local_global_offset - param_layout.param_offset
        shard = param_data.contiguous().view(-1)[
            param_offset : param_offset + info.local_numel
        ]
        nbytes = shard.numel() * shard.element_size()
        byte_storage[info.byte_offset : info.byte_offset + nbytes].copy_(
            shard.view(torch.uint8)
        )

    def _make_local_bucket_view(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
    ) -> torch.Tensor:
        bucket_layout = self._bucket_layout(infos[0])
        local_bucket_numel = bucket_layout.local_numel
        if local_bucket_numel == 0:
            return tensors[0].new_empty(0)

        bucket_storage: torch.UntypedStorage | None = None
        bucket_start_ptr: int | None = None
        bucket_tensor: torch.Tensor | None = None
        for tensor, info in zip(tensors, infos, strict=True):
            if tensor.numel() == 0:
                continue
            storage = tensor.untyped_storage()
            storage_start_ptr = tensor.data_ptr() - info.byte_offset
            if bucket_storage is None:
                bucket_storage = storage
                bucket_start_ptr = storage_start_ptr
                bucket_tensor = tensor
            elif (
                storage.data_ptr() != bucket_storage.data_ptr()
                or storage_start_ptr != bucket_start_ptr
            ):
                raise ValueError(
                    "BucketedBlockShard expected local tensors to share one "
                    "bucket storage allocation."
                )
            storage_offset_bytes = storage_start_ptr - storage.data_ptr()
            if storage_offset_bytes % tensor.element_size() != 0:
                raise AssertionError(
                    "BucketedBlockShard local bucket storage is not dtype-aligned."
                )

        if bucket_storage is None or bucket_start_ptr is None or bucket_tensor is None:
            raise ValueError(
                "BucketedBlockShard cannot create a view-in send buffer because "
                "this rank has no parameter tensor covering its non-empty bucket "
                "range."
            )
        storage_offset_bytes = bucket_start_ptr - bucket_storage.data_ptr()
        storage_offset = storage_offset_bytes // bucket_tensor.element_size()
        return bucket_tensor.new_empty(0).set_(
            bucket_storage,
            storage_offset,
            (local_bucket_numel,),
            (1,),
        )

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        dtype = _unsharded_dtype(infos[0])
        device = tensors[0].device
        with _record_copy_in_if_eager():
            send_buf = self._make_local_bucket_view(tensors, infos)
            copy_in_scratch: list[torch.Tensor] = []
            if send_buf.dtype != dtype:
                send_buf, copy_in_scratch = pack_tensors_into_flat_buffer_with_scratch(
                    [send_buf],
                    dtype,
                )
            else:
                send_buf = copy_tensor_to_dtype(send_buf, dtype)
            bucket_layout = self._bucket_layout(infos[0])
            gathered_bucket = torch.empty(
                bucket_layout.global_numel,
                dtype=dtype,
                device=device,
            )
            gathered_views = [
                gathered_bucket[offset : offset + numel]
                for offset, numel in zip(
                    bucket_layout.rank_offsets,
                    bucket_layout.rank_numels,
                    strict=True,
                )
            ]
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send_buf, gathered_bucket, *gathered_views, *copy_in_scratch],
            placement_state=BucketedBlockShard._UnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                num_gathered_views=len(gathered_views),
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        if not isinstance(prepared.placement_state, BucketedBlockShard._UnshardState):
            raise AssertionError(
                "Expected BucketedBlockShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        send_buf = prepared.buffers[0]
        gathered_views = prepared.buffers[
            2 : 2 + prepared.placement_state.num_gathered_views
        ]
        with _record_comm_if_eager(
            "FlexShard::all_gather",
            prepared.placement_state.debug_fqn,
        ):
            dist.all_gather(gathered_views, send_buf, group=prepared.placement_state.pg)

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        if not isinstance(prepared.placement_state, BucketedBlockShard._UnshardState):
            raise AssertionError(
                "Expected BucketedBlockShard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        gathered_bucket = prepared.buffers[1]
        full_params: list[torch.Tensor] = []
        for info in prepared.placement_state.infos:
            param_offset = self._param_layout(info).param_offset
            full_params.append(
                gathered_bucket[param_offset : param_offset + info.global_numel].view(
                    info.global_shape
                )
            )
        return PlacementUnshardResult(
            full_params=full_params,
            consumer_buffers=[gathered_bucket],
        )

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        world_size = mesh.size()
        dtype = _grad_reduce_dtype(infos[0])
        device = tensors[0].device
        bucket_layout = self._bucket_layout(infos[0])
        padded_segment_numel = max(bucket_layout.rank_numels)
        with _record_function_if_eager("FlexShard::reduce_scatter_copy_in", debug_fqn):
            global_grad_bucket = torch.zeros(
                bucket_layout.global_numel,
                dtype=dtype,
                device=device,
            )
            copy_dsts: list[torch.Tensor] = []
            copy_srcs: list[torch.Tensor] = []
            for tensor, info in zip(tensors, infos, strict=True):
                param_layout = self._param_layout(info)
                copy_srcs.append(tensor.reshape(-1))
                copy_dsts.append(
                    global_grad_bucket[
                        param_layout.param_offset : param_layout.param_offset
                        + info.global_numel
                    ]
                )
            foreach_copy_(copy_dsts, copy_srcs)

            send_buf = torch.zeros(
                world_size * padded_segment_numel,
                dtype=dtype,
                device=device,
            )
            send_buf_by_rank = send_buf.view(world_size, padded_segment_numel)
            copy_dsts = []
            copy_srcs = []
            for rank, (offset, numel) in enumerate(
                zip(
                    bucket_layout.rank_offsets,
                    bucket_layout.rank_numels,
                    strict=True,
                )
            ):
                if numel > 0:
                    copy_srcs.append(global_grad_bucket[offset : offset + numel])
                    copy_dsts.append(send_buf_by_rank[rank, :numel])
            foreach_copy_(copy_dsts, copy_srcs)

        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send_buf],
            placement_state=BucketedBlockShard._ReduceGradState(
                infos=infos,
                rank=mesh.get_local_rank(),
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                padded_segment_numel=padded_segment_numel,
                gradient_reduce_op=gradient_reduce_op_from_infos(infos),
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        if not isinstance(
            prepared.placement_state, BucketedBlockShard._ReduceGradState
        ):
            raise AssertionError(
                "Expected BucketedBlockShard._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        send_buf = prepared.buffers[0]
        recv_buf = torch.empty(
            state.padded_segment_numel,
            dtype=send_buf.dtype,
            device=send_buf.device,
        )
        with _record_comm_if_eager(
            "FlexShard::post_backward_reduce",
            state.debug_fqn,
        ):
            dist.reduce_scatter_tensor(
                output=recv_buf,
                input=send_buf,
                op=_to_dist_reduce_op(state.gradient_reduce_op),
                group=state.pg,
            )

        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out",
            state.debug_fqn,
        ):
            sharded_grads = [
                recv_buf[
                    info.byte_offset // info.dtype.itemsize : info.byte_offset
                    // info.dtype.itemsize
                    + info.local_numel
                ].view(info.local_shape)
                for info in state.infos
            ]
        return PlacementReduceGradResult(sharded_grads, [recv_buf])


def make_bucketed_block_placement_fn(
    *,
    dims: tuple[int, ...],
    blocks_per_rank: tuple[int, ...],
) -> PlacementFn:
    """Return a function assigning one BucketedBlockShard per bucket."""

    def bucketed_block_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        if len(blocks_per_rank) != mesh.size():
            raise ValueError(
                "BucketedBlockShard blocks_per_rank length must match mesh size: "
                f"got {len(blocks_per_rank)} block counts for mesh size "
                f"{mesh.size()}."
            )
        placement = BucketedBlockShard(dims=dims, blocks_per_rank=blocks_per_rank)
        return {fqn: (placement,) for fqn, _ in named_params}

    return bucketed_block_placements


__all__ = [
    "BucketedBlockShard",
    "make_bucketed_block_placement_fn",
    "BlockShard",
]
