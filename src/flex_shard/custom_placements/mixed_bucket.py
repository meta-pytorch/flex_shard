# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist

from ..flex_shard.bucket_storage import gradient_reduce_op_from_infos
from ..flex_shard.placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..flex_shard.utils import (
    _record_comm_if_eager,
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
    _record_function_if_eager,
)
from .block_shard import BlockShard
from .fp8_bucketed_block_shard import _align_up, _VEC_ALIGN_BYTES, Fp8BucketedBlockShard
from .owned import BucketedOwned
from .shard import Shard

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo


@dataclass(frozen=True)
class _PlacementGroup:
    placement: Placement
    indices: list[int]
    tensors: list[torch.Tensor]
    infos: list[ParamInfo]


@dataclass(frozen=True)
class _MixedUnshardGroupLayout:
    offset: int
    numel: int
    owner_rank: int | None


@dataclass(frozen=True)
class _MixedUnshardLayout:
    groups: tuple[_MixedUnshardGroupLayout, ...]
    dtype: torch.dtype
    requires_dtype_reinterpretation: bool
    row_numel: int
    padding_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _MixedUnshardGroupState:
    prepared: PlacementPreparedUnshard
    indices: tuple[int, ...]
    offset: int
    numel: int


@dataclass(frozen=True)
class _MixedUnshardState:
    groups: list[_MixedUnshardGroupState]
    world_size: int
    row_numel: int
    requires_dtype_reinterpretation: bool
    pg: Any
    debug_fqn: str | None


@dataclass(frozen=True)
class _MixedReduceGradGroupState:
    prepared: PlacementPreparedReduceGrad
    indices: list[int]
    offset: int
    numel: int


@dataclass(frozen=True)
class _MixedReduceGradState:
    groups: list[_MixedReduceGradGroupState]
    world_size: int
    row_numel: int
    pg: Any
    debug_fqn: str | None
    gradient_reduce_op: dist.ReduceOp


class _MixedBucketMember:
    _mixed_bucket: MixedBucketPlacement
    _collective_placement: Placement

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _MixedBucketMember):
            other = other._collective_placement
        return self._collective_placement == other

    def __hash__(self) -> int:
        return hash(self._collective_placement)

    def bucket_compatibility_key(self) -> object:
        return self._mixed_bucket

    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        return self._mixed_bucket.bucket_storage_layout(
            named_params,
            param_placements,
            mesh,
        )

    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        return self._mixed_bucket.prepare_unshard_bucket(
            tensors,
            infos,
            mesh,
            debug_fqn,
        )

    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        self._mixed_bucket.run_prepared_unshard(prepared)

    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        return self._mixed_bucket.finish_prepared_unshard(prepared)

    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        return self._mixed_bucket.prepare_reduce_grad(
            tensors,
            infos,
            mesh,
            debug_fqn,
        )

    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        return self._mixed_bucket.reduce_prepared_grad(prepared)


class _MixedShard0(_MixedBucketMember, Shard):
    """Optimizer-visible Shard(0) member of one mixed bucket placement."""

    __eq__ = _MixedBucketMember.__eq__
    __hash__ = _MixedBucketMember.__hash__

    def __init__(self, mixed_bucket: MixedBucketPlacement) -> None:
        Shard.__init__(self, 0)
        self._mixed_bucket = mixed_bucket
        self._collective_placement = mixed_bucket._shard


class _MixedBlockShard(_MixedBucketMember, BlockShard):
    """Optimizer-visible per-parameter BlockShard mixed-bucket member."""

    __eq__ = _MixedBucketMember.__eq__
    __hash__ = _MixedBucketMember.__hash__

    def __init__(
        self,
        mixed_bucket: MixedBucketPlacement,
        blocks_per_rank: tuple[int, ...],
        dim: int,
    ) -> None:
        BlockShard.__init__(self, blocks_per_rank=blocks_per_rank, dim=dim)
        self._mixed_bucket = mixed_bucket
        self._collective_placement = BlockShard(
            blocks_per_rank=blocks_per_rank,
            dim=dim,
        )


class _MixedBucketedOwned(_MixedBucketMember, BucketedOwned):
    """Optimizer-visible BucketedOwned member of one mixed bucket placement."""

    __eq__ = _MixedBucketMember.__eq__
    __hash__ = _MixedBucketMember.__hash__

    def __init__(
        self,
        mixed_bucket: MixedBucketPlacement,
        segments_by_fqn: dict[str, list[Any]],
    ) -> None:
        BucketedOwned.__init__(
            self,
            segments_by_fqn,
            view_kind="full_param",
        )
        self._mixed_bucket = mixed_bucket
        self._collective_placement = mixed_bucket._bucketed_owned


class _MixedFp8BucketedBlockShard(
    _MixedBucketMember,
    Fp8BucketedBlockShard,
):
    """Optimizer-visible blockwise FP8 member of one mixed bucket placement."""

    __eq__ = _MixedBucketMember.__eq__
    __hash__ = _MixedBucketMember.__hash__

    def __init__(
        self,
        mixed_bucket: MixedBucketPlacement,
        placement: Fp8BucketedBlockShard,
    ) -> None:
        Fp8BucketedBlockShard.__init__(
            self,
            world_size=placement.world_size,
            weight_factory=placement.weight_factory,
            block_size=placement.block_size,
            fp8_dtype=placement.fp8_dtype,
        )
        self._mixed_bucket = mixed_bucket
        self._collective_placement = placement


class MixedBucketPlacement(Placement):
    """One collective for the built-in mixed-bucket placement groups."""

    def __init__(self, segments_by_fqn: dict[str, list[Any]]) -> None:
        self._shard = Shard(0)
        self._bucketed_owned = BucketedOwned(
            segments_by_fqn,
            view_kind="full_param",
        )
        self.shard0 = _MixedShard0(self)
        self.bucketed_owned = _MixedBucketedOwned(
            self,
            segments_by_fqn,
        )
        self._block_shard_members_by_config: dict[
            tuple[int, tuple[int, ...]], _MixedBlockShard
        ] = {}
        self._fp8_bucketed_block_member: _MixedFp8BucketedBlockShard | None = None

    def block_shard(
        self,
        *,
        blocks_per_rank: tuple[int, ...],
        dim: int = 0,
    ) -> _MixedBlockShard:
        key = (dim, blocks_per_rank)
        member = self._block_shard_members_by_config.get(key)
        if member is None:
            member = _MixedBlockShard(self, blocks_per_rank, dim)
            self._block_shard_members_by_config[key] = member
        return member

    def fp8_bucketed_block_shard(
        self,
        placement: Fp8BucketedBlockShard,
    ) -> Fp8BucketedBlockShard:
        member = self._fp8_bucketed_block_member
        if member is None:
            member = _MixedFp8BucketedBlockShard(self, placement)
            self._fp8_bucketed_block_member = member
        elif member._collective_placement is not placement:
            raise NotImplementedError(
                "A mixed bucket supports one FP8 bucketed block placement instance."
            )
        return member

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(id(self))

    def compute_local_shape(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> torch.Size:
        raise NotImplementedError("Mixed bucket placement is bucket-scoped.")

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError("Mixed bucket placement is bucket-scoped.")

    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, torch.nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        local_params: list[tuple[str, torch.nn.Parameter, Placement]] = []
        for fqn, param in named_params:
            (placement,) = param_placements[fqn]
            if not isinstance(placement, _MixedBucketMember):
                raise TypeError(
                    "MixedBucketPlacement requires mixed member "
                    f"placements, but {fqn!r} uses {type(placement).__name__}."
                )
            if placement._mixed_bucket is not self:
                raise ValueError(
                    "MixedBucketPlacement cannot mix member "
                    "placements from different bucket instances."
                )
            local_params.append(
                (
                    fqn,
                    param,
                    _collective_placement(
                        placement,
                        fqn,
                        param.shape,
                        world_size,
                    ),
                )
            )

        param_layouts: dict[str, BucketParamStorageLayout] = {}
        byte_offset = 0
        for local_group in _group_local_params_by_placement(local_params):
            collective_placement = local_group[0][2]
            group_named_params = [(fqn, param) for fqn, param, _ in local_group]
            group_storage_layout = collective_placement.bucket_storage_layout(
                group_named_params,
                {fqn: (collective_placement,) for fqn, _ in group_named_params},
                mesh,
            )
            if group_storage_layout is not None:
                for fqn, layout in group_storage_layout.param_layouts.items():
                    param_layouts[fqn] = BucketParamStorageLayout(
                        local_shape=layout.local_shape,
                        local_numel=layout.local_numel,
                        byte_offset=byte_offset + layout.byte_offset,
                        storage_nbytes=layout.storage_nbytes,
                        bucket_layout=layout.bucket_layout,
                    )
                byte_offset += group_storage_layout.total_bytes
                continue

            for fqn, param, placement in local_group:
                local_layout = placement.local_storage_layout(
                    param.shape,
                    param.dtype,
                    rank,
                    world_size,
                )
                offset = byte_offset if local_layout.storage_nbytes > 0 else 0
                param_layouts[fqn] = BucketParamStorageLayout(
                    local_shape=local_layout.local_shape,
                    local_numel=local_layout.local_numel,
                    byte_offset=offset,
                    storage_nbytes=local_layout.storage_nbytes,
                )
                byte_offset += local_layout.storage_nbytes

        return BucketStorageLayout(
            param_layouts=param_layouts,
            total_bytes=byte_offset,
        )

    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        rank = mesh.get_local_rank()
        world_size = mesh.size()
        groups = _group_tensors_by_placement(tensors, infos, world_size)
        _validate_mixed_groups(groups)

        prepared_groups: list[tuple[PlacementPreparedUnshard, tuple[int, ...]]] = []
        group_buffers: list[torch.Tensor] = []
        device: torch.device | None = None

        with _record_copy_in_if_eager():
            for group in groups:
                prepared = _prepare_mixed_unshard_group(
                    group.placement,
                    group.tensors,
                    group.infos,
                    mesh,
                    debug_fqn,
                )
                send = prepared.buffers[0]
                if device is None:
                    device = send.device
                elif send.device != device:
                    raise ValueError(
                        "Mixed FlexShard unshard requires one device per bucket, "
                        f"but got {device} and {send.device}."
                    )
                prepared_groups.append((prepared, tuple(group.indices)))
                group_buffers.extend(prepared.buffers)

            if device is None:
                raise AssertionError("Expected at least one mixed bucket group.")
            layout = _build_mixed_unshard_layout(prepared_groups, world_size)
            group_states = [
                _MixedUnshardGroupState(
                    prepared=group_prepared,
                    indices=indices,
                    offset=group_layout.offset,
                    numel=group_layout.numel,
                )
                for (group_prepared, indices), group_layout in zip(
                    prepared_groups,
                    layout.groups,
                    strict=True,
                )
            ]
            send = torch.empty(layout.row_numel, dtype=layout.dtype, device=device)
            _zero_padding(send, layout.padding_ranges)
            for group_state, group_layout in zip(
                group_states,
                layout.groups,
                strict=True,
            ):
                if (
                    group_layout.owner_rank is not None
                    and group_layout.owner_rank != rank
                ):
                    continue
                group_send = group_state.prepared.buffers[0]
                send.narrow(0, group_state.offset, group_state.numel).copy_(
                    group_send.contiguous().view(torch.uint8).reshape(-1)
                    if layout.requires_dtype_reinterpretation
                    else group_send
                )
            gathered = torch.empty(
                world_size * layout.row_numel,
                dtype=layout.dtype,
                device=device,
            )

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send, gathered, *group_buffers],
            placement_state=_MixedUnshardState(
                groups=group_states,
                world_size=world_size,
                row_numel=layout.row_numel,
                requires_dtype_reinterpretation=(
                    layout.requires_dtype_reinterpretation
                ),
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
            ),
        )

    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        if not isinstance(prepared.placement_state, _MixedUnshardState):
            raise AssertionError(
                "Expected _MixedUnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        with _record_comm_if_eager(
            "FlexShard::mixed_all_gather",
            prepared.placement_state.debug_fqn,
        ):
            dist.all_gather_into_tensor(
                output_tensor=prepared.buffers[1],
                input_tensor=prepared.buffers[0],
                group=prepared.placement_state.pg,
            )

    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        if not isinstance(prepared.placement_state, _MixedUnshardState):
            raise AssertionError(
                "Expected _MixedUnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        gathered_by_rank = prepared.buffers[1].view(
            state.world_size,
            state.row_numel,
        )
        full_params: list[torch.Tensor | None] = [None] * sum(
            len(group.indices) for group in state.groups
        )
        finish_buffers: list[torch.Tensor] = []
        consumer_buffers: list[torch.Tensor] = []
        with _record_copy_out_if_eager():
            for group in state.groups:
                group_gathered_by_rank = gathered_by_rank[
                    :, group.offset : group.offset + group.numel
                ]
                placement = group.prepared.placement
                if isinstance(placement, Fp8BucketedBlockShard):
                    gathered = group_gathered_by_rank
                    result = placement._finish_unshard_from_rank_rows(
                        group.prepared,
                        gathered,
                    )
                else:
                    if state.requires_dtype_reinterpretation:
                        gathered = group.prepared.buffers[1]
                        gathered.view(torch.uint8).view_as(
                            group_gathered_by_rank
                        ).copy_(group_gathered_by_rank)
                    else:
                        gathered = group_gathered_by_rank.contiguous().view(-1)
                    group_prepared = PlacementPreparedUnshard(
                        placement=placement,
                        buffers=[
                            group.prepared.buffers[0],
                            gathered,
                            *group.prepared.buffers[2:],
                        ],
                        placement_state=group.prepared.placement_state,
                    )
                    result = placement.finish_prepared_unshard(group_prepared)
                finish_buffers.append(gathered)
                for index, full_param in zip(
                    group.indices,
                    result.full_params,
                    strict=True,
                ):
                    full_params[index] = full_param
                finish_buffers.extend(result.buffers)
                finish_buffers.extend(result.finish_buffers)
                consumer_buffers.extend(result.consumer_buffers)

        ordered_full_params: list[torch.Tensor] = []
        for full_param in full_params:
            if full_param is None:
                raise AssertionError("Mixed unshard did not produce every param.")
            ordered_full_params.append(full_param)
        return PlacementUnshardResult(
            full_params=ordered_full_params,
            finish_buffers=finish_buffers,
            consumer_buffers=consumer_buffers,
        )

    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        world_size = mesh.size()
        groups = _group_tensors_by_placement(tensors, infos, world_size)
        _validate_mixed_groups(groups)
        prepared_groups: list[tuple[PlacementPreparedReduceGrad, list[int], int]] = []
        group_buffers: list[torch.Tensor] = []
        dtype: torch.dtype | None = None
        device: torch.device | None = None

        with _record_function_if_eager(
            "FlexShard::mixed_reduce_scatter_copy_in",
            debug_fqn,
        ):
            for group in groups:
                prepared = group.placement.prepare_reduce_grad(
                    group.tensors,
                    group.infos,
                    mesh,
                    debug_fqn,
                )
                send = prepared.buffers[0]
                if send.numel() % world_size != 0:
                    raise AssertionError(
                        "Mixed reduce-scatter subgroup send size must be "
                        f"divisible by world size {world_size}, got "
                        f"{send.numel()}."
                    )
                if dtype is None:
                    dtype = send.dtype
                    device = send.device
                elif send.dtype != dtype or send.device != device:
                    raise ValueError(
                        "Mixed FlexShard reduce-grad requires one send dtype "
                        f"and device per bucket, but got {dtype}/{device} and "
                        f"{send.dtype}/{send.device}."
                    )
                group_numel = send.numel() // world_size
                prepared_groups.append((prepared, group.indices, group_numel))
                group_buffers.extend(prepared.buffers)

            if dtype is None or device is None:
                raise AssertionError("Expected at least one mixed bucket group.")
            (
                offsets,
                owner_ranks,
                row_numel,
                max_exclusive_payload_numel,
            ) = _plan_group_offsets(
                [prepared.placement for prepared, _, _ in prepared_groups],
                [numel for _, _, numel in prepared_groups],
                world_size,
            )
            group_states = [
                _MixedReduceGradGroupState(
                    prepared=prepared,
                    indices=indices,
                    offset=offset,
                    numel=numel,
                )
                for (prepared, indices, numel), offset in zip(
                    prepared_groups,
                    offsets,
                    strict=True,
                )
            ]
            send = torch.empty(
                world_size * row_numel,
                dtype=dtype,
                device=device,
            )
            send_rows = send.view(world_size, row_numel)
            if max_exclusive_payload_numel:
                send_rows[:, :max_exclusive_payload_numel].zero_()
            for group_state, owner_rank in zip(
                group_states,
                owner_ranks,
                strict=True,
            ):
                group_send = group_state.prepared.buffers[0].view(
                    world_size,
                    group_state.numel,
                )
                start = group_state.offset
                end = start + group_state.numel
                if owner_rank is None:
                    send_rows[:, start:end].copy_(group_send)
                else:
                    send_rows[owner_rank, start:end].copy_(group_send[owner_rank])

        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send, *group_buffers],
            placement_state=_MixedReduceGradState(
                groups=group_states,
                world_size=world_size,
                row_numel=row_numel,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                gradient_reduce_op=gradient_reduce_op_from_infos(infos),
            ),
        )

    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        if not isinstance(prepared.placement_state, _MixedReduceGradState):
            raise AssertionError(
                "Expected _MixedReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        state = prepared.placement_state
        send = prepared.buffers[0]
        recv = torch.empty(
            state.row_numel,
            dtype=send.dtype,
            device=send.device,
        )
        with _record_comm_if_eager(
            "FlexShard::mixed_reduce_scatter",
            state.debug_fqn,
        ):
            try:
                dist.reduce_scatter_tensor(
                    output=recv,
                    input=send,
                    op=state.gradient_reduce_op,
                    group=state.pg,
                )
            finally:
                for group in state.groups:
                    _release_group_scratch_lease(group.prepared)

        sharded_grads: list[torch.Tensor | None] = [None] * sum(
            len(group.indices) for group in state.groups
        )
        with _record_function_if_eager(
            "FlexShard::mixed_reduce_scatter_copy_out",
            state.debug_fqn,
        ):
            for group in state.groups:
                group_recv = recv.narrow(0, group.offset, group.numel)
                group_grads = _finish_mixed_reduce_group(
                    group.prepared,
                    group_recv,
                )
                for index, sharded_grad in zip(
                    group.indices,
                    group_grads,
                    strict=True,
                ):
                    sharded_grads[index] = sharded_grad

        ordered_sharded_grads: list[torch.Tensor] = []
        for sharded_grad in sharded_grads:
            if sharded_grad is None:
                raise AssertionError("Mixed reduce-grad did not produce every grad.")
            ordered_sharded_grads.append(sharded_grad)
        return PlacementReduceGradResult(ordered_sharded_grads, [recv])


def _prepare_mixed_unshard_group(
    placement: Placement,
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    debug_fqn: str | None,
) -> PlacementPreparedUnshard:
    if isinstance(placement, Fp8BucketedBlockShard):
        return placement._prepare_local_unshard_payload(
            tensors,
            infos,
            mesh,
            debug_fqn,
        )
    return placement.prepare_unshard_bucket(
        tensors,
        infos,
        mesh,
        debug_fqn,
    )


def _zero_padding(
    tensor: torch.Tensor,
    ranges: tuple[tuple[int, int], ...],
) -> None:
    slices = [tensor.narrow(0, offset, numel) for offset, numel in ranges]
    if torch.compiler.is_compiling():
        for slice_ in slices:
            slice_.zero_()
    elif slices:
        torch._foreach_zero_(slices)


def _build_mixed_unshard_layout(
    prepared_groups: list[tuple[PlacementPreparedUnshard, tuple[int, ...]]],
    world_size: int,
) -> _MixedUnshardLayout:
    dtypes = {group_prepared.buffers[0].dtype for group_prepared, _ in prepared_groups}
    requires_dtype_reinterpretation = len(dtypes) > 1
    dtype = torch.uint8 if requires_dtype_reinterpretation else next(iter(dtypes))
    group_numels = [
        (
            group_prepared.buffers[0].numel() * group_prepared.buffers[0].element_size()
            if requires_dtype_reinterpretation
            else group_prepared.buffers[0].numel()
        )
        for group_prepared, _ in prepared_groups
    ]
    group_placements = [prepared.placement for prepared, _ in prepared_groups]
    fp8_transport_groups = [
        isinstance(prepared.placement, Fp8BucketedBlockShard)
        and prepared.buffers[0].dtype == torch.uint8
        for prepared, _ in prepared_groups
    ]
    align_fp8_transport = any(fp8_transport_groups)
    group_alignments = [
        _VEC_ALIGN_BYTES if is_fp8_transport else 1
        for is_fp8_transport in fp8_transport_groups
    ]
    row_alignment = _VEC_ALIGN_BYTES if align_fp8_transport else 1
    offsets, owner_ranks, row_numel, max_exclusive_payload_numel = _plan_group_offsets(
        group_placements,
        group_numels,
        world_size,
        group_alignments=group_alignments,
        row_alignment=row_alignment,
    )
    groups = tuple(
        _MixedUnshardGroupLayout(
            offset=offset,
            numel=numel,
            owner_rank=owner_rank,
        )
        for offset, numel, owner_rank in zip(
            offsets,
            group_numels,
            owner_ranks,
            strict=True,
        )
    )
    padding_ranges: list[tuple[int, int]] = []
    if max_exclusive_payload_numel:
        padding_ranges.append((0, max_exclusive_payload_numel))
    next_padding_start = max_exclusive_payload_numel
    for group in groups:
        if group.owner_rank is not None:
            continue
        if next_padding_start < group.offset:
            padding_ranges.append(
                (next_padding_start, group.offset - next_padding_start)
            )
        next_padding_start = group.offset + group.numel
    if next_padding_start < row_numel:
        padding_ranges.append((next_padding_start, row_numel - next_padding_start))
    return _MixedUnshardLayout(
        groups=groups,
        dtype=dtype,
        requires_dtype_reinterpretation=requires_dtype_reinterpretation,
        row_numel=row_numel,
        padding_ranges=tuple(padding_ranges),
    )


def _group_tensors_by_placement(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    world_size: int,
) -> list[_PlacementGroup]:
    groups: list[_PlacementGroup] = []
    for index, (tensor, info) in enumerate(zip(tensors, infos, strict=True)):
        placement = _collective_placement(
            info.placement,
            info.fqn,
            info.global_shape,
            world_size,
        )
        for group in groups:
            if placement == group.placement:
                group.indices.append(index)
                group.tensors.append(tensor)
                group.infos.append(info)
                break
        else:
            groups.append(
                _PlacementGroup(
                    placement=placement,
                    indices=[index],
                    tensors=[tensor],
                    infos=[info],
                )
            )
    return groups


def _group_local_params_by_placement(
    local_params: list[tuple[str, torch.nn.Parameter, Placement]],
) -> list[list[tuple[str, torch.nn.Parameter, Placement]]]:
    groups: list[list[tuple[str, torch.nn.Parameter, Placement]]] = []
    for local_param in local_params:
        placement = local_param[2]
        for group in groups:
            if placement == group[0][2]:
                group.append(local_param)
                break
        else:
            groups.append([local_param])
    return groups


def _collective_placement(
    placement: Placement,
    fqn: str,
    global_shape: torch.Size,
    world_size: int,
) -> Placement:
    if isinstance(placement, _MixedBucketedOwned):
        return placement._lower_to_block_shards(
            {fqn: tuple(global_shape)},
            world_size,
        )[fqn]
    if isinstance(placement, _MixedBucketMember):
        return placement._collective_placement
    return placement


def _one_hot_block_owner(
    placement: Placement,
    world_size: int,
) -> int | None:
    if not isinstance(placement, BlockShard):
        return None
    counts = placement.blocks_per_rank
    if (
        len(counts) != world_size
        or counts.count(1) != 1
        or any(count not in (0, 1) for count in counts)
    ):
        return None
    return counts.index(1)


def _plan_group_offsets(
    placements: list[Placement],
    group_numels: list[int],
    world_size: int,
    *,
    group_alignments: list[int] | None = None,
    row_alignment: int = 1,
) -> tuple[list[int], list[int | None], int, int]:
    if group_alignments is None:
        group_alignments = [1] * len(placements)
    if len(group_alignments) != len(placements):
        raise AssertionError(
            "Expected one mixed-bucket alignment per placement, got "
            f"{len(group_alignments)} for {len(placements)} placements."
        )
    owner_numels = [0] * world_size
    offsets = []
    owner_ranks = []
    regular_numel = 0
    regular_alignment = 1
    for placement, group_numel, alignment in zip(
        placements,
        group_numels,
        group_alignments,
        strict=True,
    ):
        owner_rank = _one_hot_block_owner(placement, world_size)
        owner_ranks.append(owner_rank)
        if owner_rank is None:
            regular_numel = _align_up(regular_numel, alignment)
            offsets.append(regular_numel)
            regular_numel += group_numel
            regular_alignment = max(regular_alignment, alignment)
        else:
            owner_numels[owner_rank] = _align_up(
                owner_numels[owner_rank],
                alignment,
            )
            offsets.append(owner_numels[owner_rank])
            owner_numels[owner_rank] += group_numel

    regular_base = _align_up(
        max(owner_numels, default=0),
        regular_alignment,
    )
    offsets = [
        offset + regular_base if owner_rank is None else offset
        for offset, owner_rank in zip(offsets, owner_ranks, strict=True)
    ]
    return (
        offsets,
        owner_ranks,
        _align_up(regular_base + regular_numel, row_alignment),
        regular_base,
    )


def _validate_mixed_groups(groups: list[_PlacementGroup]) -> None:
    if not groups:
        raise ValueError("Mixed FlexShard bucket collectives require parameters.")
    for group in groups:
        placement = group.placement
        if not (
            isinstance(placement, BlockShard)
            or isinstance(placement, Shard)
            and placement.dim == 0
            or isinstance(placement, Fp8BucketedBlockShard)
        ):
            raise ValueError(
                "Mixed FlexShard bucket collectives currently support only "
                "Shard(0), BlockShard, and Fp8BucketedBlockShard subgroups."
            )


def _release_group_scratch_lease(prepared: PlacementPreparedReduceGrad) -> None:
    lease = getattr(prepared.placement_state, "scratch_lease", None)
    if lease is not None:
        lease.release()


def _finish_mixed_reduce_group(
    prepared: PlacementPreparedReduceGrad,
    recv: torch.Tensor,
) -> list[torch.Tensor]:
    placement = prepared.placement
    state = prepared.placement_state
    if isinstance(placement, Shard):
        return placement._unpack_reduce_scatter_grad(
            recv,
            state.infos,
            state.layout,
            state.rank,
            state.world_size,
        )
    if isinstance(placement, BlockShard):
        return placement._unpack_reduce_scatter_grad(recv, state)
    if isinstance(placement, Fp8BucketedBlockShard):
        bucket_layout = placement._bucket_layout(state.infos[0])
        rank_offset = bucket_layout.rank_offsets[state.rank]
        return [
            recv[
                placement._param_layout(info).local_global_offset
                - rank_offset : placement._param_layout(info).local_global_offset
                - rank_offset
                + info.local_numel
            ].view(info.local_shape)
            for info in state.infos
        ]
    raise TypeError(
        "Mixed FlexShard reduce-grad supports only Shard(0), BlockShard, "
        "and Fp8BucketedBlockShard, "
        f"but got {type(placement).__name__}."
    )
