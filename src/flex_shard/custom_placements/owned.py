# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TYPE_CHECKING

import torch
import torch.nn as nn
from typing_extensions import override

from ..flex_shard.placement_contract import (
    BucketStorageLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from .block_shard import BlockShard

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo, PlacementFn


@dataclass(frozen=True)
class BucketedOwnedSegmentSpec:
    """Flat slice of one parameter owned by one rank inside a BucketedOwned bucket."""

    name: str
    fqn: str
    param_offset: int
    numel: int
    owner_rank: int
    storage_order: int = 0


BucketedOwnedViewKind = Literal["full_param", "expert_block"]


class BucketedOwned(Placement):
    """Bucket-global owner-partition placement.

    Supported whole-tensor and complete dim-0 slab layouts lower privately to
    per-parameter ``BlockShard`` placements.
    """

    def __init__(
        self,
        segments_by_fqn: dict[str, list[BucketedOwnedSegmentSpec]],
        *,
        view_kind: BucketedOwnedViewKind = "full_param",
    ) -> None:
        if view_kind not in ("full_param", "expert_block"):
            raise ValueError(
                "BucketedOwned view_kind must be 'full_param' or 'expert_block', "
                f"but got {view_kind!r}."
            )
        normalized: dict[str, tuple[BucketedOwnedSegmentSpec, ...]] = {}
        for fqn, segments in segments_by_fqn.items():
            if not segments:
                raise ValueError(
                    f"BucketedOwned requires at least one segment for {fqn!r}."
                )
            normalized_segments = tuple(
                sorted(segments, key=lambda segment: segment.param_offset)
            )
            for segment in normalized_segments:
                if segment.fqn != fqn:
                    raise ValueError(
                        f"BucketedOwned segment {segment.name!r} is under key "
                        f"{fqn!r} but names FQN {segment.fqn!r}."
                    )
                if segment.numel <= 0:
                    raise ValueError(
                        f"BucketedOwned segment {segment.name!r} has invalid "
                        f"numel {segment.numel}."
                    )
            normalized[fqn] = normalized_segments
        self.segments_by_fqn = normalized
        self.view_kind = view_kind
        self._hash_key = tuple(
            (fqn, segments) for fqn, segments in sorted(normalized.items())
        )
        self._standalone_mixed_member: Placement | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BucketedOwned):
            return False
        return self.view_kind == other.view_kind and self._hash_key == other._hash_key

    def __hash__(self) -> int:
        return hash((type(self), self.view_kind, self._hash_key))

    def __repr__(self) -> str:
        return (
            f"BucketedOwned(num_params={len(self.segments_by_fqn)}, "
            f"view_kind={self.view_kind!r})"
        )

    @staticmethod
    def _validate_shape(fqn: str, shape: Sequence[int]) -> tuple[int, ...]:
        normalized = tuple(shape)
        if not normalized:
            raise ValueError(
                f"BucketedOwned cannot lower scalar parameter {fqn!r} to BlockShard."
            )
        if any(size < 0 for size in normalized):
            raise ValueError(
                f"BucketedOwned shape for {fqn!r} must be non-negative, but got "
                f"{shape}."
            )
        return normalized

    @staticmethod
    def _validate_segment(
        fqn: str,
        segment: BucketedOwnedSegmentSpec,
        world_size: int,
    ) -> None:
        if segment.fqn != fqn:
            raise ValueError(
                f"BucketedOwned segment {segment.name!r} is under key {fqn!r} "
                f"but names FQN {segment.fqn!r}."
            )
        if segment.param_offset < 0:
            raise ValueError("BucketedOwned param_offset must be non-negative.")
        if segment.numel <= 0:
            raise ValueError("BucketedOwned numel must be positive.")
        if segment.owner_rank < 0 or segment.owner_rank >= world_size:
            raise ValueError(
                f"BucketedOwned owner for segment {segment.name!r} must be in "
                f"[0, {world_size}), but got {segment.owner_rank}."
            )

    @staticmethod
    def _one_hot_blocks(owner_rank: int, world_size: int) -> tuple[int, ...]:
        return tuple(1 if rank == owner_rank else 0 for rank in range(world_size))

    @staticmethod
    def _blocks_per_rank_from_dim0_slices(
        fqn: str,
        shape: tuple[int, ...],
        segments: tuple[BucketedOwnedSegmentSpec, ...],
        world_size: int,
    ) -> tuple[int, ...]:
        slab_numel = math.prod(shape[1:])
        if slab_numel <= 0 or len(segments) != shape[0]:
            raise NotImplementedError(
                f"BucketedOwned {fqn!r} must contain one complete dim-0 slab "
                "per segment."
            )
        owners = []
        for slab_index, segment in enumerate(segments):
            if (
                segment.param_offset != slab_index * slab_numel
                or segment.numel != slab_numel
            ):
                raise NotImplementedError(
                    f"BucketedOwned {fqn!r} must contain equal complete dim-0 "
                    "slabs; partial segments cannot lower to BlockShard."
                )
            owners.append(segment.owner_rank)
        if owners != sorted(owners):
            raise NotImplementedError(
                f"BucketedOwned {fqn!r} requires contiguous, nondecreasing "
                "dim-0 slab owners to lower to BlockShard."
            )
        return tuple(owners.count(rank) for rank in range(world_size))

    def _lower_to_block_shards(
        self,
        logical_shapes_by_fqn: Mapping[str, Sequence[int]],
        world_size: int,
    ) -> dict[str, BlockShard]:
        """Lower supported ownership declarations to per-FQN BlockShard."""
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, but got {world_size}.")

        lowered: dict[str, BlockShard] = {}
        for fqn, shape_value in logical_shapes_by_fqn.items():
            if fqn not in self.segments_by_fqn:
                raise ValueError(f"BucketedOwned missing segments for {fqn!r}.")
            shape = self._validate_shape(fqn, shape_value)
            segments = self.segments_by_fqn[fqn]
            for segment in segments:
                self._validate_segment(fqn, segment, world_size)

            logical_numel = math.prod(shape)
            if (
                len(segments) == 1
                and segments[0].param_offset == 0
                and segments[0].numel == logical_numel
            ):
                blocks_per_rank = self._one_hot_blocks(
                    segments[0].owner_rank,
                    world_size,
                )
            else:
                blocks_per_rank = self._blocks_per_rank_from_dim0_slices(
                    fqn,
                    shape,
                    segments,
                    world_size,
                )
            lowered[fqn] = BlockShard(
                blocks_per_rank=blocks_per_rank,
                dim=0,
            )

        return lowered

    def _mixed_member(self) -> Placement:
        from .mixed_bucket import MixedBucketPlacement

        if self._standalone_mixed_member is None:
            mixed = MixedBucketPlacement(
                {fqn: list(segments) for fqn, segments in self.segments_by_fqn.items()}
            )
            self._standalone_mixed_member = mixed.bucketed_owned
        return self._standalone_mixed_member

    def _member_infos(self, infos: list[ParamInfo]) -> list[ParamInfo]:
        member = self._mixed_member()
        return [replace(info, placements=(member,)) for info in infos]

    @override
    def compute_local_shape(
        self,
        global_shape: torch.Size,
        rank: int,
        world_size: int,
    ) -> torch.Size:
        raise NotImplementedError(
            "BucketedOwned local shape is FQN-dependent; use bucket_storage_layout()."
        )

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "BucketedOwned local shard is FQN-dependent; use copy_param_to_storage()."
        )

    @override
    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        expected = {fqn for fqn, _ in named_params}
        actual = set(self.segments_by_fqn)
        if expected != actual:
            raise ValueError(
                "BucketedOwned segment map must match the bucket FQNs exactly: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        for fqn, _ in named_params:
            placements = param_placements[fqn]
            if placements != (self,):
                raise ValueError(
                    "BucketedOwned requires the same placement instance for every "
                    f"parameter in a bucket; {fqn!r} uses {placements!r}."
                )
        member = self._mixed_member()
        return member.bucket_storage_layout(
            named_params,
            {fqn: (member,) for fqn, _ in named_params},
            mesh,
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
        placement = self._lower_to_block_shards(
            {info.fqn: tuple(info.global_shape)},
            world_size,
        )[info.fqn]
        placement.copy_param_to_storage(
            byte_storage,
            info,
            param,
            rank,
            world_size,
        )

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        member = self._mixed_member()
        prepared = member.prepare_unshard_bucket(
            tensors,
            self._member_infos(infos),
            mesh,
            debug_fqn,
        )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=prepared.buffers,
            placement_state=prepared.placement_state,
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        self._mixed_member().run_prepared_unshard(prepared)

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        return self._mixed_member().finish_prepared_unshard(prepared)

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        member = self._mixed_member()
        prepared = member.prepare_reduce_grad(
            tensors,
            self._member_infos(infos),
            mesh,
            debug_fqn,
        )
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=prepared.buffers,
            placement_state=prepared.placement_state,
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        return self._mixed_member().reduce_prepared_grad(prepared)


def _assign_params_to_ranks(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
) -> dict[str, int]:
    """Greedily assign each full parameter to the currently least-loaded rank."""
    loads = [(0, rank) for rank in range(world_size)]
    heapq.heapify(loads)
    assignments: dict[str, int] = {}
    for fqn, param in sorted(
        named_params,
        key=lambda item: item[1].numel(),
        reverse=True,
    ):
        load, rank = heapq.heappop(loads)
        assignments[fqn] = rank
        heapq.heappush(loads, (load + param.numel(), rank))
    return assignments


def make_bucketed_owned_full_param_segments(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
) -> dict[str, list[BucketedOwnedSegmentSpec]]:
    """Build one whole-parameter BucketedOwned segment per parameter.

    Parameters are assigned to owners with greedy LPT by numel.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    assignments = _assign_params_to_ranks(named_params, world_size)
    return {
        fqn: [
            BucketedOwnedSegmentSpec(
                name=f"{fqn}#full",
                fqn=fqn,
                param_offset=0,
                numel=param.numel(),
                owner_rank=assignments[fqn],
                storage_order=param_order,
            )
        ]
        for param_order, (fqn, param) in enumerate(named_params)
    }


def make_bucketed_owned_full_param_placement_fn() -> PlacementFn:
    """Return a whole-parameter BucketedOwned placement function."""

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        segments_by_fqn = make_bucketed_owned_full_param_segments(
            named_params,
            mesh.size(),
        )
        placement = BucketedOwned(segments_by_fqn, view_kind="full_param")
        return {fqn: (placement,) for fqn, _ in named_params}

    return placement_fn


def _expert_block_order_key(fqn: str, suffix_order: tuple[str, ...]) -> int:
    for index, suffix in enumerate(suffix_order):
        if fqn.endswith(suffix):
            return index
    raise ValueError(f"Unexpected grouped expert weight FQN: {fqn}")


def make_bucketed_owned_expert_block_segments(
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
    *,
    suffix_order: tuple[str, ...] = (".w1", ".w3", ".w2"),
) -> dict[str, list[BucketedOwnedSegmentSpec]]:
    """Build per-expert BucketedOwned segments for packed 3D expert weights.

    Experts are assigned to contiguous owner ranges. ``suffix_order`` preserves
    the declarative expert-matrix order; lowering uses per-parameter BlockShard
    layouts.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, but got {world_size}.")
    if not suffix_order:
        raise ValueError("suffix_order must contain at least one suffix.")
    if not named_params:
        return {}

    bad = [
        (fqn, tuple(param.shape))
        for fqn, param in named_params
        if param.dim() != 3 or not fqn.endswith(suffix_order)
    ]
    if bad:
        raise ValueError(f"BucketedOwned expert block expects packed 3D weights: {bad}")

    num_experts = named_params[0][1].shape[0]
    if any(param.shape[0] != num_experts for _, param in named_params):
        raise ValueError("BucketedOwned expert block requires matching expert counts.")

    ordered_params = sorted(
        named_params,
        key=lambda item: _expert_block_order_key(item[0], suffix_order),
    )
    segments_by_fqn: dict[str, list[BucketedOwnedSegmentSpec]] = {
        fqn: [] for fqn, _ in ordered_params
    }
    experts_per_owner = max(1, (num_experts + world_size - 1) // world_size)
    for expert_idx in range(num_experts):
        owner = min(expert_idx // experts_per_owner, world_size - 1)
        for param_order, (fqn, param) in enumerate(ordered_params):
            expert_numel = math.prod(param.shape[1:])
            segments_by_fqn[fqn].append(
                BucketedOwnedSegmentSpec(
                    name=f"{fqn}#expert{expert_idx}",
                    fqn=fqn,
                    param_offset=expert_idx * expert_numel,
                    numel=expert_numel,
                    owner_rank=owner,
                    storage_order=expert_idx * len(ordered_params) + param_order,
                )
            )
    return segments_by_fqn


def make_bucketed_owned_expert_block_placement_fn(
    *,
    suffix_order: tuple[str, ...] = (".w1", ".w3", ".w2"),
) -> PlacementFn:
    """Return a placement function for one packed 3D expert-weight bucket."""

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        segments_by_fqn = make_bucketed_owned_expert_block_segments(
            named_params,
            mesh.size(),
            suffix_order=suffix_order,
        )
        placement = BucketedOwned(segments_by_fqn, view_kind="expert_block")
        return {fqn: (placement,) for fqn, _ in named_params}

    return placement_fn


__all__ = [
    "BucketedOwned",
    "BucketedOwnedSegmentSpec",
    "make_bucketed_owned_expert_block_placement_fn",
    "make_bucketed_owned_expert_block_segments",
    "make_bucketed_owned_full_param_placement_fn",
    "make_bucketed_owned_full_param_segments",
]
