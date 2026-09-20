# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Placement helpers for compute-ready FlexShard DistMuon storage."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard as DTensorShard

from ..custom_placements.block_shard import BucketedBlockShard
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..custom_placements.owned import BucketedOwnedSegmentSpec
from ..custom_placements.shard import Shard
from ..flex_shard import Placement, PlacementFn


@dataclass(frozen=True, slots=True)
class BlockShardPlan:
    """Describe equal logical matrices packed along tensor dimension 0."""

    num_matrices: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_matrices, bool)
            or not isinstance(self.num_matrices, int)
            or self.num_matrices <= 0
        ):
            raise ValueError("num_matrices must be a positive integer.")

    def validate_parameter(self, fqn: str, param: torch.Tensor) -> None:
        if param.ndim != 2:
            raise ValueError(
                f"Matrix-sharded parameter {fqn!r} must be 2D, but has "
                f"shape {tuple(param.shape)}."
            )
        if param.shape[0] % self.num_matrices != 0:
            raise ValueError(
                f"Matrix-sharded parameter {fqn!r} must have tensor dimension 0 "
                f"divisible by {self.num_matrices} logical matrices, "
                f"but has shape {tuple(param.shape)}."
            )

    def validate_matrix_extent(
        self,
        fqn: str,
        param: torch.Tensor,
        expected_matrix_extent: int,
    ) -> None:
        self.validate_parameter(fqn, param)
        matrix_extent = param.shape[0] // self.num_matrices
        if matrix_extent != expected_matrix_extent:
            raise ValueError(
                f"Matrix-sharded parameter {fqn!r} has inferred matrix extent "
                f"{matrix_extent}, expected {expected_matrix_extent}."
            )


def get_mesh_rank_groups(
    flat_mesh: DeviceMesh,
    partitioned_mesh: DeviceMesh,
    *,
    rank_axis_name: str,
    partition_axis_name: str,
) -> tuple[tuple[int, ...], ...]:
    mesh_axis_names = partitioned_mesh.mesh_dim_names
    if mesh_axis_names is None or set(mesh_axis_names) != {
        rank_axis_name,
        partition_axis_name,
    }:
        raise ValueError("Partitioned mesh must have exactly the requested named axes.")

    flat_global_ranks = [int(rank) for rank in flat_mesh.mesh.flatten().tolist()]
    partitioned_global_ranks = [
        int(rank) for rank in partitioned_mesh.mesh.flatten().tolist()
    ]
    if set(partitioned_global_ranks) != set(flat_global_ranks):
        raise ValueError("Flat and partitioned meshes must contain the same ranks.")

    local_rank_by_global_rank = {
        global_rank: local_rank
        for local_rank, global_rank in enumerate(flat_global_ranks)
    }
    ranks_by_axis = partitioned_mesh.mesh.permute(
        mesh_axis_names.index(rank_axis_name),
        mesh_axis_names.index(partition_axis_name),
    )
    return tuple(
        tuple(
            local_rank_by_global_rank[int(global_rank)]
            for global_rank in ranks_by_axis[:, partition].tolist()
        )
        for partition in range(ranks_by_axis.shape[1])
    )


def get_dim0_block_partitions(
    named_params: Sequence[tuple[str, nn.Parameter]],
    *,
    num_partitions: int,
    partition_axis_name: str,
) -> tuple[tuple[int, int], ...]:
    local_shapes = [
        param.to_local().shape if isinstance(param, DTensor) else param.shape
        for _, param in named_params
    ]
    if any(len(shape) != 3 for shape in local_shapes):
        raise ValueError("Dimension-0 block parameters must be 3D.")

    num_local_blocks = local_shapes[0][0]
    matrix_numel = math.prod(local_shapes[0][1:])
    if any(
        shape[0] != num_local_blocks or math.prod(shape[1:]) != matrix_numel
        for shape in local_shapes[1:]
    ):
        raise ValueError(
            "Dimension-0 block parameters must have matching block counts "
            "and per-block numel."
        )

    block_numel = len(local_shapes) * matrix_numel
    if num_partitions == 1:
        if num_local_blocks <= 0:
            raise ValueError("Dimension-0 block parameters must contain a block.")
        return ((num_local_blocks, block_numel),)

    first_param = named_params[0][1]
    if not isinstance(first_param, DTensor):
        raise ValueError("Partitioned block parameters must be DTensors.")
    global_num_blocks = int(first_param.shape[0])
    _validate_partitioned_params(
        named_params,
        global_num_blocks=global_num_blocks,
        matrix_numel=matrix_numel,
        num_partitions=num_partitions,
        partition_axis_name=partition_axis_name,
    )
    return tuple(
        (
            int(
                DTensorShard.local_shard_size_and_offset(
                    global_num_blocks,
                    num_partitions,
                    partition,
                )[0]
            ),
            block_numel,
        )
        for partition in range(num_partitions)
    )


def _validate_partitioned_params(
    named_params: Sequence[tuple[str, nn.Parameter]],
    *,
    global_num_blocks: int,
    matrix_numel: int,
    num_partitions: int,
    partition_axis_name: str,
) -> None:
    for fqn, param in named_params:
        if not isinstance(param, DTensor):
            raise ValueError(f"Partitioned block parameter {fqn!r} must be a DTensor.")
        mesh_axis_names = param.device_mesh.mesh_dim_names
        if mesh_axis_names is None or partition_axis_name not in mesh_axis_names:
            raise ValueError(
                f"Partitioned block parameter {fqn!r} must use mesh axis "
                f"{partition_axis_name!r}."
            )
        partition_axis = mesh_axis_names.index(partition_axis_name)
        placement = param.placements[partition_axis]
        if (
            not isinstance(placement, DTensorShard)
            or placement.dim != 0
            or param.device_mesh.size(partition_axis) != num_partitions
        ):
            raise ValueError(
                f"Partitioned block parameter {fqn!r} must use Shard(0) on "
                f"mesh axis {partition_axis_name!r} of size {num_partitions}."
            )
        if (
            int(param.shape[0]) != global_num_blocks
            or math.prod(param.shape[1:]) != matrix_numel
        ):
            raise ValueError(
                "Partitioned block parameters must have matching global block "
                "counts and per-block numel."
            )


def make_mixed_placement_fn(
    owned_rank_by_fqn: Mapping[str, int],
    blocks_per_rank_by_fqn: Mapping[str, tuple[int, ...]],
) -> PlacementFn:
    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        _mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        owned_params = [
            (fqn, param) for fqn, param in named_params if fqn in owned_rank_by_fqn
        ]
        segments_by_fqn = {
            fqn: [
                BucketedOwnedSegmentSpec(
                    name=f"{fqn}#full",
                    fqn=fqn,
                    param_offset=0,
                    numel=param.numel(),
                    owner_rank=owned_rank_by_fqn[fqn],
                    storage_order=order,
                )
            ]
            for order, (fqn, param) in enumerate(owned_params)
        }
        mixed = MixedBucketPlacement(segments_by_fqn)
        placements: dict[str, tuple[Placement, ...]] = {}
        for fqn, _param in named_params:
            if fqn in blocks_per_rank_by_fqn:
                placement = mixed.block_shard(
                    blocks_per_rank=blocks_per_rank_by_fqn[fqn],
                    dim=0,
                )
            elif fqn in owned_rank_by_fqn:
                placement = mixed.bucketed_owned
            else:
                placement = mixed.shard0
            placements[fqn] = (placement,)
        return placements

    return placement_fn


def shard0_placement_fn(
    named_params: list[tuple[str, nn.Parameter]],
    _mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    return {fqn: (Shard(0),) for fqn, _ in named_params}


def make_bucketed_block_placement_fn(
    blocks_per_rank: tuple[int, ...],
) -> PlacementFn:
    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        _mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        placement = BucketedBlockShard(
            dims=(0,),
            blocks_per_rank=blocks_per_rank,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return placement_fn
