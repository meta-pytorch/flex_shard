# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assign logical matrices and materialize their FlexShard storage buckets."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..flex_shard import BucketSpec, MixedPrecisionPolicy
from .placement import (
    BlockShardPlan,
    get_dim0_block_partitions,
    get_mesh_rank_groups,
    make_bucketed_block_placement_fn,
    make_mixed_placement_fn,
    shard0_placement_fn,
)


@dataclass(frozen=True, slots=True)
class WholeMatrixSpec:
    """One complete logical matrix stored on one rank."""

    name: str
    numel: int


@dataclass(frozen=True, slots=True)
class MatrixBlockGroupSpec:
    """Equal-sized logical blocks, potentially spanning multiple matrices, distributed across eligible ranks."""

    name: str
    num_blocks: int
    block_numel: int
    eligible_ranks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AssignmentGroup:
    """Whole matrices and block groups balanced at one ordered step."""

    matrices: tuple[WholeMatrixSpec, ...]
    block_groups: tuple[MatrixBlockGroupSpec, ...]


@dataclass(frozen=True, slots=True)
class MatrixAssignment:
    """Matrix owners and per-rank block counts returned by the balancer.

    Each block-count tuple follows its block group's ``eligible_ranks`` order.
    """

    rank_by_matrix: dict[str, int]
    blocks_per_rank_by_group: dict[str, tuple[int, ...]]


class MatrixAssignmentFn(Protocol):
    """Assign logical matrices using assignment-mesh-local rank indices.

    Implementations run independently on every rank and must be deterministic.
    """

    def __call__(
        self,
        groups: Sequence[AssignmentGroup],
        *,
        num_ranks: int,
    ) -> MatrixAssignment: ...


@dataclass(frozen=True, slots=True)
class PackedParameterPlan:
    fqn: str
    parameter: nn.Parameter
    block_plan: BlockShardPlan


@dataclass(frozen=True, slots=True)
class ParameterBucketPlan:
    mixed_fqns: tuple[str, ...]
    owned_parameters: tuple[tuple[str, nn.Parameter], ...]
    packed_parameters: tuple[PackedParameterPlan, ...]
    partitioned_parameters: tuple[tuple[str, nn.Parameter], ...]
    partition_group_name_prefix: str


def _validate_inputs(
    groups: Sequence[AssignmentGroup],
    num_ranks: int,
) -> None:
    if num_ranks <= 0:
        raise ValueError("num_ranks must be positive.")

    names: set[str] = set()
    for group in groups:
        for matrix in group.matrices:
            if matrix.name in names:
                raise ValueError(f"Duplicate assignment name {matrix.name!r}.")
            names.add(matrix.name)
            if matrix.numel <= 0:
                raise ValueError(
                    f"Whole matrix {matrix.name!r} must have positive numel."
                )

        for block_group in group.block_groups:
            if block_group.name in names:
                raise ValueError(f"Duplicate assignment name {block_group.name!r}.")
            names.add(block_group.name)
            if block_group.num_blocks < 0 or block_group.block_numel <= 0:
                raise ValueError(
                    f"Matrix block group {block_group.name!r} must have a "
                    "non-negative block count and positive block size."
                )
            if not block_group.eligible_ranks:
                raise ValueError(
                    f"Matrix block group {block_group.name!r} has no eligible rank."
                )
            if len(set(block_group.eligible_ranks)) != len(block_group.eligible_ranks):
                raise ValueError(
                    f"Matrix block group {block_group.name!r} repeats an eligible rank."
                )
            if any(
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 0 <= rank < num_ranks
                for rank in block_group.eligible_ranks
            ):
                raise ValueError(
                    f"Matrix block group {block_group.name!r} has an eligible "
                    "rank outside num_ranks."
                )


class _BalancedMatrixAssignment:
    def assign(
        self,
        groups: Sequence[AssignmentGroup],
        *,
        num_ranks: int,
    ) -> MatrixAssignment:
        rank_by_matrix: dict[str, int] = {}
        blocks_per_rank_by_group: dict[str, tuple[int, ...]] = {}
        rank_loads = [0] * num_ranks

        for group in groups:
            if group.matrices:
                slot_loads = [0] * num_ranks
                slot_matrices: list[list[str]] = [[] for _ in range(num_ranks)]
                slot_heap = [(0, slot) for slot in range(num_ranks)]
                heapq.heapify(slot_heap)
                for matrix in sorted(
                    group.matrices,
                    key=lambda matrix: matrix.numel,
                    reverse=True,
                ):
                    load, slot = heapq.heappop(slot_heap)
                    slot_matrices[slot].append(matrix.name)
                    slot_loads[slot] = load + matrix.numel
                    heapq.heappush(slot_heap, (load + matrix.numel, slot))

                slots = sorted(
                    zip(slot_loads, slot_matrices, strict=True),
                    key=lambda item: item[0],
                    reverse=True,
                )
                available_ranks = sorted(
                    range(num_ranks),
                    key=lambda rank: (rank_loads[rank], rank),
                )
                for (slot_load, matrix_names), rank in zip(
                    slots,
                    available_ranks,
                    strict=True,
                ):
                    for matrix_name in matrix_names:
                        rank_by_matrix[matrix_name] = rank
                    rank_loads[rank] += slot_load

            for block_group in group.block_groups:
                blocks_per_rank = [0] * len(block_group.eligible_ranks)
                load_heap = [
                    (rank_loads[rank], group_rank)
                    for group_rank, rank in enumerate(block_group.eligible_ranks)
                ]
                heapq.heapify(load_heap)
                for _ in range(block_group.num_blocks):
                    load, group_rank = heapq.heappop(load_heap)
                    blocks_per_rank[group_rank] += 1
                    load += block_group.block_numel
                    rank = block_group.eligible_ranks[group_rank]
                    rank_loads[rank] = load
                    heapq.heappush(load_heap, (load, group_rank))
                blocks_per_rank_by_group[block_group.name] = tuple(blocks_per_rank)

        return MatrixAssignment(
            rank_by_matrix=rank_by_matrix,
            blocks_per_rank_by_group=blocks_per_rank_by_group,
        )


def assign_matrices(
    groups: Sequence[AssignmentGroup],
    *,
    num_ranks: int,
) -> MatrixAssignment:
    """Assign recipe-defined groups in their explicit sequence order."""
    _validate_inputs(groups, num_ranks)
    return _BalancedMatrixAssignment().assign(
        groups,
        num_ranks=num_ranks,
    )


def _validate_assignment_names(
    expected: set[str],
    actual: set[str],
    *,
    kind: str,
) -> None:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{kind} assignment names must exactly match inputs; "
            f"missing={missing}, unexpected={unexpected}."
        )


def _validate_assignment(
    groups: Sequence[AssignmentGroup],
    assignment: MatrixAssignment,
    *,
    num_ranks: int,
) -> None:
    matrices = {matrix.name: matrix for group in groups for matrix in group.matrices}
    block_groups = {
        block_group.name: block_group
        for group in groups
        for block_group in group.block_groups
    }
    _validate_assignment_names(
        set(matrices),
        set(assignment.rank_by_matrix),
        kind="Matrix",
    )
    _validate_assignment_names(
        set(block_groups),
        set(assignment.blocks_per_rank_by_group),
        kind="Block group",
    )

    for name, rank in assignment.rank_by_matrix.items():
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 <= rank < num_ranks
        ):
            raise ValueError(
                f"Whole matrix {name!r} has invalid assigned rank {rank!r}."
            )

    for name, counts in assignment.blocks_per_rank_by_group.items():
        block_group = block_groups[name]
        if len(counts) != len(block_group.eligible_ranks):
            raise ValueError(
                f"Matrix block group {name!r} must provide one count per eligible rank."
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        ):
            raise ValueError(
                f"Matrix block group {name!r} counts must be non-negative integers."
            )
        if sum(counts) != block_group.num_blocks:
            raise ValueError(
                f"Matrix block group {name!r} counts must sum to "
                f"{block_group.num_blocks}, got {sum(counts)}."
            )


def materialize_dist_muon_buckets(
    *,
    bucket_plans: Sequence[ParameterBucketPlan],
    assignment_name_by_fqn: Mapping[str, str],
    initial_sharded_fqns: Sequence[str],
    final_sharded_fqn_groups: Sequence[Sequence[str]],
    assignment_mesh: DeviceMesh,
    partition_mesh: DeviceMesh | None,
    partition_rank_axis_name: str,
    partition_axis_name: str,
    mp_policy: MixedPrecisionPolicy,
    assignment_fn: MatrixAssignmentFn = assign_matrices,
) -> list[BucketSpec]:
    """Assign complete matrices and materialize their storage buckets.

    ``assignment_fn`` runs on every rank and must return the same assignment.
    """
    if partition_mesh is None:
        partition_storage_mesh = assignment_mesh
        rank_groups = (tuple(range(assignment_mesh.size())),)
        current_partition = 0
    else:
        partition_storage_mesh = partition_mesh[partition_rank_axis_name]
        rank_groups = get_mesh_rank_groups(
            assignment_mesh,
            partition_mesh,
            rank_axis_name=partition_rank_axis_name,
            partition_axis_name=partition_axis_name,
        )
        current_partition = partition_mesh.get_local_rank(partition_axis_name)

    assignment_groups: list[AssignmentGroup] = []
    partition_group_names_by_plan: list[tuple[str, ...]] = []
    packed_fqn_by_matrix_name: dict[str, str] = {}
    owned_fqns: list[str] = []
    for plan in bucket_plans:
        owned_matrices = []
        for fqn, parameter in plan.owned_parameters:
            assignment_name = assignment_name_by_fqn[fqn]
            owned_matrices.append(WholeMatrixSpec(assignment_name, parameter.numel()))
            owned_fqns.append(fqn)

        packed_matrices: list[WholeMatrixSpec] = []
        for packed in sorted(
            plan.packed_parameters,
            key=lambda spec: assignment_name_by_fqn[spec.fqn],
        ):
            packed.block_plan.validate_parameter(packed.fqn, packed.parameter)
            assignment_name = assignment_name_by_fqn[packed.fqn]
            matrix_numel = packed.parameter.numel() // packed.block_plan.num_matrices
            for matrix_index in range(packed.block_plan.num_matrices):
                matrix_name = f"{assignment_name}#matrix{matrix_index}"
                packed_matrices.append(WholeMatrixSpec(matrix_name, matrix_numel))
                packed_fqn_by_matrix_name[matrix_name] = packed.fqn

        partitions = (
            get_dim0_block_partitions(
                plan.partitioned_parameters,
                num_partitions=len(rank_groups),
                partition_axis_name=partition_axis_name,
            )
            if plan.partitioned_parameters
            else ()
        )
        partition_groups = (
            tuple(
                MatrixBlockGroupSpec(
                    name=f"{plan.partition_group_name_prefix}{partition_index}",
                    num_blocks=num_blocks,
                    block_numel=block_numel,
                    eligible_ranks=eligible_ranks,
                )
                for partition_index, (
                    (num_blocks, block_numel),
                    eligible_ranks,
                ) in enumerate(zip(partitions, rank_groups, strict=True))
            )
            if partitions
            else ()
        )
        assignment_groups.append(
            AssignmentGroup(
                matrices=(*owned_matrices, *packed_matrices),
                block_groups=partition_groups,
            )
        )
        partition_group_names_by_plan.append(
            tuple(group.name for group in partition_groups)
        )

    assignment_groups_tuple = tuple(assignment_groups)
    num_assignment_ranks = assignment_mesh.size()
    _validate_inputs(assignment_groups_tuple, num_assignment_ranks)
    assignment = assignment_fn(
        assignment_groups_tuple,
        num_ranks=num_assignment_ranks,
    )
    _validate_assignment(
        assignment_groups_tuple,
        assignment,
        num_ranks=num_assignment_ranks,
    )
    owned_rank_by_fqn = {
        fqn: assignment.rank_by_matrix[assignment_name_by_fqn[fqn]]
        for fqn in owned_fqns
    }
    block_counts_by_fqn: dict[str, list[int]] = {}
    for matrix_name, fqn in packed_fqn_by_matrix_name.items():
        counts = block_counts_by_fqn.setdefault(
            fqn,
            [0] * assignment_mesh.size(),
        )
        counts[assignment.rank_by_matrix[matrix_name]] += 1
    blocks_per_rank_by_fqn = {
        fqn: tuple(counts) for fqn, counts in block_counts_by_fqn.items()
    }

    buckets: list[BucketSpec] = []
    if initial_sharded_fqns:
        buckets.append(
            BucketSpec(
                tuple(initial_sharded_fqns),
                placement_fn=shard0_placement_fn,
                mesh=assignment_mesh,
                mp_policy=mp_policy,
                gradient_reduce_op=dist.ReduceOp.SUM,
                reshard_after_forward=True,
            )
        )
    for plan, partition_group_names in zip(
        bucket_plans,
        partition_group_names_by_plan,
        strict=True,
    ):
        if plan.mixed_fqns:
            buckets.append(
                BucketSpec(
                    plan.mixed_fqns,
                    placement_fn=make_mixed_placement_fn(
                        {
                            fqn: rank
                            for fqn, rank in owned_rank_by_fqn.items()
                            if fqn in plan.mixed_fqns
                        },
                        {
                            fqn: counts
                            for fqn, counts in blocks_per_rank_by_fqn.items()
                            if fqn in plan.mixed_fqns
                        },
                    ),
                    mesh=assignment_mesh,
                    mp_policy=mp_policy,
                    gradient_reduce_op=dist.ReduceOp.SUM,
                    reshard_after_forward=True,
                )
            )
        if plan.partitioned_parameters:
            group_name = partition_group_names[current_partition]
            buckets.append(
                BucketSpec(
                    tuple(fqn for fqn, _parameter in plan.partitioned_parameters),
                    placement_fn=make_bucketed_block_placement_fn(
                        assignment.blocks_per_rank_by_group[group_name]
                    ),
                    mesh=partition_storage_mesh,
                    mp_policy=mp_policy,
                    gradient_reduce_op=dist.ReduceOp.SUM,
                    reshard_after_forward=True,
                )
            )
    for fqns in final_sharded_fqn_groups:
        if fqns:
            buckets.append(
                BucketSpec(
                    tuple(fqns),
                    placement_fn=shard0_placement_fn,
                    mesh=assignment_mesh,
                    mp_policy=mp_policy,
                    gradient_reduce_op=dist.ReduceOp.SUM,
                    reshard_after_forward=False,
                )
            )
    return buckets


__all__ = [
    "assign_matrices",
    "AssignmentGroup",
    "materialize_dist_muon_buckets",
    "MatrixAssignmentFn",
    "MatrixBlockGroupSpec",
    "MatrixAssignment",
    "PackedParameterPlan",
    "ParameterBucketPlan",
    "WholeMatrixSpec",
]
