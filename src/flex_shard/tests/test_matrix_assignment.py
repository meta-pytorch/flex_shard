# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from typing import cast
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.common_utils import TestCase

from ..custom_placements.block_shard import BucketedBlockShard
from ..custom_placements.owned import BucketedOwned
from ..dist_muon import (
    assign_matrices,
    AssignmentGroup,
    materialize_dist_muon_buckets,
    MatrixAssignment,
    MatrixAssignmentFn,
    MatrixBlockGroupSpec,
    ParameterBucketPlan,
    WholeMatrixSpec,
)
from ..flex_shard.bucket_storage import BucketSpec, MixedPrecisionPolicy


def _materialize_with_assignment_fn(
    assignment_fn: MatrixAssignmentFn,
) -> tuple[list[BucketSpec], nn.Parameter, nn.Parameter, DeviceMesh]:
    assignment_mesh_mock = MagicMock(spec=DeviceMesh)
    assignment_mesh_mock.size.return_value = 2
    assignment_mesh = cast(DeviceMesh, assignment_mesh_mock)
    owned = nn.Parameter(torch.empty(2, 2))
    blocks = nn.Parameter(torch.empty(4, 2, 3))
    buckets = materialize_dist_muon_buckets(
        bucket_plans=(
            ParameterBucketPlan(
                mixed_fqns=("owned",),
                owned_parameters=(("owned", owned),),
                packed_parameters=(),
                partitioned_parameters=(("blocks", blocks),),
                partition_group_name_prefix="blocks#",
            ),
        ),
        assignment_name_by_fqn={"owned": "matrix"},
        initial_sharded_fqns=(),
        final_sharded_fqn_groups=(),
        assignment_mesh=assignment_mesh,
        partition_mesh=None,
        partition_rank_axis_name="rank",
        partition_axis_name="partition",
        mp_policy=MixedPrecisionPolicy(),
        assignment_fn=assignment_fn,
    )
    return buckets, owned, blocks, assignment_mesh


class MatrixAssignmentTest(TestCase):
    def test_caps_each_rank_share_of_a_group(self) -> None:
        # The second feed-forward matrix fits better on rank 1 by total (60
        # against 100), but both there would double the group's collective
        # padding, so it goes to rank 0.
        assignment = assign_matrices(
            (
                AssignmentGroup(
                    matrices=(WholeMatrixSpec("attention_matrix", 100),),
                    block_groups=(),
                ),
                AssignmentGroup(
                    matrices=(
                        WholeMatrixSpec("feed_forward_matrix_0", 60),
                        WholeMatrixSpec("feed_forward_matrix_1", 60),
                    ),
                    block_groups=(),
                ),
            ),
            num_ranks=2,
        )

        self.assertEqual(assignment.rank_by_matrix["attention_matrix"], 0)
        self.assertEqual(
            {
                assignment.rank_by_matrix["feed_forward_matrix_0"],
                assignment.rank_by_matrix["feed_forward_matrix_1"],
            },
            {0, 1},
        )

    def test_balances_totals_over_all_groups(self) -> None:
        # One group per layer of four matrices, on eight ranks. Placing each
        # layer onto the least loaded ranks before it gives one rank two large
        # matrices (90); taking the largest matrices of all layers first pairs
        # each with a small one from another layer (68, against a mean of 64.5).
        groups = tuple(
            AssignmentGroup(
                matrices=tuple(
                    WholeMatrixSpec(f"layer{layer}.{name}", numel)
                    for name, numel in (
                        ("fc1", 58),
                        ("qkv", 32),
                        ("fc2", 29),
                        ("proj", 10),
                    )
                ),
                block_groups=(),
            )
            for layer in range(4)
        )
        assignment = assign_matrices(groups, num_ranks=8)

        totals = [0] * 8
        for group in groups:
            shares = [0] * 8
            for matrix in group.matrices:
                rank = assignment.rank_by_matrix[matrix.name]
                totals[rank] += matrix.numel
                shares[rank] += matrix.numel
            # As when balancing the layer alone, no rank holds more of it than
            # its largest matrix, so its collective pads no further.
            self.assertEqual(max(shares), 58)
        self.assertEqual(max(totals), 68)

    def test_rejects_duplicate_assignment_names(self) -> None:
        duplicate = WholeMatrixSpec("weight", 10)
        with self.assertRaisesRegex(ValueError, "Duplicate assignment name"):
            assign_matrices(
                (
                    AssignmentGroup(matrices=(duplicate,), block_groups=()),
                    AssignmentGroup(matrices=(duplicate,), block_groups=()),
                ),
                num_ranks=2,
            )

    def test_block_groups_use_accumulated_physical_load(self) -> None:
        assignment = assign_matrices(
            (
                AssignmentGroup(
                    matrices=(WholeMatrixSpec("matrix", 100),),
                    block_groups=(
                        MatrixBlockGroupSpec(
                            name="block_group",
                            num_blocks=2,
                            block_numel=60,
                            eligible_ranks=(0, 1),
                        ),
                    ),
                ),
            ),
            num_ranks=2,
        )

        self.assertEqual(assignment.rank_by_matrix["matrix"], 0)
        self.assertEqual(
            assignment.blocks_per_rank_by_group["block_group"],
            (0, 2),
        )

    def test_block_groups_follow_all_whole_matrices(self) -> None:
        # The second group's large matrix is placed before the first group's
        # blocks, which then fill in around it.
        assignment = assign_matrices(
            (
                AssignmentGroup(
                    matrices=(WholeMatrixSpec("small", 10),),
                    block_groups=(
                        MatrixBlockGroupSpec(
                            name="block_group",
                            num_blocks=2,
                            block_numel=60,
                            eligible_ranks=(0, 1),
                        ),
                    ),
                ),
                AssignmentGroup(
                    matrices=(WholeMatrixSpec("large", 100),),
                    block_groups=(),
                ),
            ),
            num_ranks=2,
        )

        self.assertEqual(assignment.rank_by_matrix, {"large": 0, "small": 1})
        self.assertEqual(
            assignment.blocks_per_rank_by_group["block_group"],
            (0, 2),
        )

    def test_materializer_uses_custom_assignment_fn(self) -> None:
        def custom_assignment_fn(
            groups: Sequence[AssignmentGroup],
            *,
            num_ranks: int,
        ) -> MatrixAssignment:
            self.assertEqual(num_ranks, 2)
            self.assertEqual(len(groups), 1)
            return MatrixAssignment(
                rank_by_matrix={"matrix": 1},
                blocks_per_rank_by_group={"blocks#0": (1, 3)},
            )

        buckets, owned, blocks, assignment_mesh = _materialize_with_assignment_fn(
            custom_assignment_fn
        )

        owned_placements = buckets[0].placement_fn(
            [("owned", owned)],
            assignment_mesh,
        )
        owned_placement = owned_placements["owned"][0]
        self.assertIsInstance(owned_placement, BucketedOwned)
        owned_placement = cast(BucketedOwned, owned_placement)
        self.assertEqual(
            owned_placement.segments_by_fqn["owned"][0].owner_rank,
            1,
        )

        block_placements = buckets[1].placement_fn(
            [("blocks", blocks)],
            assignment_mesh,
        )
        block_placement = block_placements["blocks"][0]
        self.assertIsInstance(block_placement, BucketedBlockShard)
        block_placement = cast(BucketedBlockShard, block_placement)
        self.assertEqual(block_placement.blocks_per_rank, (1, 3))

    def test_materializer_rejects_invalid_custom_assignment(self) -> None:
        valid_blocks = {"blocks#0": (1, 3)}
        cases = (
            (
                MatrixAssignment({}, valid_blocks),
                "Matrix assignment names",
            ),
            (
                MatrixAssignment({"matrix": 2}, valid_blocks),
                "invalid assigned rank",
            ),
            (
                MatrixAssignment({"matrix": 0}, {"blocks#0": (4,)}),
                "one count per eligible rank",
            ),
        )

        for assignment, error_pattern in cases:
            invalid_assignment_fn = cast(
                MatrixAssignmentFn,
                lambda _groups, *, num_ranks, assignment=assignment: assignment,
            )
            with self.subTest(assignment=assignment):
                with self.assertRaisesRegex(ValueError, error_pattern):
                    _materialize_with_assignment_fn(invalid_assignment_fn)
