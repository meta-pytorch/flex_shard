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
    def test_balances_group_before_cumulative_mapping(self) -> None:
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

    def test_group_sequence_controls_cumulative_assignment(self) -> None:
        attention = AssignmentGroup(
            matrices=(WholeMatrixSpec("q", 100),),
            block_groups=(),
        )
        feed_forward = AssignmentGroup(
            matrices=(WholeMatrixSpec("w", 60),),
            block_groups=(),
        )

        attention_first = assign_matrices(
            (attention, feed_forward),
            num_ranks=2,
        )
        feed_forward_first = assign_matrices(
            (feed_forward, attention),
            num_ranks=2,
        )

        self.assertEqual(attention_first.rank_by_matrix, {"q": 0, "w": 1})
        self.assertEqual(feed_forward_first.rank_by_matrix, {"w": 0, "q": 1})

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
