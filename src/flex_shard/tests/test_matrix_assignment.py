# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Sequence
from typing import cast
from unittest import mock
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.testing._internal.common_utils import TestCase

from ..custom_placements.block_shard import BucketedBlockShard
from ..custom_placements.owned import (
    BucketedOwned,
    make_bucketed_owned_full_param_segments,
)
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
from ..dist_muon.placement import get_dim0_block_partitions
from ..flex_shard.bucket_storage import BucketSpec, MixedPrecisionPolicy
from ..flex_shard.placement_contract import GlobalLayout, set_global_layout
from ..flex_shard.utils import assign_to_hosts_then_ranks, mesh_rank_hosts


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

    def test_dim0_block_partitions_use_declared_global_layout(self) -> None:
        blocks = nn.Parameter(torch.empty(2, 2, 3))
        with self.assertRaisesRegex(ValueError, "global layout"):
            get_dim0_block_partitions(
                [("blocks", blocks)], num_partitions=2, partition_index=1
            )

        set_global_layout(
            blocks,
            GlobalLayout(
                global_shape=(4, 2, 3),
                global_offsets=((2, 0, 0),),
                local_offsets=((0, 0, 0),),
                local_sizes=((2, 2, 3),),
            ),
        )
        self.assertEqual(
            get_dim0_block_partitions(
                [("blocks", blocks)], num_partitions=2, partition_index=1
            ),
            ((2, 6), (2, 6)),
        )
        with self.assertRaisesRegex(ValueError, "global layout"):
            get_dim0_block_partitions(
                [("blocks", blocks)], num_partitions=4, partition_index=1
            )
        # Experts sharded along another axis sit at offsets the partition
        # axis's rank groups do not expect.
        with self.assertRaisesRegex(ValueError, "holding partition 0"):
            get_dim0_block_partitions(
                [("blocks", blocks)], num_partitions=2, partition_index=0
            )


# DeepSeek V3 671B's whole Muon matrices of one MoE layer: attention wo and
# wq_a, shared experts' gate and up, down, router.
_MOE_LAYER = (
    7168 * 16384,
    1536 * 7168,
    2048 * 7168,
    2048 * 7168,
    7168 * 2048,
    256 * 7168,
)


class TwoLevelOwnerTest(TestCase):
    def test_mesh_rank_hosts(self) -> None:
        self.assertEqual(mesh_rank_hosts(range(8), 4), (0, 0, 0, 0, 1, 1, 1, 1))
        # A mesh of global ranks, such as one pipeline stage's data-parallel
        # ranks: hosts are numbered in mesh order.
        self.assertEqual(mesh_rank_hosts((64, 65, 72, 73), 8), (0, 0, 1, 1))
        self.assertEqual(mesh_rank_hosts(range(4), 1), (0, 1, 2, 3))
        with mock.patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "2"}):
            self.assertEqual(mesh_rank_hosts(range(4)), (0, 0, 1, 1))

    def test_heavy_and_light_hosts_alternate(self) -> None:
        # A 4-layer MoE stage on 64 ranks, 8 hosts: each wo exceeds a host's
        # share (117.4M against 87.2M), so the hosts holding one alternate
        # with the others in rank order, and the next bucket flips them.
        hosts = mesh_rank_hosts(range(64), 8)
        group = AssignmentGroup(
            matrices=tuple(
                WholeMatrixSpec(f"layer{layer}.m{index}", numel)
                for layer in range(4)
                for index, numel in enumerate(_MOE_LAYER)
            ),
            block_groups=(),
        )
        assignment = assign_matrices((group,), num_ranks=64, hosts=hosts)
        self.assertEqual(
            sorted(
                hosts[assignment.rank_by_matrix[f"layer{layer}.m0"]]
                for layer in range(4)
            ),
            [0, 2, 4, 6],
        )
        loads = [0] * 8
        for matrix in group.matrices:
            loads[hosts[assignment.rank_by_matrix[matrix.name]]] += matrix.numel
        self.assertEqual(loads[0::2], [_MOE_LAYER[0]] * 4)
        self.assertEqual(loads[1::2], [sum(_MOE_LAYER) - _MOE_LAYER[0]] * 4)
        # The second bucket, balancing carried loads: wo on the other hosts.
        second = assign_to_hosts_then_ranks(
            [matrix.numel for matrix in group.matrices],
            hosts,
            carried_loads=[
                sum(
                    m.numel
                    for m in group.matrices
                    if assignment.rank_by_matrix[m.name] == rank
                )
                for rank in range(64)
            ],
            one_per_rank=True,
        )
        self.assertEqual(
            sorted(hosts[second[6 * layer]] for layer in range(4)), [1, 3, 5, 7]
        )

    def test_one_host_alternates_ranks(self) -> None:
        group = AssignmentGroup(
            matrices=tuple(
                WholeMatrixSpec(f"m{index}", numel)
                for index, numel in enumerate((100, 100, 100, 100, 10, 10, 10, 10))
            ),
            block_groups=(),
        )
        assignment = assign_matrices((group,), num_ranks=8, hosts=(0,) * 8)
        self.assertEqual(
            sorted(assignment.rank_by_matrix[f"m{index}"] for index in range(4)),
            [0, 2, 4, 6],
        )

    def test_without_hosts_keeps_rank_order(self) -> None:
        group = AssignmentGroup(
            matrices=tuple(
                WholeMatrixSpec(f"m{index}", numel)
                for index, numel in enumerate((100, 100, 10, 10, 10, 10))
            ),
            block_groups=(),
        )
        two_level = assign_matrices(
            (group,), num_ranks=8, hosts=mesh_rank_hosts(range(8), 4)
        )
        rank_order = assign_matrices((group,), num_ranks=8)
        self.assertEqual({two_level.rank_by_matrix[m] for m in ("m0", "m1")}, {0, 4})
        self.assertEqual({rank_order.rank_by_matrix[m] for m in ("m0", "m1")}, {0, 1})

    def test_bucketed_owned_segments_place_hosts_first(self) -> None:
        named_params = [
            (f"p{index}", nn.Parameter(torch.empty(numel)))
            for index, numel in enumerate((100, 100, 10, 10))
        ]
        two_level = make_bucketed_owned_full_param_segments(
            named_params, 8, hosts=mesh_rank_hosts(range(8), 4)
        )
        rank_order = make_bucketed_owned_full_param_segments(named_params, 8)
        self.assertEqual({two_level[p][0].owner_rank for p in ("p0", "p1")}, {0, 4})
        self.assertEqual({rank_order[p][0].owner_rank for p in ("p0", "p1")}, {0, 1})
