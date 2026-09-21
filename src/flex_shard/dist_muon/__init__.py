# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from .bucket_planner import (
    assign_matrices,
    AssignmentGroup,
    materialize_dist_muon_buckets,
    MatrixAssignment,
    MatrixAssignmentFn,
    MatrixBlockGroupSpec,
    PackedParameterPlan,
    ParameterBucketPlan,
    WholeMatrixSpec,
)
from .comm_free_adapter import build_local_dist_muon, LocalDistMuonBinding
from .optimizer import build_dist_muon, DistMuon
from .placement import BlockShardPlan
from .storage_layout import (
    capture_flex_shard_muon_canonical_shards,
    get_flex_shard_muon_compute_layouts,
    LocalMuonComputeLayout,
    LocalMuonStateShard,
)

__all__ = [
    "assign_matrices",
    "AssignmentGroup",
    "BlockShardPlan",
    "build_dist_muon",
    "build_local_dist_muon",
    "capture_flex_shard_muon_canonical_shards",
    "get_flex_shard_muon_compute_layouts",
    "DistMuon",
    "LocalDistMuonBinding",
    "LocalMuonComputeLayout",
    "LocalMuonStateShard",
    "materialize_dist_muon_buckets",
    "MatrixAssignment",
    "MatrixAssignmentFn",
    "MatrixBlockGroupSpec",
    "PackedParameterPlan",
    "ParameterBucketPlan",
    "WholeMatrixSpec",
]
