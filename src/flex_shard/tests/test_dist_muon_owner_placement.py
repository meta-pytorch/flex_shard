# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from ..dist_muon import Owned
from ..dist_muon.optimizer import (
    _balance_loads_across_partitions,
    _estimate_muon_compute_cost,
    _participant_hosts,
)

# DeepSeek V3 671B's whole Muon matrices of one MoE layer.
_MOE_LAYER_MATRICES = {
    "attention.wo": (7168, 16384),
    "attention.wq_a": (1536, 7168),
    "moe.shared_experts.w1": (2048, 7168),
    "moe.shared_experts.w3": (2048, 7168),
    "moe.shared_experts.w2": (7168, 2048),
    "moe.router.gate": (256, 7168),
}


def _moe_stage_owners(participants: tuple[int, ...], ranks_per_host: int) -> dict:
    """Owners of a 4-layer MoE stage's whole matrices, planned as DistMuon
    plans them: two buckets of two layers, with loads carried across them."""
    cumulative_loads = (0,) * len(participants)
    owners = {}
    for layers in ((0, 1), (2, 3)):
        names, loads = [], []
        for layer in layers:
            for name, shape in _MOE_LAYER_MATRICES.items():
                fqn = f"layers.{layer}.{name}"
                names.append(fqn)
                loads.append(
                    (
                        _estimate_muon_compute_cost(torch.Size(shape), 5),
                        shape[0] * shape[1] * 2,
                        fqn,
                    )
                )
        partitions, cumulative_loads = _balance_loads_across_partitions(
            loads,
            initial_cumulative_primary_loads=cumulative_loads,
            partition_hosts=_participant_hosts(participants, ranks_per_host),
        )
        owners.update(
            (fqn, participants[partition])
            for fqn, partition in zip(names, partitions, strict=True)
        )
    return owners


class TestDistMuonOwnerPlacement(unittest.TestCase):
    def test_participant_hosts(self):
        self.assertEqual(_participant_hosts(tuple(range(16)), 8), (0,) * 8 + (1,) * 8)
        self.assertEqual(_participant_hosts(tuple(range(4)), 1), (0, 1, 2, 3))
        # A group of global ranks, such as one pipeline rank's data-parallel
        # group: hosts are numbered in participant order.
        self.assertEqual(_participant_hosts((64, 65, 72, 73), 8), (0, 0, 1, 1))

    def test_heavy_hosts_alternate_with_light_ones(self):
        # 64 data-parallel ranks on 8 hosts, a 4-layer MoE stage planned as two
        # buckets of two layers. Each bucket puts its wo on two hosts with a
        # light host between them, and the second bucket on two others.
        participants = tuple(range(64, 128))
        owners = _moe_stage_owners(participants, 8)
        hosts = [
            sorted({owners[f"layers.{layer}.attention.wo"] // 8 - 8 for layer in pair})
            for pair in ((0, 1), (2, 3))
        ]
        self.assertEqual(len({host for pair in hosts for host in pair}), 4)
        for pair in hosts:
            self.assertGreater(pair[1] - pair[0], 1)
        # One host of 64 ranks puts them on 2 hosts.
        one_host = _moe_stage_owners(participants, 64)
        wo = [f"layers.{layer}.attention.wo" for layer in range(4)]
        self.assertEqual(len({one_host[fqn] // 8 for fqn in wo}), 2)
        # Every matrix still has one owner, and owners never share a matrix.
        self.assertEqual(set(owners), set(one_host))
        self.assertEqual(len(set(owners.values())), len(owners))

    def test_fixed_owners_keep_their_ranks(self):
        # Two hosts of two ranks. Alone, the largest load takes rank 0 and the
        # others follow; fixed on rank 3, it stays there, the others balance
        # around it, and the carried loads count it.
        loads = [(100, 1, "a"), (60, 1, "b"), (50, 1, "c"), (40, 1, "d")]
        hosts = (0, 0, 1, 1)
        balanced, _ = _balance_loads_across_partitions(
            loads,
            initial_cumulative_primary_loads=(0, 0, 0, 0),
            partition_hosts=hosts,
        )
        self.assertEqual(balanced, (0, 2, 3, 1))
        partitions, cumulative = _balance_loads_across_partitions(
            loads,
            initial_cumulative_primary_loads=(0, 0, 0, 0),
            partition_hosts=hosts,
            fixed_partitions=(3, None, None, None),
        )
        self.assertEqual(partitions, (3, 0, 1, 2))
        self.assertEqual(cumulative, (60, 50, 40, 100))

    def test_owned_block_sizes_are_validated(self):
        self.assertIsNone(Owned().block_sizes)
        self.assertEqual(Owned(block_sizes=(128, 64)).block_sizes, (128, 64))
        for block_sizes in ((), (4, 0), [4], (True,)):
            with self.assertRaisesRegex(ValueError, "Owned.block_sizes"):
                Owned(block_sizes=block_sizes)


if __name__ == "__main__":
    unittest.main()
