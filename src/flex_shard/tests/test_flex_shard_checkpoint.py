# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import (
    BucketSpec,
    get_global_layout,
    GlobalLayout,
    set_global_layout,
    set_state_dict_global_layouts,
)
from ..custom_placements.block_shard import BlockShard, BucketedBlockShard
from ..custom_placements.owned import BucketedOwned, BucketedOwnedSegmentSpec
from ..custom_placements.shard import Shard
from ..flex_shard.bucket_storage import ShardedBucketStorage
from ..flex_shard.flex_shard import _attach_flex_shard_module_state
from ..flex_shard.placement_contract import compose_global_layouts, Placement
from .common import single_rank_cpu_mesh


class _Mesh:
    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._world_size


def _global_layouts(
    named_params: list[tuple[str, nn.Parameter]],
    placements: dict[str, tuple[Placement, ...]],
    rank: int,
    world_size: int,
) -> dict[str, GlobalLayout]:
    infos, _ = ShardedBucketStorage.create_param_infos(
        named_params, cast(Any, _Mesh(rank, world_size)), placements
    )
    return {
        fqn: info.placement.global_layout(info, rank, world_size)
        for fqn, info in infos.items()
    }


def _single_param_layout(
    placement: Placement, shape: tuple[int, ...], rank: int, world_size: int
) -> GlobalLayout:
    return _global_layouts(
        [("weight", nn.Parameter(torch.empty(shape)))],
        {"weight": (placement,)},
        rank,
        world_size,
    )["weight"]


def _shard_model(model: nn.Module, placement: Placement, mesh: Any) -> None:
    named_params = list(model.named_parameters())
    storage = ShardedBucketStorage.from_bucket(
        model,
        named_params,
        {fqn: (placement,) for fqn, _ in named_params},
        mesh,
        torch.device("cpu"),
        BucketSpec(
            ["*"],
            placement_fn=lambda named_params, mesh: {
                fqn: (placement,) for fqn, _ in named_params
            },
            mesh=mesh,
            reshard_after_forward=False,
        ),
    )
    _attach_flex_shard_module_state(model, [storage])


def _checkpoint_state_dict(model: nn.Module) -> dict[str, Any]:
    state_dict = model.state_dict()
    set_state_dict_global_layouts(model, state_dict)
    return state_dict


class TestFlexShardGlobalLayout(TestCase):
    def test_shard_uneven_and_empty_ranks(self) -> None:
        self.assertEqual(
            _single_param_layout(Shard(0), (5, 4), 1, 2),
            GlobalLayout((5, 4), ((3, 0),), ((0, 0),), ((2, 4),)),
        )
        self.assertEqual(
            _single_param_layout(Shard(0), (1, 4), 2, 3),
            GlobalLayout((1, 4), (), (), ()),
        )

    def test_block_shard(self) -> None:
        placement = BlockShard(blocks_per_rank=(2, 0, 1), dim=1)
        self.assertEqual(
            _single_param_layout(placement, (2, 6), 2, 3),
            GlobalLayout((2, 6), ((0, 4),), ((0, 0),), ((2, 2),)),
        )

    def test_bucketed_owned(self) -> None:
        placement = BucketedOwned(
            {
                "slabs": [
                    BucketedOwnedSegmentSpec(f"slabs#{row}", "slabs", 4 * row, 4, owner)
                    for row, owner in enumerate((0, 0, 1))
                ],
            }
        )
        named_params = [("slabs", nn.Parameter(torch.empty(3, 4)))]
        placements = {"slabs": (placement,)}
        self.assertEqual(
            _global_layouts(named_params, placements, 1, 2)["slabs"],
            GlobalLayout((3, 4), ((2, 0),), ((0, 0),), ((1, 4),)),
        )

    def test_bucketed_block_shard(self) -> None:
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 1))
        named_params = [
            ("prefix", nn.Parameter(torch.empty(2, 3))),
            ("weight", nn.Parameter(torch.empty(4, 3))),
        ]
        placements = {fqn: (placement,) for fqn, _ in named_params}
        self.assertEqual(
            _global_layouts(named_params, placements, 1, 2)["weight"],
            GlobalLayout((4, 3), ((1, 0),), ((0, 0),), ((3, 3),)),
        )

    def test_compose_multi_chunk(self) -> None:
        # The intermediate tensor stacks global rows [1, 3) and [5, 7).
        outer = GlobalLayout(
            global_shape=(8, 6),
            global_offsets=((1, 0), (5, 0)),
            local_offsets=((0, 0), (2, 0)),
            local_sizes=((2, 6), (2, 6)),
        )
        # The local tensor holds intermediate row 1 and a column slab of rows 2-3.
        inner = GlobalLayout(
            global_shape=(4, 6),
            global_offsets=((1, 0), (2, 2)),
            local_offsets=((0, 0), (1, 2)),
            local_sizes=((1, 6), (2, 4)),
        )
        self.assertEqual(
            compose_global_layouts(outer, inner),
            GlobalLayout(
                global_shape=(8, 6),
                global_offsets=((2, 0), (5, 2)),
                local_offsets=((0, 0), (1, 2)),
                local_sizes=((1, 6), (2, 4)),
            ),
        )


class TestFlexShardCheckpointStateDict(TestCase):
    def test_dcp_round_trip_of_checkpoint_wrapped_submodule(self) -> None:
        expected = torch.arange(20, dtype=torch.float32).view(5, 4)
        linear = nn.Linear(4, 5, bias=False)
        with torch.no_grad():
            linear.weight.copy_(expected)
        model = nn.Module()
        model.layer = checkpoint_wrapper(linear)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model.layer, Shard(0), mesh)
            state_dict = _checkpoint_state_dict(model)
            self.assertEqual(
                get_global_layout(state_dict["layer.weight"]),
                GlobalLayout((5, 4), ((0, 0),), ((0, 0),), ((5, 4),)),
            )

            with TemporaryDirectory() as checkpoint_dir:
                dcp.save(state_dict, checkpoint_id=checkpoint_dir, no_dist=True)
                next(model.parameters()).detach().fill_(-1)
                load_state_dict = _checkpoint_state_dict(model)
                dcp.load(load_state_dict, checkpoint_id=checkpoint_dir, no_dist=True)
                model.load_state_dict(load_state_dict)

            self.assertEqual(next(model.parameters()), expected)

    def test_state_dict_composes_declared_outer_layout(self) -> None:
        model = nn.Module()
        model.weight = nn.Parameter(torch.empty(4, 3))
        layout = GlobalLayout((8, 3), ((4, 0),), ((0, 0),), ((4, 3),))
        set_global_layout(model.weight, layout)
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, Shard(0), mesh)
            state_dict = _checkpoint_state_dict(model)
        self.assertEqual(get_global_layout(state_dict["weight"]), layout)

    def test_unsupported_placement_raises(self) -> None:
        model = nn.Module()
        model.weight = nn.Parameter(torch.arange(24.0).view(2, 3, 4))
        with single_rank_cpu_mesh() as mesh:
            _shard_model(model, BucketedBlockShard(dims=(0, 1)), mesh)
            with self.assertRaisesRegex(
                NotImplementedError, "Distributed checkpointing is not supported"
            ):
                _checkpoint_state_dict(model)


if __name__ == "__main__":
    run_tests()
