# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native validation contracts using real CPU FlexShard replacement parameters."""

import copy
from dataclasses import replace

import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase

from .. import BucketSpec, get_shard_metadata
from ..custom_placements import BlockShard, BucketedBlockShard
from ..custom_placements.owned import make_bucketed_owned_full_param_placement_fn
from ..dist_muon import DistMuon
from ..flex_shard.bucket_storage import ShardedBucketStorage
from ..flex_shard.sharded_param import set_sharding_info
from .common import single_rank_cpu_mesh


def _parameters(mesh, shape, placement_fn, count=1):
    model = nn.Module()
    for index in range(count):
        model.register_parameter(f"weight{index}", nn.Parameter(torch.randn(shape)))
    named = list(model.named_parameters())
    spec = BucketSpec(["*"], placement_fn, mesh, reshard_after_forward=False)
    ShardedBucketStorage.from_bucket(
        model,
        named,
        placement_fn(named, mesh),
        mesh,
        torch.device("cpu"),
        spec,
    )
    return list(model.parameters())


def _set_metadata(parameter, metadata):
    set_sharding_info(
        parameter,
        metadata.placements,
        metadata.storage_shape,
        metadata.storage_stride,
        metadata.mesh,
        metadata=metadata,
    )


def _with_outer_shard(parameter, dimension):
    metadata = get_shard_metadata(parameter)
    shape = list(metadata.canonical_shape)
    offset = [0] * len(shape)
    offset[dimension] = shape[dimension]
    shape[dimension] *= 2
    regions = tuple(
        replace(region, global_offset=tuple(offset)) for region in metadata.regions
    )
    _set_metadata(
        parameter,
        replace(
            metadata,
            canonical_shape=torch.Size(shape),
            outer_offset=tuple(offset),
            regions=regions,
        ),
    )


class TestNativeMuonValidation(TestCase):
    def test_changed_metadata_fqn_fails_before_mutation(self):
        with single_rank_cpu_mesh() as mesh:
            parameters = _parameters(
                mesh, (4, 3), make_bucketed_owned_full_param_placement_fn()
            )
            parameter = parameters[0]
            optimizer = DistMuon(parameters)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            saved = copy.deepcopy(optimizer.state_dict())
            before = parameter.detach().clone()
            momentum = optimizer.state[parameter]["momentum_buffer"].clone()
            metadata = get_shard_metadata(parameter)
            _set_metadata(parameter, replace(metadata, fqn="different_weight"))
            for action in (
                optimizer.step,
                optimizer.state_dict,
                lambda: optimizer.load_state_dict(saved),
            ):
                with (
                    self.subTest(action=action),
                    self.assertRaisesRegex(ValueError, "FQN changed"),
                ):
                    action()
                self.assertEqual(parameter, before)
                self.assertEqual(
                    optimizer.state[parameter]["momentum_buffer"], momentum
                )

    def test_outer_matrix_fragments_are_rejected(self):
        cases = (
            ((4, 3), make_bucketed_owned_full_param_placement_fn(), 1),
            (
                (12, 3),
                lambda named, mesh: {n: (BlockShard((3,)),) for n, _ in named},
                1,
            ),
            (
                (2, 4, 3),
                lambda named, mesh: {
                    n: (BucketedBlockShard((0,), (1,)),) for n, _ in named
                },
                1,
            ),
            (
                (2, 4, 3),
                lambda named, mesh: {
                    n: (BucketedBlockShard((0,), (1,)),) for n, _ in named
                },
                2,
            ),
        )
        with single_rank_cpu_mesh() as mesh:
            for shape, placement_fn, dimension in cases:
                with self.subTest(shape=shape, dimension=dimension):
                    parameter = _parameters(mesh, shape, placement_fn)[0]
                    _with_outer_shard(parameter, dimension)
                    with self.assertRaisesRegex(NotImplementedError, "outer shard"):
                        DistMuon([parameter])

    def test_outer_packed_rows_retain_checkpoint_coordinates(self):
        with single_rank_cpu_mesh() as mesh:
            parameter = _parameters(
                mesh,
                (12, 3),
                lambda named, mesh: {n: (BlockShard((3,)),) for n, _ in named},
            )[0]
            _with_outer_shard(parameter, 0)
            optimizer = DistMuon([parameter])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            saved = optimizer.state_dict()
            momentum = saved["state"][0]["momentum_buffer"]
            self.assertEqual(momentum.global_shape, (24, 3))
            self.assertEqual(momentum.global_offsets, ((12, 0),))
            optimizer.load_state_dict(saved)

    def test_invalid_existing_momentum_prevents_partial_updates(self):
        with single_rank_cpu_mesh() as mesh:
            parameters = _parameters(
                mesh, (4, 3), make_bucketed_owned_full_param_placement_fn(), count=2
            )
            optimizer = DistMuon(parameters)
            for parameter in parameters:
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            before = [parameter.detach().clone() for parameter in parameters]
            momentum = optimizer.state[parameters[0]]["momentum_buffer"].clone()
            optimizer.state[parameters[1]]["momentum_buffer"] = torch.zeros(2, 3)
            with self.assertRaisesRegex(ValueError, "momentum storage layout"):
                optimizer.step()
            self.assertEqual(parameters, before)
            self.assertEqual(
                optimizer.state[parameters[0]]["momentum_buffer"], momentum
            )

    def test_replacing_storage_allocation_keeps_real_parameter_identity(self):
        with single_rank_cpu_mesh() as mesh:
            parameter = _parameters(
                mesh, (4, 3), make_bucketed_owned_full_param_placement_fn()
            )[0]
            reference = nn.Parameter(parameter.detach().clone())
            optimizer = DistMuon([parameter])
            reference_optimizer = torch.optim.Muon(
                [reference], lr=1e-3, weight_decay=0.1
            )
            for _ in range(2):
                parameter.data = parameter.detach().clone()
                gradient = torch.randn_like(parameter)
                parameter.grad = gradient.clone()
                reference.grad = gradient.clone()
                optimizer.step()
                reference_optimizer.step()
                self.assertIs(optimizer.param_groups[0]["params"][0], parameter)
                self.assertEqual(parameter, reference)

    def test_invalid_checkpoint_is_rejected_before_loading_any_state(self):
        with single_rank_cpu_mesh() as mesh:
            parameters = _parameters(
                mesh, (4, 3), make_bucketed_owned_full_param_placement_fn(), count=2
            )
            optimizer = DistMuon(parameters)
            for parameter in parameters:
                parameter.grad = torch.randn_like(parameter)
            optimizer.step()
            saved = copy.deepcopy(optimizer.state_dict())
            before = [parameter.detach().clone() for parameter in parameters]
            bad_checkpoints = []
            invalid = copy.deepcopy(saved)
            invalid["param_groups"][0]["param_names"][1] = "different_weight"
            bad_checkpoints.append(invalid)
            invalid = copy.deepcopy(saved)
            invalid["state"][1]["momentum_buffer"].global_offsets = ((1, 0),)
            bad_checkpoints.append(invalid)
            invalid = copy.deepcopy(saved)
            invalid["param_groups"][0]["lr"] = -1
            bad_checkpoints.append(invalid)
            for invalid in bad_checkpoints:
                with self.assertRaises(ValueError):
                    optimizer.load_state_dict(invalid)
                self.assertEqual(optimizer.state_dict(), saved)
                self.assertEqual(parameters, before)

    def test_checkpoint_state_follows_names_after_group_reordering(self):
        with single_rank_cpu_mesh() as mesh:
            parameters = _parameters(
                mesh, (4, 3), make_bucketed_owned_full_param_placement_fn(), count=2
            )
            optimizer = DistMuon(parameters)
            for parameter in parameters:
                parameter.grad = torch.randn_like(parameter)
            optimizer.step()
            saved = copy.deepcopy(optimizer.state_dict())
            group = saved["param_groups"][0]
            group["params"].reverse()
            group["param_names"].reverse()
            resumed = DistMuon(parameters)
            resumed.load_state_dict(saved)
            self.assertEqual(
                resumed.param_groups[0]["param_names"], ["weight0", "weight1"]
            )
            for parameter in parameters:
                self.assertEqual(
                    resumed.state[parameter]["momentum_buffer"],
                    optimizer.state[parameter]["momentum_buffer"],
                )
