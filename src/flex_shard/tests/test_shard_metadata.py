# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import FrozenInstanceError

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard as DTensorShard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, FSDPTestMultiThread
from torch.testing._internal.common_utils import TestCase

from .. import BucketSpec, flex_shard, get_global_shape, get_shard_metadata, Placement
from ..custom_placements import (
    BlockShard,
    BucketedBlockShard,
    make_bucketed_owned_full_param_segments,
    MixedBucketPlacement,
)
from ..flex_shard.bucket_storage import ShardedBucketStorage
from .common import single_rank_cuda_mesh


class _OpaquePlacement(Placement):
    """A third-party storage layout with no known coordinate mapping."""

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return hash(type(self))

    def compute_local_shape(self, global_shape, rank, world_size):
        return global_shape

    def extract_local_shard(self, param, rank, world_size):
        return param


def _parameter_module(**shapes):
    model = nn.Module()
    for name, shape in shapes.items():
        model.register_parameter(
            name,
            nn.Parameter(torch.arange(torch.Size(shape).numel()).float().view(shape)),
        )
    return model


class TestShardMetadataStorage(FSDPTestMultiThread):
    @property
    def world_size(self):
        return 2

    def _materialize(self, model, placement_fn):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        named_params = list(model.named_parameters())
        spec = BucketSpec(["*"], placement_fn, mesh, reshard_after_forward=False)
        return ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placement_fn(named_params, mesh),
            mesh,
            torch.device("cpu"),
            spec,
        )

    def _assert_region_matches(self, parameter, full):
        metadata = get_shard_metadata(parameter)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.local_shape, parameter.shape)
        if parameter.numel() == 0:
            self.assertEqual(metadata.regions, ())
            return
        self.assertEqual(len(metadata.regions), 1)
        region = metadata.regions[0]
        expected = full[
            tuple(
                slice(offset, offset + size)
                for offset, size in zip(region.global_offset, region.shape, strict=True)
            )
        ]
        self.assertEqual(parameter, expected)
        self.assertEqual(region.local_offset, (0,) * parameter.ndim)

    def test_uneven_and_empty_blocks_keep_immutable_coordinates_on_reinstall(self):
        for counts in ((2, 1), (3, 0)):
            with self.subTest(counts=counts):
                model = _parameter_module(qkv=(12, 4))
                original = model.qkv.detach().clone()
                placement = BlockShard(counts)
                storage = self._materialize(
                    model,
                    lambda params, mesh: {name: (placement,) for name, _ in params},
                )
                parameter = dict(model.named_parameters())["qkv"]
                metadata = get_shard_metadata(parameter)
                self.assertEqual(metadata.fqn, "qkv")
                self.assertEqual(metadata.storage_shape, (12, 4))
                self.assertEqual(metadata.canonical_shape, (12, 4))
                self.assertEqual(metadata.storage_stride, (4, 1))
                self.assertEqual(metadata.outer_offset, (0, 0))
                self._assert_region_matches(parameter, original)
                if parameter.numel():
                    self.assertEqual(
                        metadata.regions[0].global_offset,
                        (sum(counts[: self.rank]) * 4, 0),
                    )
                with self.assertRaises(FrozenInstanceError):
                    metadata.fqn = "changed"
                storage.install_sharded_params(torch.device("cpu"))
                replacement = dict(model.named_parameters())["qkv"]
                self.assertIsNot(replacement, parameter)
                self.assertIs(get_shard_metadata(replacement), metadata)

    def test_mixed_owned_and_block_storage_preserves_parameter_coordinates(self):
        model = _parameter_module(owned_large=(6, 4), owned_small=(4, 4), qkv=(12, 4))
        original = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        segments = make_bucketed_owned_full_param_segments(
            [
                (name, param)
                for name, param in model.named_parameters()
                if name != "qkv"
            ],
            self.world_size,
        )
        mixed = MixedBucketPlacement(segments)
        self._materialize(
            model,
            lambda params, mesh: {
                name: (mixed.block_shard(blocks_per_rank=(2, 1)),)
                if name == "qkv"
                else (mixed.bucketed_owned,)
                for name, _ in params
            },
        )
        for name, parameter in model.named_parameters():
            self._assert_region_matches(parameter, original[name])
            if name != "qkv" and parameter.numel():
                self.assertEqual(parameter.shape, original[name].shape)
                self.assertEqual(
                    get_shard_metadata(parameter).regions[0].global_offset, (0, 0)
                )

    def test_packed_matrix_batches_use_parameter_relative_offsets(self):
        model = _parameter_module(first=(3, 2, 4), second=(5, 2, 4))
        originals = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(1, 1))
        self._materialize(
            model, lambda params, mesh: {name: (placement,) for name, _ in params}
        )
        for name, parameter in model.named_parameters():
            self._assert_region_matches(parameter, originals[name])
        metadata = get_shard_metadata(dict(model.named_parameters())["second"])
        self.assertEqual(metadata.local_flat_offset, 8 if self.rank else 0)
        self.assertEqual(
            metadata.regions[0].global_offset, (1 if self.rank else 0, 0, 0)
        )

    def test_unknown_placement_remains_usable_without_inventing_coordinates(self):
        model = _parameter_module(weight=(3, 4))
        placement = _OpaquePlacement()
        self._materialize(
            model, lambda params, mesh: {name: (placement,) for name, _ in params}
        )
        metadata = get_shard_metadata(dict(model.named_parameters())["weight"])
        self.assertEqual(metadata.canonical_shape, (3, 4))
        self.assertIsNone(metadata.regions)
        self.assertIsNone(metadata.local_flat_offset)


class TestShardMetadataMaterialization(TestCase):
    def test_meta_to_empty_preserves_metadata_and_replaces_parameter_identity(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Linear(4, 6, bias=False, device="meta")
            placement = BlockShard((3,))
            flex_shard(
                model,
                buckets=[
                    BucketSpec(
                        ["*"],
                        lambda params, mesh: {name: (placement,) for name, _ in params},
                        mesh,
                        reshard_after_forward=False,
                    )
                ],
            )
            before = dict(model.named_parameters())["weight"]
            metadata = get_shard_metadata(before)
            self.assertEqual(metadata.regions[0].shape, (6, 4))
            model.to_empty(device="cuda")
            after = dict(model.named_parameters())["weight"]
            self.assertIsNot(before, after)
            self.assertIs(get_shard_metadata(after), metadata)
            nn.init.ones_(after)
            model(torch.ones(2, 4, device="cuda")).sum().backward()
            self.assertEqual(after.grad, torch.full_like(after, 2))


class TestShardMetadataOuterShard(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_flex_shard_captures_outer_dtensor_before_unwrapping(self):
        mesh = init_device_mesh("cuda", (2, 1), mesh_dim_names=("outer", "fsdp"))
        full = torch.arange(48, dtype=torch.float32, device="cuda").view(12, 4)
        outer = distribute_tensor(full, mesh["outer"], [DTensorShard(0)])
        model = nn.Module()
        model.register_parameter("weight", nn.Parameter(outer))
        placement = BlockShard((3,))
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    lambda params, mesh: {name: (placement,) for name, _ in params},
                    mesh["fsdp"],
                    reshard_after_forward=False,
                )
            ],
        )
        parameter = dict(model.named_parameters())["weight"]
        self.assertNotIsInstance(parameter, DTensor)
        metadata = get_shard_metadata(parameter)
        self.assertEqual(get_global_shape(parameter), (6, 4))
        self.assertEqual(metadata.storage_shape, (6, 4))
        self.assertEqual(metadata.canonical_shape, (12, 4))
        self.assertEqual(metadata.canonical_stride, (4, 1))
        self.assertEqual(metadata.outer_offset, (6 * self.rank, 0))
        self.assertEqual(metadata.regions[0].global_offset, (6 * self.rank, 0))
        self.assertEqual(metadata.regions[0].shape, (6, 4))
        self.assertEqual(parameter, full[6 * self.rank : 6 * (self.rank + 1)])
