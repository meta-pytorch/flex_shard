# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
    _mxfp8_quantize_reference_3d,
)
from torchao.prototype.mx_formats.utils import from_blocked

from .. import BucketSpec, flex_shard, GradientReduction
from ..custom_placements import mxfp8_bucketed_block_shard as mxfp8_module
from ..custom_placements.blockwise_quant_bucketed_block_shard import (
    _align_up,
    _VEC_ALIGN_BYTES,
)
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..custom_placements.mxfp8_bucketed_block_shard import Mxfp8BucketedBlockShard
from ..flex_shard.placement_contract import (
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from .common import FakeMesh, make_param_infos


device_type = torch.device(get_devtype())


def _ceil_to(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class _MxWeightFactoryCall:
    fp8_data: torch.Tensor
    fwd_scale: torch.Tensor
    dgrad_scale: torch.Tensor
    orig_dtype: torch.dtype
    requires_grad: bool


class _RecordingMxWeightFactory:
    def __init__(self) -> None:
        self.calls: list[_MxWeightFactoryCall] = []

    def __call__(
        self,
        fp8_data: torch.Tensor,
        fwd_scale: torch.Tensor,
        dgrad_scale: torch.Tensor,
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        self.calls.append(
            _MxWeightFactoryCall(
                fp8_data,
                fwd_scale,
                dgrad_scale,
                orig_dtype,
                requires_grad,
            )
        )
        return fp8_data


def _logical_scale(
    blocked: torch.Tensor,
    data_rows: int,
    data_cols: int,
) -> torch.Tensor:
    """Valid logical 1x32 scales of a blocked scale for 32-padded data dims."""
    return from_blocked(
        blocked.reshape(-1).view(torch.uint8),
        _ceil_to(data_rows, 32),
        _ceil_to(data_cols, 32) // 32,
    )


def _skip_unless_mxfp8_kernels(test: TestCase) -> None:
    if not _mxfp8_cutedsl_kernels_available:
        test.skipTest("MXFP8 CuTeDSL kernels require CUDA SM 10.x and CUDA 12.8+")


class TestMxfp8AllGatherLayout(TestCase):
    """Planner and scale tensor layout; no quantization kernels."""

    def test_scale_tensors_cover_whole_blocked_tiles(self) -> None:
        placement = Mxfp8BucketedBlockShard(
            world_size=2,
            weight_factory=_RecordingMxWeightFactory(),
        )
        # One 512-byte tile per 128x128 data tile, in both orientations.
        self.assertEqual(placement._chunk_scale_numels(128, 128), (512, 512))
        self.assertEqual(placement._chunk_scale_numels(256, 320), (3072, 3072))
        self.assertEqual(placement._chunk_scale_numels(72, 100), (512, 512))
        self.assertEqual(
            mxfp8_module._blocked_scale_shape(256, 320),
            (64, 48),
        )

    def test_planner_cuts_on_128_row_units(self) -> None:
        world_size = 4
        placement = Mxfp8BucketedBlockShard(
            world_size=world_size,
            weight_factory=_RecordingMxWeightFactory(),
        )
        named_params = [
            ("w1", nn.Parameter(torch.empty(512, 256))),
            ("w2", nn.Parameter(torch.empty(200, 320))),
        ]
        infos = make_param_infos(
            named_params,
            placement,
            FakeMesh(world_size),
        )
        metadata = placement._quant_metadata_from_infos(infos)
        for chunk in metadata.chunks:
            self.assertEqual(chunk.row_start % 128, 0)
            param_rows = dict(named_params)[chunk.fqn].shape[0]
            if chunk.row_end != param_rows:
                self.assertEqual(chunk.row_end % 128, 0)
        for fqn, param in named_params:
            self.assertEqual(
                sum(
                    chunk.row_end - chunk.row_start
                    for chunk in metadata.chunks
                    if chunk.fqn == fqn
                ),
                param.shape[0],
            )
        for rank, plan in enumerate(metadata.rank_pack_plans):
            fwd_numel = sum(chunk.scale_numels[0] for chunk in plan.chunks)
            dgrad_numel = sum(chunk.scale_numels[1] for chunk in plan.chunks)
            self.assertEqual(fwd_numel, dgrad_numel)
            self.assertEqual(metadata.rank_scale_numels[rank], fwd_numel * 2)
            self.assertEqual(fwd_numel % 512, 0)


class TestMxfp8FusedCopyOut(TestCase):
    """The one-kernel rank-row copy-out matches the cat fallback bitwise."""

    def test_copy_out_matches_cat_fallback(self) -> None:
        if device_type.type != "cuda":
            self.skipTest("The fused copy-out runs on CUDA.")
        world_size = 4
        placement = Mxfp8BucketedBlockShard(
            world_size=world_size,
            weight_factory=_RecordingMxWeightFactory(),
        )
        named_params = [
            (f"w{index}", nn.Parameter(torch.empty(shape, device="meta")))
            for index, shape in enumerate([(512, 256), (200, 320), (72, 100)])
        ]
        infos = make_param_infos(named_params, placement, FakeMesh(world_size))
        metadata = placement._quant_metadata_from_infos(infos)
        row_nbytes = metadata.packed_send_nbytes_per_rank
        qdata_nbytes = sum(metadata.rank_qdata_numels)
        scale_byte_offset = _align_up(qdata_nbytes, _VEC_ALIGN_BYTES)
        compact_nbytes = scale_byte_offset + sum(
            region.nbytes
            for regions in metadata.scale_regions_by_index
            for region in regions
        )
        # Row start offset and padding between rows: contiguous rows, rows
        # spaced apart as in a mixed bucket, and rows at an odd byte offset.
        for offset, padding in ((0, 0), (0, 16), (1, 3)):
            with self.subTest(offset=offset, padding=padding):
                parent = torch.randint(
                    0,
                    256,
                    (world_size, offset + row_nbytes + padding),
                    dtype=torch.uint8,
                    device=device_type,
                )
                rank_rows = parent[:, offset : offset + row_nbytes]
                fused = torch.full(
                    (compact_nbytes,),
                    0xAB,
                    dtype=torch.uint8,
                    device=device_type,
                )
                expected = fused.clone()
                self.assertTrue(
                    placement._copy_out_rank_rows_triton(metadata, rank_rows, fused)
                )
                placement._copy_out_rank_rows_cat(
                    metadata,
                    rank_rows,
                    expected,
                    qdata_nbytes=qdata_nbytes,
                    scale_byte_offset=scale_byte_offset,
                )
                self.assertTrue(torch.equal(fused, expected))


class TestMxfp8WeightQuantization(TestCase):
    def test_quantizer_matches_torchao_reference_in_both_orientations(self) -> None:
        _skip_unless_mxfp8_kernels(self)
        for shape in ((256, 256), (224, 320), (96, 10944)):
            with self.subTest(shape=shape):
                torch.manual_seed(0)
                weight = torch.randn(shape, device=device_type, dtype=torch.bfloat16)
                out_dim, in_dim = shape
                fp8_data, fwd_scale, dgrad_scale = (
                    mxfp8_module._quantize_dense_weight_to_mxfp8_32x32(
                        weight,
                        "rceil",
                    )
                )
                # torchao's reference cannot emit blocked 32x32 scales, so compare
                # against its logical (in_dim / 32, out_dim / 32) tile scales.
                ref_data_t, ref_tiles = _mxfp8_quantize_reference_3d(
                    weight.t().contiguous().unsqueeze(0),
                    block_size=32,
                    scale_block_dim1=32,
                    scale_block_dim2=32,
                    scaling_mode="rceil",
                    blocked_scale_output=False,
                )
                ref_tiles = ref_tiles.reshape(in_dim // 32, out_dim // 32).view(
                    torch.uint8
                )
                self.assertTrue(fp8_data.is_contiguous())
                self.assertTrue(
                    torch.equal(
                        fp8_data.view(torch.uint8),
                        ref_data_t[0].t().contiguous().view(torch.uint8),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        _logical_scale(fwd_scale, out_dim, in_dim),
                        ref_tiles.t().repeat_interleave(32, dim=0),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        _logical_scale(dgrad_scale, in_dim, out_dim),
                        ref_tiles.repeat_interleave(32, dim=0),
                    )
                )

    def _unshard_on_fake_ranks(
        self,
        placement: Mxfp8BucketedBlockShard,
        named_params: list[tuple[str, nn.Parameter]],
        world_size: int,
        persistent_buffers: list[torch.Tensor] | None = None,
    ) -> tuple[PlacementPreparedUnshard, PlacementUnshardResult]:
        """Pack every rank's chunks, then finish rank 0's persistent unshard."""
        prepared_by_rank = []
        for rank in range(world_size):
            mesh = FakeMesh(world_size, rank)
            infos = make_param_infos(named_params, placement, mesh)
            local_tensors = []
            for (_, param), info in zip(named_params, infos, strict=True):
                if info.local_numel == 0:
                    local_tensors.append(param.new_empty(info.local_shape))
                    continue
                param_layout = info.bucket_layout.param_layouts[info.fqn]
                in_dim = param.shape[-1]
                row_start = (
                    param_layout.local_global_offset - param_layout.param_offset
                ) // in_dim
                row_end = row_start + info.local_numel // in_dim
                local_tensors.append(
                    param.detach()
                    .reshape(-1, in_dim)[row_start:row_end]
                    .reshape(info.local_shape)
                )
            prepared_by_rank.append(
                placement.prepare_unshard_bucket(local_tensors, infos, mesh, None)
            )
        prepared = prepared_by_rank[0]
        prepared.persistent = True
        prepared.persistent_buffers = persistent_buffers
        rank_rows = torch.stack(
            [rank_prepared.buffers[0] for rank_prepared in prepared_by_rank]
        )
        return prepared, placement._finish_unshard_from_rank_rows(prepared, rank_rows)

    def _assert_matches_full_weight_quantization(
        self,
        call: _MxWeightFactoryCall,
        full_weight: torch.Tensor,
    ) -> None:
        matrix = full_weight.detach().reshape(-1, full_weight.shape[-1])
        out_dim, in_dim = matrix.shape
        ref_data, ref_fwd, ref_dgrad = (
            mxfp8_module._quantize_dense_weight_to_mxfp8_32x32(matrix, "rceil")
        )
        self.assertEqual(call.fp8_data.shape, full_weight.shape)
        self.assertEqual(call.fwd_scale.dtype, torch.float8_e8m0fnu)
        self.assertEqual(call.dgrad_scale.dtype, torch.float8_e8m0fnu)
        self.assertEqual(call.fwd_scale.shape, ref_fwd.shape)
        self.assertEqual(call.dgrad_scale.shape, ref_dgrad.shape)
        self.assertTrue(
            torch.equal(
                call.fp8_data.reshape(out_dim, in_dim).view(torch.uint8),
                ref_data.view(torch.uint8),
            )
        )
        # Byte for byte, padding tiles included.
        self.assertTrue(
            torch.equal(call.fwd_scale.view(torch.uint8), ref_fwd.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(call.dgrad_scale.view(torch.uint8), ref_dgrad.view(torch.uint8))
        )

    def test_sharded_unshard_matches_full_weight_quantization(self) -> None:
        """Rank chunks assemble into the unsharded weight's data and scales,
        and a refill updates the weights built by the first unshard."""
        _skip_unless_mxfp8_kernels(self)
        world_size = 3
        weight_factory = _RecordingMxWeightFactory()
        placement = Mxfp8BucketedBlockShard(
            world_size=world_size,
            weight_factory=weight_factory,
        )
        shapes = {
            # Splits unevenly across ranks, e.g. one 128-row unit next to three.
            "w1": (640, 256),
            # Stacked and split across ranks; the 128-row unit at matrix rows
            # 128..255 straddles both [160, 256] matrices.
            "w4": (2, 160, 256),
            # Partial 128-row tail, in_dim not a multiple of 128.
            "w2": (200, 320),
            # Dims not multiples of 32.
            "w3": (40, 100),
        }

        def make_params(seed: int) -> list[tuple[str, nn.Parameter]]:
            torch.manual_seed(seed)
            return [
                (
                    fqn,
                    nn.Parameter(
                        torch.randn(shape, device=device_type, dtype=torch.bfloat16)
                    ),
                )
                for fqn, shape in shapes.items()
            ]

        named_params = make_params(0)
        prepared, result = self._unshard_on_fake_ranks(
            placement, named_params, world_size
        )
        chunk_fqns = [chunk.fqn for chunk in prepared.placement_state.metadata.chunks]
        self.assertGreater(chunk_fqns.count("w1"), 1)
        self.assertGreater(chunk_fqns.count("w4"), 1)
        self.assertEqual(len(result.persistent_buffers), 1)
        first_calls = list(weight_factory.calls)
        for result_weight, call, (_, full_weight) in zip(
            result.full_params,
            first_calls,
            named_params,
            strict=True,
        ):
            self.assertIs(result_weight, call.fp8_data)
            self.assertEqual(call.orig_dtype, torch.bfloat16)
            self._assert_matches_full_weight_quantization(call, full_weight)

        # The refill writes into the first unshard's buffer, which every operand
        # of the first unshard's weights views.
        new_params = make_params(1)
        self._unshard_on_fake_ranks(
            placement,
            new_params,
            world_size,
            persistent_buffers=result.persistent_buffers,
        )
        for call, (_, full_weight) in zip(first_calls, new_params, strict=True):
            self._assert_matches_full_weight_quantization(call, full_weight)


class TestMxfp8AllGather(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_mixed_mxfp8_bucket_uses_one_collective_and_matches_reference(
        self,
    ) -> None:
        _skip_unless_mxfp8_kernels(self)
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

        class MixedMxfp8Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Parameter(torch.randn(256))
                self.weight = nn.Parameter(torch.randn(384, 256))

        torch.manual_seed(0)
        model = MixedMxfp8Module().to(device=device_type, dtype=torch.bfloat16)
        originals = {
            fqn: param.detach().clone() for fqn, param in model.named_parameters()
        }
        weight_factory = _RecordingMxWeightFactory()
        raw_placement = Mxfp8BucketedBlockShard(
            world_size=self.world_size,
            weight_factory=weight_factory,
        )
        mixed = MixedBucketPlacement({})
        weight_placement = mixed.blockwise_quant_bucketed_block_shard(raw_placement)
        self.assertIsInstance(weight_placement, Mxfp8BucketedBlockShard)
        placements = {
            "norm": (mixed.shard0,),
            "weight": (weight_placement,),
        }
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=lambda named_params, mesh: placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ],
        )
        storage = model.sharded_bucket_storages[0]
        infos = [storage.param_infos[fqn] for fqn in placements]
        local_params = [storage.get_local_view(info.fqn) for info in infos]

        prepared = mixed.prepare_unshard_bucket(local_params, infos, mesh, None)
        self.assertEqual(prepared.buffers[0].dtype, torch.uint8)
        with patch.object(
            dist,
            "all_gather_into_tensor",
            wraps=dist.all_gather_into_tensor,
        ) as all_gather:
            mixed.run_prepared_unshard(prepared)
        self.assertEqual(all_gather.call_count, 1)

        norm, weight = mixed.finish_prepared_unshard(prepared).full_params
        self.assertEqual(norm, originals["norm"])
        (call,) = weight_factory.calls
        self.assertIs(weight, call.fp8_data)
        ref_data, ref_fwd, ref_dgrad = (
            mxfp8_module._quantize_dense_weight_to_mxfp8_32x32(
                originals["weight"],
                "rceil",
            )
        )
        self.assertTrue(
            torch.equal(call.fp8_data.view(torch.uint8), ref_data.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(call.fwd_scale.view(torch.uint8), ref_fwd.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(call.dgrad_scale.view(torch.uint8), ref_dgrad.view(torch.uint8))
        )

        # The MXFP8 group's grads go straight into the bucket's send buffer,
        # not through a send buffer of the group's own.
        with patch.object(
            Mxfp8BucketedBlockShard,
            "prepare_reduce_grad",
            side_effect=AssertionError("expected the one-pass copy-in"),
        ):
            prepared_reduce = mixed.prepare_reduce_grad(
                [originals[info.fqn] for info in infos],
                infos,
                mesh,
                None,
            )
        with patch.object(
            dist,
            "reduce_scatter_tensor",
            wraps=dist.reduce_scatter_tensor,
        ) as reduce_scatter:
            reduced = mixed.reduce_prepared_grad(prepared_reduce, GradientReduction())
        self.assertEqual(reduce_scatter.call_count, 1)
        for grad, info in zip(reduced.sharded_grads, infos, strict=True):
            self.assertEqual(grad, storage.get_local_view(info.fqn))


if __name__ == "__main__":
    run_tests()
