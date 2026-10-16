# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from unittest.mock import patch
import weakref

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torchao.utils import is_sm_at_least_90

from .. import (
    BucketSpec,
    flex_shard,
    GlobalLayout,
    GradientReduction,
    MixedPrecisionPolicy,
)
from ..custom_placements import (
    fp8_bucketed_block_shard as fp8_bucketed_block_shard_module,
)
from ..custom_placements.block_shard import _copy_box_, BucketedBlockShard
from ..custom_placements.dense_all_gather_reference import (
    DenseAllGatherReferencePlacement,
)
from ..custom_placements.fp8_bucketed_block_shard import (
    _VEC_ALIGN_BYTES,
    Fp8BucketedBlockShard,
)
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..flex_shard.bucket_storage import ParamInfo, ShardedBucketStorage
from ..flex_shard.placement_contract import PlacementPreparedUnshard
from .common import single_rank_cpu_mesh


device_type = torch.device(get_devtype())
_REFERENCE_EPS = 1e-12


def _reference_blockwise_quant_weight(
    weight: torch.Tensor,
    block_size: int,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight without using the production TorchAO kernel."""
    fp8_bucketed_block_shard_module._validate_block_size(block_size)
    out_dim, in_dim = fp8_bucketed_block_shard_module._validate_2d_non_empty(
        weight.shape,
        "reference blockwise quantization",
    )
    padded_weight = fp8_bucketed_block_shard_module._pad_2d_to_block_shape(
        weight,
        block_size,
    )
    padded_out_dim, padded_in_dim = padded_weight.shape
    scale_rows, scale_cols = fp8_bucketed_block_shard_module._scale_shape(
        weight.shape,
        block_size,
    )
    fp8_max = torch.finfo(fp8_dtype).max
    blocks = (
        padded_weight.reshape(scale_rows, block_size, scale_cols, block_size)
        .permute(0, 2, 1, 3)
        .reshape(-1, block_size * block_size)
    )
    amax = (
        blocks.abs()
        .amax(dim=1, keepdim=True)
        .clamp(min=_REFERENCE_EPS)
        .to(torch.float64)
    )
    scale = (fp8_max / amax).to(torch.float32)
    quantized = (
        (blocks.to(torch.float32) * scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    )
    quantized = (
        quantized.reshape(scale_rows, scale_cols, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(padded_out_dim, padded_in_dim)
    )
    recip_scale = (1.0 / scale).reshape(scale_rows, scale_cols).to(torch.float32)
    return (
        quantized[:out_dim, :in_dim].contiguous(),
        recip_scale.contiguous(),
    )


def _return_fp8_data(
    fp8_data: torch.Tensor,
    recip_scale: torch.Tensor,
    block_size: int,
    *,
    orig_dtype: torch.dtype,
    requires_grad: bool,
) -> torch.Tensor:
    _ = recip_scale, block_size, orig_dtype, requires_grad
    return fp8_data


@dataclass(frozen=True)
class _WeightFactoryCall:
    fp8_data: torch.Tensor
    recip_scale: torch.Tensor
    block_size: int
    orig_dtype: torch.dtype
    requires_grad: bool


class _RecordingWeightFactory:
    def __init__(self) -> None:
        self.calls: list[_WeightFactoryCall] = []

    def __call__(
        self,
        fp8_data: torch.Tensor,
        recip_scale: torch.Tensor,
        block_size: int,
        *,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> torch.Tensor:
        self.calls.append(
            _WeightFactoryCall(
                fp8_data,
                recip_scale,
                block_size,
                orig_dtype,
                requires_grad,
            )
        )
        return fp8_data


class _FakeMesh:
    """Minimal mesh for exercising bucket_storage_layout without a real PG."""

    def __init__(self, size: int, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank

    def get_group(self):
        return None


def _make_fp8_param_info(
    fqn: str,
    param: nn.Parameter,
    placement: Fp8BucketedBlockShard,
    mesh: _FakeMesh,
) -> ParamInfo:
    return _make_fp8_param_infos([(fqn, param)], placement, mesh)[0]


def _make_fp8_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    placement: Fp8BucketedBlockShard,
    mesh: _FakeMesh,
) -> list[ParamInfo]:
    placements = {fqn: (placement,) for fqn, _ in named_params}
    layout = placement.bucket_storage_layout(
        named_params,
        placements,
        mesh,
    )
    return [
        ParamInfo(
            fqn=fqn,
            global_shape=param.shape,
            global_stride=tuple(param.stride()),
            dtype=param.dtype,
            requires_grad=param.requires_grad,
            placements=(placement,),
            local_shape=layout.param_layouts[fqn].local_shape,
            local_numel=layout.param_layouts[fqn].local_numel,
            byte_offset=layout.param_layouts[fqn].byte_offset,
            storage_nbytes=layout.param_layouts[fqn].storage_nbytes,
            global_numel=param.numel(),
            bucket_layout=layout.param_layouts[fqn].bucket_layout,
        )
        for fqn, param in named_params
    ]


def _packed_rank_views(
    packed: torch.Tensor,
    rank: int,
    metadata: Fp8BucketedBlockShard._Fp8BucketMetadata,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_nbytes = metadata.rank_fp8_numels[rank]
    scale_start = metadata.rank_scale_byte_offsets[rank]
    scale_end = scale_start + metadata.rank_scale_numels[rank] * torch.float32.itemsize
    return (
        packed[:fp8_nbytes].view(fp8_dtype),
        packed[scale_start:scale_end].view(torch.float32),
    )


def _cpu_reference_quantize_local_weight(
    placement: Fp8BucketedBlockShard,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _reference_blockwise_quant_weight(
        tensor,
        placement.block_size,
        placement.fp8_dtype,
    )


def _local_rows(param: nn.Parameter, info: ParamInfo) -> torch.Tensor:
    """Return the local shard: a row range of ``param`` viewed as a matrix."""
    if info.local_numel == 0:
        return param.new_empty(info.local_shape)
    param_layout = info.bucket_layout.param_layouts[info.fqn]
    in_dim = param.shape[-1]
    row_start = (param_layout.local_global_offset - param_layout.param_offset) // in_dim
    row_end = row_start + info.local_numel // in_dim
    return param.detach().flatten(0, -2)[row_start:row_end].reshape(info.local_shape)


def _box(offset: tuple[int, ...], size: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(
        slice(start, start + length) for start, length in zip(offset, size, strict=True)
    )


def _prepare_unshard_on_each_rank(
    placement: Fp8BucketedBlockShard,
    named_params: list[tuple[str, nn.Parameter]],
    world_size: int,
) -> list:
    prepared_by_rank = []
    for rank in range(world_size):
        mesh = _FakeMesh(world_size, rank)
        infos = _make_fp8_param_infos(named_params, placement, mesh)
        local_tensors = [
            _local_rows(param, info)
            for (_, param), info in zip(named_params, infos, strict=True)
        ]
        prepared_by_rank.append(
            placement.prepare_unshard_bucket(local_tensors, infos, mesh, None)
        )
    return prepared_by_rank


class TestFp8AllGatherLayout(TestCase):
    """CPU-only tests for contiguous block-row partitioning."""

    def setUp(self) -> None:
        super().setUp()
        quantize_patch = patch.object(
            Fp8BucketedBlockShard,
            "_quantize_local_weight",
            _cpu_reference_quantize_local_weight,
        )
        quantize_patch.start()
        self.addCleanup(quantize_patch.stop)

    def test_layout_validation_and_partitioning(self) -> None:
        block = 4
        placement = Fp8BucketedBlockShard(
            world_size=2,
            weight_factory=_return_fp8_data,
            block_size=block,
        )

        with self.assertRaisesRegex(ValueError, "2D weights"):
            placement.bucket_storage_layout(
                [("w", nn.Parameter(torch.zeros(16)))],
                {"w": (placement,)},
                _FakeMesh(2),
            )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            placement.bucket_storage_layout(
                [("w", nn.Parameter(torch.zeros(0, 8)))],
                {"w": (placement,)},
                _FakeMesh(2),
            )
        with self.assertRaisesRegex(ValueError, "one-byte"):
            Fp8BucketedBlockShard(
                world_size=2,
                weight_factory=_return_fp8_data,
                block_size=block,
                fp8_dtype=torch.float16,
            )

        named = [
            ("w1", nn.Parameter(torch.zeros(16, 8))),
            ("w2", nn.Parameter(torch.zeros(8, 8))),
        ]
        placements = {fqn: (placement,) for fqn, _ in named}
        layouts = [
            placement.bucket_storage_layout(
                named,
                placements,
                _FakeMesh(2, rank=rank),
            )
            for rank in range(2)
        ]
        self.assertEqual(
            [tuple(layout.param_layouts["w1"].local_shape) for layout in layouts],
            [(12, 8), (4, 8)],
        )
        self.assertEqual(
            [tuple(layout.param_layouts["w2"].local_shape) for layout in layouts],
            [(0, 8), (8, 8)],
        )
        bucket_layout = layouts[0].param_layouts["w1"].bucket_layout
        self.assertEqual(bucket_layout.rank_offsets, (0, 96))
        self.assertEqual(bucket_layout.rank_numels, (96, 96))
        for layout in layouts:
            rank_bucket_layout = layout.param_layouts["w1"].bucket_layout
            for fqn, param in named:
                param_layout = rank_bucket_layout.param_layouts[fqn]
                local_layout = layout.param_layouts[fqn]
                relative_start = (
                    param_layout.local_global_offset - param_layout.param_offset
                )
                row_start = relative_start // param.shape[1]
                row_end = row_start + local_layout.local_shape[0]
                valid_boundaries = {
                    min(row, param.shape[0])
                    for row in range(0, param.shape[0] + block, block)
                }
                self.assertIn(row_start, valid_boundaries)
                self.assertIn(row_end, valid_boundaries)

        partial = [("w", nn.Parameter(torch.zeros(9, 32)))]
        partial_placement = Fp8BucketedBlockShard(
            world_size=2,
            weight_factory=_return_fp8_data,
            block_size=2,
        )
        partial_placements = {"w": (partial_placement,)}
        self.assertEqual(
            tuple(
                partial_placement.bucket_storage_layout(
                    partial,
                    partial_placements,
                    _FakeMesh(2, rank=0),
                )
                .param_layouts["w"]
                .local_shape
            ),
            (4, 32),
        )
        self.assertEqual(
            tuple(
                partial_placement.bucket_storage_layout(
                    partial,
                    partial_placements,
                    _FakeMesh(2, rank=1),
                )
                .param_layouts["w"]
                .local_shape
            ),
            (5, 32),
        )

        empty_rank_placement = Fp8BucketedBlockShard(
            world_size=4,
            weight_factory=_return_fp8_data,
            block_size=block,
        )
        empty_rank_named = [("w", nn.Parameter(torch.zeros(4, 4)))]
        empty_rank_layout = empty_rank_placement.bucket_storage_layout(
            empty_rank_named,
            {"w": (empty_rank_placement,)},
            _FakeMesh(4, rank=3),
        )
        self.assertEqual(empty_rank_layout.total_bytes, 0)
        self.assertEqual(empty_rank_layout.param_layouts["w"].local_numel, 0)
        self.assertEqual(empty_rank_layout.param_layouts["w"].storage_nbytes, 0)

    def test_fp8_unshard_quantizes_param_dtype(self) -> None:
        """FP32 master shards are cast to the BF16 compute dtype before quantization."""
        block = 4
        mesh = _FakeMesh(1)
        placement = Fp8BucketedBlockShard(
            world_size=1,
            weight_factory=_return_fp8_data,
            block_size=block,
        )
        full_weight = torch.full(
            (block, block),
            0.25,
            dtype=torch.float32,
            device=device_type,
        )
        full_weight[0, 0] = 1.001
        param = nn.Parameter(full_weight.clone())
        info = _make_fp8_param_info("w", param, placement, mesh)
        info.param_dtype = torch.bfloat16

        prepared = placement.prepare_unshard_bucket(
            [full_weight],
            [info],
            mesh,
            None,
        )
        fp8_data, recip_scale = _packed_rank_views(
            prepared.buffers[0],
            0,
            prepared.placement_state.metadata,
            placement.fp8_dtype,
        )
        reference_fp8, reference_scale = _reference_blockwise_quant_weight(
            full_weight.to(torch.bfloat16),
            block,
        )
        _, fp32_scale = _reference_blockwise_quant_weight(full_weight, block)

        self.assertTrue(torch.equal(fp8_data.view(full_weight.shape), reference_fp8))
        self.assertTrue(
            torch.equal(recip_scale.view(reference_scale.shape), reference_scale)
        )
        self.assertFalse(torch.equal(recip_scale.view(fp32_scale.shape), fp32_scale))

    def test_mixed_fp8_member_with_param_dtype_overrides_is_one_group(self) -> None:
        """FP8 members quantize each param from its own dtype in one subgroup."""
        with single_rank_cpu_mesh() as mesh:
            weight_factory = _RecordingWeightFactory()
            mixed = MixedBucketPlacement({})
            fp8 = mixed.fp8_bucketed_block_shard(
                Fp8BucketedBlockShard(
                    world_size=1,
                    weight_factory=weight_factory,
                    block_size=4,
                )
            )
            named_params = [
                ("w0", nn.Parameter(torch.randn(8, 8))),
                ("w1", nn.Parameter(torch.randn(6, 8))),
                ("w2", nn.Parameter(torch.randn(4, 8))),
            ]
            param_infos, _ = ShardedBucketStorage.create_param_infos(
                named_params,
                mesh,
                {fqn: (fp8,) for fqn, _ in named_params},
                MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    param_dtype_overrides={"w1": torch.float32},
                ),
            )
            infos = list(param_infos.values())
            params = dict(named_params)
            prepared = mixed.prepare_unshard_bucket(
                [params[info.fqn].detach() for info in infos], infos, mesh, None
            )
            self.assertEqual(
                [
                    [infos[index].fqn for index in group.indices]
                    for group in prepared.placement_state.groups
                ],
                [["w0", "w2", "w1"]],
            )
            mixed.run_prepared_unshard(prepared)
            mixed.finish_prepared_unshard(prepared)
            self.assertEqual(
                [call.orig_dtype for call in weight_factory.calls],
                [torch.bfloat16, torch.bfloat16, torch.float32],
            )

    def test_fp8_unshard_quantizes_every_gather(self) -> None:
        """Forward and checkpoint recompute's backward re-gather each quantize
        the payload, to the same bytes, and the placement keeps neither alive."""
        from torch.utils.checkpoint import checkpoint

        mesh = _FakeMesh(1)
        placement = Fp8BucketedBlockShard(
            world_size=1,
            weight_factory=_return_fp8_data,
            block_size=4,
        )
        param = nn.Parameter(
            torch.arange(64, dtype=torch.float32, device=device_type).view(8, 8)
        )
        info = _make_fp8_param_info("w", param, placement, mesh)
        payloads = []
        payload_refs = []

        def function(tensor: torch.Tensor) -> torch.Tensor:
            prepared = placement.prepare_unshard_bucket(
                [param.detach()],
                [info],
                mesh,
                None,
            )
            payloads.append(prepared.buffers[0].clone())
            payload_refs.append(weakref.ref(prepared.buffers[0]))
            return tensor.sin() * tensor.cos()

        tensor = torch.randn(8, device=device_type, requires_grad=True)
        with patch.object(
            placement,
            "_quantize_local_weight",
            wraps=placement._quantize_local_weight,
        ) as quantize:
            checkpoint(function, tensor, use_reentrant=False).sum().backward()

        self.assertEqual(quantize.call_count, 2)
        self.assertTrue(torch.equal(payloads[0], payloads[1]))
        self.assertTrue(all(ref() is None for ref in payload_refs))

    def test_fp8_unshard_uses_single_packed_all_gather(self) -> None:
        """The FP8 placement launches one collective over packed bytes."""
        mesh = _FakeMesh(2)
        placement = Fp8BucketedBlockShard(
            world_size=2,
            weight_factory=_return_fp8_data,
            block_size=4,
        )
        param = nn.Parameter(torch.zeros(16, 8, device=device_type))
        info = _make_fp8_param_info("w", param, placement, mesh)
        tensor = torch.zeros(info.local_shape, device=device_type)
        prepared = placement.prepare_unshard_bucket([tensor], [info], mesh, None)
        local_packed, gathered_packed = prepared.buffers

        with patch.object(dist, "all_gather_into_tensor") as all_gather:
            placement.run_prepared_unshard(prepared)

        all_gather.assert_called_once()
        self.assertIs(all_gather.call_args.kwargs["output_tensor"], gathered_packed)
        self.assertIs(all_gather.call_args.kwargs["input_tensor"], local_packed)
        self.assertIsNone(all_gather.call_args.kwargs["group"])

    def test_finish_packed_unshard_compacts_rank_rows_into_shared_storage(
        self,
    ) -> None:
        block = 4
        world_size = 8
        weight_factory = _RecordingWeightFactory()
        placement = Fp8BucketedBlockShard(
            world_size=world_size,
            weight_factory=weight_factory,
            block_size=block,
        )
        named_params = [
            (
                "w1",
                nn.Parameter(
                    torch.arange(128, dtype=torch.float32, device=device_type).reshape(
                        16, 8
                    )
                ),
            ),
            (
                "w2",
                nn.Parameter(
                    torch.arange(64, dtype=torch.float32, device=device_type).reshape(
                        8, 8
                    )
                ),
            ),
            (
                "w3",
                nn.Parameter(
                    torch.arange(3, dtype=torch.float32, device=device_type).reshape(
                        1, 3
                    )
                ),
            ),
        ]
        prepared_by_rank = _prepare_unshard_on_each_rank(
            placement,
            named_params,
            world_size,
        )

        prepared = prepared_by_rank[0]
        metadata = prepared.placement_state.metadata
        send_nbytes_per_rank = metadata.packed_send_nbytes_per_rank
        parent = torch.full(
            (world_size, send_nbytes_per_rank + 3),
            0xFF,
            dtype=torch.uint8,
            device=device_type,
        )
        rank_rows = parent[:, 1 : send_nbytes_per_rank + 1]
        self.assertFalse(rank_rows.is_contiguous())
        self.assertEqual(rank_rows.storage_offset() % torch.float32.itemsize, 1)
        for rank, rank_prepared in enumerate(prepared_by_rank):
            local_packed = rank_prepared.buffers[0]
            fp8_end = metadata.rank_fp8_numels[rank]
            scale_start = metadata.rank_scale_byte_offsets[rank]
            payload_end = metadata.rank_packed_nbytes[rank]
            rank_rows[rank, :fp8_end].copy_(local_packed[:fp8_end])
            rank_rows[rank, scale_start:payload_end].copy_(
                local_packed[scale_start:payload_end]
            )

        prepared.persistent = True
        result = placement._finish_unshard_from_rank_rows(prepared, rank_rows)
        self.assertEqual(len(result.persistent_buffers), 1)
        compact = result.persistent_buffers[0]
        expected_fp8_nbytes = sum(param.numel() for _, param in named_params)
        self.assertNotEqual(expected_fp8_nbytes % torch.float32.itemsize, 0)
        self.assertIn(0, metadata.rank_fp8_numels)
        expected_scale_numel = sum(
            ((param.shape[0] + block - 1) // block)
            * ((param.shape[1] + block - 1) // block)
            for _, param in named_params
        )
        # The scale region starts on a 16-byte boundary so aligned scale slices
        # can use CatArrayBatchedCopy_vectorized.
        expected_scale_byte_offset = (
            (expected_fp8_nbytes + _VEC_ALIGN_BYTES - 1)
            // _VEC_ALIGN_BYTES
            * _VEC_ALIGN_BYTES
        )
        expected_compact_nbytes = (
            expected_scale_byte_offset + expected_scale_numel * torch.float32.itemsize
        )
        self.assertEqual(compact.dtype, torch.uint8)
        self.assertEqual(compact.numel(), expected_compact_nbytes)

        storage_ptr = compact.untyped_storage().data_ptr()
        for result_weight, factory_call, (_, full_weight) in zip(
            result.full_params,
            weight_factory.calls,
            named_params,
            strict=True,
        ):
            reference_fp8, reference_scale = _reference_blockwise_quant_weight(
                full_weight.detach(),
                block,
            )
            self.assertIs(result_weight, factory_call.fp8_data)
            self.assertEqual(factory_call.block_size, block)
            self.assertEqual(factory_call.orig_dtype, full_weight.dtype)
            self.assertEqual(factory_call.requires_grad, full_weight.requires_grad)
            self.assertTrue(factory_call.fp8_data.is_contiguous())
            self.assertTrue(factory_call.recip_scale.is_contiguous())
            self.assertEqual(
                factory_call.fp8_data.untyped_storage().data_ptr(),
                storage_ptr,
            )
            self.assertEqual(
                factory_call.recip_scale.untyped_storage().data_ptr(),
                storage_ptr,
            )
            self.assertTrue(
                torch.equal(
                    factory_call.fp8_data.view(torch.uint8),
                    reference_fp8.view(torch.uint8),
                )
            )
            self.assertTrue(torch.equal(factory_call.recip_scale, reference_scale))

    def test_stacked_weight_shards_rows_of_flattened_matrix(self) -> None:
        """A ``[2, 6, 8]`` weight is planned as its ``[12, 8]`` matrix, so the
        4-row block held by rank 1 straddles the two stacked matrices."""
        world_size = 3
        placement = Fp8BucketedBlockShard(
            world_size=world_size,
            weight_factory=_return_fp8_data,
            block_size=4,
        )
        param = nn.Parameter(torch.arange(96, dtype=torch.float32).reshape(2, 6, 8))
        infos = [
            _make_fp8_param_info("w13", param, placement, _FakeMesh(world_size, rank))
            for rank in range(world_size)
        ]
        self.assertEqual(
            [tuple(info.local_shape) for info in infos],
            [(1, 4, 8)] * world_size,
        )
        layouts = [
            placement.global_layout(info, rank, world_size)
            for rank, info in enumerate(infos)
        ]
        self.assertEqual(
            layouts,
            [
                GlobalLayout((2, 6, 8), ((0, 0, 0),), ((0, 0, 0),), ((1, 4, 8),)),
                GlobalLayout(
                    (2, 6, 8),
                    ((0, 4, 0), (1, 0, 0)),
                    ((0, 0, 0), (0, 2, 0)),
                    ((1, 2, 8), (1, 2, 8)),
                ),
                GlobalLayout((2, 6, 8), ((1, 2, 0),), ((0, 0, 0),), ((1, 4, 8),)),
            ],
        )
        reconstructed = torch.zeros_like(param)
        for info, layout in zip(infos, layouts, strict=True):
            local = _local_rows(param, info)
            for global_offset, local_offset, size in zip(
                layout.global_offsets,
                layout.local_offsets,
                layout.local_sizes,
                strict=True,
            ):
                reconstructed[_box(global_offset, size)] = local[
                    _box(local_offset, size)
                ]
        self.assertEqual(reconstructed, param.detach())

    def test_finish_unshard_quantizes_stacked_weight_as_flattened_matrix(
        self,
    ) -> None:
        block = 4
        world_size = 3
        weight_factory = _RecordingWeightFactory()
        placement = Fp8BucketedBlockShard(
            world_size=world_size,
            weight_factory=weight_factory,
            block_size=block,
        )
        torch.manual_seed(0)
        param = nn.Parameter(torch.randn(2, 6, 8, device=device_type))
        prepared_by_rank = _prepare_unshard_on_each_rank(
            placement,
            [("w13", param)],
            world_size,
        )
        rank_rows = torch.stack([prepared.buffers[0] for prepared in prepared_by_rank])

        placement._finish_unshard_from_rank_rows(prepared_by_rank[0], rank_rows)

        (factory_call,) = weight_factory.calls
        reference_fp8, reference_scale = _reference_blockwise_quant_weight(
            param.detach().flatten(0, -2),
            block,
        )
        self.assertEqual(factory_call.fp8_data.shape, param.shape)
        self.assertTrue(
            torch.equal(
                factory_call.fp8_data.flatten(0, -2).view(torch.uint8),
                reference_fp8.view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(factory_call.recip_scale, reference_scale))


class TestFp8FusedCopyOut(TestCase):
    """The one-kernel rank-row copy-out matches the cat fallback bitwise."""

    def setUp(self) -> None:
        super().setUp()
        if device_type.type != "cuda":
            self.skipTest("The fused copy-out runs on CUDA.")

    def _metadata(
        self,
        shapes: list[tuple[int, int]],
        placement_cls: type[Fp8BucketedBlockShard] = Fp8BucketedBlockShard,
        world_size: int = 8,
    ) -> tuple[Fp8BucketedBlockShard, Fp8BucketedBlockShard._Fp8BucketMetadata]:
        placement = placement_cls(
            world_size=world_size,
            weight_factory=_return_fp8_data,
            block_size=4,
        )
        named_params = [
            (f"w{index}", nn.Parameter(torch.empty(shape, device="meta")))
            for index, shape in enumerate(shapes)
        ]
        infos = _make_fp8_param_infos(
            named_params,
            placement,
            _FakeMesh(world_size),
        )
        return placement, placement._fp8_metadata_from_infos(infos)

    def test_copy_out_matches_cat_fallback(self) -> None:
        # Shapes, and the copy unit for 4-byte aligned rows: whole 4-byte units
        # when every region is aligned, bytes for odd FP8 byte counts.
        cases = {
            "aligned regions": ([(16, 8), (8, 8), (2, 4)], 4),
            "odd regions": ([(16, 8), (8, 8), (1, 3)], 1),
        }
        # Row start offset and padding between rows: contiguous rows, rows
        # spaced apart as in a mixed bucket, and rows at an odd byte offset.
        row_layouts = {
            "contiguous": (0, 0),
            "spaced": (0, 16),
            "odd offset": (1, 3),
        }
        for case, (shapes, aligned_unit) in cases.items():
            placement, metadata = self._metadata(shapes)
            self.assertIn(0, metadata.rank_fp8_numels)
            world_size = len(metadata.rank_fp8_numels)
            row_nbytes = metadata.packed_send_nbytes_per_rank
            fp8_nbytes = sum(metadata.rank_fp8_numels)
            scale_byte_offset = (
                (fp8_nbytes + _VEC_ALIGN_BYTES - 1)
                // _VEC_ALIGN_BYTES
                * _VEC_ALIGN_BYTES
            )
            compact_nbytes = (
                scale_byte_offset
                + sum(metadata.rank_scale_numels) * torch.float32.itemsize
            )
            for layout, (offset, padding) in row_layouts.items():
                with self.subTest(case=case, layout=layout):
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
                        placement._copy_out_rank_rows_triton(
                            metadata,
                            rank_rows,
                            fused,
                        )
                    )
                    placement._copy_out_rank_rows_cat(
                        metadata,
                        rank_rows,
                        expected,
                        fp8_nbytes=fp8_nbytes,
                        scale_byte_offset=scale_byte_offset,
                    )
                    self.assertTrue(torch.equal(fused, expected))
                    plan = placement._copy_out_plan(
                        metadata,
                        data_elem_nbytes=1,
                        with_scales=True,
                    )
                    unit_nbytes = aligned_unit if offset == 0 else 1
                    self.assertIn(
                        (rank_rows.device, rank_rows.stride(0), unit_nbytes),
                        plan.descriptors,
                    )

    def test_copy_plan_is_built_once_per_row_stride(self) -> None:
        placement, metadata = self._metadata([(16, 8), (8, 8), (6, 4)])
        world_size = len(metadata.rank_fp8_numels)
        row_nbytes = metadata.packed_send_nbytes_per_rank
        compact = torch.empty(
            row_nbytes * world_size,
            dtype=torch.uint8,
            device=device_type,
        )
        rank_rows = torch.zeros(
            (world_size, row_nbytes),
            dtype=torch.uint8,
            device=device_type,
        )
        placement._copy_out_rank_rows_triton(metadata, rank_rows, compact)
        plan = placement._copy_out_plan(metadata, data_elem_nbytes=1, with_scales=True)
        (descriptor,) = plan.descriptors.values()
        placement._copy_out_rank_rows_triton(metadata, rank_rows, compact)
        self.assertIs(
            placement._copy_out_plan(metadata, data_elem_nbytes=1, with_scales=True),
            plan,
        )
        self.assertEqual(list(plan.descriptors.values()), [descriptor])

        spaced_rows = torch.zeros(
            (world_size, row_nbytes + 16),
            dtype=torch.uint8,
            device=device_type,
        )[:, :row_nbytes]
        placement._copy_out_rank_rows_triton(metadata, spaced_rows, compact)
        self.assertEqual(len(plan.descriptors), 2)

    def test_copy_out_falls_back_while_compiling(self) -> None:
        placement, metadata = self._metadata([(16, 8)])
        rank_rows = torch.zeros(
            (len(metadata.rank_fp8_numels), metadata.packed_send_nbytes_per_rank),
            dtype=torch.uint8,
            device=device_type,
        )
        compact = torch.empty(rank_rows.numel(), dtype=torch.uint8, device=device_type)
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(
                placement._copy_out_rank_rows_triton(metadata, rank_rows, compact)
            )
        self.assertFalse(
            placement._copy_out_rank_rows_triton(metadata, rank_rows.cpu(), compact)
        )

    def test_dense_reference_joins_rank_rows(self) -> None:
        placement, metadata = self._metadata(
            [(16, 8), (8, 8), (1, 3)],
            placement_cls=DenseAllGatherReferencePlacement,
        )
        world_size = len(metadata.rank_fp8_numels)
        for row_numel in (max(metadata.rank_fp8_numels) + 5, 64):
            with self.subTest(row_numel=row_numel):
                rank_rows = torch.randn(
                    (world_size, row_numel),
                    dtype=torch.bfloat16,
                    device=device_type,
                )
                joined = rank_rows.new_empty(sum(metadata.rank_fp8_numels))
                self.assertTrue(
                    placement._copy_out_rank_rows_triton(
                        metadata,
                        rank_rows.view(torch.uint8),
                        joined.view(torch.uint8),
                        data_elem_nbytes=rank_rows.element_size(),
                        with_scales=False,
                    )
                )
                expected = torch.cat(
                    [
                        rank_rows[rank, :numel]
                        for rank, numel in enumerate(metadata.rank_fp8_numels)
                    ]
                )
                self.assertTrue(
                    torch.equal(joined.view(torch.int16), expected.view(torch.int16))
                )

    def test_segment_copy_splits_long_segments(self) -> None:
        from ..custom_placements._copy_kernels import (
            copy_segments_triton,
            SegmentCopyDescriptor,
        )

        source = torch.randint(
            -(2**31),
            2**31 - 1,
            (4096,),
            dtype=torch.int32,
            device=device_type,
        )
        # Lengths from empty to 30 blocks, so long segments split into
        # several grid rows next to short ones.
        segments = [(5, 0, 0), (100, 1, 0), (300, 240, 1), (1000, 37, 241)]
        segments.append((2000, 960, 278))
        descriptor = SegmentCopyDescriptor.create(
            src_offsets=[src for src, _, _ in segments],
            numels=[numel for _, numel, _ in segments],
            dst_offsets=[dst for _, _, dst in segments],
            device=device_type,
            block_size=32,
            max_chunks_per_segment=2,
        )
        self.assertGreater(descriptor.nsegments, len(segments))
        self.assertEqual(descriptor.max_segment_numel, 64)
        output = torch.zeros(1238, dtype=torch.int32, device=device_type)
        copy_segments_triton(source, output, descriptor)
        expected = torch.zeros_like(output)
        for src, numel, dst in segments:
            expected[dst : dst + numel] = source[src : src + numel]
        self.assertTrue(torch.equal(output, expected))


def _reference_reduce_scatter_input(
    placement: BucketedBlockShard,
    grads: list[torch.Tensor],
    infos: list[ParamInfo],
    dtype: torch.dtype,
) -> torch.Tensor:
    """The reduce-scatter input as the copies through a zeroed gradient bucket
    build it: each rank's range of the bucket, zero-padded to the longest."""
    bucket_layout = placement._bucket_layout(infos[0])
    bucket = torch.zeros(
        bucket_layout.global_numel, dtype=dtype, device=grads[0].device
    )
    for grad, info in zip(grads, infos, strict=True):
        offset = placement._param_layout(info).param_offset
        bucket[offset : offset + info.global_numel] = grad.reshape(-1)
    rows = torch.zeros(
        len(bucket_layout.rank_numels),
        max(bucket_layout.rank_numels),
        dtype=dtype,
        device=grads[0].device,
    )
    for rank, (offset, numel) in enumerate(
        zip(bucket_layout.rank_offsets, bucket_layout.rank_numels, strict=True)
    ):
        rows[rank, :numel] = bucket[offset : offset + numel]
    return rows


class TestReduceScatterCopyIn(TestCase):
    """The one-pass reduce-scatter copy-in matches the copies through a
    zeroed gradient bucket bitwise."""

    def _bucket(
        self,
        shapes: list[tuple[int, int]],
        world_size: int,
        param_dtype: torch.dtype,
    ) -> tuple[Fp8BucketedBlockShard, list[ParamInfo]]:
        placement = Fp8BucketedBlockShard(
            world_size=world_size,
            weight_factory=_return_fp8_data,
            block_size=4,
        )
        named_params = [
            (
                f"w{index}",
                nn.Parameter(torch.empty(shape, device="meta", dtype=param_dtype)),
            )
            for index, shape in enumerate(shapes)
        ]
        infos = _make_fp8_param_infos(named_params, placement, _FakeMesh(world_size))
        return placement, infos

    def _grads(self, infos: list[ParamInfo], dtype: torch.dtype) -> list[torch.Tensor]:
        torch.manual_seed(0)
        return [
            torch.randn(info.global_shape, dtype=dtype, device=device_type)
            for info in infos
        ]

    def test_matches_reference(self) -> None:
        # Shapes and world sizes: ranks with no rows and padded ranges, param
        # widths whose alignment leaves gaps between params, and odd lengths
        # that leave single-element copy units.
        cases = {
            "padded ranges": ([(16, 8), (8, 8), (12, 4)], 8),
            "gaps between params": ([(8, 12), (8, 8), (4, 12)], 3),
            "odd lengths": ([(16, 8), (5, 3)], 4),
        }
        for case, (shapes, world_size) in cases.items():
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(case=case, dtype=dtype):
                    placement, infos = self._bucket(shapes, world_size, dtype)
                    grads = self._grads(infos, dtype)
                    prepared = placement.prepare_reduce_grad(
                        grads, infos, _FakeMesh(world_size), None
                    )
                    expected = _reference_reduce_scatter_input(
                        placement, grads, infos, dtype
                    )
                    self.assertTrue(
                        torch.equal(prepared.buffers[0].view(expected.shape), expected)
                    )

    def test_matches_reference_with_gaps_between_params(self) -> None:
        # BucketedBlockShard aligns each param to the lcm of the row widths,
        # leaving zero gaps between params, and a rank may own no blocks.
        world_size = 4
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(2, 1, 0, 3))
        named_params = [
            (
                f"w{index}",
                nn.Parameter(torch.empty(shape, device="meta", dtype=torch.bfloat16)),
            )
            for index, shape in enumerate([(7, 3), (5, 4), (9, 3)])
        ]
        infos = _make_fp8_param_infos(named_params, placement, _FakeMesh(world_size))
        grads = self._grads(infos, torch.bfloat16)
        prepared = placement.prepare_reduce_grad(
            grads, infos, _FakeMesh(world_size), None
        )
        expected = _reference_reduce_scatter_input(
            placement, grads, infos, torch.bfloat16
        )
        self.assertTrue(torch.equal(prepared.buffers[0].view(expected.shape), expected))
        if device_type.type == "cuda":
            (plan,) = placement._reduce_copy_in_plans.values()
            self.assertGreater(len(plan.zero_segments), world_size)

    def test_packs_into_wider_rows(self) -> None:
        # A mixed bucket's send buffer: the group's columns start past other
        # groups' columns, and the rows are wider than the group.
        if device_type.type != "cuda":
            self.skipTest("The one-pass copy-in runs on CUDA.")
        world_size = 8
        placement, infos = self._bucket(
            [(16, 8), (8, 8), (12, 4)], world_size, torch.bfloat16
        )
        grads = self._grads(infos, torch.bfloat16)
        expected = _reference_reduce_scatter_input(
            placement, grads, infos, torch.bfloat16
        )
        group_numel = expected.size(1)
        for start, extra in ((0, 0), (4, 12), (3, 5)):
            with self.subTest(start=start, extra=extra):
                rows = torch.full(
                    (world_size, start + group_numel + extra),
                    7.0,
                    dtype=torch.bfloat16,
                    device=device_type,
                )
                send_rows = rows[:, start : start + group_numel]
                placement._pack_reduce_scatter_grad(grads, infos, world_size, send_rows)
                self.assertTrue(torch.equal(send_rows, expected))
                self.assertTrue(torch.all(rows[:, :start] == 7.0))
                self.assertTrue(torch.all(rows[:, start + group_numel :] == 7.0))

    def test_plan_is_built_once_per_row_stride(self) -> None:
        if device_type.type != "cuda":
            self.skipTest("The one-pass copy-in runs on CUDA.")
        world_size = 8
        placement, infos = self._bucket(
            [(16, 8), (8, 8), (12, 4)], world_size, torch.bfloat16
        )
        grads = self._grads(infos, torch.bfloat16)
        mesh = _FakeMesh(world_size)
        placement.prepare_reduce_grad(grads, infos, mesh, None)
        (plan,) = placement._reduce_copy_in_plans.values()
        (descriptors,) = plan.descriptors.values()
        placement.prepare_reduce_grad(grads, infos, mesh, None)
        self.assertEqual(len(placement._reduce_copy_in_plans), 1)
        self.assertIs(next(iter(plan.descriptors.values())), descriptors)
        group_numel = max(placement._bucket_layout(infos[0]).rank_numels)
        rows = torch.empty(
            (world_size, group_numel + 4), dtype=torch.bfloat16, device=device_type
        )
        placement._pack_reduce_scatter_grad(
            grads, infos, world_size, rows[:, :group_numel]
        )
        self.assertEqual(len(placement._reduce_copy_in_plans), 2)

    def test_copies_through_a_bucket_while_compiling_or_casting(self) -> None:
        world_size = 8
        placement, infos = self._bucket(
            [(16, 8), (8, 8), (12, 4)], world_size, torch.bfloat16
        )
        mesh = _FakeMesh(world_size)
        bf16_grads = self._grads(infos, torch.bfloat16)
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertFalse(
                placement._packs_reduce_scatter_grad_in_one_pass(
                    bf16_grads, torch.bfloat16
                )
            )
            prepared = placement.prepare_reduce_grad(bf16_grads, infos, mesh, None)
        expected = _reference_reduce_scatter_input(
            placement, bf16_grads, infos, torch.bfloat16
        )
        self.assertTrue(torch.equal(prepared.buffers[0].view(expected.shape), expected))
        # Grads in another dtype than the reduce dtype are cast on the copy.
        fp32_grads = self._grads(infos, torch.float32)
        self.assertFalse(
            placement._packs_reduce_scatter_grad_in_one_pass(fp32_grads, torch.bfloat16)
        )
        prepared = placement.prepare_reduce_grad(fp32_grads, infos, mesh, None)
        expected = _reference_reduce_scatter_input(
            placement, fp32_grads, infos, torch.bfloat16
        )
        self.assertTrue(torch.equal(prepared.buffers[0].view(expected.shape), expected))
        self.assertEqual(placement._reduce_copy_in_plans, {})


class TestFp8ColumnBlocks(TestCase):
    """Column blocks for long-row weights: the planner, the unshard, the
    reduce-scatter copy-in and the storage copy agree with each rank's
    global_layout boxes."""

    # The last weight's rows (24) are longer than the others' (8), and it has
    # 8 rows: its 4-column blocks hold 32 elements, as the 4-row blocks do.
    shapes = [(16, 8), (8, 8), (12, 8), (8, 24)]
    world_size = 4
    block = 4

    def _infos(self, rank: int, placement: Fp8BucketedBlockShard) -> list[ParamInfo]:
        named_params = [
            (
                f"w{index}",
                nn.Parameter(torch.empty(shape, device="meta", dtype=torch.bfloat16)),
            )
            for index, shape in enumerate(self.shapes)
        ]
        return _make_fp8_param_infos(
            named_params, placement, _FakeMesh(self.world_size, rank)
        )

    def _placement(self, *, columns: bool, factory=_return_fp8_data):
        return Fp8BucketedBlockShard(
            world_size=self.world_size,
            weight_factory=factory,
            block_size=self.block,
            column_blocks_for_long_rows=columns,
        )

    def _local(self, placement, info, rank, full: torch.Tensor) -> torch.Tensor:
        local = torch.zeros(info.local_shape, dtype=full.dtype, device=full.device)
        layout = placement.global_layout(info, rank, self.world_size)
        for global_offset, local_offset, size in zip(
            layout.global_offsets, layout.local_offsets, layout.local_sizes, strict=True
        ):
            local[tuple(slice(o, o + n) for o, n in zip(local_offset, size))] = full[
                tuple(slice(o, o + n) for o, n in zip(global_offset, size))
            ]
        return local

    def test_plans_column_blocks_for_long_rows(self) -> None:
        for columns in (False, True):
            with self.subTest(columns=columns):
                placement = self._placement(columns=columns)
                infos = self._infos(0, placement)
                metadata = placement._fp8_metadata_from_infos(infos)
                column_fqns = {
                    c.fqn for c in metadata.chunks if c.col_start is not None
                }
                self.assertEqual(column_fqns, {"w3"} if columns else set())
        # Llama 3.1 405B's layer at 256 ranks: 9.5% padding with row blocks only.
        layer = {
            "wqkv": (18432, 16384),
            "wo": (16384, 16384),
            "w13": (2, 53248, 16384),
            "w2": (16384, 53248),
        }
        named_params = [
            (fqn, nn.Parameter(torch.empty(shape, device="meta", dtype=torch.bfloat16)))
            for fqn, shape in layer.items()
        ]
        numel = sum(param.numel() for _, param in named_params)
        for columns, expected in ((False, 13_631_488), (True, 12_582_912)):
            placement = Fp8BucketedBlockShard(
                world_size=256,
                weight_factory=_return_fp8_data,
                column_blocks_for_long_rows=columns,
            )
            infos = _make_fp8_param_infos(named_params, placement, _FakeMesh(256))
            rank_numels = placement._bucket_layout(infos[0]).rank_numels
            self.assertEqual(max(rank_numels), expected)
            self.assertEqual(sum(rank_numels), numel)
        # The dense reference keeps row blocks: its dense payload handles only them.
        reference = DenseAllGatherReferencePlacement(
            world_size=self.world_size,
            weight_factory=_return_fp8_data,
            block_size=self.block,
            column_blocks_for_long_rows=True,
        )
        metadata = reference._fp8_metadata_from_infos(self._infos(0, reference))
        self.assertTrue(all(c.col_start is None for c in metadata.chunks))
        raw = self._placement(columns=True)
        member = MixedBucketPlacement({}).fp8_bucketed_block_shard(raw)
        self.assertTrue(member.column_blocks_for_long_rows)

    def test_unshard_matches_reference_quantization(self) -> None:
        if device_type.type != "cuda":
            self.skipTest("The FP8 quantization kernels run on CUDA.")
        torch.manual_seed(0)
        fulls = [
            torch.randn(shape, dtype=torch.bfloat16, device=device_type)
            for shape in self.shapes
        ]
        for compiling in (False, True):
            with self.subTest(compiling=compiling):
                factory = _RecordingWeightFactory()
                placement = self._placement(columns=True, factory=factory)
                rank_infos = [self._infos(r, placement) for r in range(self.world_size)]
                payloads = []
                for rank, infos in enumerate(rank_infos):
                    metadata = placement._fp8_metadata_from_infos(infos)
                    tensors = [
                        self._local(placement, info, rank, full)
                        for info, full in zip(infos, fulls, strict=True)
                    ]
                    payloads.append(
                        placement._pack_local_quantized_chunks(
                            tensors, infos, rank, metadata
                        )
                    )
                metadata = placement._fp8_metadata_from_infos(rank_infos[0])
                prepared = PlacementPreparedUnshard(
                    placement=placement,
                    buffers=[payloads[0], torch.stack(payloads).view(-1)],
                    placement_state=Fp8BucketedBlockShard._Fp8UnshardState(
                        infos=rank_infos[0], pg=None, debug_fqn=None, metadata=metadata
                    ),
                )
                with patch.object(
                    torch.compiler, "is_compiling", return_value=compiling
                ):
                    placement.finish_prepared_unshard(prepared)
                for call, full in zip(factory.calls, fulls, strict=True):
                    reference_fp8, reference_scale = _reference_blockwise_quant_weight(
                        full, self.block
                    )
                    self.assertTrue(
                        torch.equal(
                            call.fp8_data.view(torch.uint8),
                            reference_fp8.view(torch.uint8),
                        )
                    )
                    self.assertTrue(torch.equal(call.recip_scale, reference_scale))

    def test_reduce_scatter_rows_hold_each_ranks_boxes(self) -> None:
        torch.manual_seed(0)
        grads = [
            torch.randn(shape, dtype=torch.bfloat16, device=device_type)
            for shape in self.shapes
        ]
        placement = self._placement(columns=True)
        rank_infos = [self._infos(r, placement) for r in range(self.world_size)]
        layout = placement._bucket_layout(rank_infos[0][0])
        mesh = _FakeMesh(self.world_size)
        for compiling in (False, True):
            with self.subTest(compiling=compiling):
                with patch.object(
                    torch.compiler, "is_compiling", return_value=compiling
                ):
                    prepared = placement.prepare_reduce_grad(
                        grads, rank_infos[0], mesh, None
                    )
                rows = prepared.buffers[0].view(self.world_size, -1)
                for rank, infos in enumerate(rank_infos):
                    rank_offset = layout.rank_offsets[rank]
                    for info, grad in zip(infos, grads, strict=True):
                        if info.local_numel == 0:
                            continue
                        start = (
                            placement._param_layout(info).local_global_offset
                            - rank_offset
                        )
                        got = rows[rank, start : start + info.local_numel].view(
                            info.local_shape
                        )
                        self.assertTrue(
                            torch.equal(got, self._local(placement, info, rank, grad))
                        )
                    self.assertTrue(
                        torch.all(rows[rank, layout.rank_numels[rank] :] == 0)
                    )

    def test_storage_copy_writes_column_rectangles(self) -> None:
        torch.manual_seed(0)
        full = torch.randn(self.shapes[-1], dtype=torch.bfloat16, device=device_type)
        placement = self._placement(columns=True)
        for rank in range(self.world_size):
            info = self._infos(rank, placement)[-1]
            if info.local_numel == 0:
                continue
            storage = torch.zeros(
                info.byte_offset + info.storage_nbytes,
                dtype=torch.uint8,
                device=device_type,
            )
            placement.copy_param_to_storage(storage, info, full, rank, self.world_size)
            got = (
                storage[info.byte_offset :].view(torch.bfloat16).view(info.local_shape)
            )
            self.assertTrue(torch.equal(got, self._local(placement, info, rank, full)))

    def test_refill_writes_the_first_unshards_bytes_without_full_params(self) -> None:
        # The eager CUDA copy-out copies every column rectangle in one box
        # kernel per copy unit; while compiling, one strided copy each. A refill
        # writes the persistent buffer and makes no full params.
        if device_type.type != "cuda":
            self.skipTest("The FP8 quantization kernels run on CUDA.")
        torch.manual_seed(0)
        fulls = [
            torch.randn(shape, dtype=torch.bfloat16, device=device_type)
            for shape in self.shapes
        ]
        placement = self._placement(columns=True)
        rank_infos = [self._infos(r, placement) for r in range(self.world_size)]
        payloads = []
        for rank, infos in enumerate(rank_infos):
            metadata = placement._fp8_metadata_from_infos(infos)
            tensors = [
                self._local(placement, info, rank, full)
                for info, full in zip(infos, fulls, strict=True)
            ]
            payloads.append(
                placement._pack_local_quantized_chunks(tensors, infos, rank, metadata)
            )
        metadata = placement._fp8_metadata_from_infos(rank_infos[0])

        def finish(persistent_buffers):
            prepared = PlacementPreparedUnshard(
                placement=placement,
                buffers=[payloads[0], torch.stack(payloads).view(-1)],
                placement_state=Fp8BucketedBlockShard._Fp8UnshardState(
                    infos=rank_infos[0], pg=None, debug_fqn=None, metadata=metadata
                ),
                persistent=True,
                persistent_buffers=persistent_buffers,
            )
            return placement.finish_prepared_unshard(prepared)

        first = finish(None)
        self.assertEqual(len(first.full_params), len(self.shapes))
        (compact,) = first.persistent_buffers
        for compiling in (False, True):
            with self.subTest(compiling=compiling):
                buffer = torch.zeros_like(compact)
                with patch.object(
                    torch.compiler, "is_compiling", return_value=compiling
                ):
                    refill = finish([buffer])
                self.assertEqual(refill.full_params, [])
                self.assertEqual(len(refill.persistent_buffers), 1)
                self.assertIs(refill.persistent_buffers[0], buffer)
                self.assertTrue(torch.equal(buffer, compact))


class TestFp8TileUnits(TestFp8ColumnBlocks):
    """Tile units: the planner, and TestFp8ColumnBlocks' agreement checks with
    weights whose dims are whole blocks taking tiles, runs of them starting and
    ending inside tile rows, and one with a partial last block row taking
    column blocks."""

    shapes = [(16, 8), (8, 12), (12, 8), (6, 8)]
    world_size = 5
    block = 4
    test_plans_column_blocks_for_long_rows = None

    def _placement(self, *, columns: bool, factory=_return_fp8_data):
        del columns
        return Fp8BucketedBlockShard(
            world_size=self.world_size,
            weight_factory=factory,
            block_size=self.block,
            tile_units=True,
        )

    def test_plans_tile_units(self) -> None:
        placement = self._placement(columns=False)
        chunks = placement._fp8_metadata_from_infos(self._infos(0, placement)).chunks
        self.assertEqual(
            {c.fqn for c in chunks if c.tile_start is not None}, {"w0", "w1", "w2"}
        )
        self.assertEqual({c.fqn for c in chunks if c.col_start is not None}, {"w3"})
        tile_cols = {"w0": 2, "w1": 3, "w2": 2}
        self.assertTrue(
            any(
                c.tile_start % tile_cols[c.fqn] or c.tile_end % tile_cols[c.fqn]
                for c in chunks
                if c.tile_start is not None
            )
        )
        # DeepSeek V3's attention and shared-expert linears at 384 ranks: wo's
        # 2.1 MB block rows pad every rank's 0.6 MB share to them with row
        # blocks, and tiles bring that to under 1% padding.
        layer = {
            "wq_a": (1536, 7168),
            "wq_b": (24576, 1536),
            "wkv_a": (576, 7168),
            "wkv_b": (32768, 512),
            "wo": (7168, 16384),
            "w1": (2048, 7168),
            "w3": (2048, 7168),
            "w2": (7168, 2048),
        }
        named_params = [
            (fqn, nn.Parameter(torch.empty(shape, device="meta", dtype=torch.bfloat16)))
            for fqn, shape in layer.items()
        ]
        numel = sum(param.numel() for _, param in named_params)
        for tiles, max_padding in ((False, 2.4), (True, 0.01)):
            placement = Fp8BucketedBlockShard(
                world_size=384, weight_factory=_return_fp8_data, tile_units=tiles
            )
            infos = _make_fp8_param_infos(named_params, placement, _FakeMesh(384))
            rank_numels = placement._bucket_layout(infos[0]).rank_numels
            self.assertEqual(sum(rank_numels), numel)
            padding = max(rank_numels) * 384 / numel - 1
            if tiles:
                self.assertLess(padding, max_padding)
            else:
                self.assertGreater(padding, max_padding)
        # The dense reference keeps row blocks, and the mixed bucket's member
        # keeps tile units.
        reference = DenseAllGatherReferencePlacement(
            world_size=self.world_size,
            weight_factory=_return_fp8_data,
            block_size=self.block,
            tile_units=True,
        )
        metadata = reference._fp8_metadata_from_infos(self._infos(0, reference))
        self.assertTrue(
            all(c.col_start is None and c.tile_start is None for c in metadata.chunks)
        )
        member = MixedBucketPlacement({}).fp8_bucketed_block_shard(
            self._placement(columns=False)
        )
        self.assertTrue(member.tile_units)


class TestCopyBoxesTriton(TestCase):
    def test_copies_boxes_through_each_unit(self) -> None:
        if device_type.type != "cuda":
            self.skipTest("copy_boxes_triton is a CUDA kernel")
        from ..custom_placements._copy_kernels import (
            BoxCopyDescriptor,
            copy_boxes_triton,
        )

        generator = torch.Generator().manual_seed(0)

        def pick(choices):
            return choices[int(torch.randint(len(choices), (1,), generator=generator))]

        for unit in (8, 4, 2, 1):
            for shift in sorted({0, unit % 8}):
                with self.subTest(unit=unit, shift=shift):
                    source_buf = torch.randint(
                        0, 256, (1 << 18,), dtype=torch.uint8, device=device_type
                    )
                    output_buf = torch.zeros(
                        1 << 18, dtype=torch.uint8, device=device_type
                    )
                    source, output = source_buf[shift:], output_buf[shift:]
                    expected = output.clone()
                    boxes, src_end, dst_end = [], 0, 0
                    for _ in range(40):
                        rows = pick([1, 3, 17, 70])
                        row_numel = pick([1, 3, 16, 96])
                        src_stride = row_numel + pick([0, 5])
                        dst_stride = row_numel + pick([0, 7, 64])
                        if (dst_end + rows * dst_stride) * unit > output.numel() or (
                            src_end + rows * src_stride
                        ) * unit > source.numel():
                            break
                        boxes.append(
                            (src_end, rows, row_numel, src_stride, dst_end, dst_stride)
                        )
                        for row in range(rows):
                            src = (src_end + row * src_stride) * unit
                            dst = (dst_end + row * dst_stride) * unit
                            expected[dst : dst + row_numel * unit] = source[
                                src : src + row_numel * unit
                            ]
                        src_end += rows * src_stride
                        dst_end += rows * dst_stride
                    descriptor = BoxCopyDescriptor.create(
                        boxes, unit_nbytes=unit, device=output.device
                    )
                    self.assertGreater(descriptor.ntiles, len(boxes))
                    copy_boxes_triton(source, output, descriptor)
                    self.assertTrue(torch.equal(output, expected))


class TestCopyBox(TestCase):
    def test_copies_through_the_widest_unit(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # (dtype, source columns, destination columns, first source column,
        # first destination column, columns copied, expected unit bytes)
        cases = [
            (torch.uint8, 256, 512, 0, 128, 64, 8),
            (torch.uint8, 256, 512, 4, 128, 64, 4),
            (torch.uint8, 256, 512, 0, 0, 6, 2),
            (torch.uint8, 256, 512, 0, 0, 3, 1),
            (torch.uint8, 255, 512, 0, 0, 64, 1),
            (torch.bfloat16, 128, 256, 4, 8, 64, 8),
            (torch.bfloat16, 128, 256, 2, 8, 64, 4),
            (torch.float32, 64, 128, 0, 2, 6, 8),
        ]
        for dtype, src_cols, dst_cols, src_col, dst_col, cols, unit in cases:
            with self.subTest(
                dtype=dtype, src_cols=src_cols, src_col=src_col, cols=cols
            ):
                src = torch.randint(0, 100, (8, src_cols), device=device).to(dtype)
                dst = torch.zeros(8, dst_cols, dtype=dtype, device=device)
                expected = dst.clone()
                expected[:, dst_col : dst_col + cols] = src[:, src_col : src_col + cols]
                self.assertEqual(
                    _copy_box_(
                        dst[:, dst_col : dst_col + cols],
                        src[:, src_col : src_col + cols],
                    ),
                    unit,
                )
                self.assertTrue(torch.equal(dst, expected))
        # Different dtypes convert through a plain copy.
        src = torch.randn(4, 64, dtype=torch.bfloat16, device=device)
        dst = torch.zeros(4, 64, device=device)
        self.assertEqual(_copy_box_(dst, src), 4)
        self.assertTrue(torch.equal(dst, src.float()))


class TestFp8BlockwiseWeightQuantization(TestCase):
    def test_production_quantizer_matches_reference_for_partial_blocks(self) -> None:
        if device_type.type != "cuda":
            self.skipTest("torchao blockwise FP8 kernels require CUDA")
        if not is_sm_at_least_90():
            self.skipTest("torchao blockwise FP8 training kernels require SM90+")

        block_size = 4
        for shape in ((6, 4), (12, 6), (6, 6)):
            with self.subTest(shape=shape):
                torch.manual_seed(0)
                weight = torch.randn(
                    shape,
                    device=device_type,
                    dtype=torch.bfloat16,
                )
                result_fp8, result_scale = (
                    fp8_bucketed_block_shard_module._quantize_dense_weight_to_blockwise_fp8(
                        weight,
                        block_size,
                        torch.float8_e4m3fn,
                    )
                )
                reference_fp8, reference_scale = _reference_blockwise_quant_weight(
                    weight,
                    block_size,
                )
                self.assertTrue(
                    torch.equal(
                        result_fp8.view(torch.uint8),
                        reference_fp8.view(torch.uint8),
                    )
                )
                self.assertTrue(torch.equal(result_scale, reference_scale))


class TestFp8AllGather(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        return init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

    def _skip_unless_blockwise_fp8_supported(self) -> None:
        if not is_sm_at_least_90():
            self.skipTest("Blockwise FP8 kernels require CUDA SM90+")

    @skip_if_lt_x_gpu(2)
    def test_mixed_fp8_bucket_uses_one_collective_and_matches_reference(self) -> None:
        self._skip_unless_blockwise_fp8_supported()
        block = 4
        mesh = self._mesh()

        class MixedFp8Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.Parameter(torch.randn(8))
                self.weight = nn.Parameter(torch.randn(16, 8))

        torch.manual_seed(0)
        model = MixedFp8Module().to(device=device_type, dtype=torch.bfloat16)
        originals = {
            fqn: param.detach().clone() for fqn, param in model.named_parameters()
        }
        weight_factory = _RecordingWeightFactory()
        raw_placement = Fp8BucketedBlockShard(
            world_size=self.world_size,
            weight_factory=weight_factory,
            block_size=block,
        )
        mixed = MixedBucketPlacement({})
        weight_placement = mixed.fp8_bucketed_block_shard(raw_placement)
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

        prepared = mixed.prepare_unshard_bucket(
            local_params,
            infos,
            mesh,
            None,
        )
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
        (factory_call,) = weight_factory.calls
        self.assertIs(weight, factory_call.fp8_data)
        self.assertEqual(factory_call.orig_dtype, torch.bfloat16)
        self.assertTrue(factory_call.requires_grad)
        reference_fp8, reference_scale = _reference_blockwise_quant_weight(
            originals["weight"],
            block,
        )
        self.assertTrue(
            torch.equal(
                factory_call.fp8_data.view(torch.uint8),
                reference_fp8.view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(factory_call.recip_scale, reference_scale))

        # The FP8 group's grads go straight into the bucket's send buffer, not
        # through a send buffer of the group's own.
        with patch.object(
            Fp8BucketedBlockShard,
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

        # A persistent unshard, then a refill after the local shards change: the
        # weights built by the first unshard see the new values.
        def persistent_unshard(persistent_buffers=None):
            prepared = mixed.prepare_unshard_bucket(local_params, infos, mesh, None)
            prepared.persistent = True
            prepared.persistent_buffers = persistent_buffers
            mixed.run_prepared_unshard(prepared)
            return mixed.finish_prepared_unshard(prepared)

        first = persistent_unshard()
        with torch.no_grad():
            for local_param in local_params:
                local_param.add_(1.0)
        for buffer in first.persistent_buffers:
            nbytes = buffer.untyped_storage().size()
            buffer.untyped_storage().resize_(0)
            buffer.untyped_storage().resize_(nbytes)
        persistent_unshard(first.persistent_buffers)
        self.assertEqual(first.full_params[0], originals["norm"] + 1.0)
        reference_fp8, reference_scale = _reference_blockwise_quant_weight(
            originals["weight"] + 1.0,
            block,
        )
        first_call = weight_factory.calls[1]
        self.assertTrue(
            torch.equal(
                first_call.fp8_data.view(torch.uint8),
                reference_fp8.view(torch.uint8),
            )
        )
        self.assertTrue(torch.equal(first_call.recip_scale, reference_scale))


if __name__ == "__main__":
    run_tests()
