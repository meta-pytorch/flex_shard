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
from torch.testing._internal.common_fsdp import get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torchao.utils import is_sm_at_least_90

from ..custom_placements import (
    fp8_bucketed_block_shard as fp8_bucketed_block_shard_module,
)
from ..custom_placements.fp8_bucketed_block_shard import (
    _VEC_ALIGN_BYTES,
    Fp8BucketedBlockShard,
)
from ..flex_shard.bucket_storage import ParamInfo


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
        if device_type.type != "cuda":
            cache_context_patch = patch.object(
                Fp8BucketedBlockShard,
                "_local_packed_cache_context",
                return_value=(device_type, None),
            )
            cache_context_patch.start()
            self.addCleanup(cache_context_patch.stop)

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

    def test_fp8_cache_phase_detects_checkpoint_recompute(self) -> None:
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
        phases = []
        payloads = []

        def function(tensor: torch.Tensor) -> torch.Tensor:
            phases.append(fp8_bucketed_block_shard_module._is_backward_graph_task())
            prepared = placement.prepare_unshard_bucket(
                [param.detach()],
                [info],
                mesh,
                None,
            )
            payloads.append(prepared.buffers[0])
            return tensor.sin() * tensor.cos()

        tensor = torch.randn(8, device=device_type, requires_grad=True)
        with patch.object(
            placement,
            "_quantize_local_weight",
            wraps=placement._quantize_local_weight,
        ) as quantize:
            checkpoint(function, tensor, use_reentrant=False).sum().backward()

        self.assertEqual(phases, [False, True])
        self.assertEqual(quantize.call_count, 1)
        self.assertIs(payloads[0], payloads[1])

    def test_fp8_local_packed_cache_rebuilds_every_forward(self) -> None:
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

        with (
            patch.object(
                fp8_bucketed_block_shard_module,
                "_is_backward_graph_task",
                return_value=False,
            ),
            patch.object(
                placement,
                "_quantize_local_weight",
                wraps=placement._quantize_local_weight,
            ) as quantize,
        ):
            first = placement.prepare_unshard_bucket(
                [param.detach()],
                [info],
                mesh,
                None,
            )
            first_payload = first.buffers[0].clone()
            initial_version = param._version
            param.data[0, 0].add_(1000)
            self.assertEqual(param._version, initial_version)
            second = placement.prepare_unshard_bucket(
                [param.detach()],
                [info],
                mesh,
                None,
            )

        self.assertEqual(quantize.call_count, 2)
        self.assertIsNot(first.buffers[0], second.buffers[0])
        self.assertTrue(torch.equal(first.buffers[0], first_payload))
        self.assertFalse(torch.equal(second.buffers[0], first_payload))

        fp8_data, recip_scale = _packed_rank_views(
            second.buffers[0],
            0,
            second.placement_state.metadata,
            placement.fp8_dtype,
        )
        reference_fp8, reference_scale = _reference_blockwise_quant_weight(
            param.detach(),
            placement.block_size,
        )
        self.assertTrue(torch.equal(fp8_data.view(param.shape), reference_fp8))
        self.assertTrue(
            torch.equal(recip_scale.view(reference_scale.shape), reference_scale)
        )

    def test_fp8_local_packed_cache_is_bypassed_while_compiling(self) -> None:
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
        cached = placement.prepare_unshard_bucket(
            [param.detach()],
            [info],
            mesh,
            None,
        )

        with patch.object(torch.compiler, "is_compiling", return_value=True):
            first = placement.prepare_unshard_bucket(
                [param.detach()],
                [info],
                mesh,
                None,
            )
            second = placement.prepare_unshard_bucket(
                [param.detach()],
                [info],
                mesh,
                None,
            )

        with patch.object(
            fp8_bucketed_block_shard_module,
            "_is_backward_graph_task",
            return_value=True,
        ):
            restored = placement.prepare_unshard_bucket(
                [param.detach()],
                [info],
                mesh,
                None,
            )
        self.assertIsNot(first.buffers[0], second.buffers[0])
        self.assertIsNot(first.buffers[0], cached.buffers[0])
        self.assertIs(restored.buffers[0], cached.buffers[0])

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
        prepared_by_rank = []
        for rank in range(world_size):
            mesh = _FakeMesh(world_size, rank)
            infos = _make_fp8_param_infos(named_params, placement, mesh)
            local_tensors = []
            for (_, param), info in zip(named_params, infos, strict=True):
                if info.local_numel == 0:
                    local_tensors.append(param.new_empty(info.local_shape))
                    continue
                param_layout = info.bucket_layout.param_layouts[info.fqn]
                relative_start = (
                    param_layout.local_global_offset - param_layout.param_offset
                )
                row_start = relative_start // param.shape[1]
                row_end = row_start + info.local_shape[0]
                local_tensors.append(param.detach()[row_start:row_end].contiguous())
            prepared_by_rank.append(
                placement.prepare_unshard_bucket(
                    local_tensors,
                    infos,
                    mesh,
                    None,
                )
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

        result = placement._finish_unshard_from_rank_rows(prepared, rank_rows)
        self.assertEqual(len(result.consumer_buffers), 1)
        compact = result.consumer_buffers[0]
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


if __name__ == "__main__":
    run_tests()
