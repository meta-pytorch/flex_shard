# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.func import functional_call
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase
from torchao.utils import is_sm_at_least_90

from .. import BucketSpec, flex_shard
from ..blockwise_fp8 import (
    BlockwiseFp8Weight,
    convert_to_flex_shard_float8_blockwise_linear,
    FlexShardFloat8BlockwiseLinear,
)
from ..custom_placements.fp8_bucketed_block_shard import (
    _align_up,
    _quantize_dense_weight_to_blockwise_fp8,
    _VEC_ALIGN_BYTES,
    BlockwiseFp8WeightFactory,
    Fp8BucketedBlockShard,
    make_fp8_bucketed_block_placement_fn,
)
from ..custom_placements.shard import per_param_placements
from ..custom_placements.utils import foreach_copy_
from ..flex_shard.bucket_storage import ParamInfo, PlacementFn
from ..flex_shard.placement_contract import (
    Placement,
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from ..flex_shard.utils import (
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
)
from .common import make_test_sgd, make_transformer_model


device_type = torch.device(get_devtype())


def _make_dense_all_gather_reference_placement_fn(
    *,
    weight_factory: BlockwiseFp8WeightFactory,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    def dense_all_gather_reference_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        placement = _DenseAllGatherReferencePlacement(
            world_size=mesh.size(),
            weight_factory=weight_factory,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return dense_all_gather_reference_placements


class _DenseAllGatherReferencePlacement(Fp8BucketedBlockShard):
    """All-gather dense shards using the FP8 placement's shard layout."""

    def _dense_send_numel_per_rank(
        self,
        metadata: Fp8BucketedBlockShard._Fp8BucketMetadata,
    ) -> int:
        return _align_up(
            max(metadata.rank_fp8_numels, default=0),
            _VEC_ALIGN_BYTES,
        )

    def _pack_local_dense_chunks(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        rank: int,
        metadata: Fp8BucketedBlockShard._Fp8BucketMetadata,
    ) -> torch.Tensor:
        dtype = infos[0].unsharded_dtype
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        local_dense = torch.empty(
            self._dense_send_numel_per_rank(metadata),
            dtype=dtype,
            device=device,
        )
        payload_end = metadata.rank_fp8_numels[rank]
        if payload_end < local_dense.numel():
            local_dense[payload_end:].zero_()
        tensors_by_fqn = {
            info.fqn: tensor.reshape(info.local_shape)
            for tensor, info in zip(tensors, infos, strict=True)
        }
        infos_by_fqn = {info.fqn: info for info in infos}
        for chunk in metadata.chunks:
            if chunk.rank != rank:
                continue
            info = infos_by_fqn[chunk.fqn]
            tensor = tensors_by_fqn[chunk.fqn]
            expected_numel = self._chunk_numel(chunk, info.global_shape[1])
            if tensor.numel() != expected_numel:
                raise AssertionError(
                    "Dense reference local shard does not match planned chunk for "
                    f"{chunk.fqn!r}: expected {expected_numel} elements, got "
                    f"{tensor.numel()}."
                )
            local_dense[chunk.fp8_offset : chunk.fp8_offset + tensor.numel()].copy_(
                tensor.reshape(-1)
            )
        return local_dense

    def _prepare_local_unshard_payload(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        rank = mesh.get_local_rank()
        metadata = self._fp8_metadata_from_infos(infos)
        with _record_copy_in_if_eager():
            local_dense = self._pack_local_dense_chunks(
                tensors,
                infos,
                rank,
                metadata,
            )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_dense],
            placement_state=Fp8BucketedBlockShard._Fp8UnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                metadata=metadata,
            ),
        )

    def _finish_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
    ) -> PlacementUnshardResult:
        state = prepared.placement_state
        if not isinstance(state, Fp8BucketedBlockShard._Fp8UnshardState):
            raise AssertionError(
                "Expected Fp8BucketedBlockShard._Fp8UnshardState, "
                f"got {type(state).__name__}"
            )
        expected_shape = (
            len(state.metadata.rank_fp8_numels),
            self._dense_send_numel_per_rank(state.metadata),
        )
        if tuple(rank_rows.shape) != expected_shape:
            raise ValueError(
                f"Expected gathered rank rows with shape {expected_shape}, "
                f"got {tuple(rank_rows.shape)}."
            )

        infos_by_fqn = {info.fqn: info for info in state.infos}
        dense_by_fqn = {
            info.fqn: torch.empty(
                info.global_shape,
                dtype=rank_rows.dtype,
                device=rank_rows.device,
            )
            for info in state.infos
        }
        copy_dsts: list[torch.Tensor] = []
        copy_srcs: list[torch.Tensor] = []
        with _record_copy_out_if_eager():
            for chunk in state.metadata.chunks:
                info = infos_by_fqn[chunk.fqn]
                in_dim = info.global_shape[1]
                numel = self._chunk_numel(chunk, in_dim)
                copy_dsts.append(
                    dense_by_fqn[chunk.fqn][chunk.row_start : chunk.row_end, :]
                )
                copy_srcs.append(
                    rank_rows[
                        chunk.rank,
                        chunk.fp8_offset : chunk.fp8_offset + numel,
                    ].view(chunk.row_end - chunk.row_start, in_dim)
                )
            foreach_copy_(copy_dsts, copy_srcs)
        full_params = []
        persistent_buffers: list[torch.Tensor] = []
        for index, info in enumerate(state.infos):
            dense_weight = dense_by_fqn[info.fqn]
            dense_weight.requires_grad_(info.requires_grad)
            fp8_data, recip_scale = _quantize_dense_weight_to_blockwise_fp8(
                dense_weight,
                self.block_size,
                self.fp8_dtype,
            )
            if prepared.persistent_buffers is not None:
                # Persistent unshard: refill the previous fp8 data and scales.
                persistent_data = prepared.persistent_buffers[2 * index]
                persistent_scale = prepared.persistent_buffers[2 * index + 1]
                persistent_data.copy_(fp8_data)
                persistent_scale.copy_(recip_scale)
                fp8_data, recip_scale = persistent_data, persistent_scale
            persistent_buffers.extend((fp8_data, recip_scale))
            full_params.append(
                self._make_blockwise_fp8_weight(
                    fp8_data,
                    recip_scale,
                    orig_dtype=dense_weight.dtype,
                    requires_grad=dense_weight.requires_grad,
                )
            )
        return PlacementUnshardResult(
            full_params=full_params,
            persistent_buffers=persistent_buffers if prepared.persistent else [],
        )


def _make_blockwise_fp8_weight() -> BlockwiseFp8Weight:
    block_size = 4
    fp8_data = (
        torch.arange(128, dtype=torch.bfloat16).reshape(16, 8).to(torch.float8_e4m3fn)
    )
    recip_scale = torch.full((4, 2), 0.5, dtype=torch.float32)
    return BlockwiseFp8Weight(
        fp8_data,
        recip_scale,
        block_size,
        orig_dtype=torch.bfloat16,
        requires_grad=True,
    )


class TestBlockwiseFp8Weight(TestCase):
    def test_blockwise_fp8_weight_protocol(self) -> None:
        wrapped = _make_blockwise_fp8_weight()

        self.assertEqual(wrapped.dtype, torch.bfloat16)
        self.assertTrue(wrapped.requires_grad)
        self.assertEqual(wrapped.block_size, 4)
        self.assertEqual(wrapped.orig_dtype, torch.bfloat16)
        self.assertEqual(
            wrapped.dequantize(torch.float32),
            wrapped.fp8_data.float() * 0.5,
        )

    def test_stacked_weight_scales_cover_flattened_matrix(self) -> None:
        block_size = 4
        fp8_data = (
            torch.arange(96, dtype=torch.bfloat16)
            .reshape(2, 6, 8)
            .to(torch.float8_e4m3fn)
        )
        # One scale per 4x4 block of the [12, 8] matrix; block row 1 holds
        # rows 4..5 of the first matrix and rows 0..1 of the second.
        recip_scale = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2)
        wrapped = BlockwiseFp8Weight(
            fp8_data,
            recip_scale,
            block_size,
            orig_dtype=torch.bfloat16,
        )

        self.assertEqual(wrapped.shape, (2, 6, 8))
        matrix_scale = recip_scale.repeat_interleave(block_size, 0).repeat_interleave(
            block_size, 1
        )
        self.assertEqual(
            wrapped.dequantize(torch.float32),
            fp8_data.float() * matrix_scale.view(2, 6, 8),
        )


class TestFlexShardFloat8BlockwiseLinear(TestCase):
    def test_stacked_weight_matches_flattened_matrix(self) -> None:
        """A ``[2, F, D]`` weight runs the ``[2F, D]`` GEMM, including the block
        row that straddles both stacked matrices when ``F % 128 != 0``."""
        if device_type.type != "cuda" or not is_sm_at_least_90():
            self.skipTest("Float8BlockwiseLinear requires CUDA SM90+")
        num_linears, out_dim, in_dim = 2, 192, 256
        torch.manual_seed(0)
        fp8_data, recip_scale = _quantize_dense_weight_to_blockwise_fp8(
            torch.randn(
                num_linears * out_dim,
                in_dim,
                device=device_type,
                dtype=torch.bfloat16,
            ),
            128,
            torch.float8_e4m3fn,
        )
        linear = FlexShardFloat8BlockwiseLinear(
            in_dim,
            num_linears * out_dim,
            bias=False,
            use_triton=True,
        )
        x = torch.randn(
            2,
            64,
            in_dim,
            device=device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        grad_output = torch.randn(
            2,
            64,
            num_linears * out_dim,
            device=device_type,
            dtype=torch.bfloat16,
        )

        def run(weight_shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
            weight = BlockwiseFp8Weight(
                fp8_data.view(weight_shape),
                recip_scale,
                128,
                orig_dtype=torch.bfloat16,
                requires_grad=True,
            )
            out = functional_call(linear, {"weight": weight}, (x,))
            return out, *torch.autograd.grad(
                out,
                (x, weight),
                grad_output.view(out.shape),
            )

        out, grad_x, grad_weight = run((num_linears, out_dim, in_dim))
        flat_out, flat_grad_x, flat_grad_weight = run((num_linears * out_dim, in_dim))

        self.assertEqual(out.shape, (2, 64, num_linears, out_dim))
        self.assertEqual(grad_weight.shape, (num_linears, out_dim, in_dim))
        self.assertTrue(torch.equal(out, flat_out.view(out.shape)))
        self.assertTrue(torch.equal(grad_x, flat_grad_x))
        self.assertTrue(
            torch.equal(grad_weight, flat_grad_weight.view(grad_weight.shape))
        )


class TestBlockwiseFp8FlexShardTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        return init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

    def _skip_unless_blockwise_linear_supported(self) -> None:
        if not is_sm_at_least_90():
            self.skipTest("Float8BlockwiseLinear requires CUDA SM90+")

    def _assert_sharded_linear_state_is_bf16(self, model: nn.Module) -> None:
        for storage in model.sharded_bucket_storages:
            for info in storage.param_infos.values():
                self.assertEqual(info.dtype, torch.bfloat16)
                self.assertEqual(
                    storage.get_local_view(info.fqn).dtype,
                    torch.bfloat16,
                )

    def _make_transformer_model_pair(self, *, mesh):
        block_size = 128
        torch.manual_seed(0)
        model_args, fp8_model = make_transformer_model(
            device=device_type,
            n_layers=1,
            vocab_size=256,
            max_seq_len=8,
            dim=256,
            n_heads=4,
        )
        fp8_model.to(dtype=torch.bfloat16)
        bf16_model = copy.deepcopy(fp8_model)

        linear_weight_fqns = [
            f"{fqn}.weight"
            for fqn, module in fp8_model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        linear_weight_fqn_set = set(linear_weight_fqns)
        other_param_fqns = [
            fqn
            for fqn, _ in fp8_model.named_parameters()
            if fqn not in linear_weight_fqn_set
        ]
        convert_to_flex_shard_float8_blockwise_linear(fp8_model, use_triton=False)
        convert_to_flex_shard_float8_blockwise_linear(bf16_model, use_triton=False)

        def make_buckets(placement_fn) -> list[BucketSpec]:
            return [
                BucketSpec(
                    linear_weight_fqns,
                    placement_fn=placement_fn,
                    mesh=mesh,
                    reshard_after_forward=True,
                ),
                BucketSpec(
                    other_param_fqns,
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=True,
                ),
            ]

        flex_shard(
            bf16_model,
            buckets=make_buckets(
                _make_dense_all_gather_reference_placement_fn(
                    weight_factory=BlockwiseFp8Weight,
                    block_size=block_size,
                )
            ),
        )
        flex_shard(
            fp8_model,
            buckets=make_buckets(
                make_fp8_bucketed_block_placement_fn(
                    weight_factory=BlockwiseFp8Weight,
                    block_size=block_size,
                )
            ),
        )
        return model_args, fp8_model, bf16_model

    def _assert_named_parameter_data_equal(
        self,
        fp8_model: nn.Module,
        bf16_model: nn.Module,
    ) -> None:
        for (fp8_name, fp8_param), (bf16_name, bf16_param) in zip(
            fp8_model.named_parameters(),
            bf16_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(fp8_name, bf16_name)
            self.assertTrue(torch.equal(fp8_param, bf16_param), fp8_name)

    def _assert_named_parameter_grads_equal(
        self,
        fp8_model: nn.Module,
        bf16_model: nn.Module,
    ) -> None:
        for (fp8_name, fp8_param), (bf16_name, bf16_param) in zip(
            fp8_model.named_parameters(),
            bf16_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(fp8_name, bf16_name)
            self.assertIsNotNone(fp8_param.grad)
            self.assertIsNotNone(bf16_param.grad)
            self.assertTrue(torch.equal(fp8_param.grad, bf16_param.grad), fp8_name)

    def _temporary_prepared_unshard_bytes(self, model: nn.Module, mesh) -> int:
        storage = model.sharded_bucket_storages[0]
        infos = list(storage.param_infos.values())
        tensors = [storage.get_local_view(info.fqn) for info in infos]

        placement = infos[0].placement
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, None)
        return prepared.buffers[0].nbytes

    @skip_if_lt_x_gpu(2)
    def test_transformer_training_matches_bf16_allgather_then_quantize(self) -> None:
        """Five training steps match the BF16-all-gather FP8-GEMM reference,
        also with microbatches without gradient sync."""
        self._skip_unless_blockwise_linear_supported()
        mesh = self._mesh()
        model_args, fp8_model, bf16_model = self._make_transformer_model_pair(
            mesh=mesh,
        )
        fp8_unshard_bytes = self._temporary_prepared_unshard_bytes(fp8_model, mesh)
        bf16_unshard_bytes = self._temporary_prepared_unshard_bytes(bf16_model, mesh)
        self.assertLess(fp8_unshard_bytes, bf16_unshard_bytes)
        self._assert_sharded_linear_state_is_bf16(fp8_model)
        self._assert_named_parameter_data_equal(fp8_model, bf16_model)

        fp8_optim = make_test_sgd(fp8_model.parameters(), lr=0.01)
        bf16_optim = make_test_sgd(bf16_model.parameters(), lr=0.01)

        for step in range(5):
            torch.manual_seed(3 + step)
            tokens = torch.randint(
                0,
                model_args.vocab_size,
                (2, model_args.max_seq_len),
                device=device_type,
            )
            targets = torch.randint(
                0,
                model_args.vocab_size,
                (2, model_args.max_seq_len),
                device=device_type,
            )
            dist.broadcast(tokens, src=0)
            dist.broadcast(targets, src=0)
            fp8_optim.zero_grad(set_to_none=True)
            bf16_optim.zero_grad(set_to_none=True)
            if step in (1, 3):
                # A microbatch without sync first, keeping the params unsharded
                # after it on step 1; its grads accumulate on the unsharded
                # params, fp8 weights included.
                for model in (fp8_model, bf16_model):
                    model.set_reshard_after_backward(step != 1)
                    model.set_requires_gradient_sync(False)
                    nn.functional.cross_entropy(
                        model(tokens).flatten(0, 1), targets.flatten()
                    ).backward()
                    model.set_requires_gradient_sync(True)

            fp8_logits = fp8_model(tokens)
            bf16_logits = bf16_model(tokens)
            self.assertTrue(
                torch.equal(fp8_logits, bf16_logits),
                f"logits diverged at step {step}",
            )

            fp8_loss = nn.functional.cross_entropy(
                fp8_logits.flatten(0, 1),
                targets.flatten(),
            )
            bf16_loss = nn.functional.cross_entropy(
                bf16_logits.flatten(0, 1),
                targets.flatten(),
            )
            fp8_loss.backward()
            bf16_loss.backward()
            self._assert_named_parameter_grads_equal(fp8_model, bf16_model)

            fp8_optim.step()
            bf16_optim.step()
            self._assert_named_parameter_data_equal(fp8_model, bf16_model)

    @skip_if_lt_x_gpu(2)
    def test_column_blocks_train_like_row_blocks(self) -> None:
        """With column blocks for the long-row w2, five training steps give
        bitwise the logits and losses of row blocks only: the layouts differ,
        the math doesn't (2 ranks sum their gradients in one order)."""

        def check(chunks) -> None:
            self.assertEqual(
                {chunk.fqn for chunk in chunks if chunk.col_start is not None},
                {"layers.0.feed_forward.w2.weight"},
            )

        self._assert_layout_trains_like_row_blocks(
            check, column_blocks_for_long_rows=True
        )

    @skip_if_lt_x_gpu(2)
    def test_tile_units_train_like_row_blocks(self) -> None:
        """With tile units, where each rank owns a run of 128x128 tiles that
        starts and ends inside tile rows, five training steps give bitwise the
        logits and losses of row blocks only."""

        def check(chunks) -> None:
            tile_chunks = [chunk for chunk in chunks if chunk.tile_start is not None]
            self.assertEqual(len(tile_chunks), len(chunks))
            self.assertTrue(
                any(chunk.tile_start % 2 or chunk.tile_end % 2 for chunk in tile_chunks)
                or len({chunk.fqn for chunk in tile_chunks}) > 1
            )

        self._assert_layout_trains_like_row_blocks(check, tile_units=True)

    def _assert_layout_trains_like_row_blocks(self, check, **layout) -> None:
        """Train a one-layer transformer whose linears' FP8 bucket takes
        ``layout`` (Fp8BucketedBlockShard options) next to one with row blocks
        only; ``check`` gets the laid-out model's FP8 chunks."""
        self._skip_unless_blockwise_linear_supported()
        mesh = self._mesh()
        torch.manual_seed(0)
        model_args, row_model = make_transformer_model(
            device=device_type,
            n_layers=1,
            vocab_size=256,
            max_seq_len=8,
            dim=256,
            n_heads=4,
        )
        row_model.to(dtype=torch.bfloat16)
        layout_model = copy.deepcopy(row_model)
        linear_weight_fqns = [
            f"{fqn}.weight"
            for fqn, module in row_model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        other_param_fqns = [
            fqn
            for fqn, _ in row_model.named_parameters()
            if fqn not in set(linear_weight_fqns)
        ]
        for model, kwargs in ((row_model, {}), (layout_model, layout)):
            convert_to_flex_shard_float8_blockwise_linear(model, use_triton=False)
            flex_shard(
                model,
                buckets=[
                    BucketSpec(
                        linear_weight_fqns,
                        placement_fn=make_fp8_bucketed_block_placement_fn(
                            weight_factory=BlockwiseFp8Weight,
                            block_size=128,
                            **kwargs,
                        ),
                        mesh=mesh,
                        reshard_after_forward=True,
                    ),
                    BucketSpec(
                        other_param_fqns,
                        placement_fn=per_param_placements,
                        mesh=mesh,
                        reshard_after_forward=True,
                    ),
                ],
            )
        infos = list(layout_model.sharded_bucket_storages[0].param_infos.values())
        check(infos[0].placement._fp8_metadata_from_infos(infos).chunks)

        row_optim = make_test_sgd(row_model.parameters(), lr=0.01)
        layout_optim = make_test_sgd(layout_model.parameters(), lr=0.01)
        for step in range(5):
            torch.manual_seed(3 + step)
            tokens = torch.randint(
                0,
                model_args.vocab_size,
                (2, model_args.max_seq_len),
                device=device_type,
            )
            targets = torch.randint(
                0,
                model_args.vocab_size,
                (2, model_args.max_seq_len),
                device=device_type,
            )
            dist.broadcast(tokens, src=0)
            dist.broadcast(targets, src=0)
            results = []
            for model, optim in ((row_model, row_optim), (layout_model, layout_optim)):
                optim.zero_grad(set_to_none=True)
                logits = model(tokens)
                loss = nn.functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten()
                )
                loss.backward()
                optim.step()
                results.append((logits.detach(), loss.detach()))
            (row_logits, row_loss), (layout_logits, layout_loss) = results
            self.assertTrue(
                torch.equal(row_logits, layout_logits),
                f"logits diverged at step {step}",
            )
            self.assertTrue(torch.equal(row_loss, layout_loss))


class TestBlockwiseFp8GroupedMM(TestCase):
    """The routed experts' FP8 grouped GEMM: gathered and per-call weights
    compute the same bytes, with or without padding, its BF16 weight gradient
    is PyTorch's, and DeepGEMM matches TorchAO's emulated backend."""

    num_groups, out_dim, in_dim = 3, 256, 384

    def _run(self, x, weight, offsets, *, aligned, preference):
        from ..blockwise_fp8.fp8_blockwise_grouped_mm import blockwise_fp8_grouped_mm

        x = x.detach().requires_grad_(True)
        if isinstance(weight, BlockwiseFp8Weight):
            weight.grad = None
        else:
            weight = weight.detach().requires_grad_(True)
        out = blockwise_fp8_grouped_mm(
            x,
            weight,
            offsets,
            num_groups=self.num_groups,
            groups_aligned=aligned,
            kernel_preference=preference,
        )
        generator = torch.Generator(device=out.device).manual_seed(1)
        out.backward(torch.randn(out.shape, generator=generator, device=out.device))
        return out.detach(), x.grad.clone(), weight.grad.clone()

    def test_gathered_and_per_call_weights_match(self) -> None:
        if device_type.type != "cuda" or not is_sm_at_least_90():
            self.skipTest("The FP8 grouped GEMM runs on SM90+ CUDA.")
        import importlib.util

        from torchao.quantization.quantize_.common import KernelPreference

        torch.manual_seed(0)
        rows = 256 * self.num_groups
        x = torch.randn(rows, self.in_dim, dtype=torch.bfloat16, device=device_type)
        weight = 0.02 * torch.randn(
            self.num_groups * self.out_dim,
            self.in_dim,
            dtype=torch.bfloat16,
            device=device_type,
        )
        fp8_data, recip_scale = _quantize_dense_weight_to_blockwise_fp8(
            weight, 128, torch.float8_e4m3fn
        )
        gathered = BlockwiseFp8Weight(fp8_data, recip_scale, 128, torch.bfloat16)
        gathered.requires_grad_(True)
        cases = {
            "aligned": (torch.tensor([256, 512, 768]), True),
            "uneven": (torch.tensor([100, 500, 768]), False),
        }
        preferences = [KernelPreference.EMULATED]
        if importlib.util.find_spec("deep_gemm") is not None:
            preferences.append(KernelPreference.AUTO)
        results = {}
        for preference in preferences:
            for name, (offsets, aligned) in cases.items():
                with self.subTest(preference=preference.name, groups=name):
                    offsets = offsets.to(device_type, torch.int32)
                    per_call = self._run(
                        x, weight, offsets, aligned=aligned, preference=preference
                    )
                    pre = self._run(
                        x, gathered, offsets, aligned=aligned, preference=preference
                    )
                    for a, b in zip(per_call, pre, strict=True):
                        self.assertTrue(torch.equal(a, b))
                    if aligned:
                        padded = self._run(
                            x, gathered, offsets, aligned=False, preference=preference
                        )
                        for a, b in zip(pre, padded, strict=True):
                            self.assertTrue(torch.equal(a, b))
                    # The BF16 weight gradient is PyTorch's grouped GEMM's.
                    leaf = weight.detach().requires_grad_(True)
                    reference = torch._grouped_mm(
                        x,
                        leaf.view(self.num_groups, self.out_dim, self.in_dim).mT,
                        offs=offsets,
                        out_dtype=torch.bfloat16,
                    )
                    generator = torch.Generator(device=x.device).manual_seed(1)
                    reference.backward(
                        torch.randn(
                            reference.shape, generator=generator, device=x.device
                        )
                    )
                    self.assertTrue(torch.equal(pre[2], leaf.grad))
                    results[preference, name] = pre
        if KernelPreference.AUTO in preferences:
            for name in cases:
                deepgemm = results[KernelPreference.AUTO, name]
                emulated = results[KernelPreference.EMULATED, name]
                for got, expected in zip(deepgemm[:2], emulated[:2], strict=True):
                    error = (
                        got.float() - expected.float()
                    ).norm() / expected.float().norm()
                    self.assertLess(error.item(), 1e-2)


if __name__ == "__main__":
    run_tests()
