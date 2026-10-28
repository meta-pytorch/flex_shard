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
    _quantize_dense_weight_to_blockwise_fp8,
    BlockwiseFp8WeightFactory,
    Fp8BucketedBlockShard,
    make_fp8_bucketed_block_placement_fn,
)
from ..custom_placements.shard import per_param_placements
from ..flex_shard.bucket_storage import PlacementFn
from ..flex_shard.placement_contract import Placement
from .common import make_test_sgd, make_transformer_model

from .blockwise_quant_reference import DenseAllGatherReferenceMixin


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


class _DenseAllGatherReferencePlacement(
    DenseAllGatherReferenceMixin,
    Fp8BucketedBlockShard,
):
    """All-gather dense shards using the FP8 placement's shard layout."""


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


if __name__ == "__main__":
    run_tests()
