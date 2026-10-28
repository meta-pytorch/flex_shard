# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.func import functional_call
from torch.testing._internal.common_utils import run_tests, TestCase
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    _mxfp8_cutedsl_kernels_available,
)
from torchao.prototype.mx_formats.utils import to_blocked

from .. import BucketSpec, flex_shard
from ..custom_placements.mxfp8_bucketed_block_shard import (
    _blocked_scale_shape,
    _quantize_dense_weight_to_mxfp8_32x32,
    make_mxfp8_bucketed_block_placement_fn,
    Mxfp8BucketedBlockShard,
)
from ..custom_placements.shard import per_param_placements
from .common import make_test_sgd, make_transformer_model
from ..blockwise_mxfp8 import (
    BlockwiseMxfp8Weight,
    convert_to_flex_shard_mxfp8_blockwise_linear,
    FlexShardMxfp8BlockwiseLinear,
)

from .blockwise_quant_reference import DenseAllGatherReferenceMixin


device_type = torch.device(get_devtype())


class _MxDenseAllGatherReferencePlacement(
    DenseAllGatherReferenceMixin,
    Mxfp8BucketedBlockShard,
):
    """All-gather dense shards using the MXFP8 placement's shard layout."""


def _skip_unless_mxfp8_kernels(test: TestCase) -> None:
    if not _mxfp8_cutedsl_kernels_available:
        test.skipTest("MXFP8 CuTeDSL kernels require CUDA SM 10.x and CUDA 12.8+")


def _sqnr_db(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference = reference.float()
    noise = (reference - actual.float()).norm()
    return (20 * torch.log10(reference.norm() / noise)).item()


def _zeros(shape: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.uint8).view(dtype)


def _quantized_weight(weight: torch.Tensor) -> BlockwiseMxfp8Weight:
    fp8_data, fwd_scale, dgrad_scale = _quantize_dense_weight_to_mxfp8_32x32(
        weight.detach(),
        "rceil",
    )
    return BlockwiseMxfp8Weight(
        fp8_data,
        fwd_scale,
        dgrad_scale,
        weight.dtype,
        requires_grad=True,
    )


class BlockwiseMxfp8WeightTest(TestCase):
    def test_weight_protocol(self) -> None:
        out_dim, in_dim = 256, 128
        weight = BlockwiseMxfp8Weight(
            _zeros((out_dim, in_dim), torch.float8_e4m3fn),
            _zeros(_blocked_scale_shape(out_dim, in_dim), torch.float8_e8m0fnu),
            _zeros(_blocked_scale_shape(in_dim, out_dim), torch.float8_e8m0fnu),
            torch.bfloat16,
            requires_grad=True,
        )
        self.assertEqual(weight.shape, (out_dim, in_dim))
        self.assertEqual(weight.dtype, torch.bfloat16)
        self.assertTrue(weight.requires_grad)

        names, metadata = weight.__tensor_flatten__()
        self.assertEqual(names, ["_data", "_fwd_scale", "_dgrad_scale"])
        rebuilt = BlockwiseMxfp8Weight.__tensor_unflatten__(
            {name: getattr(weight, name) for name in names},
            metadata,
            weight.shape,
            weight.stride(),
        )
        self.assertIsInstance(rebuilt, BlockwiseMxfp8Weight)
        self.assertTrue(rebuilt.requires_grad)
        self.assertIs(rebuilt.fwd_scale, weight.fwd_scale)

        detached = weight.detach()
        self.assertIsInstance(detached, BlockwiseMxfp8Weight)
        self.assertFalse(detached.requires_grad)
        self.assertEqual(
            detached.dgrad_scale.data_ptr(),
            weight.dgrad_scale.data_ptr(),
        )
        with self.assertRaises(NotImplementedError):
            weight + 1

    def test_stacked_weight_scales_cover_flattened_matrix(self) -> None:
        num_linears, out_dim, in_dim = 2, 64, 64
        matrix_rows = num_linears * out_dim
        fp8_data = (
            torch.arange(matrix_rows * in_dim, dtype=torch.float32)
            .remainder(7)
            .reshape(num_linears, out_dim, in_dim)
            .to(torch.float8_e4m3fn)
        )
        # One e8m0 exponent per 1x32 block of the [128, 64] matrix.
        exponents = torch.arange(matrix_rows * in_dim // 32).remainder(9) + 123
        exponents = exponents.to(torch.uint8).reshape(matrix_rows, in_dim // 32)
        weight = BlockwiseMxfp8Weight(
            fp8_data,
            to_blocked(exponents)
            .view(torch.float8_e8m0fnu)
            .view(_blocked_scale_shape(matrix_rows, in_dim)),
            _zeros(_blocked_scale_shape(in_dim, matrix_rows), torch.float8_e8m0fnu),
        )

        self.assertEqual(weight.shape, (num_linears, out_dim, in_dim))
        scale = torch.exp2(exponents.float() - 127).repeat_interleave(32, dim=1)
        self.assertEqual(
            weight.dequantize(torch.float32),
            (fp8_data.float().view(matrix_rows, in_dim) * scale).view(fp8_data.shape),
        )

    def test_rejects_mismatched_scale_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "dgrad_scale shape"):
            BlockwiseMxfp8Weight(
                _zeros((256, 128), torch.float8_e4m3fn),
                _zeros(_blocked_scale_shape(256, 128), torch.float8_e8m0fnu),
                _zeros(_blocked_scale_shape(256, 128), torch.float8_e8m0fnu),
            )


class FlexShardMxfp8LinearTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        _skip_unless_mxfp8_kernels(self)

    def _check_linear(self, out_dim: int, in_dim: int, bias: bool) -> None:
        torch.manual_seed(0)
        dense = nn.Linear(
            in_dim,
            out_dim,
            bias=bias,
            device="cuda",
            dtype=torch.bfloat16,
        )
        module = nn.Sequential(dense)
        convert_to_flex_shard_mxfp8_blockwise_linear(module)
        linear = module[0]
        self.assertIsInstance(linear, FlexShardMxfp8BlockwiseLinear)
        weight = _quantized_weight(dense.weight)
        self.assertGreater(_sqnr_db(dense.weight, weight.dequantize()), 25)
        # FlexShard installs the gathered weight as a plain tensor attribute.
        del linear._parameters["weight"]
        linear.weight = weight

        x = torch.randn(
            2,
            200,
            in_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        grad_output = torch.randn(2, 200, out_dim, device="cuda", dtype=torch.bfloat16)
        out = linear(x)
        inputs = [x, weight] + ([linear.bias] if bias else [])
        grads = torch.autograd.grad(out, inputs, grad_output)

        # High-precision reference against the dequantized weight isolates the
        # activation and gradient quantization error.
        weight_ref = weight.dequantize(torch.float32).requires_grad_()
        x_ref = x.detach().float().requires_grad_()
        out_ref = x_ref @ weight_ref.t()
        if bias:
            out_ref = out_ref + linear.bias.detach().float()
        grad_x_ref, grad_weight_ref = torch.autograd.grad(
            out_ref,
            [x_ref, weight_ref],
            grad_output.float(),
        )
        self.assertEqual(out.shape, out_ref.shape)
        self.assertEqual(grads[1].shape, (out_dim, in_dim))
        self.assertTrue(grads[1].is_contiguous())
        self.assertGreater(_sqnr_db(out_ref, out), 25)
        self.assertGreater(_sqnr_db(grad_x_ref, grads[0]), 25)
        self.assertGreater(_sqnr_db(grad_weight_ref, grads[1]), 25)
        if bias:
            self.assertEqual(grads[2], grad_output.sum(dim=(0, 1)))

    def test_linear_matches_high_precision_reference(self) -> None:
        self._check_linear(256, 384, bias=False)

    def test_linear_pads_unaligned_dims(self) -> None:
        self._check_linear(200, 320, bias=True)

    def test_stacked_weight_matches_flattened_matrix(self) -> None:
        """A ``[2, F, D]`` weight runs the ``[2F, D]`` GEMM, including the
        128-row scale tile that spans both stacked matrices."""
        num_linears, out_dim, in_dim = 2, 192, 256
        torch.manual_seed(0)
        fp8_data, fwd_scale, dgrad_scale = _quantize_dense_weight_to_mxfp8_32x32(
            torch.randn(
                num_linears * out_dim,
                in_dim,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            "rceil",
        )
        linear = FlexShardMxfp8BlockwiseLinear(
            in_dim,
            num_linears * out_dim,
            bias=False,
            device="meta",
        )
        x = torch.randn(
            2,
            64,
            in_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        grad_output = torch.randn(
            2,
            64,
            num_linears * out_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )

        def run(weight_shape: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
            weight = BlockwiseMxfp8Weight(
                fp8_data.view(weight_shape),
                fwd_scale,
                dgrad_scale,
                torch.bfloat16,
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

    def test_weight_grad_is_emitted_in_grad_dtype(self) -> None:
        """An FP32 grad_dtype gets the WGRAD GEMM's FP32 result, not the BF16
        result upcast."""
        torch.manual_seed(0)
        dense = nn.Linear(384, 256, bias=False, device="cuda", dtype=torch.bfloat16)
        module = nn.Sequential(dense)
        convert_to_flex_shard_mxfp8_blockwise_linear(module)
        linear = module[0]
        del linear._parameters["weight"]
        x = torch.randn(400, 384, device="cuda", dtype=torch.bfloat16)
        grad_output = torch.randn(400, 256, device="cuda", dtype=torch.bfloat16)

        grad_weights = {}
        for grad_dtype in (torch.bfloat16, torch.float32):
            weight = _quantized_weight(dense.weight)
            weight.grad_dtype = grad_dtype
            linear.weight = weight
            (grad_weights[grad_dtype],) = torch.autograd.grad(
                linear(x), [weight], grad_output
            )

        bf16_grad = grad_weights[torch.bfloat16]
        fp32_grad = grad_weights[torch.float32]
        self.assertEqual(bf16_grad.dtype, torch.bfloat16)
        self.assertEqual(fp32_grad.dtype, torch.float32)
        # One GEMM: the BF16 gradient is the FP32 one rounded.
        self.assertTrue(torch.equal(fp32_grad.to(torch.bfloat16), bf16_grad))
        self.assertFalse(torch.equal(fp32_grad, bf16_grad.float()))


class BlockwiseMxfp8FlexShardTrainingTest(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _make_transformer_model_pair(self, mesh):
        torch.manual_seed(0)
        model_args, mx_model = make_transformer_model(
            device=device_type,
            n_layers=1,
            vocab_size=256,
            max_seq_len=8,
            dim=256,
            n_heads=4,
        )
        mx_model.to(dtype=torch.bfloat16)
        reference_model = copy.deepcopy(mx_model)
        linear_weight_fqns = [
            f"{fqn}.weight"
            for fqn, module in mx_model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        other_param_fqns = [
            fqn
            for fqn, _ in mx_model.named_parameters()
            if fqn not in set(linear_weight_fqns)
        ]
        convert_to_flex_shard_mxfp8_blockwise_linear(mx_model)
        convert_to_flex_shard_mxfp8_blockwise_linear(reference_model)

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

        def reference_placement_fn(named_params, mesh):
            placement = _MxDenseAllGatherReferencePlacement(
                world_size=mesh.size(),
                weight_factory=BlockwiseMxfp8Weight,
            )
            return {fqn: (placement,) for fqn, _ in named_params}

        flex_shard(reference_model, buckets=make_buckets(reference_placement_fn))
        flex_shard(
            mx_model,
            buckets=make_buckets(
                make_mxfp8_bucketed_block_placement_fn(
                    weight_factory=BlockwiseMxfp8Weight,
                )
            ),
        )
        return model_args, mx_model, reference_model

    @staticmethod
    def _prepared_unshard_nbytes(model: nn.Module, mesh) -> int:
        storage = model.sharded_bucket_storages[0]
        infos = list(storage.param_infos.values())
        tensors = [storage.get_local_view(info.fqn) for info in infos]
        prepared = infos[0].placement.prepare_unshard_bucket(tensors, infos, mesh, None)
        return prepared.buffers[0].nbytes

    def _assert_params_equal(self, model: nn.Module, reference: nn.Module) -> None:
        for (name, param), (ref_name, ref_param) in zip(
            model.named_parameters(),
            reference.named_parameters(),
            strict=True,
        ):
            self.assertEqual(name, ref_name)
            self.assertTrue(torch.equal(param, ref_param), name)

    def _assert_grads_equal(self, model: nn.Module, reference: nn.Module) -> None:
        for (name, param), (_, ref_param) in zip(
            model.named_parameters(),
            reference.named_parameters(),
            strict=True,
        ):
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.equal(param.grad, ref_param.grad), name)

    @skip_if_lt_x_gpu(2)
    def test_transformer_training_matches_bf16_allgather_then_quantize(self) -> None:
        """Five training steps match the BF16-all-gather MXFP8-GEMM reference."""
        _skip_unless_mxfp8_kernels(self)
        mesh = init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )
        model_args, mx_model, reference_model = self._make_transformer_model_pair(mesh)
        self.assertLess(
            self._prepared_unshard_nbytes(mx_model, mesh),
            self._prepared_unshard_nbytes(reference_model, mesh),
        )
        self._assert_params_equal(mx_model, reference_model)

        mx_optim = make_test_sgd(mx_model.parameters(), lr=0.01)
        reference_optim = make_test_sgd(reference_model.parameters(), lr=0.01)
        for step in range(5):
            torch.manual_seed(3 + step)
            tokens, targets = (
                torch.randint(
                    0,
                    model_args.vocab_size,
                    (2, model_args.max_seq_len),
                    device=device_type,
                )
                for _ in range(2)
            )
            dist.broadcast(tokens, src=0)
            dist.broadcast(targets, src=0)
            mx_optim.zero_grad(set_to_none=True)
            reference_optim.zero_grad(set_to_none=True)

            mx_logits = mx_model(tokens)
            reference_logits = reference_model(tokens)
            self.assertTrue(
                torch.equal(mx_logits, reference_logits),
                f"logits diverged at step {step}",
            )
            for logits in (mx_logits, reference_logits):
                nn.functional.cross_entropy(
                    logits.flatten(0, 1),
                    targets.flatten(),
                ).backward()
            self._assert_grads_equal(mx_model, reference_model)

            mx_optim.step()
            reference_optim.step()
            self._assert_params_equal(mx_model, reference_model)


if __name__ == "__main__":
    run_tests()
