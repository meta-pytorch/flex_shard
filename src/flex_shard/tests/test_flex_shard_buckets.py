#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for FlexShard BucketSpec and bucket validation.

Usage after installing the public package with its test dependencies:
    # Single-process tests (no GPU/NCCL required):
    python -m pytest --pyargs flex_shard.tests.test_flex_shard_buckets \
      -v -k "not Distributed"

    # Distributed correctness tests:
    python -m pytest --pyargs flex_shard.tests.test_flex_shard_buckets \
      -q -k Distributed
"""

import copy
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)

from .. import BucketSpec, flex_shard, is_flex_shard_param, Placement
from ..custom_placements.block_shard import BlockShard
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..custom_placements.owned import make_bucketed_owned_full_param_segments
from ..custom_placements.shard import per_param_placements, Shard
from ..flex_shard.bucket_comm import prepare_reduce_grad
from ..flex_shard.bucket_storage import (
    _assign_params_to_buckets,
    ParamInfo,
    ShardedBucketStorage,
)
from ..flex_shard.flex_shard import (
    _materialize_bucket_storages,
    PreparedFlexShardInputs,
)
from .common import (
    expected_shard,
    make_test_sgd,
    make_transformer_model,
    single_rank_cpu_mesh,
    single_rank_cuda_mesh,
    transformer_bucket_specs,
    transformer_inputs,
)


device_type = torch.device(get_devtype())


class _IncompletePlacement(Placement):
    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IncompletePlacement)

    def __hash__(self) -> int:
        return hash(type(self))


# Shard mesh tests
# ---------------------------------------------------------------------------


class TestFlexShardMesh(TestCase):
    """Test FlexShard shard mesh validation."""

    def test_rejects_multi_dim_mesh(self):
        from ..flex_shard.utils import _validate_flex_shard_mesh

        with single_rank_cuda_mesh():
            mesh = init_device_mesh(
                "cuda",
                (1, 1),
                mesh_dim_names=("fsdp", "tp"),
            )
            with self.assertRaisesRegex(ValueError, "1D DeviceMesh"):
                _validate_flex_shard_mesh(mesh)


# ---------------------------------------------------------------------------
# Bucket assignment tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketAssignment(TestCase):
    """Test _assign_params_to_buckets."""

    def test_assigns_params_to_correct_bucket(self):
        """Params match the right bucket by fnmatch."""
        from ..flex_shard.bucket_storage import _assign_params_to_buckets

        fqns = ["attn.weight", "attn.bias", "ffn.weight", "ffn.bias"]
        with single_rank_cpu_mesh() as mesh:
            buckets = [
                BucketSpec(
                    ["attn.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                ),
                BucketSpec(
                    ["ffn.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                ),
            ]
            result = _assign_params_to_buckets(fqns, buckets)

            self.assertEqual(result[0], ["attn.weight", "attn.bias"])
            self.assertEqual(result[1], ["ffn.weight", "ffn.bias"])

    def test_rejects_orphan_params(self):
        """Params matching zero buckets raise ValueError."""
        from ..flex_shard.bucket_storage import _assign_params_to_buckets

        fqns = ["attn.weight", "norm.weight"]
        with single_rank_cpu_mesh() as mesh:
            buckets = [
                BucketSpec(
                    ["attn.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ]
            with self.assertRaises(ValueError, msg="not covered by any bucket"):
                _assign_params_to_buckets(fqns, buckets)

    def test_rejects_overlapping_params(self):
        """Params matching multiple buckets raise ValueError."""
        from ..flex_shard.bucket_storage import _assign_params_to_buckets

        fqns = ["attn.weight"]
        with single_rank_cpu_mesh() as mesh:
            buckets = [
                BucketSpec(
                    ["attn.*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                ),
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                ),
            ]
            with self.assertRaises(ValueError, msg="matched multiple buckets"):
                _assign_params_to_buckets(fqns, buckets)


# ---------------------------------------------------------------------------
# Placement consistency tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketPlacementValidation(TestCase):
    """Test explicit placement and bucket validation."""

    @staticmethod
    def _named_params(
        dtypes: dict[str, torch.dtype] | None = None,
    ) -> list[tuple[str, nn.Parameter]]:
        dtypes = dtypes or {}
        return [
            (
                fqn,
                nn.Parameter(torch.empty(2, 2, dtype=dtypes.get(fqn, torch.float32))),
            )
            for fqn in ("a.weight", "b.weight")
        ]

    def test_rejects_missing_or_extra_placements(self):
        """Placement validation requires exact managed parameter coverage."""
        from ..flex_shard.utils import _validate_placements

        with single_rank_cpu_mesh():
            named_params = self._named_params()
            with self.assertRaisesRegex(ValueError, "missing placements"):
                _validate_placements(
                    {"a.weight": (Shard(0),)},
                    named_params,
                )

            with self.assertRaisesRegex(ValueError, "unexpected placements"):
                _validate_placements(
                    {
                        "a.weight": (Shard(0),),
                        "b.weight": (Shard(0),),
                        "extra.weight": (Shard(0),),
                    },
                    named_params,
                )

    def test_rejects_non_placement_object(self):
        """Placement validation requires Placement instances."""
        from ..flex_shard.utils import _validate_placements

        with single_rank_cpu_mesh():
            named_params = self._named_params()
            with self.assertRaisesRegex(TypeError, "Placement instances"):
                _validate_placements(
                    {
                        "a.weight": (object(),),
                        "b.weight": (object(),),
                    },
                    named_params,
                )

    def test_rejects_incomplete_placement_contract(self):
        """Placement subclasses must implement the storage layout contract."""
        with single_rank_cpu_mesh() as mesh:
            named_params = self._named_params()
            with self.assertRaisesRegex(TypeError, "storage layout contract"):
                ShardedBucketStorage.create_param_infos(
                    named_params,
                    mesh,
                    {
                        "a.weight": (_IncompletePlacement(),),
                        "b.weight": (_IncompletePlacement(),),
                    },
                )

    def test_rejects_shard_dim_out_of_range(self):
        """Placement layout validation happens during bucket storage planning."""

        with single_rank_cpu_mesh() as mesh:
            with self.assertRaisesRegex(ValueError, "invalid for parameter"):
                ShardedBucketStorage.create_param_infos(
                    [("scalar", nn.Parameter(torch.empty(())))],
                    mesh,
                    {"scalar": (Shard(0),)},
                )

    def test_rejects_mixed_dtypes(self):
        """Parameters in one bucket must share the same storage dtype."""
        from ..flex_shard.utils import _validate_bucket_uniform_dtype_and_placement

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(0),),
        }
        with single_rank_cpu_mesh() as mesh:
            buckets = [
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ]
            with self.assertRaisesRegex(ValueError, "mixed parameter dtypes"):
                _validate_bucket_uniform_dtype_and_placement(
                    assignments,
                    placements,
                    buckets,
                    self._named_params({"b.weight": torch.bfloat16}),
                )

    def test_rejects_unsupported_mixed_placements_in_one_bucket(self):
        """Buckets reject placements that do not share one bucket placement."""
        from ..flex_shard.utils import _validate_bucket_uniform_dtype_and_placement

        assignments = [["a.weight", "b.weight"]]
        placements = {
            "a.weight": (Shard(0),),
            "b.weight": (Shard(1),),
        }
        with single_rank_cpu_mesh() as mesh:
            buckets = [
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ]
            with self.assertRaisesRegex(ValueError, "mixed placements"):
                _validate_bucket_uniform_dtype_and_placement(
                    assignments,
                    placements,
                    buckets,
                    self._named_params(),
                )

    def test_flex_shard_rejects_unsupported_mixed_placements_before_materializing(self):
        """Invalid bucket placement config should not partially shard the module."""

        class TwoParamModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Parameter(torch.empty(2, 2))
                self.b = nn.Parameter(torch.empty(2, 2))

        def mixed_placements(named_params, mesh):
            return {
                "a": (Shard(0),),
                "b": (Shard(1),),
            }

        with single_rank_cuda_mesh() as mesh:
            model = TwoParamModule()
            with self.assertRaisesRegex(ValueError, "mixed placements"):
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=mixed_placements,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )
            self.assertFalse(hasattr(model, "_sharded_bucket_storages"))

    def test_flex_shard_rejects_shard1_before_materializing(self):
        class OneParamModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty(2, 2))

        def shard1_placement(named_params, mesh):
            del mesh
            return {fqn: (Shard(1),) for fqn, _ in named_params}

        with single_rank_cuda_mesh() as mesh:
            model = OneParamModule()
            with self.assertRaisesRegex(NotImplementedError, "only Shard\\(0\\)"):
                flex_shard(
                    model,
                    buckets=[
                        BucketSpec(
                            ["*"],
                            placement_fn=shard1_placement,
                            mesh=mesh,
                            reshard_after_forward=False,
                        )
                    ],
                )
            self.assertFalse(hasattr(model, "_sharded_bucket_storages"))


# ---------------------------------------------------------------------------
# Bucket storage layout tests (single-process, no NCCL)
# ---------------------------------------------------------------------------


class TestBucketStorageLayout(FSDPTestMultiThread):
    """Test ParamInfo and ShardedBucketStorage layout for bucket materialization."""

    @property
    def world_size(self) -> int:
        return 2

    def test_materialized_params_are_views_into_bucket_storage(self):
        mesh = init_device_mesh("cpu", (self.world_size,), mesh_dim_names=("fsdp",))
        args, model = make_transformer_model()
        named_params = list(model.named_parameters())
        placements = {fqn: (Shard(0),) for fqn, _ in named_params}
        buckets = transformer_bucket_specs(
            args.n_layers,
            mesh,
            reshard_after_forward=False,
        )
        assignments = _assign_params_to_buckets(
            [fqn for fqn, _ in named_params],
            buckets,
        )

        inputs = PreparedFlexShardInputs(
            named_params=named_params,
            device=torch.device("cpu"),
            param_placements=placements,
            bucket_assignments=assignments,
        )
        bucket_storages, fqn_to_bucket_spec = _materialize_bucket_storages(
            model,
            inputs,
            buckets,
        )

        self.assertEqual(len(bucket_storages), len(buckets))
        self.assertIs(fqn_to_bucket_spec["tok_embeddings.weight"], buckets[0])
        self.assertIs(fqn_to_bucket_spec["output.weight"], buckets[-1])

        current_params = dict(model.named_parameters())
        for bucket_storage in bucket_storages:
            storage_ptr = bucket_storage.byte_storage.untyped_storage().data_ptr()
            for fqn, info in bucket_storage.param_infos.items():
                param = current_params[fqn]
                local_view = bucket_storage.get_local_view(fqn)

                self.assertEqual(param.shape, info.local_shape)
                self.assertEqual(
                    param.untyped_storage().data_ptr(),
                    storage_ptr,
                )
                self.assertEqual(
                    local_view.untyped_storage().data_ptr(),
                    storage_ptr,
                )
                self.assertEqual(param, local_view)
                self.assertTrue(is_flex_shard_param(param))


class TestMixedBucketComposition(TestCase):
    def test_mixed_payload_dtypes_use_one_byte_all_gather(self) -> None:
        with single_rank_cpu_mesh() as mesh:
            mixed = MixedBucketPlacement({})
            placements = [
                mixed.shard0,
                mixed.block_shard(blocks_per_rank=(1,)),
            ]
            local_params = [
                torch.arange(3, dtype=torch.float32),
                torch.arange(8, dtype=torch.float64).view(2, 4),
            ]
            infos = [
                ParamInfo(
                    fqn=f"param_{index}",
                    global_shape=param.shape,
                    global_stride=param.stride(),
                    dtype=param.dtype,
                    requires_grad=False,
                    placements=(placement,),
                    local_shape=param.shape,
                    local_numel=param.numel(),
                    storage_nbytes=param.nbytes,
                    global_numel=param.numel(),
                )
                for index, (param, placement) in enumerate(
                    zip(local_params, placements, strict=True)
                )
            ]

            prepared = mixed.prepare_unshard_bucket(
                local_params,
                infos,
                mesh,
                None,
            )

            self.assertEqual(prepared.buffers[0].dtype, torch.uint8)
            self.assertEqual(
                prepared.buffers[0].numel(),
                sum(param.nbytes for param in local_params),
            )
            with mock.patch.object(
                dist,
                "all_gather_into_tensor",
                wraps=dist.all_gather_into_tensor,
            ) as all_gather:
                mixed.run_prepared_unshard(prepared)
            self.assertEqual(all_gather.call_count, 1)

            result = mixed.finish_prepared_unshard(prepared)

            self.assertEqual(result.full_params, local_params)

    def test_single_rank_uint8_group_buffer_is_float32_aligned(self) -> None:
        with single_rank_cpu_mesh() as mesh:
            mixed = MixedBucketPlacement({})
            placements = [
                mixed.shard0,
                mixed.block_shard(blocks_per_rank=(1,)),
            ]
            scales = torch.tensor([1.0, 2.0], dtype=torch.float32)
            local_params = [
                torch.arange(3, dtype=torch.bfloat16),
                scales.view(torch.uint8).view(2, 4),
            ]
            infos = [
                ParamInfo(
                    fqn=f"param_{index}",
                    global_shape=param.shape,
                    global_stride=param.stride(),
                    dtype=param.dtype,
                    requires_grad=False,
                    placements=(placement,),
                    local_shape=param.shape,
                    local_numel=param.numel(),
                    storage_nbytes=param.nbytes,
                    global_numel=param.numel(),
                )
                for index, (param, placement) in enumerate(
                    zip(local_params, placements, strict=True)
                )
            ]

            prepared = mixed.prepare_unshard_bucket(
                local_params,
                infos,
                mesh,
                None,
            )
            mixed.run_prepared_unshard(prepared)

            result = mixed.finish_prepared_unshard(prepared)

            self.assertEqual(result.full_params, local_params)
            uint8_groups = [
                buffer
                for buffer in result.finish_buffers
                if buffer.dtype == torch.uint8
            ]
            self.assertEqual(len(uint8_groups), 1)
            self.assertEqual(
                uint8_groups[0].view(torch.float32),
                scales,
            )


# ---------------------------------------------------------------------------
# Distributed per-bucket ShardedBucketStorage tests
# ---------------------------------------------------------------------------


class TestDistributedBuckets(FSDPTest):
    """Multi-process correctness tests for bucketed FlexShard.

    Run with:
        python -m pytest --pyargs flex_shard.tests.test_flex_shard_buckets \
          -q -k Distributed
    """

    @property
    def world_size(self) -> int:
        return 2

    def _init_mesh(self):
        return init_device_mesh(
            device_type.type,
            (self.world_size,),
            mesh_dim_names=("fsdp",),
        )

    @skip_if_lt_x_gpu(2)
    def test_mixed_payload_dtypes_reconstruct_rank_distinct_values(self) -> None:
        mesh = self._init_mesh()
        device = torch.device(device_type.type, torch.cuda.current_device())
        mixed = MixedBucketPlacement({})
        placements = [
            mixed.shard0,
            mixed.block_shard(blocks_per_rank=(1, 1)),
        ]
        local_params = [
            torch.arange(3, dtype=torch.float32, device=device) + 10 * self.rank,
            (
                torch.arange(4, dtype=torch.float64, device=device) + 100 * self.rank
            ).view(1, 4),
        ]
        expected_params = [
            torch.cat(
                [
                    torch.arange(3, dtype=torch.float32, device=device) + 10 * rank
                    for rank in range(self.world_size)
                ]
            ),
            torch.cat(
                [
                    (
                        torch.arange(4, dtype=torch.float64, device=device) + 100 * rank
                    ).view(1, 4)
                    for rank in range(self.world_size)
                ]
            ),
        ]
        infos = [
            ParamInfo(
                fqn=f"param_{index}",
                global_shape=expected.shape,
                global_stride=expected.stride(),
                dtype=expected.dtype,
                requires_grad=False,
                placements=(placement,),
                local_shape=local.shape,
                local_numel=local.numel(),
                storage_nbytes=local.nbytes,
                global_numel=expected.numel(),
            )
            for index, (local, expected, placement) in enumerate(
                zip(local_params, expected_params, placements, strict=True)
            )
        ]

        prepared = mixed.prepare_unshard_bucket(
            local_params,
            infos,
            mesh,
            None,
        )
        self.assertEqual(prepared.buffers[0].dtype, torch.uint8)

        mixed.run_prepared_unshard(prepared)
        result = mixed.finish_prepared_unshard(prepared)

        self.assertEqual(result.full_params, expected_params)

    @skip_if_lt_x_gpu(2)
    def test_mixed_bucket_combines_shard_block_and_bucketed_owned(self):
        mesh = self._init_mesh()
        device = torch.device(device_type.type, torch.cuda.current_device())

        class MixedModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.head = nn.Parameter(torch.arange(6.0, device=device).view(2, 3))
                self.wo = nn.Parameter(torch.arange(6.0, device=device).view(3, 2))
                self.norm = nn.Parameter(torch.arange(3.0, device=device))
                self.dense = nn.Parameter(torch.arange(4.0, device=device).view(2, 2))

        model = MixedModule()
        named_params = list(model.named_parameters())
        originals = {fqn: param.detach().clone() for fqn, param in named_params}
        mixed = MixedBucketPlacement(
            make_bucketed_owned_full_param_segments(
                [("head", model.head), ("dense", model.dense)],
                self.world_size,
            )
        )
        block = mixed.block_shard(blocks_per_rank=(1, 0))
        wo_block = mixed.block_shard(blocks_per_rank=(1, 0), dim=1)
        plain_block = BlockShard(blocks_per_rank=(1, 0))
        other_block = mixed.block_shard(blocks_per_rank=(0, 1))
        self.assertEqual(block, plain_block)
        self.assertEqual(plain_block, block)
        self.assertEqual(hash(block), hash(plain_block))
        self.assertNotEqual(block, other_block)
        self.assertNotEqual(block, wo_block)
        self.assertNotEqual(block, mixed.shard0)
        self.assertIs(block.bucket_compatibility_key(), mixed)
        self.assertIs(wo_block.bucket_compatibility_key(), mixed)
        self.assertIs(mixed.shard0.bucket_compatibility_key(), mixed)
        placements = {
            "head": (mixed.bucketed_owned,),
            "wo": (wo_block,),
            "norm": (mixed.shard0,),
            "dense": (mixed.bucketed_owned,),
        }
        bucket_spec = BucketSpec(
            ["*"],
            placement_fn=lambda named_params, mesh: placements,
            mesh=mesh,
            reshard_after_forward=False,
        )
        storage = ShardedBucketStorage.from_bucket(
            model,
            named_params,
            placements,
            mesh,
            device,
            bucket_spec,
        )
        infos = [storage.param_infos[fqn] for fqn, _ in named_params]
        local_params = [storage.get_local_view(fqn) for fqn, _ in named_params]

        expected_head = originals["head"] if self.rank == 0 else originals["head"][:0]
        self.assertEqual(storage.get_local_view("head"), expected_head)
        expected_wo = originals["wo"] if self.rank == 0 else originals["wo"][:, :0]
        self.assertEqual(storage.get_local_view("wo"), expected_wo)
        self.assertEqual(
            storage.param_infos["head"].storage_nbytes,
            expected_head.numel() * expected_head.element_size(),
        )
        self.assertEqual(
            storage.param_infos["wo"].storage_nbytes,
            expected_wo.numel() * expected_wo.element_size(),
        )
        self.assertEqual(
            storage.param_infos["norm"].storage_nbytes,
            2 * originals["norm"].element_size(),
        )

        prepared_unshard = mixed.prepare_unshard_bucket(
            local_params,
            infos,
            mesh,
            None,
        )
        self.assertEqual(prepared_unshard.placement_state.row_numel, 14)
        self.assertEqual(prepared_unshard.buffers[0].dtype, torch.float32)
        with mock.patch.object(
            dist,
            "all_gather_into_tensor",
            wraps=dist.all_gather_into_tensor,
        ) as all_gather:
            mixed.run_prepared_unshard(prepared_unshard)
        self.assertEqual(all_gather.call_count, 1)
        unsharded = mixed.finish_prepared_unshard(prepared_unshard)
        self.assertEqual(
            unsharded.full_params,
            [originals[fqn] for fqn, _ in named_params],
        )

        prepared_reduce = prepare_reduce_grad(
            [originals[fqn] for fqn, _ in named_params],
            infos,
            mesh,
            None,
        ).prepared
        self.assertEqual(prepared_reduce.placement_state.row_numel, 14)
        shard_reduce_group = next(
            group
            for group in prepared_reduce.placement_state.groups
            if isinstance(group.prepared.placement, Shard)
        )
        self.assertGreater(shard_reduce_group.offset, 0)

        with mock.patch.object(
            dist,
            "reduce_scatter_tensor",
            wraps=dist.reduce_scatter_tensor,
        ) as reduce_scatter:
            reduced = mixed.reduce_prepared_grad(prepared_reduce)
        self.assertEqual(reduce_scatter.call_count, 1)
        for local_grad, (fqn, _) in zip(
            reduced.sharded_grads,
            named_params,
            strict=True,
        ):
            self.assertEqual(
                local_grad,
                storage.get_local_view(fqn),
                msg=fqn,
            )

    def _flex_shard(self, model, mesh, **kwargs):
        kwargs.setdefault(
            "buckets",
            [
                BucketSpec(
                    ["*"],
                    placement_fn=per_param_placements,
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ],
        )
        return flex_shard(
            model,
            **kwargs,
        )

    @skip_if_lt_x_gpu(2)
    def test_multi_bucket_forward_correct(self):
        """Model with explicit buckets produces correct forward output."""
        mesh = self._init_mesh()
        args, model = make_transformer_model(device=device_type.type)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        x = transformer_inputs(args, device=device_type.type)
        dist.broadcast(x, src=0)
        ref_output = model(x).clone()

        self._flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(
                args.n_layers,
                mesh,
                reshard_after_forward=False,
            ),
        )

        self.assertEqual(len(model.sharded_bucket_storages), 5)
        output = model(x)
        self.assertEqual(output, ref_output)

    @skip_if_lt_x_gpu(2)
    def test_multi_bucket_state_dict_sharded(self):
        """state_dict returns sharded params across all buckets."""
        mesh = self._init_mesh()
        args, model = make_transformer_model(device=device_type.type)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

        self._flex_shard(
            model,
            mesh,
            buckets=transformer_bucket_specs(
                args.n_layers,
                mesh,
                reshard_after_forward=False,
            ),
        )

        sd = model.state_dict()
        self.assertEqual(
            sd["tok_embeddings.weight"].shape,
            (args.vocab_size // self.world_size, args.dim),
        )
        self.assertEqual(
            sd["output.weight"].shape,
            (args.vocab_size // self.world_size, args.dim),
        )


# ---------------------------------------------------------------------------
# Per-bucket mesh: experts on a 1-D efsdp axis, dense on a 1-D dp axis
# ---------------------------------------------------------------------------


def _multi_mesh_moe_args() -> ModelArgs:
    # weight_tying=False: flex_shard rejects shared params (output<->tok_emb).
    return ModelArgs(
        n_layers=2,
        vocab_size=16,
        max_seq_len=16,
        dim=16,
        n_heads=4,
        dropout_p=0.0,
        num_experts=8,
        weight_tying=False,
    )


def _multi_mesh_moe_bucket_specs(
    args: ModelArgs,
    dp_mesh,
    efsdp_mesh,
) -> list[BucketSpec]:
    # Flat buckets mirroring root -> layer -> MoE: experts on efsdp, rest on dp.
    buckets = [
        BucketSpec(
            [pattern],
            placement_fn=per_param_placements,
            mesh=dp_mesh,
            reshard_after_forward=False,
        )
        for pattern in (
            "tok_embeddings.*",
            "pos_embeddings.*",
            "norm.*",
            "output.*",
        )
    ]
    for i in range(args.n_layers):
        # Dense attention q/k/v/o + attention/ffn norms -> one bucket on dp.
        buckets.append(
            BucketSpec(
                [
                    f"layers.{i}.attention.*",
                    f"layers.{i}.attention_norm.*",
                    f"layers.{i}.ffn_norm.*",
                ],
                placement_fn=per_param_placements,
                mesh=dp_mesh,
                reshard_after_forward=False,
            )
        )
        # MoE expert FFN stacks (experts.w1, experts.w2) -> efsdp.
        buckets.append(
            BucketSpec(
                [f"layers.{i}.expert_layer.*"],
                placement_fn=per_param_placements,
                mesh=efsdp_mesh,
                reshard_after_forward=False,
            )
        )
    return buckets


def _is_common_dtensor_expert_param(fqn: str) -> bool:
    return "expert_layer" in fqn  # expert_layer.experts.{w1,w2}


class TestMultiMeshBuckets(FSDPTest):
    """Per-bucket mesh: different buckets shard on different 1-D sub-meshes.

    The plain-tensor analog of fully_shard's per-param-mesh MoE setup (experts on
    an expert-FSDP axis, dense params on the data-parallel axis), expressed with
    flat, FQN-patterned buckets instead of nested wrapping + shard_placement_fn.
    Uses the same toy ``Transformer`` (with ``num_experts``) that fully_shard's
    ``test_shard_placement_fn_tp_ep`` uses, so the dense/expert split is
    representative: attention ``wq/wk/wv/wo`` and ``attention_norm``/``ffn_norm``
    are dense; ``expert_layer.experts.{w1,w2}`` are 3-D expert stacks.

    The world (4 ranks) is factored as efsdp(2) x ep(2); same rank set, two 1-D
    meshes:

      * dense params shard ``Shard(0)`` over the full ``dp`` mesh (size 4)
      * expert params shard ``Shard(0)`` over the ``efsdp`` sub-mesh (size 2)
        -- replicated across ``ep``, sharded within ``efsdp``

    Verifies the layout (experts<->efsdp, dense<->dp), that each bucket carries the
    right mesh, and that gathering each shard over its bucket's mesh reconstructs
    the full param. (This toy Transformer's MoE has no router gate -- it runs and
    averages all experts -- so only the expert FFN weights live on efsdp.)
    """

    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_experts_on_efsdp_dense_on_dp(self) -> None:
        dp_mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("dp",)
        )
        sparse_mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("efsdp", "ep")
        )
        efsdp_mesh = sparse_mesh["efsdp"]
        self.assertEqual(dp_mesh.size(), 4)
        self.assertEqual(efsdp_mesh.size(), 2)

        args = _multi_mesh_moe_args()
        model = Transformer(args).to(device_type.type)
        for p in model.parameters():  # identical full params on every rank
            dist.broadcast(p.data, src=0)
        reference = copy.deepcopy(model)

        flex_shard(
            model,
            buckets=_multi_mesh_moe_bucket_specs(args, dp_mesh, efsdp_mesh),
        )

        # Grouping: each bucket storage carries the mesh its params shard on.
        for storage in model.sharded_bucket_storages:
            for fqn in storage._param_infos:
                expected_mesh = (
                    efsdp_mesh if _is_common_dtensor_expert_param(fqn) else dp_mesh
                )
                self.assertIs(storage._mesh, expected_mesh, fqn)

        # Sharding: experts <-> efsdp, dense <-> dp, byte-for-byte (pre-forward,
        # while params are still sharded).
        ref_params = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            ref = ref_params[name].detach()
            param_mesh = (
                efsdp_mesh if _is_common_dtensor_expert_param(name) else dp_mesh
            )
            want = expected_shard(
                ref, rank=param_mesh.get_local_rank(), world_size=param_mesh.size()
            )
            self.assertEqual(param.detach(), want, name)

        # Expert leading dim follows efsdp (size 2), not dp (size 4).
        experts_w1 = dict(model.named_parameters())["layers.0.expert_layer.experts.w1"]
        self.assertEqual(experts_w1.shape[0], args.num_experts // efsdp_mesh.size())

        # Runtime: gathering each local shard over its bucket's mesh reconstructs
        # the full reference param -- experts gather over efsdp(2), dense over
        # dp(4) -- exercising the two meshes' all-gather process groups.
        def _gather_full(local: torch.Tensor, mesh, full_dim0: int) -> torch.Tensor:
            parts = [torch.empty_like(local) for _ in range(mesh.size())]
            dist.all_gather(parts, local.contiguous(), group=mesh.get_group())
            return torch.cat(parts, dim=0)[:full_dim0]

        for name, param in model.named_parameters():
            ref = ref_params[name].detach()
            param_mesh = (
                efsdp_mesh if _is_common_dtensor_expert_param(name) else dp_mesh
            )
            self.assertEqual(
                _gather_full(param.detach(), param_mesh, ref.shape[0]), ref, name
            )

    @skip_if_lt_x_gpu(4)
    def test_train_parity(self) -> None:
        """Full fwd/bwd/SGD loop matches a single-device reference, with dense
        blocks on dp and expert FFNs on efsdp.

        The upstream toy Transformer reads expert weights more than once during
        forward, so this exercises FlexShard's forward-scoped param access cache.
        All ranks share the input, so per-rank grads are identical and flex's
        reduce-scatter (mean) is a no-op + chunk, giving exact per-shard parity
        after each SGD step.
        """
        dp_mesh = init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("dp",)
        )
        efsdp_mesh = init_device_mesh(
            device_type.type, (2, 2), mesh_dim_names=("efsdp", "ep")
        )["efsdp"]

        torch.manual_seed(0)
        args = _multi_mesh_moe_args()
        model = Transformer(args).to(device_type.type)
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        reference = copy.deepcopy(model)

        flex_shard(
            model,
            buckets=_multi_mesh_moe_bucket_specs(args, dp_mesh, efsdp_mesh),
        )

        flex_optim = make_test_sgd(model.parameters(), lr=0.1)
        ref_optim = make_test_sgd(reference.parameters(), lr=0.1)

        torch.manual_seed(7)
        x = torch.randint(
            0,
            args.vocab_size,
            (2, args.max_seq_len),
            device=device_type.type,
        )
        target = torch.randint(
            0,
            args.vocab_size,
            (2, args.max_seq_len),
            device=device_type.type,
        )
        dist.broadcast(x, src=0)
        dist.broadcast(target, src=0)
        cross_entropy = nn.functional.cross_entropy

        for step in range(3):
            flex_optim.zero_grad()
            ref_optim.zero_grad()
            out = model(x)
            ref_out = reference(x)
            # Forward parity: dp and efsdp all-gathers both reconstruct the params.
            self.assertEqual(out, ref_out, f"forward step {step}")
            loss = cross_entropy(out.reshape(-1, args.vocab_size), target.reshape(-1))
            ref_loss = cross_entropy(
                ref_out.reshape(-1, args.vocab_size),
                target.reshape(-1),
            )
            self.assertEqual(loss, ref_loss, f"loss step {step}")
            loss.backward()
            ref_loss.backward()
            flex_optim.step()
            ref_optim.step()

            # After the step, each local shard equals the reference param chunked
            # on its bucket's mesh -- experts on efsdp(2), dense on dp(4).
            ref_now = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                param_mesh = (
                    efsdp_mesh if _is_common_dtensor_expert_param(name) else dp_mesh
                )
                want = expected_shard(
                    ref_now[name].detach(),
                    rank=param_mesh.get_local_rank(),
                    world_size=param_mesh.size(),
                )
                self.assertEqual(param.detach(), want, f"{name} after step {step}")


if __name__ == "__main__":
    run_tests()
