# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.protocol import CheckpointableTensor
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_utils import run_tests, TestCase

from ..custom_placements.block_shard import BlockShard, BucketedBlockShard
from ..custom_placements.owned import BucketedOwned, BucketedOwnedSegmentSpec
from ..dist_muon.comm_free_adapter import _build_parameter_binding, LocalDistMuonBinding
from ..dist_muon.storage_layout import (
    _block_shard_layout,
    _bucketed_block_shard_layout,
    capture_flex_shard_muon_canonical_shards,
    get_flex_shard_muon_compute_layouts,
    LocalMuonComputeLayout,
    LocalMuonStateShard,
)
from ..flex_shard.bucket_storage import BucketLayout, BucketParamLayout, ParamInfo
from ..flex_shard.sharded_param import set_sharding_info
from .common import single_rank_cpu_mesh


def _sgd_binding(mesh, parameters=None):
    if parameters is None:
        parameters = {
            "weight": nn.Parameter(torch.arange(6).reshape(2, 3).float()),
            "experts": nn.Parameter(torch.arange(12).reshape(2, 2, 3).float()),
        }
    layouts = (
        LocalMuonComputeLayout(
            kind="row_blocks",
            block_size=2,
            state_checkpoint_shard=LocalMuonStateShard(
                global_shape=torch.Size((12, 3)),
                global_offset=(4, 0),
                local_shape=torch.Size((2, 3)),
            ),
        ),
        LocalMuonComputeLayout(
            kind="matrix_batch",
            state_checkpoint_shard=LocalMuonStateShard(
                global_shape=torch.Size((8, 2, 3)),
                global_offset=(3, 0, 0),
                local_shape=torch.Size((2, 2, 3)),
            ),
        ),
    )
    bindings = tuple(
        _build_parameter_binding(name, parameter, layout, mesh)
        for (name, parameter), layout in zip(parameters.items(), layouts, strict=True)
    )
    optimizer = torch.optim.SGD(  # noqa: CITRINE(missing_for_each_optimizer)
        [
            {
                "params": [binding.proxy_param for binding in bindings],
                "param_names": [binding.fqn for binding in bindings],
            }
        ],
        lr=0.125,
        momentum=0.9,
        foreach=False,
    )
    return parameters, bindings, LocalDistMuonBinding(optimizer, bindings)


class TestDistMuonStorageLayout(TestCase):
    def test_checkpoint_wrapped_names_resolve_to_owned_layout(self):
        with single_rank_cpu_mesh() as mesh:
            model = nn.Module()
            model._orig_mod = checkpoint_wrapper(nn.Linear(3, 4, bias=False))
            capture_flex_shard_muon_canonical_shards(model)
            raw_name, parameter = next(model.named_parameters())
            canonical_name = "_orig_mod.weight"
            placement = BucketedOwned(
                {
                    canonical_name: [
                        BucketedOwnedSegmentSpec(
                            canonical_name, canonical_name, 0, parameter.numel(), 0
                        )
                    ]
                }
            )
            set_sharding_info(
                parameter,
                placements=(placement,),
                global_shape=parameter.shape,
                global_stride=parameter.stride(),
                mesh=mesh,
            )
            raw, canonical = get_flex_shard_muon_compute_layouts(
                model, [(raw_name, parameter), (canonical_name, parameter)]
            )
            self.assertEqual(raw, canonical)
            self.assertEqual(canonical.kind, "owned")
            self.assertIsNone(canonical.block_size)
            self.assertEqual(canonical.state_checkpoint_shard.global_shape, (4, 3))
            self.assertEqual(canonical.state_checkpoint_shard.global_offset, (0, 0))
            self.assertEqual(canonical.state_checkpoint_shard.local_shape, (4, 3))
            with self.assertRaises(KeyError):
                get_flex_shard_muon_compute_layouts(model, [("weight", parameter)])
            with self.assertRaisesRegex(ValueError, "before calling flex_shard"):
                capture_flex_shard_muon_canonical_shards(model)
            for shape in ((2, 3), (0, 3)):
                with self.subTest(local_shape=shape):
                    partial = nn.Parameter(torch.empty(shape))
                    set_sharding_info(
                        partial,
                        placements=(placement,),
                        global_shape=parameter.shape,
                        global_stride=parameter.stride(),
                        mesh=mesh,
                    )
                    with self.assertRaisesRegex(
                        NotImplementedError, "complete 2D matrix"
                    ):
                        get_flex_shard_muon_compute_layouts(
                            model, [(canonical_name, partial)]
                        )

    def test_row_blocks_add_outer_offset_and_use_global_matrix_rows(self):
        layout = _block_shard_layout(
            "packed",
            torch.empty(4, 4),
            torch.Size((6, 4)),
            LocalMuonStateShard(
                global_shape=torch.Size((18, 4)),
                global_offset=(6, 0),
                local_shape=torch.Size((6, 4)),
            ),
            BlockShard(blocks_per_rank=(1, 2), dim=0),
            rank=1,
        )
        self.assertEqual(layout.kind, "row_blocks")
        self.assertEqual(layout.block_size, 2)
        self.assertEqual(layout.state_checkpoint_shard.global_shape, (18, 4))
        self.assertEqual(layout.state_checkpoint_shard.global_offset, (8, 0))
        self.assertEqual(layout.state_checkpoint_shard.local_shape, (4, 4))

    def test_matrix_batch_offsets_exclude_preceding_bucket_parameters(self):
        parameter = torch.empty(3, 2, 3)
        placement = BucketedBlockShard(dims=(0,), blocks_per_rank=(3, 3))
        info = ParamInfo(
            fqn="experts",
            global_shape=torch.Size((4, 2, 3)),
            global_stride=(6, 3, 1),
            dtype=parameter.dtype,
            requires_grad=False,
            placements=(placement,),
            bucket_layout=BucketLayout(
                global_numel=36,
                local_numel=18,
                rank_offsets=(0, 18),
                rank_numels=(18, 18),
                param_layouts={
                    "prefix": BucketParamLayout(0, 18),
                    "experts": BucketParamLayout(12, 18),
                },
            ),
        )
        layout = _bucketed_block_shard_layout(
            "experts",
            parameter,
            info.global_shape,
            LocalMuonStateShard(
                global_shape=torch.Size((8, 2, 3)),
                global_offset=(4, 0, 0),
                local_shape=torch.Size((4, 2, 3)),
            ),
            placement,
            info,
        )
        self.assertEqual(layout.kind, "matrix_batch")
        self.assertIsNone(layout.block_size)
        self.assertEqual(layout.state_checkpoint_shard.global_shape, (8, 2, 3))
        self.assertEqual(layout.state_checkpoint_shard.global_offset, (5, 0, 0))
        self.assertEqual(layout.state_checkpoint_shard.local_shape, (3, 2, 3))


class TestDistMuonBinding(TestCase):
    def test_missing_gradient_rejects_step_before_any_parameter_update(self):
        with single_rank_cpu_mesh() as mesh:
            parameters, bindings, binding = _sgd_binding(mesh)
            before = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
            parameters["weight"].grad = torch.ones_like(parameters["weight"])
            with self.assertRaisesRegex(RuntimeError, "missing gradients:.*experts"):
                binding.step()
            self.assertEqual(binding.optimizer.state, {})
            for name, parameter in parameters.items():
                self.assertEqual(parameter, before[name])
            for parameter_binding in bindings:
                self.assertIsNone(parameter_binding.proxy_param.grad)

    def test_momentum_checkpoint_roundtrip_preserves_next_step(self):
        with single_rank_cpu_mesh() as mesh:
            parameters, bindings, binding = _sgd_binding(mesh)
            reference = {
                name: nn.Parameter(parameter.detach().clone())
                for name, parameter in parameters.items()
            }
            reference_optimizer = torch.optim.SGD(  # noqa: CITRINE(missing_for_each_optimizer)
                reference.values(), lr=0.125, momentum=0.9, foreach=False
            )
            for parameter_binding in bindings:
                self.assertIsInstance(parameter_binding.proxy_param, DTensor)
                self.assertEqual(
                    parameter_binding.proxy_param.to_local().data_ptr(),
                    parameter_binding.real_param.data_ptr(),
                )
            for scale in (1.0, 2.0):
                for index, (name, parameter) in enumerate(parameters.items(), 1):
                    gradient = torch.full_like(parameter, scale * index)
                    parameter.grad = gradient
                    reference[name].grad = gradient.clone()
                binding.step()
                reference_optimizer.step()
            checkpoint = copy.deepcopy(binding.optimizer.state_dict())
            state_ids = checkpoint["param_groups"][0]["params"]
            for parameter_binding, state_id in zip(bindings, state_ids, strict=True):
                momentum = checkpoint["state"][state_id]["momentum_buffer"]
                shard = parameter_binding.layout.state_checkpoint_shard
                self.assertNotIsInstance(momentum, DTensor)
                self.assertIsInstance(momentum, CheckpointableTensor)
                self.assertEqual(momentum.global_shape, tuple(shard.global_shape))
                self.assertEqual(momentum.global_offsets, (shard.global_offset,))
                self.assertEqual(momentum.local_offsets, ((0,) * momentum.ndim,))
                self.assertEqual(momentum.local_sizes, (tuple(shard.local_shape),))
                self.assertEqual(
                    momentum,
                    reference_optimizer.state[reference[parameter_binding.fqn]][
                        "momentum_buffer"
                    ],
                )
            resumed_parameters = {
                name: nn.Parameter(parameter.detach().clone())
                for name, parameter in parameters.items()
            }
            _, resumed_bindings, resumed = _sgd_binding(mesh, resumed_parameters)
            resumed.optimizer.load_state_dict(checkpoint)
            for parameter_binding in resumed_bindings:
                self.assertIsInstance(
                    resumed.optimizer.state[parameter_binding.proxy_param][
                        "momentum_buffer"
                    ],
                    DTensor,
                )
            for index, (name, parameter) in enumerate(parameters.items(), 1):
                gradient = torch.full_like(parameter, -0.5 * index)
                parameter.grad = gradient
                resumed_parameters[name].grad = gradient.clone()
                reference[name].grad = gradient.clone()
            binding.step()
            resumed.step()
            reference_optimizer.step()
            for name, parameter in parameters.items():
                self.assertEqual(parameter, reference[name])
                self.assertEqual(resumed_parameters[name], reference[name])
            for parameter_binding in resumed_bindings:
                self.assertEqual(
                    resumed.optimizer.state[parameter_binding.proxy_param][
                        "momentum_buffer"
                    ].to_local(),
                    reference_optimizer.state[reference[parameter_binding.fqn]][
                        "momentum_buffer"
                    ],
                )


if __name__ == "__main__":
    run_tests()
