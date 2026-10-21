# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @lint-ignore-every CITRINE

import math
import unittest
from unittest import mock

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from ..dist_muon import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from ..dist_muon.optimizer import (
    _adjust_muon_learning_rate,
    DistMuon,
)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestDistMuon(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_type(self):
        return "cuda"

    @parametrize("block_sizes", [(4,), (4, 2)])
    @with_comms
    def test_matches_plain_muon_across_state_dict_round_trip(
        self, block_sizes: tuple[int, ...]
    ):
        lr = 0.03
        num_repeats = 3
        matrix_row_sizes = block_sizes * num_repeats
        num_rows = sum(matrix_row_sizes)
        # Both cases split a four-row matrix at the storage shard boundary.
        # Variable blocks give compute owners 10 and 8 rows, with the second
        # owner starting at the second position of the repeating pattern.
        weight_decay = 0.2
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)

        def make_parameter(value: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                distribute_tensor(value.clone(), mesh, (Shard(0),))
            )

        def make_optimizer(
            redistributed: torch.nn.Parameter,
            local_blocks: torch.nn.Parameter,
            ns_steps: int = 2,
        ):
            redistributed_fqn = "layers.0.redistributed"
            local_blocks_fqn = "layers.0.local_blocks"
            return DistMuon(
                [
                    {
                        "params": [redistributed, local_blocks],
                        "param_names": [redistributed_fqn, local_blocks_fqn],
                    }
                ],
                compute_sharding_by_fqn={
                    redistributed_fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "dp_shard": Owned(),
                        },
                    ),
                    local_blocks_fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "dp_shard": BlockShard(
                                dim=0,
                                block_sizes=block_sizes,
                            )
                        },
                    ),
                },
                bucket_configs=[
                    BucketConfig(
                        patterns=("layers.0.*",),
                        name="layers.0",
                    )
                ],
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.8,
                nesterov=True,
                ns_steps=ns_steps,
            )

        redistributed_value = (
            torch.arange(12, device=device).reshape(4, 3).float().div_(10).add_(1)
        )
        local_blocks_value = (
            torch.arange(12, 12 + num_rows * 3, device=device)
            .reshape(num_rows, 3)
            .float()
            .div_(10)
        )
        redistributed = make_parameter(redistributed_value)
        local_blocks = make_parameter(local_blocks_value)
        optimizer = make_optimizer(redistributed, local_blocks)
        self.assertIs(type(optimizer), DistMuon)
        with self.assertRaisesRegex(RuntimeError, "parameter groups are frozen"):
            optimizer.add_param_group({"params": []})

        reference_redistributed = torch.nn.Parameter(redistributed_value.clone())
        reference_local_blocks = tuple(
            torch.nn.Parameter(block.clone())
            for block in local_blocks_value.split(matrix_row_sizes)
        )
        reference_optimizer = torch.optim.Muon(
            [reference_redistributed, *reference_local_blocks],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
        )

        def step_and_assert(
            current_optimizer,
            current_redistributed: torch.nn.Parameter,
            current_local_blocks: torch.nn.Parameter,
            redistributed_grad: torch.Tensor,
            local_blocks_grad: torch.Tensor,
        ) -> None:
            local_blocks_before = current_local_blocks.to_local().clone()
            reference_local_blocks_before = tuple(
                parameter.detach().clone() for parameter in reference_local_blocks
            )
            current_redistributed.grad = distribute_tensor(
                redistributed_grad.clone(), mesh, (Shard(0),)
            )
            current_local_blocks.grad = distribute_tensor(
                local_blocks_grad.clone(), mesh, (Shard(0),)
            )
            reference_redistributed.grad = redistributed_grad.clone()
            for parameter, grad in zip(
                reference_local_blocks,
                local_blocks_grad.split(matrix_row_sizes),
                strict=True,
            ):
                parameter.grad = grad.clone()

            current_optimizer.step()
            reference_optimizer.step()

            rank = mesh.get_local_rank()
            expected_redistributed = reference_redistributed.detach().chunk(
                self.world_size, dim=0
            )[rank]
            torch.testing.assert_close(
                current_redistributed.to_local(),
                expected_redistributed,
                rtol=0,
                atol=0,
            )

            expected_local_blocks = torch.cat(
                tuple(parameter.detach() for parameter in reference_local_blocks)
            ).chunk(self.world_size)[rank]
            expected_local_blocks_before = torch.cat(
                reference_local_blocks_before
            ).chunk(self.world_size)[rank]
            decay = 1 - lr * weight_decay
            adjusted_lrs = torch.cat(
                tuple(
                    parameter.new_full(
                        (parameter.shape[0], 1),
                        _adjust_muon_learning_rate(lr, None, parameter.shape),
                    )
                    for parameter in reference_local_blocks
                )
            ).chunk(self.world_size)[rank]
            actual_update = (
                local_blocks_before * decay - current_local_blocks.to_local()
            ) / adjusted_lrs
            expected_update = (
                expected_local_blocks_before * decay - expected_local_blocks
            ) / adjusted_lrs
            # Batched BF16 Newton-Schulz can differ slightly across GEMM schedules.
            torch.testing.assert_close(
                actual_update,
                expected_update,
                rtol=0,
                atol=2e-2,
            )

        first_redistributed_grad = (
            torch.arange(1, 13, device=device).reshape(4, 3).float().div_(17)
        )
        first_local_blocks_grad = (
            torch.arange(13, 13 + num_rows * 3, device=device)
            .reshape(num_rows, 3)
            .float()
            .div_(19)
        )
        step_and_assert(
            optimizer,
            redistributed,
            local_blocks,
            first_redistributed_grad,
            first_local_blocks_grad,
        )

        state_dict = optimizer.state_dict()
        resumed_redistributed = make_parameter(redistributed.full_tensor().detach())
        resumed_local_blocks = make_parameter(local_blocks.full_tensor().detach())
        resumed_optimizer = make_optimizer(
            resumed_redistributed,
            resumed_local_blocks,
            ns_steps=3,
        )
        resumed_optimizer.load_state_dict(state_dict)

        second_redistributed_grad = first_redistributed_grad.flip(0).contiguous()
        second_local_blocks_grad = first_local_blocks_grad.flip(0).contiguous()
        step_and_assert(
            resumed_optimizer,
            resumed_redistributed,
            resumed_local_blocks,
            second_redistributed_grad,
            second_local_blocks_grad,
        )


@instantiate_parametrized_tests
@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistMuonNativeMatrixBatch(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @parametrize(
        "storage_placement,compute_sharding",
        [
            subtest((Shard(1), Owned()), name="rows_to_owned"),
            subtest((Shard(1), Shard(0)), name="rows_to_matrix_shards"),
            subtest((Shard(2), Shard(0)), name="columns_to_matrix_shards"),
            subtest((Replicate(), Shard(0)), name="replicated_to_matrix_shards"),
        ],
    )
    @with_comms
    def test_gate_up_matches_independent_matrix_updates(
        self, storage_placement, compute_sharding
    ):
        lr = 0.03
        weight_decay = 0.2
        mesh = init_device_mesh(
            self.device_type,
            (self.world_size,),
            mesh_dim_names=("dp_shard",),
        )
        device = torch.device(self.device_type, self.rank)
        value = torch.arange(70, device=device).reshape(2, 7, 5).float().div_(17)
        storage_placements = (storage_placement,)
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, storage_placements)
        )
        fqn = "layers.0.feed_forward.w13.weight"
        optimizer = DistMuon(
            [{"params": [parameter], "param_names": [fqn]}],
            compute_sharding_by_fqn={
                fqn: ComputeLayout(
                    shardings_by_mesh_axis={"dp_shard": compute_sharding},
                )
            },
            bucket_configs=[BucketConfig(patterns=(fqn,))],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
            adjust_lr_fn="match_rms_adamw",
        )
        reference_parameters = [torch.nn.Parameter(matrix.clone()) for matrix in value]
        reference_optimizer = torch.optim.Muon(
            reference_parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
            adjust_lr_fn="match_rms_adamw",
        )
        adjusted_lr = lr * 0.2 * max(value.shape[-2:]) ** 0.5
        for step in range(3):
            gradient = value.mul(step + 0.7).add_(0.2).sin_()
            gradient[1].mul_(7)
            local_gradient = distribute_tensor(
                gradient.clone(), mesh, storage_placements
            ).to_local()
            optimizer.zero_grad()
            (parameter.to_local() * local_gradient).sum().backward()
            if storage_placement == Shard(2) and self.rank == self.world_size - 1:
                # The padded empty column shard and autograd disagree on strides.
                self.assertEqual(parameter.to_local().numel(), 0)
                self.assertNotEqual(
                    parameter.to_local().stride(), parameter.grad.to_local().stride()
                )
            before = parameter.to_local().clone()
            reference_before = torch.stack(
                [reference.detach().clone() for reference in reference_parameters]
            )
            for reference, reference_grad in zip(
                reference_parameters, gradient, strict=True
            ):
                reference.grad = reference_grad.clone()

            with mock.patch.object(
                optimizer, "_compute_update", wraps=optimizer._compute_update
            ) as compute_update:
                optimizer.step()
            if compute_sharding == Shard(0):
                self.assertEqual(compute_update.call_count, int(self.rank < 2))
                if self.rank < 2:
                    self.assertEqual(compute_update.call_args.args[1].shape, (1, 7, 5))
            reference_optimizer.step()

            expected = torch.stack(
                [reference.detach() for reference in reference_parameters]
            )
            actual_update = (
                before * (1 - lr * weight_decay) - parameter.to_local()
            ) / adjusted_lr
            expected_update = (
                reference_before * (1 - lr * weight_decay) - expected
            ) / adjusted_lr
            expected_local_update = distribute_tensor(
                expected_update, mesh, storage_placements
            ).to_local()
            # Batched BF16 Newton-Schulz can differ across GEMM schedules.
            torch.testing.assert_close(
                actual_update,
                expected_local_update,
                rtol=0,
                atol=2e-2,
            )
            self.assertEqual(parameter.placements, storage_placements)


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistMuonInitialExpertStorageContract(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    @with_comms
    def test_preserves_ep_shard_during_edp_shard_redistribution(self):
        lr = 0.03
        weight_decay = 0.2
        mesh = init_device_mesh(
            self.device_type,
            (2, 2),
            mesh_dim_names=("edp_shard", "ep"),
        )
        num_experts = 3
        self.assertLess(
            num_experts,
            mesh["edp_shard"].size() * mesh["ep"].size(),
        )
        device = torch.device(self.device_type, self.rank)
        value = (
            torch.arange(num_experts * 2 * 5 * 3, device=device)
            .reshape(num_experts, 2, 5, 3)
            .float()
            .div_(13)
        )
        storage_placements = (Shard(2), Shard(0))
        parameter = torch.nn.Parameter(
            distribute_tensor(value.clone(), mesh, storage_placements)
        )
        fqn = "layers.0.routed_experts.w13.weight"

        def make_optimizer(param, shard_order_by_tensor_dim):
            return DistMuon(
                [{"params": [param], "param_names": [fqn]}],
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.0,
                nesterov=False,
                ns_steps=2,
                compute_sharding_by_fqn={
                    fqn: ComputeLayout(
                        shardings_by_mesh_axis={
                            "edp_shard": Shard(0),
                            "ep": Shard(0),
                        },
                        shard_order_by_tensor_dim=shard_order_by_tensor_dim,
                    )
                },
                bucket_configs=[BucketConfig(patterns=(fqn,))],
            )

        expected_shard_order = {0: ("ep", "edp_shard")}
        # The storage-mesh order shards over edp_shard first, which loses the exact
        # EP-axis ownership that the redistribution has to preserve.
        for default_order in ({}, {0: ("edp_shard", "ep")}):
            with self.assertRaisesRegex(
                ValueError,
                r"must declare shard_order_by_tensor_dim=\{0: \('ep', 'edp_shard'\)\}",
            ):
                make_optimizer(parameter, default_order)

        optimizer = make_optimizer(parameter, expected_shard_order)
        grad = (
            torch.arange(value.numel(), device=device)
            .reshape_as(value)
            .float()
            .mul_(0.37)
            .add_(0.2)
            .sin_()
        )
        parameter.grad = distribute_tensor(grad.clone(), mesh, storage_placements)

        mesh_coordinate = mesh.get_coordinate()
        assert mesh_coordinate is not None
        edp_shard_coordinate, ep_coordinate = mesh_coordinate
        ep_num_experts, ep_offset = Shard.local_shard_size_and_offset(
            num_experts,
            mesh["ep"].size(),
            ep_coordinate,
        )
        edp_shard_num_experts, edp_shard_offset = Shard.local_shard_size_and_offset(
            ep_num_experts,
            mesh["edp_shard"].size(),
            edp_shard_coordinate,
        )
        compute_offset = ep_offset + edp_shard_offset
        expected_compute = grad.narrow(
            0,
            compute_offset,
            edp_shard_num_experts,
        ).contiguous()
        expected_direction = grad.clone().mul_(0.5).add_(0.25)
        expected_parameter = value.clone().mul_(1 - lr * weight_decay)
        expected_parameter.add_(
            expected_direction,
            alpha=-_adjust_muon_learning_rate(lr, None, value.shape[1:]),
        )
        captured_compute = None

        def capture_compute(_compute_layout, compute):
            nonlocal captured_compute
            captured_compute = compute.clone()
            compute.mul_(0.5).add_(0.25)

        with mock.patch.object(
            optimizer,
            "_compute_update",
            side_effect=capture_compute,
        ):
            optimizer.step()

        if expected_compute.numel():
            self.assertIsNotNone(captured_compute)
            torch.testing.assert_close(
                captured_compute,
                expected_compute,
                rtol=0,
                atol=0,
            )
        else:
            self.assertIsNone(captured_compute)

        torch.testing.assert_close(
            parameter.full_tensor(),
            expected_parameter,
            rtol=0,
            atol=0,
        )
        self.assertEqual(parameter.placements, storage_placements)

        compute_ready_local = value.narrow(
            0,
            compute_offset,
            edp_shard_num_experts,
        ).contiguous()
        # The compute-ready storage already carries the declared shard order.
        compute_ready_parameter = torch.nn.Parameter(
            DTensor.from_local(
                compute_ready_local,
                mesh,
                (_StridedShard(0, split_factor=mesh["ep"].size()), Shard(0)),
                shape=value.shape,
                stride=value.stride(),
                run_check=False,
            )
        )
        compute_ready_optimizer = make_optimizer(
            compute_ready_parameter,
            expected_shard_order,
        )
        compute_ready_layout = compute_ready_optimizer._parameter_compute_layouts[0]
        self.assertTrue(compute_ready_layout.storage_is_compute_ready)


_WQ_B = "layers.0.attention.wq_b.weight"
_WO = "layers.0.attention.wo.weight"
_W13 = "layers.0.feed_forward.w13.weight"
_EXPERTS = "layers.0.moe.routed_experts.w13"
_NEXT_WO = "layers.1.attention.wo.weight"
# Three heads of [nope; rope] rows, a matrix, gate and up, one expert per rank,
# and the next block's matrix.
_SHAPES = {
    _WQ_B: (18, 5),
    _WO: (6, 9),
    _W13: (2, 7, 5),
    _EXPERTS: (4, 6, 5),
    _NEXT_WO: (9, 6),
}
_HEAD_BLOCKS = (4, 2)


@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestDistMuonOwners(DTensorTestBase):
    """Owned blocks and fixed owners, on two blocks with the routed experts'
    local bucket between them."""

    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    def _values(self) -> dict[str, torch.Tensor]:
        device = torch.device(self.device_type, self.rank)
        return {
            fqn: torch.arange(math.prod(shape), device=device)
            .reshape(shape)
            .float()
            .mul_(0.37 + index)
            .sin_()
            for index, (fqn, shape) in enumerate(_SHAPES.items())
        }

    def _optimizer(
        self,
        params: dict[str, torch.nn.Parameter],
        *,
        owner_by_fqn: dict[str, int],
        wq_b_sharding: Owned | BlockShard = Owned(block_sizes=_HEAD_BLOCKS),
    ) -> DistMuon:
        def layout(sharding) -> ComputeLayout:
            return ComputeLayout(shardings_by_mesh_axis={"dp_shard": sharding})

        return DistMuon(
            [{"params": list(params.values()), "param_names": list(params)}],
            compute_sharding_by_fqn={
                _WQ_B: layout(wq_b_sharding),
                _WO: layout(Owned()),
                _W13: layout(Owned()),
                _EXPERTS: layout(Shard(0)),
                _NEXT_WO: layout(Owned()),
            },
            bucket_configs=[
                BucketConfig(
                    patterns=("layers.0.attention.*", "layers.0.feed_forward.*"),
                    name="layers.0",
                ),
                BucketConfig(patterns=("layers.0.moe.*",), name="layers.0.experts"),
                BucketConfig(patterns=("layers.1.*",), name="layers.1"),
            ],
            lr=0.03,
            weight_decay=0.2,
            momentum=0.8,
            nesterov=True,
            ns_steps=2,
            owner_by_fqn=owner_by_fqn,
        )

    def _train(
        self, optimizer: DistMuon, params: dict[str, torch.nn.Parameter]
    ) -> dict[str, torch.Tensor]:
        mesh = next(iter(params.values())).device_mesh
        for step in range(3):
            for index, (fqn, param) in enumerate(params.items()):
                grad = self._values()[fqn].mul(step + 1.3).add_(index).cos_()
                param.grad = distribute_tensor(grad, mesh, (Shard(0),))
            optimizer.step()
        return {fqn: param.full_tensor() for fqn, param in params.items()}

    def _params(self, mesh) -> dict[str, torch.nn.Parameter]:
        return {
            fqn: torch.nn.Parameter(distribute_tensor(value, mesh, (Shard(0),)))
            for fqn, value in self._values().items()
        }

    @with_comms
    def test_owned_blocks_on_fixed_owners(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )
        params = self._params(mesh)
        optimizer = self._optimizer(params, owner_by_fqn={_WQ_B: 3, _WO: 1})
        layouts = {
            layout.fqn: layout for layout in optimizer._parameter_compute_layouts
        }
        self.assertEqual(layouts[_WQ_B].compute_sharding, Owned(_HEAD_BLOCKS))
        self.assertEqual(layouts[_WQ_B].lr_reference_shape, (6, 5))
        computed = []

        def record(compute_layout, compute):
            computed.append((compute_layout.fqn, tuple(compute.shape)))
            return DistMuon._compute_update(optimizer, compute_layout, compute)

        with mock.patch.object(optimizer, "_compute_update", side_effect=record):
            actual = self._train(optimizer, params)
        # The owners compute whole parameters; the owner of wq_b per head:
        # a batch of 3 nope blocks and a batch of 3 rope blocks.
        self.assertEqual((_WQ_B, (18, 5)) in computed, self.rank == 3)
        self.assertEqual((_WO, (6, 9)) in computed, self.rank == 1)
        if self.rank == 3:
            self.assertEqual(
                [view.shape for view in optimizer._matrix_views_by_fqn[_WQ_B]],
                [torch.Size((3, 4, 5)), torch.Size((3, 2, 5))],
            )
        computed_by_rank = [None] * self.world_size
        torch.distributed.all_gather_object(computed_by_rank, computed)
        for fqn in (_W13, _NEXT_WO):
            owners = [
                rank
                for rank, calls in enumerate(computed_by_rank)
                if any(name == fqn for name, _shape in calls)
            ]
            self.assertEqual(len(owners), 1)

        # Per-head Newton-Schulz on the owner: the same update as BlockShard's
        # heads on the data-parallel ranks, within BF16 kernel differences.
        reference_params = self._params(mesh)
        reference = self._train(
            self._optimizer(
                reference_params,
                owner_by_fqn={_WO: 1},
                wq_b_sharding=BlockShard(dim=0, block_sizes=_HEAD_BLOCKS),
            ),
            reference_params,
        )
        for fqn, expected in reference.items():
            torch.testing.assert_close(actual[fqn], expected, rtol=0, atol=2e-3)

    @with_comms
    def test_rejects_invalid_fixed_owners(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("dp_shard",)
        )
        with self.assertRaisesRegex(ValueError, "mesh axis has 4 ranks"):
            self._optimizer(self._params(mesh), owner_by_fqn={_WO: 4})
        with self.assertRaisesRegex(ValueError, "not Owned"):
            self._optimizer(
                self._params(mesh),
                owner_by_fqn={_EXPERTS: 0},
            )
        with self.assertRaisesRegex(ValueError, "whole repetitions"):
            self._optimizer(
                self._params(mesh),
                owner_by_fqn={},
                wq_b_sharding=Owned(block_sizes=(4, 3)),
            )


instantiate_parametrized_tests(TestDistMuon)


if __name__ == "__main__":
    unittest.main()
