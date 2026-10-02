# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent unsharded parameters (FSDP2-style ``resize_``)."""

import copy

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import BucketSpec, flex_shard, is_flex_shard_param
from ..custom_placements.block_shard import BlockShard, BucketedBlockShard
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..custom_placements.shard import per_param_placements, Shard
from .common import single_rank_cuda_mesh


class _ScaleReadingWeightInBackward(torch.autograd.Function):
    """``x * weight`` whose backward reads ``module.weight`` instead of saving it."""

    @staticmethod
    def forward(ctx, x, weight, module):
        ctx.save_for_backward(x)
        ctx.module = module
        return x * weight

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        weight = ctx.module.weight
        return grad_output * weight, (grad_output * x).sum(0), None


class _BackwardWeightReader(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ScaleReadingWeightInBackward.apply(x, self.weight, self)


class _ScaleWithParamsAsInputs(torch.autograd.Function):
    """Like TransformerEngine's operation fuser: ``module.parameters()`` are the
    autograd inputs, while the computation reads ``module.weight``."""

    @staticmethod
    def forward(ctx, x, module, *params):
        weight = module.weight
        ctx.save_for_backward(x, weight)
        return x * weight

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        return grad_output * weight, None, (grad_output * x).sum(0)


class _ParamsAsInputsScale(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.params_match_weight: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = list(self.parameters())
        if not torch.compiler.is_compiling():
            # A growing Python list would make Dynamo recompile every step.
            self.params_match_weight.append(params[0] is self.weight)
        return _ScaleWithParamsAsInputs.apply(x, self, *params)


class _CallTwice(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.inner = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(self.inner(x))


class _WeightObserver(nn.Linear):
    """Records what ``self.weight`` is during forward."""

    def __init__(self, dim: int) -> None:
        super().__init__(dim, dim, bias=False)
        self.seen_weights: list[torch.Tensor] = []
        self.seen_storage_nbytes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_weights.append(self.weight)
        self.seen_storage_nbytes.append(self.weight.untyped_storage().size())
        return super().forward(x)


def _bucket(patterns, mesh, *, reshard_after_forward):
    return BucketSpec(
        patterns,
        placement_fn=per_param_placements,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
    )


def _per_child_buckets(model: nn.Module, mesh, reshard_after_forward: bool):
    return [
        _bucket([f"{name}.*"], mesh, reshard_after_forward=reshard_after_forward)
        for name, _ in model.named_children()
    ]


def _buckets(model: nn.Module):
    return [
        bucket
        for context in model._flex_shard_eager_comm_contexts.values()
        for bucket in context.buckets
    ]


def _persistent_params(model: nn.Module) -> list[nn.Parameter]:
    """Persistent params (created by each bucket's first unshard)."""
    return [param for bucket in _buckets(model) for param in bucket.unsharded_params]


def _inner_tensors(model: nn.Module) -> list[torch.Tensor]:
    return [
        inner for bucket in _buckets(model) for inner in bucket.unsharded_inner_tensors
    ]


class TestPersistentUnshardedParams(TestCase):
    def _assert_same_params(self, params, expected):
        self.assertEqual(len(params), len(expected))
        self.assertTrue(all(a is b for a, b in zip(params, expected, strict=True)))

    def _assert_grads_match(self, model, reference):
        for param, ref_param in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(param.grad, ref_param.grad)

    def test_backward_reads_param(self):
        for reshard_after_forward in (False, True):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                with single_rank_cuda_mesh() as mesh:
                    torch.manual_seed(0)
                    model = nn.Sequential(
                        _BackwardWeightReader(8), _BackwardWeightReader(8)
                    )
                    reference = copy.deepcopy(model).cuda()
                    flex_shard(
                        model,
                        buckets=_per_child_buckets(model, mesh, reshard_after_forward),
                    )
                    x = torch.randn(4, 8, device="cuda")

                    model(x).sum().backward()
                    reference(x).sum().backward()

                    self._assert_grads_match(model, reference)

    def test_parameters_as_autograd_inputs(self):
        for reshard_after_forward in (False, True):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                with single_rank_cuda_mesh() as mesh:
                    torch.manual_seed(0)
                    model = nn.Sequential(
                        _ParamsAsInputsScale(8), _ParamsAsInputsScale(8)
                    )
                    reference = copy.deepcopy(model).cuda()
                    flex_shard(
                        model,
                        buckets=_per_child_buckets(model, mesh, reshard_after_forward),
                    )
                    x = torch.randn(4, 8, device="cuda")

                    model(x).sum().backward()
                    reference(x).sum().backward()

                    for module in model:
                        self.assertEqual(module.params_match_weight, [True])
                    self._assert_grads_match(model, reference)

    def test_storage_lifecycle(self):
        for reshard_after_forward in (False, True):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                with single_rank_cuda_mesh() as mesh:
                    torch.manual_seed(0)
                    model = nn.Sequential(_WeightObserver(8), _WeightObserver(8))
                    flex_shard(
                        model,
                        buckets=_per_child_buckets(model, mesh, reshard_after_forward),
                    )
                    persistent = None
                    for _step in range(2):
                        loss = model(torch.randn(4, 8, device="cuda")).sum()
                        if persistent is None:
                            # Created by the first unshard, then reused.
                            persistent = _persistent_params(model)
                        # Same objects (identity; their storage may be freed).
                        self._assert_same_params(_persistent_params(model), persistent)
                        for unsharded in persistent:
                            self.assertEqual(
                                unsharded.untyped_storage().size() == 0,
                                reshard_after_forward,
                            )
                        loss.backward()
                        for inner in _inner_tensors(model):
                            self.assertEqual(inner.untyped_storage().size(), 0)
                        for module, unsharded in zip(model, persistent, strict=True):
                            self.assertEqual(unsharded.untyped_storage().size(), 0)
                            self.assertIsNone(unsharded.grad)
                            shard = module._parameters["weight"]
                            self.assertTrue(is_flex_shard_param(shard))
                            self.assertIsNotNone(shard.grad)

                    for module, unsharded in zip(model, persistent, strict=True):
                        # The module saw the same persistent parameter every
                        # forward, with storage allocated.
                        self.assertTrue(
                            all(w is unsharded for w in module.seen_weights)
                        )
                        self.assertTrue(all(n > 0 for n in module.seen_storage_nbytes))

    def test_module_called_twice(self):
        for reshard_after_forward in (False, True):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                with single_rank_cuda_mesh() as mesh:
                    torch.manual_seed(0)
                    model = _CallTwice(8)
                    reference = copy.deepcopy(model).cuda()
                    flex_shard(
                        model,
                        buckets=[
                            _bucket(
                                ["inner.*"],
                                mesh,
                                reshard_after_forward=reshard_after_forward,
                            )
                        ],
                    )
                    x = torch.randn(4, 8, device="cuda")

                    model(x).sum().backward()
                    reference(x).sum().backward()

                    self._assert_grads_match(model, reference)
                    for unsharded in _persistent_params(model):
                        self.assertEqual(unsharded.untyped_storage().size(), 0)

    def test_no_grad_forward_reshards(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward=False),
            )
            with torch.no_grad():
                model(torch.randn(4, 8, device="cuda"))
            for unsharded in _persistent_params(model):
                self.assertEqual(unsharded.untyped_storage().size(), 0)
            for module in model:
                self.assertTrue(is_flex_shard_param(module._parameters["weight"]))

    def test_training_steps_match_reference(self):
        for reshard_after_forward in (False, True):
            with self.subTest(reshard_after_forward=reshard_after_forward):
                with single_rank_cuda_mesh() as mesh:
                    torch.manual_seed(0)
                    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
                    reference = copy.deepcopy(model).cuda()
                    flex_shard(
                        model,
                        buckets=[
                            _bucket(
                                [f"{idx}.*"],
                                mesh,
                                reshard_after_forward=reshard_after_forward,
                            )
                            for idx in (0, 2)
                        ],
                    )
                    for param in model.parameters():
                        self.assertTrue(is_flex_shard_param(param))
                    optim = torch.optim.SGD(model.parameters(), lr=0.1)
                    ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
                    for _step in range(2):
                        x = torch.randn(4, 8, device="cuda")
                        optim.zero_grad()
                        ref_optim.zero_grad()
                        loss = model(x).sum()
                        ref_loss = reference(x).sum()
                        torch.testing.assert_close(loss, ref_loss)
                        loss.backward()
                        ref_loss.backward()
                        self._assert_grads_match(model, reference)
                        optim.step()
                        ref_optim.step()

    def test_persistent_placements(self):
        mixed = MixedBucketPlacement({})
        cases = (
            ("shard", lambda fqn: Shard(0)),
            ("block_shard", lambda fqn: BlockShard(blocks_per_rank=(1,))),
            (
                "bucketed_block_shard",
                lambda fqn: BucketedBlockShard(dims=(0,), blocks_per_rank=(1,)),
            ),
            (
                "mixed",
                lambda fqn: (
                    mixed.shard0
                    if fqn.endswith("weight")
                    else mixed.block_shard(blocks_per_rank=(1,))
                ),
            ),
        )
        for name, make_placement in cases:
            with self.subTest(placement=name):
                self._check_persistent_placement(make_placement)

    def _check_persistent_placement(self, make_placement):
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8))
            reference = copy.deepcopy(model).cuda()
            flex_shard(
                model,
                buckets=[
                    BucketSpec(
                        [f"{idx}.*"],
                        placement_fn=lambda named_params, mesh: {
                            fqn: (make_placement(fqn),) for fqn, _ in named_params
                        },
                        mesh=mesh,
                        # The backward re-gather refills inner tensors that back
                        # params autograd saved in forward.
                        reshard_after_forward=True,
                    )
                    for idx in (0, 1)
                ],
            )
            persistent = None
            for _step in range(2):
                x = torch.randn(4, 8, device="cuda")
                for param in (*model.parameters(), *reference.parameters()):
                    param.grad = None
                model(x).sum().backward()
                reference(x).sum().backward()
                self._assert_grads_match(model, reference)
                if persistent is None:
                    persistent = _persistent_params(model)
                # Same objects (identity; their storage may be freed).
                self._assert_same_params(_persistent_params(model), persistent)
                for inner in _inner_tensors(model):
                    self.assertEqual(inner.untyped_storage().size(), 0)
            # While unsharded, each persistent param is backed by an inner tensor.
            inner_ptrs = set()
            for bucket in _buckets(model):
                bucket.pre_forward_persistent()
                inner_ptrs.update(
                    inner.untyped_storage().data_ptr()
                    for inner in bucket.unsharded_inner_tensors
                )
                for param in bucket.unsharded_params:
                    self.assertIn(param.untyped_storage().data_ptr(), inner_ptrs)
                bucket.reshard_persistent()

    def test_persistent_param_inherits_attributes(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(_WeightObserver(8))
            model[0].weight.custom_tag = "kept"
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward=False),
            )
            model(torch.randn(4, 8, device="cuda")).sum().backward()
            self.assertEqual(model[0].seen_weights[0].custom_tag, "kept")

    def test_fullgraph_capture(self):
        cases = (
            ("linear", lambda: nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))),
            (
                "backward_reader",
                lambda: nn.Sequential(
                    _BackwardWeightReader(8), _BackwardWeightReader(8)
                ),
            ),
            (
                "params_as_inputs",
                lambda: nn.Sequential(_ParamsAsInputsScale(8), _ParamsAsInputsScale(8)),
            ),
        )
        for name, make_model in cases:
            for reshard_after_forward in (False, True):
                with self.subTest(
                    model=name, reshard_after_forward=reshard_after_forward
                ):
                    self._check_fullgraph_capture(make_model, reshard_after_forward)

    def _check_fullgraph_capture(self, make_model, reshard_after_forward):
        torch._dynamo.reset()
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = make_model()
            reference = copy.deepcopy(model).cuda()
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward),
            )
            graphs = []

            def capture_backend(gm, example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled = torch.compile(model, backend=capture_backend, fullgraph=True)
            for _step in range(2):
                x = torch.randn(4, 8, device="cuda")
                for param in (*model.parameters(), *reference.parameters()):
                    param.grad = None
                compiled(x).sum().backward()
                reference(x).sum().backward()
                self._assert_grads_match(model, reference)

            # One graph across both steps: no graph break, no recompile.
            self.assertEqual(1, len(graphs))
            targets = set()
            for _, submodule in graphs[0].named_modules():
                if isinstance(submodule, torch.fx.GraphModule):
                    targets.update(str(node.target) for node in submodule.graph.nodes)
            top_targets = [str(node.target) for node in graphs[0].graph.nodes]
            unshards = [t for t in top_targets if t == "autograd_function_apply"]
            self.assertGreaterEqual(len(unshards), len(model.sharded_bucket_storages))
            self.assertIn("_c10d_functional.all_gather_into_tensor", targets)
            self.assertIn("_c10d_functional.reduce_scatter_tensor", targets)


if __name__ == "__main__":
    run_tests()
