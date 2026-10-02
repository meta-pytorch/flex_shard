# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persistent unsharded parameters (FSDP2-style ``resize_``)."""

import copy
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import BucketSpec, flex_shard, is_flex_shard_param
from ..custom_placements.block_shard import BlockShard, BucketedBlockShard
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..custom_placements.shard import per_param_placements, Shard
from ..flex_shard.bucket_runtime import BucketRuntime
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


class _FirstChildOnly(nn.Module):
    """Uses one of its two linears, like a layer whose expert got no tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.used = nn.Linear(dim, dim)
        self.unused = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x)


class _LayerList(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _ViewOutput(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).view(-1)


class _TwoStageBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.first = nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim))
        self.second = nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


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


def _persistent_buffers(model: nn.Module) -> list[torch.Tensor]:
    return [
        buffer for bucket in _buckets(model) for buffer in bucket.persistent_buffers
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
                        for buffer in _persistent_buffers(model):
                            self.assertEqual(buffer.untyped_storage().size(), 0)
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
                        # The backward re-gather refills persistent buffers that back
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
                for buffer in _persistent_buffers(model):
                    self.assertEqual(buffer.untyped_storage().size(), 0)
            # While unsharded, each persistent param is backed by a persistent buffer.
            buffer_ptrs = set()
            for bucket in _buckets(model):
                bucket.unshard()
                buffer_ptrs.update(
                    buffer.untyped_storage().data_ptr()
                    for buffer in bucket.persistent_buffers
                )
                for param in bucket.unsharded_params:
                    self.assertIn(param.untyped_storage().data_ptr(), buffer_ptrs)
                bucket.reshard()

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

    def test_unused_param_does_not_delay_reshard(self):
        # The input-grad trigger reduces and reshards once the module's backward
        # is done, although one of its params got no grad this step.
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 8), _FirstChildOnly(8))
            reference = copy.deepcopy(model).cuda()
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward=True),
            )
            bucket = _buckets(model)[1]
            unsharded_in_earlier_backward = []

            def probe(grad):
                unsharded_in_earlier_backward.append(bucket.is_unsharded)
                return grad

            def register_probe(module, args, output):
                output.register_hook(probe)

            model[0].register_forward_hook(register_probe)
            x = torch.randn(4, 8, device="cuda", requires_grad=True)
            model(x).sum().backward()
            reference(x).sum().backward()

            self.assertEqual(unsharded_in_earlier_backward, [False])
            self._assert_grads_match(model, reference)

    def test_bucket_hook_module(self):
        # A bucket of one module at a dotted path hooks that module; a bucket
        # spanning a ModuleList (no forward) hooks its nearest ancestor.
        cases = (
            ("per_layer", [["layers.0.*"], ["layers.1.*"]], lambda m: list(m.layers)),
            ("whole_list", [["layers.*"]], lambda m: [m]),
        )
        for name, patterns, expected_targets in cases:
            with self.subTest(buckets=name), single_rank_cuda_mesh() as mesh:
                torch.manual_seed(0)
                model = _LayerList(8)
                reference = copy.deepcopy(model).cuda()
                flex_shard(
                    model,
                    buckets=[
                        _bucket(p, mesh, reshard_after_forward=True) for p in patterns
                    ],
                )
                buckets = _buckets(model)
                self._assert_same_params(
                    [bucket.forward_hook_module() for bucket in buckets],
                    expected_targets(model),
                )
                x = torch.randn(4, 8, device="cuda")
                model(x).sum().backward()
                reference(x).sum().backward()
                self.assertTrue(all(b.unsharded_params is not None for b in buckets))
                self._assert_grads_match(model, reference)

    def test_inference_mode(self):
        # Persistent storage is never an inference tensor: repeated inference
        # forwards refill it, and training works after inference ran first.
        for steps in (
            ("infer", "infer", "train"),
            ("train", "infer", "infer", "train"),
        ):
            with self.subTest(steps=steps), single_rank_cuda_mesh() as mesh:
                torch.manual_seed(0)
                model = nn.Sequential(nn.Linear(8, 8))
                reference = copy.deepcopy(model).cuda()
                flex_shard(
                    model,
                    buckets=_per_child_buckets(model, mesh, reshard_after_forward=True),
                )
                x = torch.randn(4, 8, device="cuda")
                for step in steps:
                    if step == "infer":
                        with torch.inference_mode():
                            torch.testing.assert_close(model(x), reference(x))
                        continue
                    for param in (*model.parameters(), *reference.parameters()):
                        param.grad = None
                    model(x).sum().backward()
                    reference(x).sum().backward()
                    self._assert_grads_match(model, reference)

    def test_to_empty_after_hooks_installed(self):
        # to_empty() re-installs the local-shard params after the runtime
        # captured them; the runtime re-reads them.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward=True),
            )
            model.to_empty(device="cuda")
            torch.manual_seed(0)
            for param in model.parameters():
                nn.init.uniform_(param, -0.1, 0.1)
            # With one rank, each local shard is the full parameter.
            reference = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).cuda()
            with torch.no_grad():
                for ref_param, param in zip(
                    reference.parameters(), model.parameters(), strict=True
                ):
                    ref_param.copy_(param)
            x = torch.randn(4, 8, device="cuda")
            model(x).sum().backward()
            reference(x).sum().backward()
            self._assert_grads_match(model, reference)

    def test_forward_without_backward(self):
        # A reshard_after_forward=False bucket stays unsharded after a forward
        # until its backward: state_dict() reshards first, and reshard() does so
        # when that backward will not run.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8))
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward=False),
            )
            (bucket,) = _buckets(model)
            x = torch.randn(4, 8, device="cuda")
            model(x)
            self.assertTrue(bucket.is_unsharded)
            state_dict = model.state_dict(keep_vars=True)
            self.assertIs(state_dict["0.weight"], bucket.sharded_params[0])
            self.assertIs(state_dict["0.bias"], bucket.sharded_params[1])

            out = model(x)
            with torch.no_grad():
                for param in bucket.sharded_params:  # stands in for optimizer.step()
                    param.add_(1.0)
            model.reshard()
            self.assertFalse(bucket.is_unsharded)
            self.assertFalse(torch.equal(model(x), out))

    def test_freezing_after_flex_shard(self):
        # The unsharded params follow the local shards' requires_grad.
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 8))
            reference = copy.deepcopy(model).cuda()
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward=True),
            )
            x = torch.randn(4, 8, device="cuda")
            for frozen in (True, False):
                model[0].bias.requires_grad_(not frozen)
                reference[0].bias.requires_grad_(not frozen)
                for param in (*model.parameters(), *reference.parameters()):
                    param.grad = None
                model(x).sum().backward()
                reference(x).sum().backward()
                self.assertEqual(model[0].bias.grad is None, frozen)
                self._assert_grads_match(model, reference)

    def test_inplace_on_view_output(self):
        # An in-place op on a view output replaces the view's autograd node; the
        # pre-backward hook sits on the view's base, so it still re-gathers.
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(_ViewOutput(8))
            reference = copy.deepcopy(model).cuda()
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward=True),
            )
            x = torch.randn(4, 8, device="cuda", requires_grad=True)
            ref_x = x.detach().clone().requires_grad_()
            for inp, module in ((x, model), (ref_x, reference)):
                out = module(inp)
                out.mul_(2)
                out.sum().backward()
            torch.testing.assert_close(x.grad, ref_x.grad)
            self._assert_grads_match(model, reference)

    def test_recompute_consumes_backward_prefetch(self):
        # Recompute of a bucket nested in a checkpointed block runs before that
        # bucket's pre-backward hook and consumes its backward prefetch.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(checkpoint_wrapper(_TwoStageBlock(8)))
            prefix = "0._checkpoint_wrapped_module"
            flex_shard(
                model,
                buckets=[
                    _bucket([f"{prefix}.{stage}.*"], mesh, reshard_after_forward=True)
                    for stage in ("first", "second")
                ],
            )
            begin_unshard = BucketRuntime.begin_unshard
            x = torch.randn(4, 8, device="cuda", requires_grad=True)
            with patch.object(
                BucketRuntime, "begin_unshard", autospec=True, side_effect=begin_unshard
            ) as unshards:
                model(x).sum().backward()
            # One unshard per bucket in forward, one re-gather in backward.
            self.assertEqual(unshards.call_count, 4)

    def test_prefetch_follows_execution_order(self):
        # BucketSpec order need not match execution order: prefetch follows the
        # order learned from the first forward, so none is wasted.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(*(nn.Linear(8, 8) for _ in range(3)))
            flex_shard(
                model,
                buckets=[
                    _bucket([f"{idx}.*"], mesh, reshard_after_forward=True)
                    for idx in (2, 1, 0)
                ],
            )
            x = torch.randn(4, 8, device="cuda", requires_grad=True)
            model(x).sum().backward()
            begin_unshard = BucketRuntime.begin_unshard
            with patch.object(
                BucketRuntime, "begin_unshard", autospec=True, side_effect=begin_unshard
            ) as unshards:
                model(x).sum().backward()
            self.assertEqual(unshards.call_count, 6)
            self.assertIsNone(_buckets(model)[0].context.pending_unshard)

    def test_earlier_pre_forward_hook_sees_unsharded_param(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8))
            seen = []
            model[0].register_forward_pre_hook(
                lambda module, args: seen.append(module.weight)
            )
            flex_shard(
                model,
                buckets=_per_child_buckets(model, mesh, reshard_after_forward=True),
            )
            model(torch.randn(4, 8, device="cuda"))
            (bucket,) = _buckets(model)
            self.assertIs(seen[0], bucket.unsharded_params[0])

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
