# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""No-sync gradient accumulation and reshard-after-backward (persistent params)."""

import copy
from unittest import mock

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import BucketSpec, flex_shard, MixedPrecisionPolicy
from ..custom_placements.shard import per_param_placements
from ..flex_shard.bucket_runtime import BucketRuntime
from .common import single_rank_cuda_mesh


def _shard_per_child(model, mesh, **bucket_kwargs):
    flex_shard(
        model,
        buckets=[
            BucketSpec(
                [f"{name}.*"],
                placement_fn=per_param_placements,
                mesh=mesh,
                **bucket_kwargs,
            )
            for name, _ in model.named_children()
            if any(True for _ in _.parameters())
        ],
    )


def _persistent_params(model):
    return [
        bucket_param.unsharded_param
        for context in model._flex_shard_eager_comm_contexts.values()
        for bucket in context.buckets
        for bucket_param in bucket.bucket_params
    ]


class _Counters:
    """Count bucket unshards and reduce-grads."""

    def __init__(self):
        self.unshards = 0
        self.reduces = 0

    def patch(self):
        original_unshard = BucketRuntime.unshard_into_persistent
        original_reduce = BucketRuntime.schedule_reduce_grad_tensors
        counters = self

        def unshard(bucket, result):
            counters.unshards += 1
            return original_unshard(bucket, result)

        def reduce(bucket, *args):
            counters.reduces += 1
            return original_reduce(bucket, *args)

        return (
            mock.patch.object(BucketRuntime, "unshard_into_persistent", unshard),
            mock.patch.object(BucketRuntime, "schedule_reduce_grad_tensors", reduce),
        )


class _TwoBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x, use_b: bool):
        out = self.a(x)
        if use_b:
            out = out + self.b(x)
        return out


class TestFlexShardNoSync(TestCase):
    def _assert_grads_match(self, model, reference, **tol):
        for param, ref_param in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(param.grad, ref_param.grad, **tol)

    def test_no_sync_matches_accumulated_reference(self):
        for reshard_after_forward in (False, True):
            for keep_params in (False, True):
                with self.subTest(
                    reshard_after_forward=reshard_after_forward,
                    keep_params=keep_params,
                ):
                    self._check_no_sync(reshard_after_forward, keep_params)

    def _check_no_sync(self, reshard_after_forward, keep_params):
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
            reference = copy.deepcopy(model).cuda()
            _shard_per_child(model, mesh, reshard_after_forward=reshard_after_forward)
            num_buckets = len(model.sharded_bucket_storages)
            counters = _Counters()
            inputs = [torch.randn(4, 8, device="cuda") for _ in range(3)]

            # Kept-unsharded buckets expose their persistent params through
            # model.parameters() between microbatches, so capture the shards.
            shards = list(model.parameters())
            patch_unshard, patch_reduce = counters.patch()
            with patch_unshard, patch_reduce:
                model.set_reshard_after_backward(not keep_params)
                with model.no_sync():
                    for x in inputs[:2]:
                        model(x).sum().backward()
                        for shard in shards:
                            self.assertIsNone(shard.grad)
                model.set_reshard_after_backward(True)
                model(inputs[2]).sum().backward()
            for x in inputs:
                reference(x).sum().backward()

            self._assert_grads_match(model, reference)
            # One reduce-scatter per bucket per optimizer step.
            self.assertEqual(counters.reduces, num_buckets)
            if keep_params and not reshard_after_forward:
                # Params stay unsharded across microbatches: one all-gather
                # per bucket per step.
                self.assertEqual(counters.unshards, num_buckets)
            for unsharded in _persistent_params(model):
                self.assertEqual(unsharded.untyped_storage().size(), 0)
                self.assertIsNone(unsharded.grad)

    def test_kept_params_regather_after_shard_update(self):
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
            reference = copy.deepcopy(model).cuda()
            _shard_per_child(model, mesh, reshard_after_forward=False)
            model.set_reshard_after_backward(False)
            optim = torch.optim.SGD(model.parameters(), lr=0.1)
            ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
            counters = _Counters()
            patch_unshard, patch_reduce = counters.patch()
            with patch_unshard, patch_reduce:
                for step in range(3):
                    x = torch.randn(4, 8, device="cuda")
                    unshards_before = counters.unshards
                    loss = model(x).sum()
                    ref_loss = reference(x).sum()
                    torch.testing.assert_close(loss, ref_loss)
                    if step > 0:
                        # The optimizer updated the shards in place, so the
                        # kept unsharded params were re-gathered.
                        self.assertEqual(counters.unshards - unshards_before, 2)
                    loss.backward()
                    ref_loss.backward()
                    optim.step()
                    ref_optim.step()
                    optim.zero_grad()
                    ref_optim.zero_grad()

    def test_unused_param_in_final_microbatch(self):
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = _TwoBranch()
            reference = copy.deepcopy(model).cuda()
            _shard_per_child(model, mesh, reshard_after_forward=False)
            x0, x1 = (torch.randn(4, 8, device="cuda") for _ in range(2))
            with model.no_sync():
                model(x0, use_b=True).sum().backward()
            model(x1, use_b=False).sum().backward()
            reference(x0, use_b=True).sum().backward()
            reference(x1, use_b=False).sum().backward()
            self._assert_grads_match(model, reference)

    def test_no_sync_accumulates_in_reduce_dtype(self):
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).to(torch.bfloat16)
            reference = copy.deepcopy(model).float().cuda()
            _shard_per_child(
                model,
                mesh,
                reshard_after_forward=False,
                mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
            )
            inputs = [torch.randn(4, 8, device="cuda") for _ in range(3)]
            with model.no_sync():
                for x in inputs[:2]:
                    model(x.bfloat16()).sum().backward()
                for unsharded in _persistent_params(model):
                    self.assertEqual(unsharded.grad.dtype, torch.float32)
            model(inputs[2].bfloat16()).sum().backward()
            for unsharded in _persistent_params(model):
                self.assertIsNone(unsharded.grad)
                self.assertIsNone(unsharded.grad_dtype)
            for x in inputs:
                reference(x).sum().backward()
            for param, ref_param in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(
                    param.grad.float(), ref_param.grad, atol=5e-2, rtol=5e-2
                )

    def test_no_sync_restores_flags(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8))
            _shard_per_child(model, mesh, reshard_after_forward=False)
            storage = model.sharded_bucket_storages[0]
            with self.assertRaises(ValueError):
                with model.no_sync():
                    self.assertFalse(storage._requires_gradient_sync)
                    raise ValueError
            self.assertTrue(storage._requires_gradient_sync)

    def test_no_sync_rejects_legacy_buckets(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8))
            _shard_per_child(
                model,
                mesh,
                reshard_after_forward=False,
                persistent_unsharded_params=False,
            )
            with self.assertRaisesRegex(NotImplementedError, "legacy path"):
                model.set_requires_gradient_sync(False)

    def test_no_sync_rejects_compile(self):
        torch._dynamo.reset()
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(nn.Linear(8, 8))
            _shard_per_child(model, mesh, reshard_after_forward=False)
            compiled = torch.compile(model, backend="eager", fullgraph=True)
            with model.no_sync():
                with self.assertRaisesRegex(Exception, "eager-only"):
                    compiled(torch.randn(4, 8, device="cuda"))


if __name__ == "__main__":
    run_tests()
