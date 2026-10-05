# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import warnings
from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import BucketSpec, flex_shard, is_flex_shard_param
from ..custom_placements.block_shard import (
    BucketedBlockShard,
    make_bucketed_block_placement_fn,
)
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..custom_placements.shard import per_param_placements
from ..flex_shard import bucket_runtime
from .common import (
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cuda_mesh,
    transformer_inputs,
)


class _TEStyleScaleFn(torch.autograd.Function):
    """Like TransformerEngine: ``module.parameters()`` are the autograd inputs,
    and backward reads ``module.weight`` instead of saving it."""

    @staticmethod
    def forward(ctx, x, module, *params):
        assert params[0] is module.weight
        ctx.save_for_backward(x)
        ctx.module = module
        ctx.param = params[0]
        return x * module.weight

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        weight = ctx.module.weight
        # With one rank the local shard equals the full param, so check identity.
        assert weight is ctx.param and weight.untyped_storage().size() > 0
        return grad_output * weight, None, (grad_output * x).sum(0)


@dataclasses.dataclass
class _Output:
    value: torch.Tensor


class _TEStyleScale(nn.Module):
    def __init__(self, dim: int, dataclass_output: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))
        self.dataclass_output = dataclass_output

    def forward(self, x: torch.Tensor) -> torch.Tensor | _Output:
        out = _TEStyleScaleFn.apply(x, self, *self.parameters())
        return _Output(out) if self.dataclass_output else out


class _ViewHead(nn.Module):
    """Returns a view, and leaves one linear unused like an unrouted expert."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.used = nn.Linear(dim, dim)
        self.unused = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.used(x).view(-1)


class _PersistentParamsNet(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_TEStyleScale(dim, dataclass_output=True)]
            + [_TEStyleScale(dim) for _ in range(2)]
        )
        self.head = _ViewHead(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x).value
        x = self.layers[2](self.layers[1](x))
        # The head runs in two parallel branches, and the later branch's view
        # output is modified in place.
        out = self.head(x)
        later = self.head(2 * x)
        later.mul_(2)
        return out + later


class _TwoStageBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.first = nn.Linear(dim, dim)
        self.second = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


class _SkipNet(nn.Module):
    """Three linears; ``skip`` leaves the middle one out of the forward."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(3))
        self.skip = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0](x)
        if not self.skip:
            x = self.layers[1](x)
        return self.layers[2](x)


class _TiedNet(nn.Module):
    """The head shares the embedding's weight, so one bucket holds both names."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, dim, bias=False)
        self.mid = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, dim, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.mid(self.embed(x))))


class _ReentrantCheckpointNet(nn.Module):
    """A block under reentrant activation checkpointing: its forward runs
    without grad, and backward recomputes it, as Megatron's recompute does."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = _TwoStageBlock(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.utils.checkpoint.checkpoint(self.block, x, use_reentrant=True)
        return self.head(x)


def _bucket(patterns, mesh, reshard_after_forward, placement_fn=per_param_placements):
    return BucketSpec(
        patterns,
        placement_fn=placement_fn,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
    )


def _buckets(model: nn.Module):
    return [
        bucket
        for context in model._flex_shard_eager_comm_contexts.values()
        for bucket in context.buckets
    ]


class TestFlexShardEagerRuntime(TestCase):
    def test_local_shard_shares_parameter_version_counter(self):
        module = nn.Linear(2, 2, bias=False)
        runtime = bucket_runtime.BucketRuntime(
            bucket_storage=Mock(),
            bucket_params=[
                bucket_runtime.BucketParam(
                    param_owner=bucket_runtime.ParamOwnerRef(module, "weight"),
                    param_info=Mock(),
                    sharded_param=module.weight,
                )
            ],
            context=Mock(),
            debug_fqn=None,
        )

        (local_shard,) = runtime._local_shards(use_autograd=False)
        original_version = local_shard._version
        with torch.no_grad():
            module.weight.add_(1)

        self.assertFalse(local_shard.requires_grad)
        self.assertEqual(local_shard._version, original_version + 1)

    def test_meta_to_empty_materializes_bucket_storage_and_runtime(self):
        with single_rank_cuda_mesh() as mesh:
            with torch.device("meta"):
                args, model = make_transformer_model()
            model.output.weight.grad_dtype = torch.bfloat16

            flex_shard_cuda(model, mesh)
            for storage in model.sharded_bucket_storages:
                self.assertEqual(storage.byte_storage.device.type, "meta")

            # The second to_empty() replaces the params the runtime captured.
            for _ in range(2):
                model.to_empty(device="cuda")
                for storage in model.sharded_bucket_storages:
                    self.assertEqual(storage.byte_storage.device.type, "cuda")
                for param in model.parameters():
                    self.assertTrue(is_flex_shard_param(param))
                    nn.init.uniform_(param, -0.1, 0.1)
                    param.grad = None
                output_weight = model.output._parameters["weight"]
                self.assertEqual(output_weight.grad_dtype, torch.bfloat16)

                loss = model(transformer_inputs(args, device="cuda")).sum()
                loss.backward()

                for param in model.parameters():
                    self.assertIsNotNone(param.grad)
                self.assertEqual(output_weight.grad.dtype, torch.bfloat16)

    def test_meta_to_empty_keeps_tied_weights_shared(self):
        # to_empty() from meta gives each slot of a shared parameter its own new
        # parameter; FlexShard re-installs one sharded parameter in every slot.
        with single_rank_cuda_mesh() as mesh:
            with torch.device("meta"):
                args, model = make_transformer_model(weight_tying=True)
            flex_shard_cuda(model, mesh)
            for _ in range(2):
                model.to_empty(device="cuda")
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                for param in model.parameters():
                    nn.init.uniform_(param, -0.1, 0.1)
                    param.grad = None
                model(transformer_inputs(args, device="cuda")).sum().backward()
                self.assertIs(model.output.weight, model.tok_embeddings.weight)
                self.assertIsNotNone(model.tok_embeddings.weight.grad)

    def test_torch_compile_tied_weights(self):
        # Under compile, _BucketUnshard's outputs swap into every slot of a
        # shared parameter, so the tied output layer reads the gathered weight.
        # The tied bucket is hooked on the root, around the per-layer buckets.
        with single_rank_cuda_mesh() as mesh:
            torch.manual_seed(0)
            args, model = make_transformer_model(
                device="cuda", n_layers=2, weight_tying=True
            )
            reference = copy.deepcopy(model)
            spec = dict(
                placement_fn=per_param_placements,
                mesh=mesh,
                reshard_after_forward=False,
            )
            flex_shard_cuda(
                model,
                mesh,
                buckets=[
                    BucketSpec(
                        [
                            "tok_embeddings.*",
                            "pos_embeddings.*",
                            "norm.*",
                            "output.*",
                        ],
                        **spec,
                    ),
                    *(
                        BucketSpec([f"layers.{idx}.*"], **spec)
                        for idx in range(args.n_layers)
                    ),
                ],
            )
            compiled_model = torch.compile(model, backend="eager", fullgraph=True)
            x = transformer_inputs(args, device="cuda")
            loss = compiled_model(x).sum()
            ref_loss = reference(x).sum()
            self.assertEqual(loss, ref_loss)
            loss.backward()
            ref_loss.backward()
            self.assertIs(model.output.weight, model.tok_embeddings.weight)
            self.assertEqual(
                model.tok_embeddings.weight.grad, reference.tok_embeddings.weight.grad
            )

    def test_persistent_unsharded_params(self):
        for reshard_after_forward in (False, True):
            with (
                self.subTest(reshard_after_forward=reshard_after_forward),
                single_rank_cuda_mesh() as mesh,
            ):
                self._check_persistent_unsharded_params(mesh, reshard_after_forward)

    def _check_persistent_unsharded_params(self, mesh, reshard_after_forward):
        torch.manual_seed(0)
        model = _PersistentParamsNet(8)
        reference = copy.deepcopy(model).cuda()
        model.layers[0].weight.custom_tag = "kept"
        seen_weights = []
        # Registered before flex_shard(), whose pre-forward hook still runs first.
        model.layers[0].register_forward_pre_hook(
            lambda module, args: seen_weights.append(module.weight)
        )
        mixed = MixedBucketPlacement({})

        def head_placements(named_params, mesh):
            return {
                fqn: (
                    # BlockShard refills with torch.cat(out=), which bumps
                    # version counters of weights autograd saved.
                    mixed.block_shard(blocks_per_rank=(1,))
                    if fqn.endswith("weight")
                    else mixed.shard0,
                )
                for fqn, _ in named_params
            }

        flex_shard(
            model,
            buckets=[
                # One module at a dotted path, and a bucket spanning a
                # ModuleList, which has no forward of its own.
                _bucket(["layers.0.*"], mesh, reshard_after_forward),
                _bucket(
                    ["layers.1.*", "layers.2.*"],
                    mesh,
                    reshard_after_forward,
                    make_bucketed_block_placement_fn(dims=(0,), blocks_per_rank=(1,)),
                ),
                _bucket(["head.*"], mesh, reshard_after_forward, head_placements),
            ],
        )
        buckets = _buckets(model)
        for bucket, hook_module in zip(
            buckets, (model.layers[0], model, model.head), strict=True
        ):
            self.assertIs(bucket.forward_hook_module(), hook_module)
        head_resharded = []

        def probe(module, args, output):
            # Runs after the head's and layers[2]'s backward, before layers[1]'s.
            if output.requires_grad:
                output.register_hook(
                    lambda grad: head_resharded.append(not buckets[2].is_unsharded)
                )

        model.layers[1].register_forward_hook(probe)

        x = torch.randn(4, 8, device="cuda")
        with torch.inference_mode():  # the first unshard runs in inference mode
            model(x)
        persistent = [param for bucket in buckets for param in bucket.unsharded_params]
        buffers = [buffer for bucket in buckets for buffer in bucket.persistent_buffers]
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
        for step in range(3):
            if step == 1:
                # Freezing a local shard after flex_shard() takes effect. The
                # frozen weight is read in backward after its bucket's other
                # grads have landed.
                model.layers[1].weight.requires_grad_(False)
                reference.layers[1].weight.requires_grad_(False)
            optim.zero_grad()
            ref_optim.zero_grad()
            loss = model(x).sum()
            self.assertTrue(
                all(
                    (buffer.untyped_storage().size() == 0) == reshard_after_forward
                    for buffer in buffers
                )
            )
            loss.backward()
            reference(x).sum().backward()
            for param, ref_param in zip(
                model.parameters(), reference.parameters(), strict=True
            ):
                # Trainable params unused in forward get zero grads, as on main.
                ref_grad = ref_param.grad
                if ref_grad is None and ref_param.requires_grad:
                    ref_grad = torch.zeros_like(ref_param)
                torch.testing.assert_close(param.grad, ref_grad)
            optim.step()
            ref_optim.step()
            # The same persistent params, their storage freed after backward.
            self.assertTrue(
                all(
                    a is b
                    for a, b in zip(
                        (p for bucket in buckets for p in bucket.unsharded_params),
                        persistent,
                        strict=True,
                    )
                )
            )
            self.assertTrue(all(b.untyped_storage().size() == 0 for b in buffers))
        self.assertEqual(head_resharded, [True] * 3)
        self.assertTrue(all(weight is persistent[0] for weight in seen_weights))
        self.assertEqual(persistent[0].custom_tag, "kept")
        with torch.inference_mode():
            torch.testing.assert_close(model(x), reference(x))
        model(x)  # a forward whose backward will not run
        model.reshard()
        self.assertTrue(all(b.untyped_storage().size() == 0 for b in buffers))

        # FlexShard does not recover from errors: after a backward that raises
        # partway, the next forward and reshard() raise too.
        def fail(grad):
            raise RuntimeError("injected backward failure")

        model.layers[1].register_forward_hook(
            lambda module, args, output: output.register_hook(fail) and None
        )
        with self.assertRaisesRegex(RuntimeError, "injected backward failure"):
            model(x).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "a previous backward raised"):
            model(x)
        with self.assertRaisesRegex(RuntimeError, "a previous backward raised"):
            model.reshard()

    def test_reshard_after_forward_holds_one_bucket(self):
        # Each bucket is freed before the next one is re-gathered, in forward
        # and in backward, as in FSDP2.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(*(nn.Linear(8, 8) for _ in range(3)))
            flex_shard(
                model,
                buckets=[
                    _bucket([f"{idx}.*"], mesh, reshard_after_forward=True)
                    for idx in range(3)
                ],
            )
            buckets = _buckets(model)
            others_unsharded = []
            finish_unshard = bucket_runtime.BucketRuntime.finish_unshard

            def counting_finish_unshard(bucket, result):
                others_unsharded.append(
                    sum(b.is_unsharded for b in buckets if b is not bucket)
                )
                return finish_unshard(bucket, result)

            with patch.object(
                bucket_runtime.BucketRuntime, "finish_unshard", counting_finish_unshard
            ):
                model(torch.randn(4, 8, device="cuda")).sum().backward()
            self.assertEqual(others_unsharded, [0] * 6)

    def test_prefetch_follows_learned_order(self):
        # BucketSpec lists the buckets in reverse execution order. From the
        # second step, prefetch follows the order learned in the first, and the
        # recompute inside backward consumes the checkpointed block's backward
        # prefetch: one unshard per bucket in forward and backward, none wasted.
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(
                nn.Linear(8, 8), checkpoint_wrapper(_TwoStageBlock(8))
            )
            block = "1._checkpoint_wrapped_module"
            flex_shard(
                model,
                buckets=[
                    _bucket([pattern], mesh, reshard_after_forward=True)
                    for pattern in (f"{block}.second.*", f"{block}.first.*", "0.*")
                ],
            )
            x = torch.randn(4, 8, device="cuda")
            model(x).sum().backward()
            begin_unshard = bucket_runtime.BucketRuntime.begin_unshard
            take = bucket_runtime.BucketCommContext.take_pending_unshard
            hits = []

            def counting_take(context, bucket):
                result = take(context, bucket)
                hits.append(result is not None)
                return result

            with (
                patch.object(
                    bucket_runtime.BucketRuntime,
                    "begin_unshard",
                    autospec=True,
                    side_effect=begin_unshard,
                ) as unshards,
                patch.object(
                    bucket_runtime.BucketCommContext,
                    "take_pending_unshard",
                    counting_take,
                ),
            ):
                model(x).sum().backward()
            self.assertEqual(unshards.call_count, 6)
            # All but the first unshard of forward and of backward were prefetched.
            self.assertEqual(sum(hits), 4)

    def test_released_in_place_refill_frees_storage(self):
        # A prefetch that refills a bucket's persistent storage in place and is
        # then released unused frees that storage, and later steps still match.
        with single_rank_cuda_mesh() as mesh:
            model = _SkipNet(8)
            reference = copy.deepcopy(model).cuda()
            placement_fn = make_bucketed_block_placement_fn(
                dims=(0,), blocks_per_rank=(1,)
            )
            flex_shard(
                model,
                buckets=[
                    _bucket([f"layers.{idx}.*"], mesh, True, placement_fn)
                    for idx in range(3)
                ],
            )
            buckets = _buckets(model)
            optim = torch.optim.SGD(model.parameters(), lr=0.1)
            ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
            x = torch.randn(4, 8, device="cuda")
            release = bucket_runtime.BucketRuntime.free_persistent_storage
            released = []

            def recording_release(bucket):
                released.append(bucket)
                return release(bucket)

            for skip in (False, True, False):
                model.skip = reference.skip = skip
                optim.zero_grad()
                ref_optim.zero_grad()
                with patch.object(
                    bucket_runtime.BucketRuntime,
                    "free_persistent_storage",
                    recording_release,
                ):
                    output = model(x)
                    reference_output = reference(x)
                    torch.testing.assert_close(output, reference_output)
                    output.sum().backward()
                reference_output.sum().backward()
                optim.step()
                ref_optim.step()
                self.assertTrue(
                    all(
                        buffer.untyped_storage().size() == 0
                        for bucket in buckets
                        for buffer in bucket.persistent_buffers
                    )
                )
            # Skipping layers.1, layers.0's prefetch of it is released.
            self.assertIn(buckets[1], released)

    def test_gradient_bucket_matches_reference(self):
        # The reduce-scatter reads the grads' gradient bucket as is, and grads
        # match an unsharded reference: with a param or a whole bucket frozen
        # after the first step, a shared (tied) param, and a reentrant activation
        # checkpoint,
        # whose forward runs without grad so only its recompute allocates the
        # gradient bucket.
        placement_fn = make_bucketed_block_placement_fn(dims=(0,), blocks_per_rank=(1,))
        cases = {
            "plain": (lambda: nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)), ["*"]),
            "frozen": (lambda: nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)), ["*"]),
            # A whole bucket frozen: its post-backward reduces no grads.
            "frozen_bucket": (
                lambda: nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)),
                ["0.*", "1.*"],
            ),
            "tied": (lambda: _TiedNet(8), ["*"]),
            "reentrant_checkpoint": (
                lambda: _ReentrantCheckpointNet(8),
                ["block.*", "head.*"],
            ),
        }
        gradient_bucket_of = BucketedBlockShard._gradient_bucket_of
        with single_rank_cuda_mesh() as mesh:
            for case, (make_model, patterns) in cases.items():
                for reshard_after_forward in (False, True):
                    with self.subTest(
                        case=case, reshard_after_forward=reshard_after_forward
                    ):
                        torch.manual_seed(0)
                        model = make_model()
                        reference = copy.deepcopy(model).cuda()
                        flex_shard(
                            model,
                            buckets=[
                                BucketSpec(
                                    [pattern],
                                    placement_fn=placement_fn,
                                    mesh=mesh,
                                    reshard_after_forward=reshard_after_forward,
                                    gradient_bucket=True,
                                )
                                for pattern in patterns
                            ],
                        )
                        optim = torch.optim.SGD(model.parameters(), lr=0.1)
                        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
                        reduced_bucket = []

                        def recording(placement, *args):
                            bucket = gradient_bucket_of(placement, *args)
                            reduced_bucket.append(bucket is not None)
                            return bucket

                        with (
                            patch.object(
                                BucketedBlockShard, "_gradient_bucket_of", recording
                            ),
                            warnings.catch_warnings(),
                        ):
                            warnings.simplefilter("error", UserWarning)
                            for step in range(3):
                                if case == "frozen" and step == 1:
                                    model[0].weight.requires_grad_(False)
                                    reference[0].weight.requires_grad_(False)
                                if case == "frozen_bucket" and step == 1:
                                    for param in (
                                        *model[0].parameters(),
                                        *reference[0].parameters(),
                                    ):
                                        param.requires_grad_(False)
                                optim.zero_grad()
                                ref_optim.zero_grad()
                                x = torch.randn(4, 8, device="cuda", requires_grad=True)
                                output = model(x)
                                reference_output = reference(x)
                                torch.testing.assert_close(output, reference_output)
                                output.square().sum().backward()
                                reference_output.square().sum().backward()
                                for param, ref_param in zip(
                                    model.parameters(),
                                    reference.parameters(),
                                    strict=True,
                                ):
                                    torch.testing.assert_close(
                                        param.grad, ref_param.grad
                                    )
                                optim.step()
                                ref_optim.step()
                        self.assertTrue(reduced_bucket)
                        self.assertTrue(all(reduced_bucket), reduced_bucket)

    def test_gradient_bucket_fallback_warns(self):
        # Grads replaced after the pre-backward hook, or a placement without
        # gradient bucket views, fall back to copying them in, once warned.
        with single_rank_cuda_mesh() as mesh:
            for case in ("replaced", "shard"):
                with self.subTest(case=case):
                    torch.manual_seed(0)
                    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
                    reference = copy.deepcopy(model).cuda()

                    def replace_grads(named_params):
                        for _, param in named_params:
                            param.grad = torch.zeros_like(param)

                    placement_fn = (
                        per_param_placements
                        if case == "shard"
                        else make_bucketed_block_placement_fn(
                            dims=(0,), blocks_per_rank=(1,)
                        )
                    )
                    flex_shard(
                        model,
                        buckets=[
                            BucketSpec(
                                ["*"],
                                placement_fn=placement_fn,
                                mesh=mesh,
                                reshard_after_forward=False,
                                gradient_bucket=True,
                                pre_backward_hook=(
                                    replace_grads if case == "replaced" else None
                                ),
                            )
                        ],
                    )
                    x = torch.randn(4, 8, device="cuda")
                    with self.assertWarnsRegex(UserWarning, "gradient_bucket=True"):
                        model(x).sum().backward()
                    reference(x).sum().backward()
                    for param, ref_param in zip(
                        model.parameters(), reference.parameters(), strict=True
                    ):
                        torch.testing.assert_close(param.grad, ref_param.grad)

    def test_torch_compile_forward_backward_on_cuda_mesh(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            compiled_model = torch.compile(model, backend="eager", fullgraph=True)

            loss = compiled_model(transformer_inputs(args, device="cuda")).sum()
            loss.backward()

            for param in model.parameters():
                self.assertIsNotNone(param.grad)

            model.set_requires_gradient_sync(False)
            with self.assertRaisesRegex(Exception, "eager-only"):
                compiled_model(transformer_inputs(args, device="cuda"))

    def test_torch_compile_traces_per_bucket_collectives(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            graphs = []

            def capture_backend(gm, example_inputs):
                graphs.append(gm)
                return gm.forward

            compiled_model = torch.compile(
                model, backend=capture_backend, fullgraph=True
            )
            compiled_model(transformer_inputs(args, device="cuda")).sum().backward()

            # fullgraph=True must produce exactly one graph with no breaks.
            self.assertEqual(1, len(graphs))
            root = graphs[0]
            top_targets = [str(node.target) for node in root.graph.nodes]

            # Dynamo traces into the bucket forward pre-hook rather than treating
            # it as opaque, so bucket member params are lifted as graph inputs.
            self.assertTrue(
                any("bucket_params" in target for target in top_targets),
                f"no bucket params lifted as graph inputs: {top_targets}",
            )

            # One unshard autograd function per bucket, not one for the whole model.
            unshards = [t for t in top_targets if t == "autograd_function_apply"]
            self.assertEqual(len(model.sharded_bucket_storages), len(unshards))

            # Collectives live in the fwd_body_*/bwd_body_* subgraphs that
            # autograd_function_apply dispatches to, not at the top level.
            subgraph_targets = set()
            for _, submodule in root.named_modules():
                if isinstance(submodule, torch.fx.GraphModule):
                    subgraph_targets.update(
                        str(node.target) for node in submodule.graph.nodes
                    )

            # Functional collectives keep the comm reorderable by graph passes.
            self.assertIn("_c10d_functional.all_gather_into_tensor", subgraph_targets)
            self.assertIn("_c10d_functional.reduce_scatter_tensor", subgraph_targets)
            self.assertIn("_c10d_functional.wait_tensor", subgraph_targets)

    def test_torch_compile_frozen_params_do_not_widen_reduce_dtype(self):
        with single_rank_cuda_mesh() as mesh:
            model = nn.Sequential(
                nn.Linear(8, 8, bias=False),
                nn.Linear(8, 8, bias=False),
            ).to(device="cuda", dtype=torch.bfloat16)
            # Explicit None stores the reduced grad in the reduce dtype.
            model[0].weight.grad_dtype = None
            model[1].weight.grad_dtype = torch.float32
            model[1].weight.requires_grad_(False)
            flex_shard_cuda(model, mesh)
            compiled_model = torch.compile(model, backend="eager", fullgraph=True)

            x = torch.ones(2, 8, dtype=torch.bfloat16, device="cuda")
            compiled_model(x).float().sum().backward()

            # Compile backward gets the frozen param's zero grad; like eager,
            # its fp32 grad_dtype does not widen the bucket's bf16.
            params = dict(model.named_parameters())
            self.assertIsNone(params["1.weight"].grad)
            self.assertEqual(params["0.weight"].grad.dtype, torch.bfloat16)


if __name__ == "__main__":
    run_tests()
