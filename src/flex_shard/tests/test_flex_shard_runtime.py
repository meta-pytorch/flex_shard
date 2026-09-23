# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import Mock, patch

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import is_flex_shard_param
from ..flex_shard import bucket_runtime, unsharded_param_getters
from .common import (
    flex_shard_cuda,
    flex_shard_transformer_model,
    make_transformer_model,
    single_rank_cuda_mesh,
    transformer_inputs,
)


class TestFlexShardEagerRuntime(TestCase):
    def test_local_shard_shares_parameter_version_counter(self):
        module = nn.Linear(2, 2, bias=False)
        runtime = bucket_runtime.BucketRuntime(
            bucket_storage=Mock(),
            bucket_params=[
                bucket_runtime.BucketParam(
                    param_owner=bucket_runtime.ParamOwnerRef(module, "weight"),
                    unsharded_param_slot=unsharded_param_getters.UnshardedParamSlot(
                        param_fqn="weight",
                        bucket_fqn=None,
                    ),
                    param_info=Mock(),
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

    def test_raf_saved_tensor_registry_ignores_reused_python_id(self):
        context = unsharded_param_getters._RafSavedTensorContext()
        registered_tensor = torch.ones(1)
        handle = object()

        with patch.object(unsharded_param_getters, "id", return_value=1, create=True):
            context.register(registered_tensor, handle)
            del registered_tensor
            unrelated_tensor = torch.ones(1)
            self.assertIs(context.pack(unrelated_tensor), unrelated_tensor)

    def test_meta_to_empty_materializes_bucket_storage_and_runtime(self):
        with single_rank_cuda_mesh() as mesh:
            with torch.device("meta"):
                args, model = make_transformer_model()

            flex_shard_cuda(model, mesh)
            for storage in model.sharded_bucket_storages:
                self.assertEqual(storage.byte_storage.device.type, "meta")

            model.to_empty(device="cuda")
            for storage in model.sharded_bucket_storages:
                self.assertEqual(storage.byte_storage.device.type, "cuda")
            for param in model.parameters():
                self.assertTrue(is_flex_shard_param(param))
                nn.init.uniform_(param, -0.1, 0.1)

            loss = model(transformer_inputs(args, device="cuda")).sum()
            loss.backward()

            for param in model.parameters():
                self.assertIsNotNone(param.grad)

    def test_torch_compile_forward_backward_on_cuda_mesh(self):
        with single_rank_cuda_mesh() as mesh:
            args, model = flex_shard_transformer_model(mesh)

            compiled_model = torch.compile(model, backend="eager", fullgraph=True)

            loss = compiled_model(transformer_inputs(args, device="cuda")).sum()
            loss.backward()

            for param in model.parameters():
                self.assertIsNotNone(param.grad)

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


if __name__ == "__main__":
    run_tests()
