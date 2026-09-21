# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train TinyMoETransformer with AdamW, optionally inspecting the trace."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from flex_shard import BucketSpec, flex_shard
from flex_shard.custom_placements.shard import per_param_placements
from tiny_moe_transformer import TinyMoETransformer


def build_buckets(model, dp_mesh, efsdp_mesh):
    def bucket(patterns, mesh):
        return BucketSpec(
            patterns,
            mesh=mesh,
            placement_fn=per_param_placements,
            reshard_after_forward=False,
        )

    buckets = [
        bucket([pattern], dp_mesh)
        for pattern in ("tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*")
    ]
    for index in range(len(model.layers)):
        buckets.extend(
            [
                bucket(
                    [
                        f"layers.{index}.{part}.*"
                        for part in ("self_attn", "norm1", "norm2", "router")
                    ],
                    dp_mesh,
                ),
                bucket([f"layers.{index}.experts.*"], efsdp_mesh),
            ]
        )
    return buckets


def inspect_trace(graphs, bucket_count, output_dir, rank):
    if len(graphs) != 1:
        raise AssertionError(f"Expected one Dynamo graph, got {len(graphs)}")
    root = graphs[0]
    subgraphs = {
        name or "root": module
        for name, module in root.named_modules()
        if isinstance(module, torch.fx.GraphModule)
    }
    targets = [
        str(node.target) for module in subgraphs.values() for node in module.graph.nodes
    ]
    counts = {
        name: targets.count(f"_c10d_functional.{name}")
        for name in ("all_gather_into_tensor", "reduce_scatter_tensor", "wait_tensor")
    }
    unshards = sum(
        str(node.target) == "autograd_function_apply" for node in root.graph.nodes
    )
    assert unshards == bucket_count, (unshards, bucket_count)
    assert counts["all_gather_into_tensor"] == bucket_count, counts
    assert counts["reduce_scatter_tensor"] == bucket_count, counts
    assert counts["wait_tensor"] > 0, counts
    destination = Path(output_dir) / f"rank{rank}"
    destination.mkdir(parents=True, exist_ok=True)
    for name, module in subgraphs.items():
        (destination / f"{name}.py").write_text(module.code)
    summary = {"dynamo_graphs": len(graphs), "bucket_unshards": unshards, **counts}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"rank={rank} trace={summary} output={destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace", action="store_true", help="Capture and inspect the graph"
    )
    parser.add_argument("--graph-dir", default="trace_output")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        if world_size != 4:
            raise ValueError("Run this example with four GPU processes")
        device = torch.device("cuda", local_rank)
        dp_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        grid = init_device_mesh("cuda", (2, 2), mesh_dim_names=("efsdp", "replica"))
        efsdp_mesh = grid["efsdp"]

        torch.manual_seed(42)
        model = TinyMoETransformer().to(device).train()
        buckets = build_buckets(model, dp_mesh, efsdp_mesh)
        flex_shard(model, buckets=buckets)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False
        )

        graphs = []

        def capture_backend(graph_module, example_inputs):
            graphs.append(graph_module)
            return graph_module.forward

        run_model = (
            torch.compile(model, backend=capture_backend, fullgraph=True)
            if args.trace
            else model
        )
        # MoE replicas share a batch; the two efsdp coordinates get different batches.
        data_rank = efsdp_mesh.get_local_rank()
        rng = torch.Generator(device=device).manual_seed(1234 + data_rank)
        for step in range(3):
            tokens = torch.randint(32, (2, 9), generator=rng, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = run_model(tokens[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(f"step={step} rank0_loss={loss.item():.4f}", flush=True)
        if args.trace:
            inspect_trace(graphs, len(buckets), args.graph_dir, rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
