# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from flex_shard import BucketSpec, flex_shard
from flex_shard.custom_placements.mixed_bucket import MixedBucketPlacement
from flex_shard.custom_placements.owned import make_bucketed_owned_full_param_segments
from flex_shard.dist_muon import DistMuon
from tiny_transformer import TinyTransformer


MUON_KWARGS = dict(
    lr=0.01,
    weight_decay=0.01,
    momentum=0.95,
    nesterov=True,
    ns_coefficients=(3.4445, -4.7750, 2.0315),
    ns_steps=5,
    eps=1e-7,
    adjust_lr_fn="original",
)
ADAMW_KWARGS = dict(lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False)
LARGE_ADAMW = {"tok_embeddings.weight", "pos_embeddings.weight", "output.weight"}


def transformer_placements(named_params, mesh):
    # Every member of this bucket must come from the same mixed placement.
    owned = [
        (name, parameter)
        for name, parameter in named_params
        if not name.endswith("self_attn.in_proj_weight") and name not in LARGE_ADAMW
    ]
    mixed = MixedBucketPlacement(
        make_bucketed_owned_full_param_segments(owned, mesh.size())
    )
    placements = {}
    for name, _ in named_params:
        if name.endswith("self_attn.in_proj_weight"):
            placement = mixed.block_shard(blocks_per_rank=(2, 1), dim=0)
        elif name in LARGE_ADAMW:
            placement = mixed.block_shard(blocks_per_rank=(1, 1), dim=0)
        else:
            placement = mixed.bucketed_owned
        placements[name] = (placement,)
    return placements


def configure_muon(model, mesh):
    assert mesh.size() == 2, "This placement example uses exactly two GPUs."
    # Record optimizer roles before FlexShard replaces the parameters.
    muon_names = {
        name
        for name, parameter in model.named_parameters()
        if name.startswith("layers.") and parameter.ndim == 2
    }
    adamw_names = {name for name, _ in model.named_parameters()} - muon_names
    patterns = [
        "tok_embeddings.*",
        "pos_embeddings.*",
        *[f"layers.{i}.*" for i in range(len(model.layers))],
        "norm.*",
        "output.*",
    ]
    flex_shard(
        model,
        buckets=[
            BucketSpec(
                [pattern],
                mesh=mesh,
                placement_fn=transformer_placements,
                reshard_after_forward=False,
            )
            for pattern in patterns
        ],
    )
    named_params = list(model.named_parameters())
    muon = DistMuon(
        [p for name, p in named_params if name in muon_names],
        **MUON_KWARGS,
    )
    adamw = torch.optim.AdamW(
        [p for name, p in named_params if name in adamw_names],
        **ADAMW_KWARGS,
    )
    return muon, adamw


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", local_rank)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        torch.manual_seed(42)
        model = TinyTransformer().to(device).train()
        muon, adamw = configure_muon(model, mesh)
        rng = torch.Generator(device=device).manual_seed(1234 + rank)
        for step in range(3):
            tokens = torch.randint(32, (2, 9), generator=rng, device=device)
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
            loss.backward()
            muon.step()
            adamw.step()
            if rank == 0:
                print(f"step={step} rank0_loss={loss.item():.4f}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
