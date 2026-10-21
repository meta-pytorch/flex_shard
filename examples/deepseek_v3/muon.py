# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muon without communication, from flex_shard's DistMuon.

Muon orthogonalizes each matrix's momentum with a Newton-Schulz iteration, so
it needs every matrix whole. ``train.py --optimizer muon`` stores each one whole
on one rank. ``build_local_dist_muon`` then runs DistMuon on each rank's own
storage, on a one-rank mesh, where its storage already is its compute layout:
the step moves nothing between ranks.
"""

from torch import nn
from torch.distributed.tensor import Shard

from flex_shard.dist_muon import (
    BlockShard,
    BucketConfig,
    build_local_dist_muon,
    ComputeLayout,
    DistMuon,
    Owned,
)


def kimi_matrices(fqn: str, config):
    """How Muon splits a DeepSeek V3 parameter into matrices, as Kimi K2 does,
    or ``None`` for the parameters AdamW trains: embeddings, norms and the
    output projection.

    Query and key-value projections split per head and role into row blocks;
    routed experts are batches of matrices, one per expert and projection;
    everything else, including ``[2, F, D]`` gate and up, is computed whole.
    """
    nope, rope = config.qk_nope_head_dim, config.qk_rope_head_dim
    if fqn.endswith("attention.wq.weight"):
        return BlockShard(dim=0, block_sizes=(nope, rope))
    if fqn.endswith("attention.wkv_a.weight"):
        return BlockShard(dim=0, block_sizes=(config.kv_lora_rank, rope))
    if fqn.endswith("attention.wkv_b.weight"):
        return BlockShard(dim=0, block_sizes=(nope, config.v_head_dim))
    if ".routed_experts." in fqn:
        return Shard(0)
    if fqn.endswith(("attention.wo.weight", "w13.weight", "w2.weight", "gate.weight")):
        return Owned()
    return None


class _LocalDistMuon(DistMuon):
    """DistMuon on one rank's stand-ins for its stored matrices
    (``build_local_dist_muon``): its step first hands them their parameters'
    gradients, and ``zero_grad`` clears both."""

    binding = None

    def step(self, closure=None):
        self.binding.copy_grads_to_proxies()
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.binding.zero_real_grads(set_to_none)


def build_muon(model: nn.Module, *, lr: float, weight_decay: float) -> DistMuon:
    """DistMuon over the Muon matrices this rank stores, with Kimi K2's update
    scaling (``match_rms_adamw``), so it takes AdamW's learning rate."""
    matrices = {
        fqn: layout
        for fqn, _ in model.named_parameters()
        if (layout := kimi_matrices(fqn, model.config)) is not None
    }

    def optimizer_factory(param_groups, _layouts):
        (group,) = param_groups
        (axis,) = group["params"][0].device_mesh.mesh_dim_names
        names = group["param_names"]
        return _LocalDistMuon(
            param_groups,
            compute_sharding_by_fqn={
                fqn: ComputeLayout(shardings_by_mesh_axis={axis: matrices[fqn]})
                for fqn in names
            },
            bucket_configs=[
                BucketConfig(name=f"layers.{i}", patterns=(f"layers.{i}.*",))
                for i in sorted({int(fqn.split(".")[1]) for fqn in names})
            ],
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )

    binding = build_local_dist_muon(
        model,
        [(fqn, p) for fqn, p in model.named_parameters() if fqn in matrices],
        optimizer_factory=optimizer_factory,
    )
    if binding is None:
        raise ValueError("This rank stores no Muon matrix; use fewer ranks.")
    binding.optimizer.binding = binding
    return binding.optimizer
