# FlexShard

FlexShard is an experimental parameter-sharding library for PyTorch. It groups
model parameters into buckets and lets placement implementations control how
each bucket stores shards, gathers parameters, and reduces gradients.

## Requirements

| Component | Requirement |
| --- | --- |
| Python | 3.10 or later |
| PyTorch | `>=2.14,<2.15` |
| Platform | Linux with NVIDIA GPUs for training |
| CUDA and NCCL | A CUDA-enabled PyTorch build with NCCL and a compatible NVIDIA driver |
| Triton | `~=3.8.0` on Linux |

FlexShard does not pin a separate CUDA toolkit version. Use the CUDA runtime
supported by your PyTorch build and a compatible driver. The examples below use
two GPUs; some tests require four. CPU-only tests cover a subset of the library,
not the CUDA training runtime.

## Installation

Install a CUDA-enabled PyTorch build satisfying the version range above, then
install FlexShard from this repository:

```bash
git clone https://github.com/meta-pytorch/flex_shard.git
cd flex_shard
python -m pip install -e .
```

The distribution is named `flex-shard`; the Python import is `flex_shard`.

## Communication-free Muon with BlockShard + Owned

Muon operates on matrices: splitting a matrix arbitrarily across ranks can
require communication to compute its update. FlexShard can instead place
complete logical matrices on their owner ranks, together with their reduced
gradients. This example combines `BlockShard` and whole-parameter Owned storage
with [TorchTitan's DistMuon](https://github.com/pytorch/torchtitan/tree/610bb6f6b99d16f2314f9ddf520ab3cd2423ebc0/torchtitan/distributed/flex_shard).

Here, **communication-free means no inter-rank communication during the Muon
optimizer step**. FlexShard still gathers parameters for forward and reduces
gradients during backward. Constructing the optimizer also creates process
groups. This is a placement and optimizer recipe for training with Muon; it does
not remove communication from the training step as a whole.

### Shared Transformer

Both examples use this small causal Transformer, built from public PyTorch
modules. Its two blocks use fused Q/K/V projections, dropout is disabled, and
the output projection does not share weights with the token embedding. Save
this as `tiny_transformer.py` beside the training scripts below:

```python
import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=32, seq_len=8, d_model=16, n_heads=4, n_layers=2):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward=2 * d_model,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.tok_embeddings(tokens) + self.pos_embeddings(positions)
        causal_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=tokens.device
        ).triu(1)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=causal_mask, is_causal=True)
        return self.output(self.norm(hidden))
```

### Choose storage and optimizer together

For two GPUs, assign parameters as follows:

| Parameters | Storage placement | Optimizer |
| --- | --- | --- |
| Each `layers.{i}.self_attn.in_proj_weight` | `BlockShard(blocks_per_rank=(2, 1), dim=0)` | DistMuon, updating Q, K, and V separately |
| Each attention `out_proj.weight`, `linear1.weight`, and `linear2.weight` | Whole-matrix Owned | DistMuon on the owner rank |
| `tok_embeddings.weight`, `pos_embeddings.weight`, and `output.weight` | `BlockShard(blocks_per_rank=(1, 1), dim=0)` | AdamW on balanced row shards |
| All biases and normalization parameters | Whole-parameter Owned, distributed across ranks | AdamW on the owner rank |

The fused Q/K/V parameter has shape `[3 * d_model, d_model]`. Its three row
blocks each contain one complete `[d_model, d_model]` matrix: rank 0 owns Q and
K, and rank 1 owns V. Orthogonalizing those matrices separately defines this
Muon recipe; it differs from treating the whole fused tensor as one matrix.
The other Muon parameters remain complete matrices on their owner ranks.

AdamW is elementwise, so its large parameters can use row blocks chosen for
storage balance. Their row counts are divisible by two in this model. Keeping
those tensors distributed also distributes their AdamW state. Small biases
and normalization tensors use Owned storage. Optimizer membership is chosen
explicitly by parameter name and role: both optimizers manage a mixture of
BlockShard and Owned parameters.

The script uses `make_bucketed_owned_full_param_segments()`, the same
whole-parameter assignment helper used by
`make_bucketed_owned_full_param_placement_fn()`. It balances owned parameters
by size with a deterministic assignment on every rank. Each of the six
execution-unit buckets uses one `MixedBucketPlacement` instance, whose
`block_shard` and `bucketed_owned` members combine the two storage layouts.
Every parameter belongs to exactly one bucket and optimizer.

### Bind DistMuon and train

FlexShard provides the local-storage adapter, not the DistMuon implementation.
This example uses TorchTitan revision
`610bb6f6b99d16f2314f9ddf520ab3cd2423ebc0` and requires Python 3.11 or later for
that optional dependency. After installing FlexShard, install the pinned
TorchTitan package:

```bash
python -m pip install tyro==1.0.15
python -m pip install --no-deps \
  'https://github.com/pytorch/torchtitan/archive/610bb6f6b99d16f2314f9ddf520ab3cd2423ebc0.zip'
```

The pinned package reports TorchTitan version `0.2.2`. Use this revision for
the compute-layout API shown below. This optimizer import needs PyTorch and `tyro`;
`--no-deps` avoids installing TorchTitan's full training application stack.
The per-parameter migration example in the next chapter does not use TorchTitan.

Capture canonical matrix shapes before applying `flex_shard`, then reacquire
parameters by name afterward because sharding replaces them. The factory
translates FlexShard's `row_blocks` and `owned` layout metadata into DistMuon
compute layouts, preserving the adapter's single ordered parameter group.
`build_local_dist_muon` supplies replicated DTensor views on a singleton mesh;
the views share storage with the local FlexShard parameters. A rank with no
local Muon parameters receives `None` and still participates in training.

Save the following as `train_muon.py` beside `tiny_transformer.py`:

```python
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from flex_shard import BucketSpec, flex_shard
from flex_shard.custom_placements.mixed_bucket import MixedBucketPlacement
from flex_shard.custom_placements.owned import make_bucketed_owned_full_param_segments
from flex_shard.dist_muon import (
    build_local_dist_muon,
    capture_flex_shard_muon_canonical_shards,
)
from torchtitan.distributed.flex_shard import (
    BlockShard as MuonBlockShard,
    BucketConfig,
    ComputeLayout,
    Owned as MuonOwned,
    build_dist_muon,
)
from tiny_transformer import TinyTransformer


MUON_KWARGS = dict(
    lr=0.01, weight_decay=0.01, momentum=0.95, nesterov=True,
    ns_coefficients=(3.4445, -4.7750, 2.0315), ns_steps=5, eps=1e-7,
    adjust_lr_fn="original",
)
ADAMW_KWARGS = dict(lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False)
LARGE_ADAMW = {"tok_embeddings.weight", "pos_embeddings.weight", "output.weight"}


def make_muon(param_groups, layouts):
    # Keep one ordered group and storage-shaped momentum for the binding.
    assert len(param_groups) == 1
    group = param_groups[0]
    compute_layouts = {}
    for name, parameter in zip(group["param_names"], group["params"], strict=True):
        assert parameter.device_mesh.mesh_dim_names == ("local",)
        assert parameter.device_mesh.size() == 1
        layout = layouts[name]
        if layout.kind == "row_blocks":
            # Each complete Q/K/V matrix is orthogonalized independently.
            compute = MuonBlockShard(dim=0, block_size=layout.block_size)
        elif layout.kind == "owned":
            compute = MuonOwned()
        else:
            raise ValueError(f"Unexpected Muon layout for {name}: {layout.kind}")
        compute_layouts[name] = ComputeLayout(
            shardings_by_mesh_axis={"local": compute}
        )
    return build_dist_muon(
        param_groups,
        compute_sharding_by_fqn=compute_layouts,
        bucket_configs=(BucketConfig(patterns=("*",), name="local_muon"),),
        **MUON_KWARGS,
    )


def transformer_placements(named_params, mesh):
    # Every member of this bucket must come from the same mixed placement.
    owned = [
        (name, parameter) for name, parameter in named_params
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
        name for name, parameter in model.named_parameters()
        if name.startswith("layers.") and parameter.ndim == 2
    }
    adamw_names = {name for name, _ in model.named_parameters()} - muon_names
    capture_flex_shard_muon_canonical_shards(model)
    patterns = [
        "tok_embeddings.*", "pos_embeddings.*",
        *[f"layers.{i}.*" for i in range(len(model.layers))],
        "norm.*", "output.*",
    ]
    flex_shard(model, buckets=[
        BucketSpec(
            [pattern], mesh=mesh, placement_fn=transformer_placements,
            gradient_reduce_op=dist.ReduceOp.AVG, reshard_after_forward=False,
        )
        for pattern in patterns
    ])
    named_params = list(model.named_parameters())
    muon = build_local_dist_muon(
        model, [(name, p) for name, p in named_params if name in muon_names],
        optimizer_factory=make_muon,
    )
    adamw = torch.optim.AdamW(
        [p for name, p in named_params if name in adamw_names], **ADAMW_KWARGS,
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
            if muon is not None:
                muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
            loss.backward()
            if muon is not None:
                muon.step()
            adamw.step()
            if rank == 0:
                print(f"step={step} rank0_loss={loss.item():.4f}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Run one process per GPU:

```bash
torchrun --standalone --nproc-per-node=2 train_muon.py
```

The script initializes identical parameters on every rank and uses different
synthetic token sequences on each rank. Both optimizers are constructed after
sharding. It clears gradients, predicts the next token, averages gradients
across ranks, and performs local optimizer updates for three steps. Printed
losses are rank 0's local batch losses, not a global average.

Muon uses learning rate `0.01`, weight decay `0.01`, momentum `0.95`,
Nesterov momentum, and five Newton–Schulz iterations with coefficients
`(3.4445, -4.7750, 2.0315)`, epsilon `1e-7`, and the `original` learning-rate
adjustment. TorchTitan runs the Newton–Schulz computation in BF16. AdamW uses
learning rate `1e-3`, epsilon `1e-6`, and weight decay `0.01`.

### Validation

Three updates matched an unsharded reference using public `torch.optim.Muon`
on separate Q, K, and V matrices and the same AdamW groups. The reference
averaged gradients across ranks. Losses, gradients, parameters, Muon momentum,
and AdamW first and second moments all had maximum absolute error zero on
both ranks. Validation also checked matrix boundaries, exclusive optimizer
membership, gradient clearing, and the adapter's `None` result in a separate
case where one rank owns no Muon parameters.

A CPU/CUDA profiler recorded forward/backward and a steady-state Muon step in
separate regions, with CUDA synchronized at the boundaries:

| Profiled region | Inter-rank collectives per rank |
| --- | --- |
| Forward and backward | 6 all-gathers + 6 reduce-scatters, with 12 corresponding NCCL GPU kernels |
| `muon.step()` after backward | 0 c10d collectives and 0 NCCL GPU kernels |

Optimizer initialization and AdamW's step are outside the Muon-step region.
The zero-collective result applies to this measured Muon step with the shown
layouts, not to forward/backward or the entire training iteration.

The run used two NVIDIA B200 GPUs, Python 3.12.12, PyTorch
`2.15.0a0+git1807a24`, CUDA 13.0, NCCL 2.30.7, and the pinned TorchTitan `0.2.2`
revision above. This was a source-checkout validation on a PyTorch build
outside FlexShard's declared dependency range. The installation commands
above describe the declared package requirements, not a reproduction of this
development environment.

## Migrating from FSDP2 with per-parameter sharding

If you already use FSDP2, dimension-0 parameter sharding is a starting point
for moving to FlexShard. Use the same Transformer and AdamW algorithm, then
express the communication groups with explicit buckets. Once that baseline
works, customize placements for optimizers such as the Muon example above.

| FSDP2 | FlexShard |
| --- | --- |
| Apply `fully_shard` to each Transformer block, then the root for remaining parameters | Call `flex_shard` once at the root with explicit `BucketSpec`s |
| Shard each parameter along dimension 0 | Set `placement_fn=per_param_placements`, assigning `Shard(0)` to each parameter |
| Choose communication groups through the modules passed to `fully_shard` | Choose bucket membership with parameter-name patterns |
| Construct the optimizer after `fully_shard` | Construct the optimizer after `flex_shard` |
| Optimizer parameters are DTensors | Optimizer parameters are local tensors carrying sharding metadata |

An FSDP2 version of this model would use the following setup after creating its
one-dimensional CUDA mesh. It also constructs AdamW after sharding:

```python
from torch.distributed.fsdp import fully_shard

model = TinyTransformer().to(device).train()
for layer in model.layers:
    fully_shard(layer, mesh=mesh, reshard_after_forward=False)
fully_shard(model, mesh=mesh, reshard_after_forward=False)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, eps=1e-6,
    weight_decay=0.01, foreach=False,
)
```

The FlexShard version uses six buckets, following the
[Transformer test layout](src/flex_shard/tests/common.py): token embeddings,
position embeddings, one per Transformer block, final normalization, and the
output projection. FSDP2's remaining root parameters therefore become four
explicit buckets. These examples share dimension-0 parameter sharding and
averaged-gradient semantics; their collective grouping and checkpoint
representations need not be identical.

Save this as `train_per_param.py` beside the same `tiny_transformer.py`:

```python
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from flex_shard import BucketSpec, flex_shard
from flex_shard.custom_placements.shard import per_param_placements
from tiny_transformer import TinyTransformer


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", local_rank)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        torch.manual_seed(42)  # Identical initial parameters on every rank.
        model = TinyTransformer().to(device).train()
        patterns = [
            "tok_embeddings.*",
            "pos_embeddings.*",
            *[f"layers.{i}.*" for i in range(len(model.layers))],
            "norm.*",
            "output.*",
        ]
        flex_shard(model, buckets=[
            BucketSpec(
                [pattern],
                mesh=mesh,
                placement_fn=per_param_placements,
                gradient_reduce_op=dist.ReduceOp.AVG,
                reshard_after_forward=False,
            )
            for pattern in patterns
        ])
        # Construct the optimizer with FlexShard's replacement parameters.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, eps=1e-6,
            weight_decay=0.01, foreach=False,
        )
        rng = torch.Generator(device=device).manual_seed(1234 + rank)
        for step in range(3):
            tokens = torch.randint(32, (2, 9), generator=rng, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(f"step={step} rank0_loss={loss.item():.4f}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Run it on two GPUs:

```bash
torchrun --standalone --nproc-per-node=2 train_per_param.py
```

Every parameter matches exactly one bucket. `per_param_placements` shards
parameters independently along dimension 0; the AdamW update remains the same
elementwise algorithm. Code that calls DTensor methods on FSDP2 parameters
needs adapting for FlexShard's local tensor representation.

### Parameter retention and checkpoints

Both chapters set `reshard_after_forward=False`. Gathered parameters remain
available through backward, using more memory to avoid gathering them again.
With `reshard_after_forward=True`, FlexShard's saved-tensor hooks replay
parameter unshards in backward; this does not by itself recompute the whole
Transformer block. If activation checkpointing is already applied, FlexShard
composes with its recomputation policy.

A regular FlexShard `state_dict()` contains rank-local shards. It is not a
complete gathered model checkpoint. Existing FSDP2 checkpoint save/load code
requires an explicit compatibility check or conversion; do not load FSDP2
sharded checkpoints unchanged on the assumption that the parameter split is
the same.

### Validation

Both complete training scripts were extracted from this README and run with
`torchrun --standalone --nproc-per-node=2`. Each completed all three steps.
Rank 0 printed losses `3.5078`, `3.5276`, `3.3849` for the Muon example and
`3.5078`, `3.5287`, `3.3812` for the per-parameter example.

The six-bucket FlexShard configuration, FSDP2 configuration, and unsharded
reference were compared over three training steps with identical initial
weights, rank-specific token sequences, and the AdamW settings shown above.
The reference explicitly averaged gradients across ranks. Losses, gradients,
parameters, and AdamW first and second moments matched exactly on both ranks;
all reported maximum absolute differences were zero. Gradient clearing and
actual parameter updates were also checked.

The run used two NVIDIA B200 GPUs, Python 3.12.12, PyTorch
`2.15.0a0+gitd307e02`, and CUDA 13.0. This validates the example against that
source-checkout environment; the PyTorch build is outside the package's
declared dependency range. The package requirements above are unchanged.

## Limitations

- The API is experimental and may change. It uses private PyTorch APIs, so the
  declared PyTorch version range matters.
- Training currently requires a one-dimensional CUDA device mesh. CPU offload
  is not supported.
- Every parameter must match exactly one bucket. The migration example's
  dimension-0 sharding placement does not support scalar parameters.
- A regular `state_dict()` contains rank-local shards. Checkpoint conversion
  and gathering require separate handling.

## Development and testing

From the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

GPU tests skip when the required hardware is unavailable; a run with skipped
GPU tests does not validate distributed training. Use a host with at least four
GPUs for the full suite.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development and pull-request
workflow and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

## License

FlexShard is BSD-3-Clause licensed, as found in the [LICENSE](LICENSE) file.
