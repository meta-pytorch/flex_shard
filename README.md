# FlexShard

FlexShard is an experimental parameter-sharding library for PyTorch. Explicit
`BucketSpec`s choose which parameters communicate together, which device mesh
they use, and how they are sharded. Optimizers work on ordinary local tensors.

Start with AdamW and per-parameter `Shard(0)` to reproduce FSDP2-style training.
The same dense and MoE examples can then expose their bucket collectives in a
full model trace, following the approach in
[`test_torch_compile_traces_per_bucket_collectives`](src/flex_shard/tests/test_flex_shard_runtime.py).

## Requirements and installation

| Component | Requirement |
| --- | --- |
| Python | 3.10 or later |
| PyTorch | `>=2.14,<2.15` |
| Platform | Linux with NVIDIA GPUs |
| CUDA and NCCL | A CUDA-enabled PyTorch build with NCCL and a compatible NVIDIA driver |
| Triton | `~=3.8.0` on Linux |

Install a CUDA-enabled PyTorch build satisfying this range, then install
FlexShard from the repository:

```bash
git clone https://github.com/meta-pytorch/flex_shard.git
cd flex_shard
python -m pip install -e .
```

The distribution is named `flex-shard`; import it as `flex_shard`. The dense
example needs two GPUs and the MoE example needs four. FlexShard uses the CUDA
runtime supported by your PyTorch build rather than pinning a separate toolkit.

## FSDP2-style training with AdamW

Both examples use small causal Transformers built from public PyTorch modules:
two blocks, hidden size 16, four attention heads, no dropout, and untied output
weights. They predict the next token for three steps, average gradients over
each parameter's mesh, and update local shards with AdamW.

| FSDP2 | FlexShard |
| --- | --- |
| Apply `fully_shard` to child modules, then the root | Call `flex_shard` once with explicit buckets |
| Shard parameters along dimension 0 | Use `per_param_placements`, which assigns each parameter `Shard(0)` |
| Give expert modules their own FSDP mesh | Give expert buckets their own `efsdp` mesh |
| Construct AdamW after sharding | Construct AdamW after sharding |
| AdamW manages DTensor parameters | AdamW manages ordinary local parameter shards |

### Dense parameters: per-parameter Shard(0)

[`TinyTransformer`](examples/tiny_transformer.py) has six communication buckets:
token embeddings, position embeddings, one per Transformer block, final
normalization, and the output projection. Every parameter belongs to exactly
one bucket and is sharded independently along dimension 0 over the full `dp`
mesh.

The setup in [`examples/train_adamw.py`](examples/train_adamw.py) is equivalent
to the following configuration, after creating a model and a one-dimensional
CUDA mesh:

```python
import torch
import torch.distributed as dist
from flex_shard import BucketSpec, flex_shard
from flex_shard.custom_placements.shard import per_param_placements

patterns = [
    "tok_embeddings.*", "pos_embeddings.*",
    *[f"layers.{i}.*" for i in range(len(model.layers))],
    "norm.*", "output.*",
]
flex_shard(model, buckets=[
    BucketSpec(
        [pattern], mesh=dp_mesh, placement_fn=per_param_placements,
        gradient_reduce_op=dist.ReduceOp.AVG, reshard_after_forward=False,
    )
    for pattern in patterns
])
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False,
)
```

Run the complete example from the repository root:

```bash
torchrun --standalone --nproc-per-node=2 examples/train_adamw.py
```

The corresponding FSDP2 setup on a fresh copy of the model is:

```python
from torch.distributed.fsdp import fully_shard

for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh, reshard_after_forward=False)
fully_shard(model, mesh=dp_mesh, reshard_after_forward=False)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False,
)
```

FSDP2 groups the remaining parameters at the root; this FlexShard example puts
them in four explicit buckets. The parameter split and averaged-gradient
update agree even though the collective grouping differs.

### MoE: dense parameters on dp, expert parameters on efsdp

[`TinyMoETransformer`](examples/tiny_moe_transformer.py) replaces each block's
feed-forward network with a softmax router and four expert MLPs. It evaluates
all experts and combines their outputs by router weight. This small example
demonstrates expert-FSDP parameter storage; it does not implement sparse token
dispatch or expert parallelism.

The four ranks form an `efsdp × replica` grid:

| Parameters | Mesh | Storage |
| --- | --- | --- |
| Embeddings, attention, router, norms, output | `dp = [0, 1, 2, 3]` | Per-parameter `Shard(0)` across four ranks |
| Expert matrices `[experts, rows, columns]` | `efsdp = [0, 2]` or `[1, 3]` | Per-parameter `Shard(0)` across two ranks |

```python
from torch.distributed.device_mesh import init_device_mesh

dp_mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
grid = init_device_mesh("cuda", (2, 2), mesh_dim_names=("efsdp", "replica"))
efsdp_mesh = grid["efsdp"]
```

The expert bucket for block `i` selects `layers.{i}.experts.*` and uses
`mesh=efsdp_mesh`. The block's attention, router, and norms form a separate
bucket on `dp_mesh`. Along with the four root buckets, this gives eight
buckets. After all-gather, each expert-FSDP group can evaluate all four experts.

The two expert replicas receive paired batches: ranks 0 and 1 share one batch,
and ranks 2 and 3 share another. Batches differ along `efsdp`, while corresponding
expert replicas see the same averaged gradients and remain synchronized.
Arbitrary independent batches across replicas would require additional replica
gradient synchronization.

Run the same AdamW training loop with the MoE model and meshes:

```bash
torchrun --standalone --nproc-per-node=4 examples/train_adamw.py --moe
```

On a fresh, unsharded `TinyMoETransformer`, the FSDP2 reference uses the same
mesh assignment:

```python
for layer in model.layers:
    fully_shard(layer.experts, mesh=efsdp_mesh, reshard_after_forward=False)
    fully_shard(layer, mesh=dp_mesh, reshard_after_forward=False)
fully_shard(model, mesh=dp_mesh, reshard_after_forward=False)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False,
)
```

### Numerical comparison

Both layouts were compared with FSDP2 and unsharded references over three
AdamW steps, checking losses, gradients, parameters, and both optimizer
moments. Validation used Python 3.10.18, PyTorch 2.14.0+cu130, CUDA 13.0,
NCCL 2.30.7, and NVIDIA B200 GPUs.

| Example | GPUs | Rank-0 losses | Largest absolute parameter difference |
| --- | --- | --- | --- |
| Dense | 2 | `3.5078`, `3.5287`, `3.3812` | `0` |
| MoE | 4 | `3.4172`, `3.5311`, `3.2894` | `3.55e-7` |

Losses agreed exactly in both cases. Dense gradients and AdamW moments also
agreed exactly; MoE differences were at most `1.49e-8` for gradients,
`1.87e-9` for first moments, and `3.64e-12` for second moments.

## Full model tracing with visible collectives

[SimpleFSDP](https://github.com/pytorch/torchtitan/blob/610bb6f6b99d16f2314f9ddf520ab3cd2423ebc0/torchtitan/experiments/graph_trainer/simple_fsdp.py)
expresses parameter reconstruction through traceable DTensor operations.
FlexShard exposes this capability through its explicit buckets: the model's
forward graph contains per-bucket autograd operations whose forward and
backward subgraphs contain functional collectives.

Following the unit test, capture the same sharded model with:

```python
graphs = []

def capture_backend(graph_module, example_inputs):
    graphs.append(graph_module)
    return graph_module.forward

run_model = torch.compile(model, backend=capture_backend, fullgraph=True)
optimizer.zero_grad(set_to_none=True)
logits = run_model(tokens[:, :-1])
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), tokens[:, 1:].flatten(),
)
loss.backward()
optimizer.step()

assert len(graphs) == 1
targets = {
    str(node.target)
    for submodule in graphs[0].modules()
    if isinstance(submodule, torch.fx.GraphModule)
    for node in submodule.graph.nodes
}
assert "_c10d_functional.all_gather_into_tensor" in targets
assert "_c10d_functional.reduce_scatter_tensor" in targets
assert "_c10d_functional.wait_tensor" in targets
```

`fullgraph=True` requires the entire model forward to capture without graph
breaks. Inspect the nested forward/backward graphs as well as the root:
looking only at root-level nodes misses the collectives. The capture backend
executes the graph directly; loss calculation and AdamW remain outside the
captured model. This is a model-tracing example, not a single flat graph of the
entire training loop.

Run and inspect both layouts:

```bash
torchrun --standalone --nproc-per-node=2 examples/train_adamw.py \
    --trace --graph-dir /tmp/flex_shard_dense_trace
torchrun --standalone --nproc-per-node=4 examples/train_adamw.py \
    --moe --trace --graph-dir /tmp/flex_shard_moe_trace
```

Each run performs three AdamW steps, checks that exactly one Dynamo graph was
captured, verifies the per-bucket collectives, and writes the graph code and
`summary.json` under `rank0/`, `rank1/`, and so on.

The traced examples retained the numerical agreement above. Each rank's
captured graphs contained:

| Example | Dynamo graphs | Bucket autograd operations | All-gathers | Reduce-scatters | Waits |
| --- | --- | --- | --- | --- | --- |
| Dense | 1 | 6 | 6 | 6 | 12 |
| MoE | 1 | 8 | 8 | 8 | 16 |

These are graph-node counts, not profiler measurements. The MoE trace includes
communication over both the four-rank `dp` and two-rank `efsdp` meshes.

This demonstrates the shared ability to expose communication to graph
inspection and transformations. It does not establish equal throughput,
memory use, or equivalence to GraphTrainer's scheduling and optimization passes.

## Communication-free Muon with BlockShard + Owned

The AdamW examples establish per-parameter sharding and tracing. The next
step is to choose storage around an optimizer's computation. Muon operates
on matrices, so assigning complete logical matrices to ranks lets each rank
compute its Muon updates locally after gradient reduction.

[`examples/train_muon.py`](examples/train_muon.py) uses the same dense
`TinyTransformer` with FlexShard's native `DistMuon` plus AdamW:

| Parameters | Storage | Optimizer |
| --- | --- | --- |
| Each fused attention Q/K/V weight | `BlockShard(blocks_per_rank=(2, 1), dim=0)` | DistMuon on separate Q, K, and V matrices |
| Attention output and feed-forward weight matrices | Whole-matrix Owned | DistMuon on the owner |
| Token/position embeddings and output weight | `BlockShard(blocks_per_rank=(1, 1), dim=0)` | AdamW on balanced row shards |
| Biases and normalization parameters | Whole-parameter Owned | AdamW on the owner |

For a fused `[3 * d_model, d_model]` Q/K/V weight, rank 0 owns the complete
Q and K matrices and rank 1 owns V. Updating these three matrices separately
defines this Muon recipe. It differs from orthogonalizing the fused tensor as
one matrix. Owned assignments use the same deterministic balancing helper as
the placement tests; each bucket combines its layouts with
`MixedBucketPlacement`.

Optimizer roles are selected by parameter name before sharding. After
`flex_shard`, construct `DistMuon` with the actual replacement parameters:

```python
from flex_shard.dist_muon import DistMuon

# muon_names records the selected names before sharding.
# The full script supplies the BlockShard + Owned bucket configuration.
muon = DistMuon(
    [p for name, p in model.named_parameters() if name in muon_names],
    lr=0.01, weight_decay=0.01, momentum=0.95, nesterov=True,
    ns_coefficients=(3.4445, -4.7750, 2.0315), ns_steps=5, eps=1e-7,
    adjust_lr_fn="original",
)
```

Parameters, gradients, and momentum are ordinary local tensors. FlexShard
records the matrix layout and canonical checkpoint coordinates automatically;
no DTensor adapter, manual capture call, or optimizer process group is needed.
Empty owners still construct a valid no-op optimizer and create no momentum
for empty shards. Native DistMuon rejects partial-matrix placements requiring
redistribution.

Run the complete two-optimizer example:

```bash
torchrun --standalone --nproc-per-node=2 examples/train_muon.py
```

Here, **communication-free refers to the Muon optimizer step**. Forward still
gathers parameters and backward still reduces gradients. DistMuon computes
Newton–Schulz iterations in BF16 and keeps momentum in the parameter's storage
shape. AdamW uses the same settings as the preceding examples.

The three-step example was validated on Python 3.10.18, PyTorch 2.14.0+cu130,
and two B200 GPUs without TorchTitan or tyro installed. Rank-0 losses were
`3.5078`, `3.5276`, and `3.3849`. Losses, gradients, parameters, Muon momentum,
and AdamW moments matched an unsharded reference with separate Q/K/V updates
exactly. A steady-state CPU/CUDA profile recorded zero Muon communication
events; forward/backward recorded six all-gathers and six reduce-scatters
per rank.

DistMuon is maintained within FlexShard; see its
[source provenance](src/flex_shard/dist_muon/UPSTREAM.md). Tests cover native
checkpoint resume, migration from legacy adapter checkpoints, empty owners,
and execution with DTensor and optimizer process-group creation disabled.


## Parameter retention and checkpoints

These examples use `reshard_after_forward=False`: gathered parameters remain
available through backward, trading memory for fewer all-gathers. With
`reshard_after_forward=True`, FlexShard's saved-tensor hooks replay parameter
unshards in backward. That does not itself recompute the whole Transformer
block; existing activation checkpointing can compose with that policy.

A regular FlexShard `state_dict()` contains rank-local shards. It is not a
gathered model checkpoint. Existing FSDP2 checkpoint code needs an explicit
compatibility check or conversion, even when the parameter split agrees.

## Limitations

- The API is experimental and uses private PyTorch APIs; the declared version
  range matters.
- Each bucket requires a one-dimensional CUDA mesh. Different buckets may use
  different submeshes, as in the MoE example. CPU offload is not supported.
- Every parameter must match exactly one bucket. The examples' `Shard(0)`
  placement does not support scalar parameters.
- Numerical and graph comparisons here concern the small examples and the
  stated configuration; they are not general FSDP2 or SimpleFSDP benchmarks.

## Development and testing

```bash
python -m pip install -e '.[test]'
python -m pytest
```

Use at least four GPUs for the full suite. Tests skip when required hardware
is unavailable; CPU-only checks do not validate distributed training.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

## License

FlexShard is BSD-3-Clause licensed; see [LICENSE](LICENSE).
