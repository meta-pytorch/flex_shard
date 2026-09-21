# FlexShard

FlexShard is an experimental parameter-sharding library for PyTorch. Explicit
`BucketSpec`s choose which parameters communicate together, which device mesh
they use, and how they are sharded. Optimizers work on ordinary local tensors.

Train `TinyMoETransformer` with AdamW, sharding dense parameters over `dp` and
expert parameters over `efsdp` to reproduce FSDP2-style training. Then expose
the same model's bucket collectives in a full model trace, following
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

The distribution is named `flex-shard`; import it as `flex_shard`. The AdamW
example needs four GPUs and uses the CUDA runtime supported by your PyTorch
build.

## FSDP2-style training with AdamW

[`TinyMoETransformer`](examples/tiny_moe_transformer.py) is a small causal model
built from public PyTorch modules: two blocks, hidden size 16, four attention
heads, no dropout, and untied output weights. Each block has a softmax router
and four expert MLPs. The model evaluates all experts and combines their outputs
by router weight; it demonstrates expert-FSDP storage without sparse token
dispatch or expert parallelism.

The example predicts the next token for three steps, averages gradients over
each parameter's mesh, and updates local shards with AdamW.

| FSDP2 | FlexShard |
| --- | --- |
| Apply `fully_shard` to child modules, then the root | Call `flex_shard` once with explicit buckets |
| Shard parameters along dimension 0 | Use `per_param_placements`, which assigns each parameter `Shard(0)` |
| `fully_shard(layer.experts, mesh=efsdp_mesh)` | Expert `BucketSpec(..., mesh=efsdp_mesh)` |
| Construct AdamW after sharding | Construct AdamW after sharding |
| AdamW manages DTensor parameters | AdamW manages ordinary local parameter shards |

### Dense parameters on dp, expert parameters on efsdp

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

[`examples/train_adamw.py`](examples/train_adamw.py) builds these buckets and
constructs AdamW after sharding:

```python
import torch
from flex_shard import BucketSpec, flex_shard
from flex_shard.custom_placements.shard import per_param_placements


def bucket(patterns, mesh):
    return BucketSpec(
        patterns, mesh=mesh, placement_fn=per_param_placements,
        reshard_after_forward=False,
    )


buckets = [
    bucket([pattern], dp_mesh)
    for pattern in ("tok_embeddings.*", "pos_embeddings.*", "norm.*", "output.*")
]
for i in range(len(model.layers)):
    buckets.extend([
        bucket(
            [f"layers.{i}.{part}.*" for part in
             ("self_attn", "norm1", "norm2", "router")],
            dp_mesh,
        ),
        bucket([f"layers.{i}.experts.*"], efsdp_mesh),
    ])
flex_shard(model, buckets=buckets)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False,
)
```

The two expert replicas receive paired batches: ranks 0 and 1 share one batch,
and ranks 2 and 3 share another. Batches differ along `efsdp`, while corresponding
expert replicas see the same averaged gradients and remain synchronized.
Arbitrary independent batches across replicas would require additional replica
gradient synchronization.

Run the complete example from the repository root:

```bash
torchrun --standalone --nproc-per-node=4 examples/train_adamw.py
```

On a fresh, unsharded `TinyMoETransformer`, the FSDP2 reference uses the same
mesh assignment and AdamW settings:

```python
from torch.distributed.fsdp import fully_shard

for layer in model.layers:
    fully_shard(layer.experts, mesh=efsdp_mesh, reshard_after_forward=False)
    fully_shard(layer, mesh=dp_mesh, reshard_after_forward=False)
fully_shard(model, mesh=dp_mesh, reshard_after_forward=False)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False,
)
```

### Numerical comparison

The example was compared with FSDP2 and an unsharded reference over three
AdamW steps, checking losses, gradients, parameters, and both optimizer
moments. Validation used four NVIDIA B200 GPUs, Python 3.10.18,
PyTorch 2.14.0+cu130, CUDA 13.0, and NCCL 2.30.7.

Rank-0 losses were `3.9559`, `3.6024`, and `3.7062`. Maximum absolute differences
were `2.39e-7` for losses, `1.50e-8` for gradients, `4.26e-7` for parameters,
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

Run and inspect the same `TinyMoETransformer`:

```bash
torchrun --standalone --nproc-per-node=4 examples/train_adamw.py \
    --trace --graph-dir /tmp/flex_shard_moe_trace
```

The run performs three AdamW steps, checks that exactly one Dynamo graph was
captured, verifies the per-bucket collectives, and writes the graph code and
`summary.json` under `rank0/`, `rank1/`, and so on.

Each rank's captured graphs contain one Dynamo graph, eight bucket autograd
operations, eight all-gathers, eight reduce-scatters, and sixteen waits.

These are graph-node counts, not profiler measurements. The MoE trace includes
communication over both the four-rank `dp` and two-rank `efsdp` meshes.

This demonstrates the shared ability to expose communication to graph
inspection and transformations. It does not establish equal throughput,
memory use, or equivalence to GraphTrainer's scheduling and optimization passes.

## Communication-free Muon with BlockShard + Owned

The AdamW example establishes per-parameter sharding and tracing. The next
step is to choose storage around an optimizer's computation. Muon operates
on matrices, so assigning complete logical matrices to ranks lets each rank
compute its Muon updates locally after gradient reduction.

[`examples/train_muon.py`](examples/train_muon.py) is a separate dense-model
example using FlexShard's native `DistMuon` plus AdamW:

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

`BucketSpec` defaults to `gradient_reduce_op=dist.ReduceOp.AVG`, so the examples
omit that argument. Its `reshard_after_forward` default is `True`; the examples
explicitly set it to `False` so gathered parameters remain available through
backward, trading memory for fewer all-gathers. With
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
