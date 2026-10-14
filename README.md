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
| PyTorch | 2.15 or later; use a nightly build until 2.15 is released |
| Platform | Linux with NVIDIA GPUs |
| CUDA and NCCL | A CUDA-enabled PyTorch build with NCCL and a compatible NVIDIA driver |
| Triton | `~=3.8.0` on Linux |

`Shard` placements call FSDP's native collective copies:
`fsdp::_all_gather_copy_out_` from
[pytorch/pytorch#197204](https://github.com/pytorch/pytorch/pull/197204), and
`fsdp::chunk_cat_mixed_dtype` with `num_leading_dims` from
[pytorch/pytorch#200179](https://github.com/pytorch/pytorch/pull/200179). Until a
PyTorch release includes them, they need a PyTorch built with those pull requests.

Install a CUDA-enabled PyTorch build satisfying this requirement, then install
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
| `shard_placement_fn` returning `Shard(i)` | A `placement_fn` returning `Shard(i)`; one bucket may mix dims |
| `fully_shard(layer.experts, mesh=efsdp_mesh)` | Expert `BucketSpec(..., mesh=efsdp_mesh)` |
| `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` | `set_buckets_to_forward_prefetch` / `set_buckets_to_backward_prefetch` on entries of `sharded_bucket_storages` |
| `unshard()`, `reshard()` and `set_reshard_after_forward` on one `fully_shard` group | The same methods on its bucket's storage, from `sharded_bucket_storages` or `model.bucket_storage_of(param)` |
| Construct AdamW after sharding | Construct AdamW after sharding |
| AdamW manages DTensor parameters | AdamW manages ordinary local parameter shards |
| DCP checkpoints DTensor state dicts | DCP checkpoints local shards that declare their layouts; see [Distributed checkpoints](#distributed-checkpoints) |

For training that is bitwise identical to FSDP2, give each `fully_shard` group
one `BucketSpec` naming the same modules, with the same `Shard(i)` placements,
and set `fsdp2_compatible=True`. The bucket then orders its parameters as
`fully_shard` does and, like FSDP2, reduce-scatters only the parameters that got
a gradient, so its all-gather and reduce-scatter buffers match FSDP2's byte for
byte.

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

This eager example uses the `BucketSpec` defaults: averaged gradients and
`reshard_after_forward=True`. Gathered parameters are released after forward
and reconstructed as needed during backward.

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
    fully_shard(layer.experts, mesh=efsdp_mesh)
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh, reshard_after_forward=True)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False,
)
```

FSDP2 defaults to resharding child modules. The root explicitly uses `True`
to match the eager FlexShard example's policy.

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

The tracing path uses the same model and mesh assignment. Under
`torch.compile`, the traced graph owns buffer lifetimes, so
`reshard_after_forward` does not apply. Following the unit test, configure a
fresh model for tracing, then capture it:

```python
from examples.tiny_moe_transformer import TinyMoETransformer

model = TinyMoETransformer().to(device).train()
flex_shard(model, buckets=buckets)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False,
)

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

## Parameter retention and checkpoints

`BucketSpec` defaults to `gradient_reduce_op=dist.ReduceOp.AVG`, so the examples
omit that argument. Its `reshard_after_forward` default is `True`, which the
eager AdamW example uses.

`AVG` divides the gradient sum over the bucket's mesh by
`gradient_divide_factor`, which defaults to the mesh size, like FSDP2's
`set_gradient_divide_factor`. With expert parallelism, where a token dispatcher
sends each token to the rank that owns its expert, set it to the dense
data-parallel size on expert buckets: an expert's local gradient already sums
the tokens its expert-parallel peers routed to it. The example evaluates every
expert locally, so the default fits it.

As in FSDP2, each managed parameter has a persistent unsharded `nn.Parameter`.
While its bucket is unsharded, the owning module's `_parameters` holds it, so
module code sees a regular parameter in forward and backward; otherwise it holds
the rank-local shard. With `reshard_after_forward=True`, FlexShard frees the
unsharded storage with `untyped_storage().resize_(0)` after the bucket's
forward and re-gathers it in a pre-backward hook. This composes with existing
activation checkpointing without wrapping the model.

Setting `reshard_after_forward=False` keeps gathered parameters available
through backward, trading memory for fewer all-gathers. Such a bucket stays
unsharded after a forward until its backward; call `model.reshard()` when that
backward will not run, as with FSDP2's `FSDPModule.reshard()`. Like the methods
of one `fully_shard` group, each bucket storage also has `unshard()`,
`reshard()` and `set_reshard_after_forward()`; `model.bucket_storage_of(param)`
finds the bucket of a parameter.

For gradient accumulation, call `model.set_requires_gradient_sync(is_last)`
before each microbatch, as with FSDP2; there is no `no_sync()` context
manager. Backwards without sync skip the reduce-scatter and keep full
gradients on the unsharded parameters, accumulating in the bucket's
`reduce_dtype` when it is wider, and the next syncing backward reduce-scatters
them. A bucket called more than once per forward, such as an output projection
shared by multi-token prediction, reduce-scatters the kept gradients on their
own when the syncing backward first reaches it, and the new ones in a second
reduce-scatter, as FSDP2's post-backward per call does.
`model.set_reshard_after_backward(False)` also keeps the parameters
unsharded between those microbatches, so only the first one all-gathers when
`reshard_after_forward=False`. Unlike FSDP2, a syncing backward always
reshards, so the optimizer step never leaves stale unsharded parameters. Both
are eager-only. Like FSDP2's per-module setters, both also exist per bucket,
on the entries of `model.sharded_bucket_storages` (one per non-empty
`BucketSpec`, in order). For example, a model can keep reduce-scattering expert
buckets whose full gradients would be large. FlexShard does not recover from errors:
after a backward that raises, the next forward or `model.reshard()` raises too,
and training has to restart.

To reduce-scatter the accumulated gradients outside a backward, turn sync back
on and call `model.finalize_backward()`, as with FSDP2. With
`async_op=True` it returns a handle without waiting; call its `wait()` before
the optimizer step or the next forward. A pipeline stage can then reduce its
last microbatch's gradients while other stages still compute.
`model.set_manual_backward_finalization(True)` goes further: backwards finish
nothing at their end, and `finalize_backward()` does it once per step.

Some kernels compute weight gradients after their module's backward, such as
TransformerEngine's `delay_wgrad_compute`, whose `backward_dw()` runs later.
A bucket with `BucketSpec(defer_post_backward=True)` then skips its usual
post-backward, which reduce-scatters as soon as its module's backward is done.
Instead, the caller calls `model.finish_deferred_backward(param)` with any of
its parameters once the late gradients exist, during that backward or before
`finalize_backward()`. A syncing backward that ends with the bucket unfinished
raises, instead of reduce-scattering without the late gradients.

Some schedules run a module's computation directly, bypassing the forward hooks
that gather its bucket. One example is Megatron-LM's EP all-to-all overlap,
which calls each layer's sub-modules. Like FSDP2, call `model.unshard()` first:
it gathers every bucket and keeps it gathered until the end of a backward that
re-gathers a bucket, or `finalize_backward()`, finishes it. With
`async_op=True` it only starts the all-gathers and returns a handle, as FSDP2's
does, so a multi-stage pipeline schedule can gather a stage while others
compute: call the handle's `wait()` before using the parameters directly, or
run the forward, which finishes its buckets' all-gathers.

Like `fully_shard` on a list of modules, a bucket whose patterns name several
modules handles forwards that skip some of them and calls of one of them on its
own. torchtitan's chunked loss does both: the decoder's forward returns the
norm's output without the output projection that shares its bucket, and the
loss then applies the projection to each chunk of the hidden states, with a
backward per chunk. The root module's post-forward completes the bucket's
forward, so the norm's backward re-gathers it. A call of the projection on its
own does not complete the bucket: its outputs get no pre-backward hook, and its
post-backward reduce-scatters, or keeps the gradients without sync, as soon as
its input gradients are computed. As FSDP2's final callback, the end of a
backward finishes the buckets it left unsharded and resets per-backward state
only if the backward re-gathered a bucket in a pre-backward hook; the backward
of such a call only waits for its reduce-scatters. Combined with the per-bucket
methods above, this reduce-scatters like FSDP2 under the chunked loss, bit for
bit.

## Outer shardings

FlexShard shards each parameter over its bucket's 1D mesh. A parameter that is
already a local shard of another sharding, such as tensor or expert
parallelism, is passed in as that plain local tensor:

- `set_global_layout(param, layout)`, before `flex_shard()`, declares where it
  sits in the full parameter. `get_outer_layout` returns the layout from the
  sharded parameter, and checkpoints compose it with FlexShard's own split.
- `set_partial_grad_group(param, group)` declares a group whose ranks each
  compute only part of the parameter's grad. One example is a norm weight
  under sequence parallelism, where each rank sees its own tokens. FlexShard
  sums the unsharded grad over that group before the reduce-scatter, as FSDP2
  does with a `Partial` grad on a mesh dim it does not shard over.

## Distributed checkpoints

A FlexShard `state_dict()` holds each rank's local shards, which share storage
with the sharded parameters. To checkpoint them with PyTorch distributed
checkpointing (DCP), declare where each shard sits in the full tensor, through
the fields of DCP's `CheckpointableTensor` protocol:

- `set_state_dict_global_layouts(model, state_dict)` declares them on a model
  state dict's parameters.
- `register_optimizer_checkpoint_hook(optimizer, model)` makes
  `optimizer.state_dict()` declare them on the optimizer's states. Register it
  once the parameters are sharded, as right after `flex_shard()`.

```python
import torch.distributed.checkpoint as dcp
from flex_shard import register_optimizer_checkpoint_hook, set_state_dict_global_layouts

register_optimizer_checkpoint_hook(optimizer, model)


def checkpoint_state_dict():
    model_state_dict = model.state_dict()
    set_state_dict_global_layouts(model, model_state_dict)
    return {"model": model_state_dict, "optim": optimizer.state_dict()}


dcp.save(checkpoint_state_dict(), checkpoint_id=path)

state_dict = checkpoint_state_dict()
dcp.load(state_dict, checkpoint_id=path)
model.load_state_dict(state_dict["model"])
optimizer.load_state_dict(state_dict["optim"])
```

DCP saves each shard as a chunk of the full tensor and loads chunks by their
global offsets. A checkpoint therefore loads at a different number of ranks,
and FSDP2 checkpoints of the same model load into FlexShard and back, as
[`test_flex_shard_checkpoint.py`](src/flex_shard/tests/test_flex_shard_checkpoint.py)
checks with AdamW. As for any `dcp.load` into `optimizer.state_dict()`, the
optimizer needs its states first, for example from a step with zero gradients.

The optimizer hook supports Adam and AdamW, whose states other than `step` are
elementwise and have their parameter's shape. `step` is saved as replicated.

The declarations are Python attributes of the state-dict tensors, so DCP has to
receive those tensor objects:

- A copy, such as a `.to(dtype)` cast before saving, needs
  `set_state_dict_global_layouts` again.
- DCP's process-based `async_save` (`AsyncCheckpointerType.PROCESS`) sends the
  state dict to its checkpoint process through `torch.multiprocessing`, whose
  tensor pickling keeps only the data. That process would save each shard as
  if it were the full tensor, without raising. The default thread-based
  `async_save` keeps the declarations.

## Limitations

- The API is experimental and uses private PyTorch APIs; the declared version
  range matters.
- Each bucket requires a one-dimensional CUDA mesh. Different buckets may use
  different submeshes, as in the MoE example. CPU offload is not supported.
- Every parameter must match exactly one bucket. The examples' `Shard(0)`
  placement does not support scalar parameters.
- Every rank must run the same graph. A bucket's parameters may get gradients
  on some ranks only (the others reduce zeros), but a bucket whose outputs
  only some ranks' losses use can reduce at a different point of backward on
  each rank, or, with `reshard_after_forward=True`, re-gather only on those
  ranks, so the collectives mismatch.
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
