# FlexShard

FlexShard is an experimental parameter-sharding library for PyTorch. Explicit
`BucketSpec`s choose which parameters communicate together, which device mesh
they use, and how they are sharded. Optimizers work on ordinary local tensors.

Because each bucket decides how its parameters are stored, it can store them the
way the optimizer computes on them. [Train DeepSeek V3 16B](#train-deepseek-v3-16b)
shows both sides: AdamW with expert parallelism and buckets that match FSDP2 bit
for bit, and Muon with every matrix stored whole on the GPU that orthogonalizes
it, so its optimizer step needs no communication.

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

The distribution is named `flex-shard`; import it as `flex_shard`.

## Train DeepSeek V3 16B

[`examples/deepseek_v3`](examples/deepseek_v3) trains torchtitan's DeepSeek V3
16B in plain PyTorch, without torchtitan, on one host with 8 GPUs. The model has
27 blocks, one dense and 26 with 64 routed experts (top 6) and 2 shared experts,
multi-head latent attention and 15.7 billion parameters, 2.7 billion of them
active per token. It uses deepseek-moe-16b-base's tokenizer, as torchtitan does;
that model's weights don't fit this architecture.

```bash
pip install -r examples/deepseek_v3/requirements.txt  # datasets and tokenizers, for C4
torchrun --standalone --nproc-per-node 8 examples/deepseek_v3/train.py \
    --data c4 --local-batch-size 4 --steps 1000 --optimizer muon
```

Leave out `--optimizer muon` for AdamW, and `--data c4` for random tokens and no
extra packages. `--ep` sets the expert-parallel degree, `--ep 1` turning expert
parallelism off; `--checkpoint-dir` saves DCP checkpoints and resumes from the
latest; `--deterministic` makes runs bitwise reproducible; `--help` lists the
rest.

### Expert parallelism, and buckets like FSDP2's

The routed experts use expert parallelism (EP) over `--ep` GPUs, half of them by
default: GPUs 0 to 3 split each block's 64 experts, 16 each, and so do GPUs 4 to
7. [`expert_parallel.py`](examples/deepseek_v3/expert_parallel.py) ports
torchtitan's all-to-all token dispatch: each routed copy of a token goes to the
GPU holding its expert and its output comes back, one all-to-all each way.
FlexShard then shards each GPU's 16 experts over the GPUs that hold the same
ones (expert FSDP, EFSDP): GPUs 0 and 4, 1 and 5, and so on, 8 experts each.
Everything else is sharded over all 8 GPUs (FSDP).

The model is built on the meta device, sharded, then initialized with
`init_from_full`, which has every rank draw each full parameter and keep its
part, so the weights don't depend on the number of GPUs. `apply_expert_parallel`
declares where each GPU's experts sit in the full weights (`set_global_layout`),
so initialization and checkpoints see full weights there too. Each bucket is one
group of torchtitan's FSDP2 setup:

```python
mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("dp",))
expert_mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("efsdp", "ep"))
apply_expert_parallel(model, expert_mesh["ep"])  # keep this GPU's experts


def bucket(patterns, reshard_after_forward=True, mesh=mesh):
    return BucketSpec(
        patterns,
        placement_fn=placement_fn,  # Shard(1) for [2, F, D] gate/up weights, else Shard(0)
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        gradient_divide_factor=1.0,  # the loss is divided by the global token count
        reshard_after_forward=reshard_after_forward,
        fsdp2_compatible=True,
    )


buckets = [bucket(["tok_embeddings"])]
for i, block in enumerate(model.layers):
    names = [f"layers.{i}.{name}" for name, _ in block.named_parameters()]
    experts = [name for name in names if ".routed_experts." in name]
    buckets.append(bucket([name for name in names if name not in experts]))
    if experts:
        buckets.append(bucket(experts, mesh=expert_mesh["efsdp"]))
buckets.append(bucket(["norm", "lm_head"], reshard_after_forward=False))
flex_shard(model, buckets=buckets)
```

`fsdp2_compatible=True` orders a bucket's parameters as `fully_shard` does, so
its all-gather and reduce-scatter buffers match that FSDP2 group's byte for
byte. As torchtitan, the example prefetches the next block's buckets explicitly
under EP, and clips with the experts' gradient norm taken over their two mesh
dims in order, as DTensor reduces it. From torchtitan's initial weights and on
the same batches, losses and gradient norms are bitwise equal to torchtitan's
FSDP2 training with the same EP at every step of a 10-step deterministic run,
with SDPA and with FlexAttention, and so they are with `--ep 1`.

| FSDP2 | FlexShard |
| --- | --- |
| Apply `fully_shard` to child modules, then the root | Call `flex_shard` once with explicit buckets |
| `shard_placement_fn` returning `Shard(i)` | A `placement_fn` returning `Shard(i)`; one bucket may mix dims |
| `fully_shard(layer.experts, mesh=efsdp_mesh)` | Expert `BucketSpec(..., mesh=efsdp_mesh)` |
| `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` | `set_buckets_to_forward_prefetch` / `set_buckets_to_backward_prefetch` on entries of `sharded_bucket_storages` |
| `unshard()`, `reshard()` and `set_reshard_after_forward` on one `fully_shard` group | The same methods on its bucket's storage, from `sharded_bucket_storages` or `model.bucket_storage_of(param)` |
| The optimizer manages DTensor parameters | The optimizer manages ordinary local shards; construct it after sharding |
| DCP checkpoints DTensor state dicts | DCP checkpoints local shards that declare their layouts; see [Distributed checkpoints](#distributed-checkpoints) |

### Muon without communication

Muon orthogonalizes each matrix's momentum with a Newton-Schulz iteration, so
each matrix has to be whole on the GPU that runs it. FSDP2 stores a slice of
every matrix on every GPU, so each optimizer step would gather every matrix to
one GPU and send the result back. With `--optimizer muon`, FlexShard stores each
matrix where Muon computes it instead:

- A mixture-of-experts block's routed experts already have a bucket of their
  own, whose `Shard(0)` keeps experts whole: each GPU stores 8.
- After initialization or loading, `relayout_bucket` moves the rest of each block
  to whole-parameter owners (`BucketedOwned`), balanced over the GPUs across
  blocks. Forward and backward gather each parameter from its owner and reduce
  its gradient back to it.

```python
placement_fn = make_bucketed_owned_full_param_placement_fn(carry_loads=True)
for storage in owner_buckets:  # each block's bucket other than the routed experts
    relayout_bucket(model, storage, placement_fn)
```

`build_local_dist_muon` then runs `flex_shard.dist_muon.DistMuon` on each GPU's
own storage, on a one-GPU mesh where storage already is the compute layout, so
the step moves nothing ([`muon.py`](examples/deepseek_v3/muon.py)). Profiles of
the optimizer step show no NCCL kernel on any GPU, and on a three-block cut of
the model its updates equal DistMuon's on the full parameters and gradients,
bit for bit. The compute layouts are Kimi K2's matrices (attention projections
per head and role, each expert, the rest whole), with its update scaling,
`0.2 * sqrt(max(rows, columns))`, so Muon takes AdamW's learning rate; AdamW
trains the embedding, norms and output projection. Checkpoints hold the
owner-stored weights and DistMuon's momentum.

### Results on 8 H100s

1,000 steps on C4 with the command above and `--ep 1` (FlexAttention, 4 x 4,096
tokens per GPU, torchtitan's learning-rate schedule):

| | AdamW | Muon |
| --- | --- | --- |
| Mean loss, steps 451 to 500 | 4.69 | 4.66 |
| Mean loss, steps 951 to 1,000 | 4.18 | 4.14 |
| Tokens per second per GPU, median | 10,417 | 9,589 |
| Peak memory, active and reserved | 42.9 and 58.1 GiB | 36.4 and 51.8 GiB |
| Optimizer step on the GPU | 26 ms | 204 ms, 97% busy |

On one host, the exchange the owners remove would cost little: this run shows
the mechanism, not its speedup. Muon's step is Newton-Schulz matmuls at 646
TFLOP/s (112 ms) and element-wise updates (85 ms), against AdamW's 26 ms, so
Muon trains 8% fewer tokens per second than AdamW here. With `--deterministic`,
a resumed run matches an uninterrupted one bit for bit, with either optimizer.

Attention is FlexAttention, compiled without autotuning: torchtitan's kernel,
which keeps packed documents apart and is deterministic; the rest of the model
runs eagerly. FlashAttention needs equal query and value head dims, so
`--attention sdpa` runs multi-head latent attention (192 and 128) on cuDNN,
which is faster but not deterministic. Per block at 4,096 tokens, forward and
backward on the GPU: cuDNN 0.9 ms, FlexAttention 2.0 ms, and SDPA's
memory-efficient kernel, which it uses with `--deterministic`, 27.6 ms. End to
end, FlexAttention trains 8% fewer tokens per second than SDPA on cuDNN.

## Smaller example and tracing

[`examples/train_adamw.py`](examples/train_adamw.py) trains a two-block toy MoE
for three AdamW steps on 4 GPUs: dense parameters on a 4-rank `dp` mesh, each
block's experts on a 2-rank `efsdp` submesh. With `--trace`, it captures the
model with `torch.compile(fullgraph=True)` and checks that each bucket's
all-gather and reduce-scatter appear as functional collectives in the graph,
as [`test_torch_compile_traces_per_bucket_collectives`](src/flex_shard/tests/test_flex_shard_runtime.py)
does. Under `torch.compile`, the traced graph owns buffer lifetimes, so
`reshard_after_forward` doesn't apply.

```bash
torchrun --standalone --nproc-per-node=4 examples/train_adamw.py --trace --graph-dir /tmp/trace
```

## Gradients, resharding and accumulation

- `gradient_reduce_op` defaults to `dist.ReduceOp.AVG`, the sum over the bucket's
  mesh divided by `gradient_divide_factor`, the mesh size by default, as FSDP2's
  `set_gradient_divide_factor`. With expert parallelism, set it to the dense
  data-parallel size on expert buckets.
- Each managed parameter has a persistent unsharded `nn.Parameter`, which the
  owning module holds while its bucket is gathered. `reshard_after_forward=True`
  frees it after forward and re-gathers it before backward, composing with
  activation checkpointing. With `False`, call `model.reshard()` when the
  backward won't run.
- Gradient accumulation: `model.set_requires_gradient_sync(is_last)` before each
  microbatch, as with FSDP2. Backwards without sync keep full gradients on the
  unsharded parameters; `model.set_reshard_after_backward(False)` also keeps the
  parameters gathered between microbatches. Both setters also exist per bucket.
- `model.finalize_backward()` reduce-scatters accumulated gradients outside a
  backward (`async_op=True` returns a handle).
  `model.set_manual_backward_finalization(True)` makes backwards finish nothing,
  leaving it to one `finalize_backward()` per step.
- `BucketSpec(defer_post_backward=True)` waits for
  `model.finish_deferred_backward(param)`, for kernels that compute weight
  gradients late, such as TransformerEngine's `delay_wgrad_compute`.
- `model.unshard()` gathers every bucket for schedules that bypass forward hooks;
  `async_op=True` returns a handle, as in FSDP2.
- A bucket naming several modules handles forwards that skip some of them and
  calls of one of them alone, as torchtitan's chunked loss does, and
  reduce-scatters like FSDP2 bit for bit.
- FlexShard does not recover from errors: after a backward that raises, the next
  forward or `model.reshard()` raises too.

## Outer shardings

A parameter that is already a local shard of another sharding, such as tensor or
expert parallelism, is passed in as that plain local tensor:

- `set_global_layout(param, layout)`, before `flex_shard()`, declares where it
  sits in the full parameter; checkpoints compose it with FlexShard's split.
- `set_partial_grad_group(param, group)` declares a group whose ranks each
  compute only part of its gradient, such as a norm weight under sequence
  parallelism; FlexShard sums it over that group before the reduce-scatter.

## Distributed checkpoints

A FlexShard `state_dict()` holds each rank's local shards. To checkpoint them
with PyTorch distributed checkpointing (DCP), declare where each shard sits in
the full tensor:

```python
import torch.distributed.checkpoint as dcp
from flex_shard import register_optimizer_checkpoint_hook, set_state_dict_global_layouts

register_optimizer_checkpoint_hook(optimizer, model)  # once, after flex_shard()


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

DCP loads shards by their global offsets, so a checkpoint loads at another
number of ranks, and FSDP2 checkpoints of the same model load into FlexShard
and back. As for any `dcp.load` into `optimizer.state_dict()`, the optimizer
needs its states first, for example from a step with zero gradients. The
optimizer hook supports Adam and AdamW. The declarations are attributes of the
state-dict tensors: a copy, such as a `.to(dtype)` cast, needs
`set_state_dict_global_layouts` again. DCP's process-based `async_save` drops
them and saves each shard as if it were the full tensor, without raising; the
default thread-based one keeps them.

## Limitations

- The API is experimental and uses private PyTorch APIs; the declared version
  range matters.
- Each bucket requires a one-dimensional CUDA mesh; different buckets may use
  different submeshes. CPU offload is not supported.
- Every parameter must match exactly one bucket.
- Every rank must run the same graph. A bucket whose outputs only some ranks'
  losses use can reduce at a different point of backward on each rank, so the
  collectives mismatch.
- The DeepSeek V3 example runs on one host, with expert parallelism but no
  tensor, pipeline or context parallelism; `--ep` must divide the GPUs and the
  64 experts.
- Numerical comparisons here concern the stated configurations; they are not
  general FSDP2 benchmarks.

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
