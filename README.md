# FlexShard

FlexShard is an experimental data-parallel sharding library for PyTorch. Like
FSDP2, it shards parameters, gradients and optimizer states, all-gathers
parameters for forward and backward, and reduce-scatters gradients. Unlike FSDP2,
you choose how each group of parameters is stored: a bucket names the parameters
that communicate together, their device mesh and their placement. Optimizers see
ordinary local tensors.

- **FSDP2 parity.** With `Shard` placements, a bucket can reproduce an FSDP2
  `fully_shard` group's collectives byte for byte, so training is bitwise
  identical.
- **Muon without communication.** Owner storage and `BlockShard` keep each
  matrix whole on the GPU that orthogonalizes it.
- **FP8 all-gathers (experimental).** `Fp8BucketedBlockShard` keeps whole
  quantization blocks on each GPU, which quantizes locally and gathers FP8.
- **Your own placements**, on one runtime for bucketing, prefetching and overlap.

## Why FlexShard

### Muon needs whole matrices

<p align="center"><img src="docs/images/muon_whole_matrices.svg" width="760" alt="FSDP2's Shard(0) stores 896 rows of a [7168, 2048] matrix on each of 8 ranks; FlexShard's Owned stores the whole matrix on rank 3"></p>

Muon orthogonalizes each weight matrix, so the matrix must be whole on one GPU.
FSDP2's `Shard(0)` puts a slice of every matrix on every GPU, so each optimizer
step gathers matrices and sends their updates back. FlexShard's `BucketedOwned`
keeps whole matrices on owner GPUs, and `BlockShard` keeps whole blocks of rows,
such as attention heads, so the optimizer step runs on local storage.

<p align="center"><img src="docs/images/dist_muon_timeline.svg" width="760" alt="With FSDP2, each bucket's exchange waits for that bucket's slowest owner, so a made-up example takes 6 + 5 + 4 = 15 units plus 6 exchanges; with FlexShard each rank runs its own matrices and the slowest rank takes 8"></p>

Removing the exchange also removes the waiting. With exchanges, every bucket
waits for its slowest GPU, so the step takes the sum of those waits; with local
storage, it takes the slowest GPU's own total. On DeepSeek V3 671B with 256
H100s, the slowest GPU's optimizer step went from 527 ms to 301 ms, 43% shorter
(timed on the GPU; the 69-second training step varies more than that between
jobs).

### Blockwise quantization needs whole tiles

<p align="center"><img src="docs/images/block_shard_tiles.svg" width="760" alt="Shard(0) cuts some 128 x 128 tiles between ranks; BlockShard cuts only between tiles"></p>

Blockwise FP8 gives each 128 x 128 tile of a weight its own scale. FSDP2's
equal shards can split a tile between GPUs, so the weight is either gathered in
BF16 and quantized after, or quantized after an all-reduce of each tile's amax.
`Fp8BucketedBlockShard` gives each GPU whole 128-row blocks instead. It is
experimental; the newest FP8 work is on the
[`fp8-stack`](https://github.com/meta-pytorch/flex_shard/tree/fp8-stack) branch.

<p align="center"><img src="docs/images/fp8_all_gather.svg" width="760" alt="Three pipelines: BF16 all-gather then quantize; FSDP2's amax all-reduce, quantize, FP8 all-gather; FlexShard's local quantize then FP8 all-gather"></p>

Each GPU quantizes its own tiles, and the all-gather moves FP8 data plus one
scale per tile, about half the bytes of BF16. The gathered weight is bitwise
equal to gathering BF16 and quantizing after.

### Placements are pluggable

A placement decides which part of each parameter a GPU stores and how its bucket
gathers parameters and reduces gradients. One runtime handles the rest:
bucketing, prefetching, overlap, resharding, gradient accumulation and
checkpoints. A new storage scheme is a `Placement` subclass
([`placement_contract.py`](src/flex_shard/flex_shard/placement_contract.py)).
Built in:

| Placement | Each GPU stores | As in |
| --- | --- | --- |
| `Shard(i)` | an equal slice of each parameter, along dim `i` | FSDP2 |
| `BucketedBlockShard` | a range of one flat buffer per bucket | FSDP1, Megatron's distributed optimizer, veScale-FSDP |
| `BucketedOwned` | whole parameters, or whole slabs along dim 0 | new |
| `BlockShard` | whole blocks of a parameter, along one dim | new |
| `Fp8BucketedBlockShard` | whole 128-row blocks, gathered as FP8 | new |

## Install

FlexShard needs Linux with NVIDIA GPUs, Python 3.10 or later, Triton `~=3.8.0`,
and a CUDA build of PyTorch 2.16 or later; until 2.16 is released, use a nightly
from 2.16.0.dev20261009 on.

```bash
git clone https://github.com/meta-pytorch/flex_shard.git
cd flex_shard
python -m pip install -e .  # installs flex-shard; import flex_shard
```

## Concepts

| Term | Meaning |
| --- | --- |
| Bucket | Parameters that share one all-gather and one reduce-scatter, named by `BucketSpec(patterns, ...)` |
| Storage sharding | How a bucket stores its parameters between uses: its `placement_fn` and `mesh` |
| Compute sharding | How parameters are laid out when used: replicated in forward and backward, sharded further by tensor or expert parallelism ([outer shardings](#outer-shardings)), or in an optimizer's layout, such as DistMuon's |

Coming from FSDP2:

| FSDP2 | FlexShard |
| --- | --- |
| `fully_shard` on child modules, then the root | One `flex_shard` call with explicit buckets |
| `shard_placement_fn` returning `Shard(i)` | A `placement_fn` returning `Shard(i)`; one bucket may mix dims |
| `fully_shard(layer.experts, mesh=efsdp_mesh)` | An expert `BucketSpec(..., mesh=efsdp_mesh)` |
| `set_modules_to_forward_prefetch` / `set_modules_to_backward_prefetch` | `set_buckets_to_forward_prefetch` / `set_buckets_to_backward_prefetch` on `sharded_bucket_storages` entries |
| `unshard()`, `reshard()`, `set_reshard_after_forward` on a group | The same methods on a bucket's storage, from `sharded_bucket_storages` or `model.bucket_storage_of(param)` |
| The optimizer holds DTensor parameters | The optimizer holds local shards; build it after sharding |
| DCP saves DTensor state dicts | DCP saves local shards that declare their layouts ([Distributed checkpoints](#distributed-checkpoints)) |

## Example: DeepSeek V3 16B

[`examples/deepseek_v3`](examples/deepseek_v3) trains torchtitan's DeepSeek V3
16B in plain PyTorch on one host with 8 GPUs: 27 blocks, 26 of them with 64
routed experts, 15.7 billion parameters with 2.7 billion active per token, and
deepseek-moe-16b-base's tokenizer, as in torchtitan.

```bash
pip install -r examples/deepseek_v3/requirements.txt  # for --data c4
torchrun --standalone --nproc-per-node 8 examples/deepseek_v3/train.py \
    --data c4 --local-batch-size 4 --steps 1000 --optimizer muon
```

Drop `--optimizer muon` for AdamW, and `--data c4` for random tokens. Other
flags: `--ep` sets the expert-parallel degree (`--ep 1` turns it off),
`--checkpoint-dir` saves and resumes, `--deterministic` makes runs bitwise
reproducible, and `--help` lists the rest.

### Expert parallelism plus FSDP

By default, expert parallelism (EP) spans half the GPUs: GPUs 0 to 3 split each
block's 64 experts, 16 each, and so do GPUs 4 to 7.
[`expert_parallel.py`](examples/deepseek_v3/expert_parallel.py) ports
torchtitan's all-to-all token dispatch. FlexShard shards each GPU's 16 experts
over the GPUs holding the same ones (GPUs 0 and 4, 1 and 5, and so on), and
everything else over all 8 GPUs:

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
        gradient_divide_factor=1.0,  # sum gradients: the loss is divided by all GPUs' tokens
        reshard_after_forward=reshard_after_forward,
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

`apply_expert_parallel` declares where each GPU's experts sit in the full
weights (`set_global_layout`), so `init_from_full` and checkpoints still see full
weights. Gradients are summed rather than averaged because each expert's
gradient already includes the tokens of its EP peers; the loss is divided by all
GPUs' tokens, as in torchtitan. Against torchtitan's FSDP2 with the same EP,
from the same weights and batches, losses and gradient norms are bitwise equal
over 10 deterministic steps, with SDPA and with FlexAttention, and with or
without EP.

### Muon without communication

With `--optimizer muon`, each matrix is stored where Muon computes it. The routed
experts already sit whole on their GPUs, 8 per GPU, and after initialization
`relayout_bucket` moves the rest of each block to whole-parameter owners:

```python
placement_fn = make_bucketed_owned_full_param_placement_fn(carry_loads=True)
for storage in owner_buckets:  # each block's non-expert bucket
    relayout_bucket(model, storage, placement_fn)
```

`build_local_dist_muon` then runs `flex_shard.dist_muon.DistMuon` on each GPU's
own storage ([`muon.py`](examples/deepseek_v3/muon.py)). The step launches no
NCCL kernel, and on a three-block model its updates equal DistMuon's on the full
tensors, bit for bit. The compute layouts follow Kimi K2 (attention per head and
role, routed experts per expert, the rest whole), with its update scale,
`0.2 * sqrt(max(rows, columns))`; AdamW trains the embedding, norms and output
projection.

## Gradients, resharding and accumulation

- **Reduction.** `gradient_reduce_op` defaults to `AVG`: the sum over the
  bucket's mesh divided by `gradient_divide_factor`, the mesh size by default,
  like FSDP2's `set_gradient_divide_factor`. With expert parallelism, set it to
  the dense data-parallel size on expert buckets.
- **Resharding.** Each parameter has a persistent unsharded `nn.Parameter`.
  `reshard_after_forward=True` frees it after forward and re-gathers it for
  backward, also under activation checkpointing; with `False`, call
  `model.reshard()` when no backward follows.
- **Accumulation.** Call `model.set_requires_gradient_sync(is_last)` before each
  microbatch, as in FSDP2; `model.set_reshard_after_backward(False)` also keeps
  parameters gathered between microbatches. Both exist per bucket too.
- **Finalization.** `model.finalize_backward()` reduce-scatters accumulated
  gradients outside a backward; after
  `model.set_manual_backward_finalization(True)`, backwards leave it to one call
  per step.
- **Late weight gradients.** `BucketSpec(defer_post_backward=True)` waits for
  `model.finish_deferred_backward(param)`, for kernels such as
  TransformerEngine's `delay_wgrad_compute`.
- **Manual gathers.** `model.unshard()` gathers every bucket, for schedules that
  bypass forward hooks.
- **Partial forwards.** A bucket over several modules handles forwards that skip
  some of them, as torchtitan's chunked loss does.
- **Errors.** FlexShard doesn't recover: after a backward raises, the next
  forward or `model.reshard()` raises too.

`finalize_backward()` and `unshard()` take `async_op=True` and then return a
handle.

## Outer shardings

A parameter that is already a local shard of another sharding, such as tensor or
expert parallelism, goes in as that plain local tensor, declared before
`flex_shard()`:

- `set_global_layout(param, layout)` says where it sits in the full parameter;
  checkpoints compose it with FlexShard's split.
- `set_partial_grad_group(param, group)` names a group whose ranks each compute
  part of its gradient, such as a norm weight under sequence parallelism;
  FlexShard sums over that group before the reduce-scatter.

## Distributed checkpoints

A FlexShard `state_dict()` holds local shards. For PyTorch distributed
checkpointing (DCP), declare where each shard sits in the full tensor:

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

- DCP loads shards by global offset, so checkpoints load at another number of
  GPUs, and FSDP2 checkpoints of the same model load into FlexShard and back.
- As with any `dcp.load` into `optimizer.state_dict()`, the optimizer needs its
  states first, for example from a step with zero gradients. The optimizer hook
  supports Adam and AdamW.
- The layouts are attributes of the state-dict tensors: after a copy, such as a
  `.to(dtype)` cast, call `set_state_dict_global_layouts` again. DCP's
  process-based `async_save` silently drops them; the default thread-based one
  keeps them.

## Used in

- [weifengpy/torchtitan-flexshard](https://github.com/weifengpy/torchtitan-flexshard):
  torchtitan with `parallelism.fsdp_backend="flex_shard"`, bitwise like its FSDP2
  backend, and DistMuon from `flex_shard.dist_muon`.
- [weifengpy/Megatron-flexshard](https://github.com/weifengpy/Megatron-flexshard):
  Megatron-LM with FlexShard as a data-parallel backend (`--use-flex-shard`).

## Limitations

- The API is experimental and uses private PyTorch APIs, so the PyTorch version
  matters.
- Each bucket needs a one-dimensional CUDA mesh; buckets may use different
  submeshes. No CPU offload.
- Every parameter must match exactly one bucket.
- Every rank must run the same graph: if only some ranks' losses use a bucket's
  outputs, it may reduce at a different point of each rank's backward, and the
  collectives mismatch.
- The DeepSeek V3 example runs on one host, with expert parallelism but no
  tensor, pipeline or context parallelism; `--ep` must divide the GPUs and the
  64 experts.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
```

The full suite needs at least four GPUs; tests skip on missing hardware, and
CPU-only runs don't validate distributed training.
[`docs/images/make_figures.py`](docs/images/make_figures.py) redraws the figures
(with matplotlib). See [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

FlexShard is BSD-3-Clause licensed; see [LICENSE](LICENSE).
