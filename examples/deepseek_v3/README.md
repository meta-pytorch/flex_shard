# Train DeepSeek V3 16B with FlexShard

This example trains torchtitan's DeepSeek V3 16B, the `"16B"` flavor of
torchtitan's `deepseek_v3` model, on one host with 8 GPUs. It needs only PyTorch
and FlexShard; training on C4 adds `datasets` and `tokenizers`.

| File | Contents |
| --- | --- |
| [`model.py`](model.py) | The model in plain PyTorch |
| [`train.py`](train.py) | Sharding, the training loop, checkpoints |
| [`data.py`](data.py) | Seeded random tokens, or C4 packed into rows |
| [`muon.py`](muon.py) | Muon for parameters that hold batches of matrices |

| Model setting | Value |
| --- | --- |
| Blocks | 27: 1 dense, 26 mixture of experts |
| Hidden size | 2,048 |
| Vocabulary | 102,400, with an untied output projection |
| Attention | Multi-head latent attention, 16 heads; key-value latent of 512; query and key head dims 128 + 64 rotary, value head dim 128 |
| Rotary embedding | YaRN: theta 10,000, factor 40, original length 4,096 |
| Dense feed-forward | SwiGLU, hidden size 10,944 |
| Mixture of experts | 64 routed experts, top 6, hidden size 1,408, sigmoid scores normalized over the top 6; 2 shared experts; expert-bias load balancing and a balance loss with coefficient 0.001 |
| Parameters | 15.7 billion, 14.4 billion of them in routed experts; 2.7 billion active per token |

The tokenizer is deepseek-moe-16b-base's, as in torchtitan. That model's
architecture differs from this one's, so its weights don't load here.

## Run

From the repository root, with FlexShard installed:

```bash
torchrun --standalone --nproc-per-node 8 examples/deepseek_v3/train.py --steps 10
```

This trains on random tokens, at about 6,100 tokens per second per GPU with
38 GiB of GPU memory reserved. To train on C4 with FlexAttention, as
torchtitan's 16B recipe does, here with 4 sequences of 4,096 tokens per GPU
(the recipe's sequences have 16,384), first install the data requirements. The
tokenizer and C4 stream from the Hugging Face Hub.

```bash
pip install -r examples/deepseek_v3/requirements.txt
torchrun --standalone --nproc-per-node 8 examples/deepseek_v3/train.py \
    --data c4 --attention flex --local-batch-size 4 --steps 1000
```

Add `--optimizer muon` to train the matrices with Muon instead of AdamW.

| Option | Default | Meaning |
| --- | --- | --- |
| `--optimizer` | `adamw` | `adamw`, or `muon` for Muon on every matrix (AdamW keeps the embedding, norms and output projection) |
| `--attention` | `sdpa` | `sdpa`: causal within each sequence; `flex`: FlexAttention, causal within each document |
| `--data` | `synthetic` | `synthetic` or `c4` |
| `--tokenizer` | `deepseek-ai/deepseek-moe-16b-base` | Hub repo or local directory with `tokenizer.json` and `tokenizer_config.json` |
| `--steps` | 1,000 | Training steps; the learning-rate schedule spans them |
| `--seq-len`, `--local-batch-size` | 4,096, 1 | Sequence length and sequences per GPU per step |
| `--lr`, `--warmup-steps`, `--decay-ratio`, `--min-lr-factor` | 2.2e-4, 200, 0.8, 0.1 | torchtitan's schedule: linear warmup, then cosine decay to 0.1x over the last 80% of steps |
| `--weight-decay`, `--max-norm` | 0.1, 1.0 | Weight decay; gradient clipping norm |
| `--deterministic` | off | Bitwise reproducible runs, slower |
| `--checkpoint-dir`, `--checkpoint-interval` | none, 500 | Save checkpoints there, and resume from the latest |
| `--init-checkpoint` | none | Start from the model weights of a checkpoint |
| `--profile-dir` | none | Save each rank's profiler trace of step 5 |
| `--no-activation-checkpointing` | off | Keep all activations instead of recomputing each block |

Each step prints the loss averaged over all GPUs' tokens, the gradient norm, the
learning rate, tokens per second per GPU and peak memory.

## How the model is sharded

FlexShard shards every parameter over the 8 GPUs, in the groups torchtitan's
FSDP2 setup uses: one bucket for the embedding, one per block, and one for the
final norm with the output projection, which stays gathered between forward
and backward. `fsdp2_compatible=True` makes each bucket's all-gather and
reduce-scatter match that FSDP2 group's byte for byte.

```python
def bucket(patterns, reshard_after_forward=True):
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
buckets += [bucket([f"layers.{i}"]) for i in range(len(model.layers))]
buckets.append(bucket(["norm", "lm_head"], reshard_after_forward=False))
flex_shard(model, buckets=buckets)
```

The model is built on the meta device, sharded, and then initialized:
`init_from_full` has every rank draw each full parameter and keep its own part,
so the weights don't depend on the number of GPUs. Compute is in BF16;
gradients are reduced and AdamW updates in FP32.

There is no expert parallelism: every GPU gathers each layer's 64 experts,
about 1.1 GB in BF16, and runs its own tokens through them with
`torch._grouped_mm`. That is the same compute as expert parallelism, without
the all-to-all exchange of tokens.

## Muon without communication

Muon orthogonalizes each matrix's momentum with a Newton-Schulz iteration, so it
needs every matrix whole. `--optimizer muon` uses Kimi K2's matrices: each
attention projection split per head and per role (`wq` into its 128 no-rotary
and 64 rotary rows per head, `wkv_a` into the 512-row latent and the 64 rotary
rows, `wkv_b` into keys and values per head), `wo`, each feed-forward matrix,
the router, and each routed expert's matrices.

After initialization, `relayout_bucket` moves each block's parameters other than
the routed experts to whole-parameter owners (`BucketedOwned`), balanced over
the GPUs across blocks. The routed experts keep `Shard(0)`: each GPU stores 8
whole experts. Every Muon matrix then sits whole on one GPU, which updates it
without communication: profiles of the full model's optimizer step show no NCCL
kernel on any GPU. On a three-block cut of the model, its updates are bitwise
equal to the same Muon run on one GPU on the full parameters and gradients.

A mixture-of-experts block then has two buckets, the routed experts and the rest
of the block. Muon learns at the AdamW learning rate: each matrix's step is
scaled by `0.2 * sqrt(max(rows, columns))` to match AdamW's update size, as in
Kimi K2. The other Muon settings are Nesterov momentum 0.95, 5 Newton-Schulz
steps in BF16 and weight decay 0.1.

On 8 GPUs, Muon trains 8% fewer tokens per second than AdamW (see the results
below). Its optimizer step launches about 2,300 small kernels, 80 ms of GPU
work spread over about 200 ms; AdamW's step takes 26 ms. Without owners, each
step would also gather every matrix for its Newton-Schulz iteration and send
the update back to the matrix's shards.

## Checkpoints

With `--checkpoint-dir`, the example saves a PyTorch distributed checkpoint
(DCP) every `--checkpoint-interval` steps and at the end: the model, the
optimizer states, the learning-rate schedule, the step and each GPU's position
in its data. Rerunning the same command resumes from the latest one. A
checkpoint takes 176 GiB with AdamW and 119 GiB with Muon, whose weights are
then stored on their owners. With `--deterministic`, 4 steps, a resume and 4 more
steps match 8 straight steps bit for bit, with either optimizer, on C4.

`--init-checkpoint` loads only the model weights from a checkpoint whose
parameter names match this model's, which are torchtitan's. The one exception
is the buffer of expert biases: `expert_bias` here, `expert_bias_E` in
torchtitan.

## Matching torchtitan

The model follows torchtitan's op for op, including the router gate's FP32 output
and close-to-FP32 gradients, the routed tokens' order and the expert weighting
in FP32. A developer check, not part of this example, compares it with
torchtitan itself:
- torchtitan's trainer runs the 16B recipe with FSDP2 on 8 GPUs, without expert
  parallelism, and saves its initial weights;
- this example loads them with `--init-checkpoint`;
- both run 10 steps with `--deterministic` on the same random batches of 4,096
  tokens per GPU.

Losses and gradient norms are bitwise equal at every step, with SDPA (swapped
into torchtitan, which no longer offers it for language models) and with
FlexAttention. For the comparison, torchtitan runs plain cross entropy instead
of its chunked loss and no compiled regions; its selective activation
checkpointing doesn't change the numbers.

## Attention

FlashAttention needs equal query and value head dims, so multi-head latent
attention, with 192 and 128, runs on cuDNN or on the memory-efficient kernel.
Attention forward and backward for one block (16 heads, 4,096 tokens), measured
on one H100:

| Kernel | Time | Deterministic |
| --- | --- | --- |
| SDPA, cuDNN (`--attention sdpa`) | 6.3 ms | No; under `--deterministic` SDPA takes the memory-efficient kernel |
| SDPA, memory-efficient | 27.9 ms | Yes |
| FlexAttention (`--attention flex`) | 3.1 ms | Yes |

FlexAttention is compiled, without autotuning; the rest of the model runs
eagerly. It is the kernel torchtitan trains with, and it keeps documents apart
in packed C4 rows. SDPA attends causally across a whole row.

## Results on 8 H100s

1,000 steps on C4 with the command above (FlexAttention, 4 x 4,096 tokens per
GPU, 131,072 tokens per step), AdamW and Muon at the same learning rate:

| | AdamW | Muon |
| --- | --- | --- |
| Mean loss, steps 51 to 100 | 6.58 | 8.36 |
| Mean loss, steps 201 to 250 | 5.47 | 5.65 |
| Mean loss, steps 451 to 500 | 4.69 | 4.66 |
| Mean loss, steps 701 to 750 | 4.34 | 4.28 |
| Mean loss, steps 951 to 1,000 | 4.18 | 4.14 |
| Tokens per second per GPU, median of steps 11 to 1,000 | 10,417 | 9,602 |
| Peak GPU memory, active and reserved | 42.9 and 58.1 GiB | 35.8 and 51.4 GiB |
| Time for 1,000 steps, with startup | 27.5 min | 30.0 min |

Muon starts slower, passes AdamW by step 500 and ends lower. It needs less
memory because it keeps one state per matrix, where AdamW keeps two. The active
memory comes from 40-step runs with the same settings.

## Limitations

- One host with 8 GPUs; no expert, tensor, pipeline or context parallelism.
- No gradient accumulation, no chunked loss, and activation checkpointing of
  whole blocks only.
- Tested on H100s with PyTorch 2.16 (development build).
