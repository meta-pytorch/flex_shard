# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Train DeepSeek V3 16B with FlexShard on the GPUs of one node.

    torchrun --nproc-per-node 8 examples/deepseek_v3/train.py --steps 10

FlexShard shards the parameters the way torchtitan's FSDP2 setup does, with
BF16 compute, FP32 gradient reduction and FP32 AdamW. The routed experts use
expert parallelism (EP): each group of ``--ep`` ranks splits them, and FlexShard
shards each rank's experts over the ranks holding the same ones (expert FSDP,
EFSDP). Everything else is sharded over all ranks (FSDP), one bucket per
transformer block plus the embedding and the output layers; a MoE block's
routed experts get a bucket of their own. Each rank trains on its own batches;
the loss is averaged over all ranks' tokens. With ``--optimizer muon``, each
block's matrices move to owner ranks and flex_shard's DistMuon updates them
there, without communication.
"""

import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from data import C4Batches, SyntheticBatches
from expert_parallel import apply_expert_parallel
from flex_shard import (
    BucketSpec,
    flex_shard,
    get_flex_shard_global_layouts,
    init_from_full,
    MixedPrecisionPolicy,
    register_optimizer_checkpoint_hook,
    relayout_bucket,
    set_state_dict_global_layouts,
)
from flex_shard.custom_placements import make_bucketed_owned_full_param_placement_fn
from flex_shard.custom_placements.shard import Shard
from flex_shard.dist_muon import capture_flex_shard_muon_canonical_shards
from model import (
    DeepSeekV3,
    DeepSeekV3Config,
    GateUpLinear,
    param_init_fns,
    update_expert_biases,
)
from muon import build_muon, kimi_matrices

IGNORE_INDEX = -100


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument(
        "--local-batch-size", type=int, default=1, help="sequences per rank per step"
    )
    parser.add_argument(
        "--ep",
        type=int,
        help="expert-parallel degree; by default half the ranks, so that each "
        "rank's experts are also FSDP-sharded over 2 ranks (EFSDP)",
    )
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--attention", choices=["flex", "sdpa"], default="flex")
    parser.add_argument("--data", choices=["synthetic", "c4"], default="synthetic")
    parser.add_argument(
        "--tokenizer",
        default="deepseek-ai/deepseek-moe-16b-base",
        help="Hugging Face Hub repo or local directory with tokenizer.json, for --data c4",
    )
    parser.add_argument("--lr", type=float, default=2.2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument(
        "--decay-ratio",
        type=float,
        default=0.8,
        help="fraction of steps spent decaying",
    )
    parser.add_argument("--min-lr-factor", type=float, default=0.1)
    parser.add_argument(
        "--max-norm", type=float, default=1.0, help="gradient clipping norm"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    parser.add_argument(
        "--deterministic", action="store_true", help="bitwise reproducible runs, slower"
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="save here every --checkpoint-interval steps; resume from the latest",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument(
        "--init-checkpoint", help="DCP checkpoint whose model weights to start from"
    )
    parser.add_argument(
        "--profile-dir", help="save a profiler trace of step 5 per rank here"
    )
    return parser.parse_args()


def shard_model(model: DeepSeekV3, mesh, *, muon: bool, expert_mesh=None) -> list[int]:
    """Shard the parameters over ``mesh`` in the groups torchtitan's FSDP2 setup
    uses. ``fsdp2_compatible`` makes each bucket's all-gather and reduce-scatter
    match that FSDP2 group's byte for byte.

    With expert parallelism, or for Muon, a MoE block's routed experts get a
    bucket of their own: over ``expert_mesh``, the ranks holding the same
    experts, or over ``mesh`` without expert parallelism. For Muon, the rest of
    each block can then move to owners (``store_on_owners``). Returns the
    indices of those buckets.
    """
    # Shard the [2, F, D] gate/up weights along F, so that each rank keeps rows
    # of both projections. Everything else, routed experts included, shards
    # along dim 0.
    shard_dims = {m.weight: 1 for m in model.modules() if isinstance(m, GateUpLinear)}

    def placement_fn(named_params, mesh):
        return {fqn: (Shard(shard_dims.get(p, 0)),) for fqn, p in named_params}

    def bucket(patterns, reshard_after_forward=True, mesh=mesh):
        return BucketSpec(
            patterns,
            placement_fn=placement_fn,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
            # The loss is already divided by the global token count: sum grads.
            gradient_divide_factor=1.0,
            reshard_after_forward=reshard_after_forward,
            fsdp2_compatible=True,
        )

    buckets = [bucket(["tok_embeddings"])]
    block_buckets, expert_buckets, owner_buckets = [], [], []
    for i, block in enumerate(model.layers):
        names = [f"layers.{i}.{name}" for name, _ in block.named_parameters()]
        experts = [name for name in names if ".routed_experts." in name]
        if not muon and not (experts and expert_mesh is not None):
            block_buckets.append([len(buckets)])
            buckets.append(bucket([f"layers.{i}"]))
            continue
        block_buckets.append([len(buckets)])
        if muon:
            owner_buckets.append(len(buckets))
        buckets.append(bucket([name for name in names if name not in experts]))
        if experts:
            block_buckets[-1].append(len(buckets))
            expert_buckets.append(len(buckets))
            buckets.append(bucket(experts, mesh=expert_mesh or mesh))
    # The output layers run last in forward and first in backward, so they stay
    # gathered in between.
    buckets.append(bucket(["norm", "lm_head"], reshard_after_forward=False))
    if muon:
        # build_local_dist_muon reads each parameter's coordinates from here.
        capture_flex_shard_muon_canonical_shards(model)
    flex_shard(model, buckets=buckets)
    if expert_mesh is not None:
        prefetch_blocks_in_order(
            list(model.sharded_bucket_storages), block_buckets, expert_buckets
        )
    return owner_buckets


def prefetch_blocks_in_order(storages, block_buckets, expert_buckets) -> None:
    """Prefetch the next block's buckets in forward and the previous block's in
    backward, as torchtitan does with expert parallelism: the dispatch waits
    for token counts on the host, which keeps the CPU from issuing the next
    all-gathers early otherwise.

    A block prefetches from its first bucket, which runs first; its expert
    bucket prefetches nothing.
    """
    blocks = [[storages[i] for i in buckets] for buckets in block_buckets]
    embedding, head = storages[0], storages[-1]
    for i in expert_buckets:
        storages[i].set_buckets_to_forward_prefetch([])
        storages[i].set_buckets_to_backward_prefetch([])
    embedding.set_buckets_to_forward_prefetch(blocks[0])
    head.set_buckets_to_backward_prefetch(blocks[-1])
    for i, block in enumerate(blocks):
        block[0].set_buckets_to_forward_prefetch(
            blocks[i + 1] if i + 1 < len(blocks) else [head]
        )
        block[0].set_buckets_to_backward_prefetch(
            blocks[i - 1] if i > 0 else [embedding]
        )


def store_on_owners(model, bucket_indices) -> None:
    """Move each bucket's parameters to whole-parameter owners, balanced over
    ranks across buckets. Muon then finds each matrix whole on one rank. Call
    it after initializing or loading the weights, before the first forward."""
    placement_fn = make_bucketed_owned_full_param_placement_fn(carry_loads=True)
    storages = list(model.sharded_bucket_storages)
    for index in bucket_indices:
        relayout_bucket(model, storages[index], placement_fn)


@torch.no_grad()
def init_weights(model: DeepSeekV3, seed: int) -> None:
    """Initialize every rank's shards as one device would initialize the full
    model: each rank draws each full parameter and keeps its own part."""
    torch.manual_seed(seed)
    init_fns = param_init_fns(model.config)
    layouts = get_flex_shard_global_layouts(model)
    for fqn, param in model.named_parameters():
        init_from_full(param, layouts[fqn], init_fns[fqn])


def model_state_dict(model):
    state_dict = model.state_dict()
    set_state_dict_global_layouts(model, state_dict)
    return state_dict


def build_lr_lambda(*, total_steps, warmup_steps, decay_ratio, min_lr_factor):
    """torchtitan's schedule: linear warmup, constant, then cosine decay to
    ``min_lr_factor`` over the last ``decay_ratio`` of the steps."""
    warmup_steps = min(warmup_steps, total_steps)
    decay_steps = round(total_steps * decay_ratio)
    if warmup_steps + decay_steps > total_steps:
        decay_steps = total_steps - warmup_steps
    # One virtual step past the end keeps the last LR above min_lr_factor.
    stable_steps = total_steps + 1 - warmup_steps - decay_steps

    def lr_lambda(step):
        warmup_stable_steps = warmup_steps + stable_steps
        step = min(step, warmup_stable_steps + decay_steps - 1)
        if step < warmup_steps:
            return float((step + 1) / warmup_steps)
        if step < warmup_stable_steps:
            return 1.0
        progress = float(step + 1 - warmup_stable_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1 - min_lr_factor) * cosine

    return lr_lambda


def sharded_grad_norm(params, groups) -> torch.Tensor:
    """The L2 norm of the gradients that ``params`` shard over ``groups``.

    As for FSDP2's DTensor gradients: take each gradient's shard norm, then for
    each mesh dim in order, sum the squares over its group and take the root;
    finally take the norm of the per-gradient norms.
    """
    # Every rank lists every parameter, even one it stores none of.
    grads = [p.grad if p.grad is not None else p.new_zeros(0) for p in params]
    norms = torch.stack(torch._foreach_norm(grads, 2.0))
    for group in groups:
        norms = norms**2.0
        dist.all_reduce(norms, group=group)
        norms = norms ** (1.0 / 2.0)
    return torch.linalg.vector_norm(norms, 2.0)


@torch.no_grad()
def clip_grad_norm_(params, max_norm, expert_params=(), expert_groups=()):
    """Clip gradients to a global L2 norm and return it.

    Each rank holds a shard of every gradient. With expert parallelism, the
    routed experts' gradients, sharded over ``expert_groups``, get their own
    norm first, as torchtitan computes it.
    """
    if not expert_params:
        norm = sharded_grad_norm(params, [dist.group.WORLD])
    else:
        experts = set(expert_params)
        dense_norm = sharded_grad_norm(
            [p for p in params if p not in experts], [dist.group.WORLD]
        )
        expert_norm = sharded_grad_norm(
            [p for p in params if p in experts], expert_groups
        )
        norm = (expert_norm**2.0 + dense_norm**2.0) ** (1.0 / 2.0)
    torch.nn.utils.clip_grads_with_norm_(params, max_norm, norm, foreach=True)
    return norm


def build_optimizers(model, *, muon: bool, lr: float, weight_decay: float):
    """AdamW for every parameter, or DistMuon for the matrices and AdamW for
    the rest. Build them after sharding: they hold the local shards."""
    optimizers = {
        "adamw": torch.optim.AdamW(
            [
                p
                for fqn, p in model.named_parameters()
                if not (muon and kimi_matrices(fqn, model.config) is not None)
            ],
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
            fused=True,
        )
    }
    register_optimizer_checkpoint_hook(optimizers["adamw"], model)
    if muon:
        optimizers["muon"] = build_muon(model, lr=lr, weight_decay=weight_decay)
    return optimizers


def train_step(
    model,
    batch,
    *,
    optimizers,
    schedulers,
    max_norm,
    use_positions,
    expert_params=(),
    expert_groups=(),
):
    """One step on this rank's batch. Returns the loss averaged over all ranks'
    tokens and the gradient norm before clipping."""
    num_tokens = batch["input"].numel()
    # Divide the summed losses by the tokens of all ranks.
    token_counts = torch.stack(
        [
            (batch["labels"] != IGNORE_INDEX).sum(),
            torch.tensor(num_tokens, device=batch["input"].device),
        ]
    )
    dist.all_reduce(token_counts)
    loss_tokens, routed_tokens = token_counts
    positions = batch["positions"] if use_positions else None
    logits = model(batch["input"], positions, aux_loss_denominator=routed_tokens)
    loss = (
        F.cross_entropy(
            logits.float(),
            batch["labels"].flatten(),
            reduction="sum",
            ignore_index=IGNORE_INDEX,
        )
        / loss_tokens
    )
    del logits
    loss.backward()
    grad_norm = clip_grad_norm_(
        list(model.parameters()), max_norm, expert_params, expert_groups
    )
    update_expert_biases(model)
    with torch.profiler.record_function("optimizer"):
        for optimizer in optimizers.values():
            optimizer.step()
    for scheduler in schedulers.values():
        scheduler.step()
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    loss = loss.detach()
    dist.all_reduce(loss)
    return loss, grad_norm


def optimizer_state_dict(name, optimizer):
    state_dict = optimizer.state_dict()
    if name != "muon":
        return state_dict
    # DistMuon holds only the matrices this rank stores: key their momentum by
    # name. The tensors are the live states, with their layouts declared.
    (group,) = state_dict["param_groups"]
    names = group["param_names"]
    return {
        "momentum": {
            names[i]: s["momentum_buffer"] for i, s in state_dict["state"].items()
        },
        "hyperparameters": {
            k: v for k, v in group.items() if k not in ("params", "param_names")
        },
    }


def checkpoint_state(model, optimizers, schedulers, step):
    return {
        "model": model_state_dict(model),
        **{name: optimizer_state_dict(name, o) for name, o in optimizers.items()},
        **{
            f"{name}_lr": scheduler.state_dict()
            for name, scheduler in schedulers.items()
        },
        "step": step,
    }


def save_checkpoint(path, *, model, optimizers, schedulers, data, step):
    dcp.save(checkpoint_state(model, optimizers, schedulers, step), checkpoint_id=path)
    # Each rank reads different data, so each saves its own position.
    torch.save(data.state_dict(), os.path.join(path, f"data_rank{dist.get_rank()}.pt"))


def load_checkpoint(path, *, model, optimizers, schedulers, data):
    # Loading into an optimizer needs its states: take a step that changes
    # nothing, with zero gradients and a zero learning rate.
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    for optimizer in optimizers.values():
        lrs = [group["lr"] for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        optimizer.step()
        for group, lr in zip(optimizer.param_groups, lrs):
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

    state = checkpoint_state(model, optimizers, schedulers, step=0)
    dcp.load(state, checkpoint_id=path)
    model.load_state_dict(state["model"])
    for name, optimizer in optimizers.items():
        if name == "muon":  # dcp.load restored the live momentum in place
            optimizer.param_groups[0].update(state[name]["hyperparameters"])
        else:
            optimizer.load_state_dict(state[name])
    for name, scheduler in schedulers.items():
        scheduler.load_state_dict(state[f"{name}_lr"])
    data.load_state_dict(
        torch.load(
            os.path.join(path, f"data_rank{dist.get_rank()}.pt"), weights_only=False
        )
    )
    return state["step"]


def latest_checkpoint(checkpoint_dir):
    if checkpoint_dir is None or not os.path.isdir(checkpoint_dir):
        return None
    steps = [
        int(name.removeprefix("step-"))
        for name in os.listdir(checkpoint_dir)
        if name.startswith("step-")
    ]
    return os.path.join(checkpoint_dir, f"step-{max(steps)}") if steps else None


def main():
    args = parse_args()
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.utils.deterministic.fill_uninitialized_memory = False

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    ep = args.ep if args.ep is not None else max(world_size // 2, 1)
    if world_size % ep:
        raise ValueError(f"--ep {ep} must divide the {world_size} ranks")
    expert_mesh = None
    if ep > 1:
        # Ranks [i * ep, (i + 1) * ep) split the experts; ranks i, i + ep, ...
        # hold the same experts and shard them.
        expert_mesh = init_device_mesh(
            "cuda", (world_size // ep, ep), mesh_dim_names=("efsdp", "ep")
        )

    config = DeepSeekV3Config(
        max_seq_len=args.seq_len,
        attention=args.attention,
        activation_checkpointing=not args.no_activation_checkpointing,
    )
    with torch.device("meta"):
        model = DeepSeekV3(config)
    if expert_mesh is not None:
        apply_expert_parallel(model, expert_mesh["ep"])
    owner_buckets = shard_model(
        model,
        mesh,
        muon=args.optimizer == "muon",
        expert_mesh=None if expert_mesh is None else expert_mesh["efsdp"],
    )
    model.to_empty(device=device)
    model.init_buffers(device)
    if args.init_checkpoint:
        state_dict = model_state_dict(model)
        dcp.load({"model": state_dict}, checkpoint_id=args.init_checkpoint)
        model.load_state_dict(state_dict)
    else:
        init_weights(model, args.seed)
    store_on_owners(model, owner_buckets)
    model.train()
    expert_params, expert_groups = [], []
    if expert_mesh is not None:
        expert_params = [
            p for fqn, p in model.named_parameters() if ".routed_experts." in fqn
        ]
        expert_groups = [expert_mesh.get_group("efsdp"), expert_mesh.get_group("ep")]

    optimizers = build_optimizers(
        model, muon=args.optimizer == "muon", lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lambda = build_lr_lambda(
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        decay_ratio=args.decay_ratio,
        min_lr_factor=args.min_lr_factor,
    )
    schedulers = {
        name: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        for name, optimizer in optimizers.items()
    }
    batch_args = dict(batch_size=args.local_batch_size, seq_len=args.seq_len, rank=rank)
    if args.data == "c4":
        data = C4Batches(tokenizer=args.tokenizer, world_size=world_size, **batch_args)
    else:
        data = SyntheticBatches(
            vocab_size=config.vocab_size, seed=args.seed, **batch_args
        )

    step = 0
    if (resume_from := latest_checkpoint(args.checkpoint_dir)) is not None:
        step = load_checkpoint(
            resume_from,
            model=model,
            optimizers=optimizers,
            schedulers=schedulers,
            data=data,
        )
        # Release the loading buffers before training.
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"resumed from {resume_from}", flush=True)

    profiler = None
    if args.profile_dir:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1, warmup=2, active=1, repeat=1, skip_first=1
            ),
            on_trace_ready=lambda p: p.export_chrome_trace(
                os.path.join(args.profile_dir, f"rank{rank}_trace.json")
            ),
        )
        profiler.start()

    num_tokens = args.local_batch_size * args.seq_len
    while step < args.steps:
        step += 1
        if profiler is not None:
            torch.cuda.synchronize()
            dist.barrier()
        start = time.perf_counter()
        batch = {k: v.to(device, non_blocking=True) for k, v in next(data).items()}
        lr = schedulers["adamw"].get_last_lr()[0]
        loss, grad_norm = train_step(
            model,
            batch,
            optimizers=optimizers,
            schedulers=schedulers,
            max_norm=args.max_norm,
            # SDPA attends across documents, so positions just count up each row.
            use_positions=args.attention == "flex",
            expert_params=expert_params,
            expert_groups=expert_groups,
        )
        loss, grad_norm = loss.item(), grad_norm.item()
        elapsed = time.perf_counter() - start
        if profiler is not None:
            torch.cuda.synchronize()
            dist.barrier()
            profiler.step()
        if rank == 0:
            active = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            print(
                f"step {step:4d}  loss {loss!r}  grad_norm {grad_norm!r}  lr {lr:.3e}"
                f"  tokens/s/GPU {num_tokens / elapsed:,.0f}"
                f"  peak memory {active:.1f} GiB active, {reserved:.1f} GiB reserved",
                flush=True,
            )
        torch.cuda.reset_peak_memory_stats()

        if args.checkpoint_dir and (
            step % args.checkpoint_interval == 0 or step == args.steps
        ):
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"step-{step}"),
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                data=data,
                step=step,
            )

    if profiler is not None:
        profiler.stop()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
