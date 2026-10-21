# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek V3 example (``examples/deepseek_v3``) at small sizes: its
bucket layout trains bitwise like FSDP2, expert parallelism moves each routed
token to its expert and back exactly, its Muon step is local and exact, its
checkpoints resume bitwise, and activation checkpointing changes nothing."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TestCase

sys.path.insert(0, str(Path(__file__).parents[3] / "examples" / "deepseek_v3"))
import train  # noqa: E402
from data import SyntheticBatches  # noqa: E402
from expert_parallel import AllToAllDispatcher, apply_expert_parallel  # noqa: E402
from model import DeepSeekV3, DeepSeekV3Config, GateUpLinear, param_init_fns  # noqa: E402
from muon import kimi_matrices  # noqa: E402

from .. import get_flex_shard_global_layouts, init_from_full  # noqa: E402
from ..dist_muon import BlockShard, BucketConfig, ComputeLayout, DistMuon, Owned  # noqa: E402


def _config(**kwargs) -> DeepSeekV3Config:
    """The 16B architecture at small sizes: one dense and two MoE blocks, with
    SDPA, which needs no compile."""
    return DeepSeekV3Config(
        attention="sdpa",
        vocab_size=128,
        dim=64,
        n_layers=3,
        n_heads=4,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        dense_hidden_dim=96,
        moe_hidden_dim=32,
        num_experts=8,
        top_k=2,
        max_seq_len=64,
        **kwargs,
    )


def _full_weights(config: DeepSeekV3Config, seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = DeepSeekV3(config)
    weights = {}
    for fqn, param in model.named_parameters():
        weights[fqn] = torch.empty(param.shape, device="cuda")
        param_init_fns(config)[fqn](weights[fqn])
    return weights


def _batches(config: DeepSeekV3Config, rank: int, steps: int):
    data = SyntheticBatches(
        vocab_size=config.vocab_size, batch_size=2, seq_len=32, rank=rank, seed=1
    )
    return [{k: v.cuda() for k, v in next(data).items()} for _ in range(steps)]


class TestDeepSeekV3Model(TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_activation_checkpointing_is_bitwise(self) -> None:
        # Recomputing each block changes no logit or gradient, and the expert
        # counts include each token once.
        torch.use_deterministic_algorithms(True)
        weights = _full_weights(_config())
        results = []
        for checkpointing in (False, True):
            with torch.device("cuda"):
                model = DeepSeekV3(_config(activation_checkpointing=checkpointing))
            model.load_state_dict(weights, strict=False)
            for param in model.parameters():
                param.data = param.data.bfloat16()
            batch = _batches(model.config, rank=0, steps=1)[0]
            logits = model(
                batch["input"], aux_loss_denominator=torch.tensor(64, device="cuda")
            )
            logits.float().logsumexp(-1).sum().backward()
            counts = [
                m.moe.router.tokens_per_expert.clone()
                for m in model.layers
                if m.moe_enabled
            ]
            results.append((logits, [p.grad for p in model.parameters()], counts))
        (logits, grads, counts), (logits_ac, grads_ac, counts_ac) = results
        self.assertTrue(torch.equal(logits, logits_ac))
        for grad, grad_ac in zip(grads, grads_ac, strict=True):
            self.assertTrue(torch.equal(grad, grad_ac))
        for count, count_ac in zip(counts, counts_ac, strict=True):
            self.assertTrue(torch.equal(count, count_ac))
            self.assertEqual(count.sum().item(), 64 * 2)

    def test_lr_schedule_matches_torchtitan(self) -> None:
        # torchtitan's LRSchedulersContainer with warmup_steps=3,
        # decay_ratio=0.8, decay_type="cosine", min_lr_factor=0.1 over 12 steps.
        expected = [
            0.3333333333333333, 0.6666666666666666, 1.0, 1.0,
            0.9728616793536587, 0.8947199994035401, 0.775, 0.6281416799501187,
            0.4718583200498814, 0.3250000000000001, 0.20528000059645996,
            0.12713832064634126,
        ]  # fmt: skip
        lr_lambda = train.build_lr_lambda(
            total_steps=12, warmup_steps=3, decay_ratio=0.8, min_lr_factor=0.1
        )
        self.assertEqual([lr_lambda(step) for step in range(12)], expected)

    def test_kimi_matrices(self) -> None:
        # Muon's matrices: per head and role for attention, per expert, the rest
        # whole; AdamW for embeddings, norms and the output projection.
        config = _config()
        with torch.device("meta"):
            model = DeepSeekV3(config)
        expected = {
            "attention.wq.weight": BlockShard(dim=0, block_sizes=(16, 8)),
            "attention.wkv_a.weight": BlockShard(dim=0, block_sizes=(32, 8)),
            "attention.wkv_b.weight": BlockShard(dim=0, block_sizes=(16, 16)),
            "attention.wo.weight": Owned(),
            "feed_forward.w13.weight": Owned(),
            "feed_forward.w2.weight": Owned(),
            "moe.router.gate.weight": Owned(),
            "moe.routed_experts.w13.weight": Shard(0),
            "moe.routed_experts.w2.weight": Shard(0),
            "moe.shared_experts.w13.weight": Owned(),
            "moe.shared_experts.w2.weight": Owned(),
        }
        for fqn, _ in model.named_parameters():
            suffix = fqn.split(".", 2)[-1] if fqn.startswith("layers.") else fqn
            self.assertEqual(kimi_matrices(fqn, config), expected.get(suffix), fqn)


class TestDeepSeekV3Training(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    def _sharded(self, config, weights, *, muon: bool, ep: int = 1):
        """The model as train.py builds it, from ``weights``, and the expert
        parameters and groups of its gradient norm."""
        mesh = init_device_mesh("cuda", (self.world_size,))
        with torch.device("meta"):
            model = DeepSeekV3(config)
        expert_mesh = None
        if ep > 1:
            expert_mesh = init_device_mesh(
                "cuda", (self.world_size // ep, ep), mesh_dim_names=("efsdp", "ep")
            )
            apply_expert_parallel(model, expert_mesh["ep"])
        owner_buckets = train.shard_model(
            model,
            mesh,
            muon=muon,
            expert_mesh=None if expert_mesh is None else expert_mesh["efsdp"],
        )
        model.to_empty(device="cuda")
        model.init_buffers(torch.device("cuda"))
        layouts = get_flex_shard_global_layouts(model)
        with torch.no_grad():
            for fqn, param in model.named_parameters():
                init_from_full(
                    param, layouts[fqn], lambda full, fqn=fqn: full.copy_(weights[fqn])
                )
        train.store_on_owners(model, owner_buckets)
        optimizers = train.build_optimizers(model, muon=muon, lr=1e-3, weight_decay=0.1)
        lr_lambda = train.build_lr_lambda(
            total_steps=4, warmup_steps=1, decay_ratio=0.8, min_lr_factor=0.1
        )
        schedulers = {
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            for name, optimizer in optimizers.items()
        }
        experts = {}
        if expert_mesh is not None:
            experts = {
                "expert_params": [
                    p
                    for fqn, p in model.named_parameters()
                    if ".routed_experts." in fqn
                ],
                "expert_groups": [
                    expert_mesh.get_group("efsdp"),
                    expert_mesh.get_group("ep"),
                ],
            }
        return model, optimizers, schedulers, experts

    def _steps(self, model, optimizers, schedulers, experts, batches):
        return [
            torch.stack(
                train.train_step(
                    model,
                    batch,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    max_norm=1.0,
                    use_positions=False,
                    **experts,
                )
            )
            for batch in batches
        ]

    @skip_if_lt_x_gpu(4)
    def test_adamw_buckets_match_fsdp2(self) -> None:
        # The example's buckets train as FSDP2's fully_shard groups do, bit for
        # bit: losses and gradient norms over 3 steps.
        torch.use_deterministic_algorithms(True)
        config = _config()
        weights = _full_weights(config)
        batches = _batches(config, self.rank, steps=3)
        flex = self._steps(*self._sharded(config, weights, muon=False), batches)

        with torch.device("cuda"):
            reference = DeepSeekV3(config)
        reference.load_state_dict(weights, strict=False)
        mesh = init_device_mesh("cuda", (self.world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        gate_up = {m.weight for m in reference.modules() if isinstance(m, GateUpLinear)}

        def placement(param):
            return Shard(1) if param in gate_up else None

        fully_shard(reference.tok_embeddings, mesh=mesh, mp_policy=mp_policy)
        for block in reference.layers:
            fully_shard(
                block, mesh=mesh, mp_policy=mp_policy, shard_placement_fn=placement
            )
        fully_shard(
            [reference.norm, reference.lm_head],
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
        )
        fully_shard(reference, mesh=mesh, mp_policy=mp_policy)
        for module in reference.modules():
            if hasattr(module, "set_gradient_divide_factor"):
                module.set_gradient_divide_factor(1.0)
        adamw = torch.optim.AdamW(
            reference.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
            fused=True,
        )
        lr_lambda = train.build_lr_lambda(
            total_steps=4, warmup_steps=1, decay_ratio=0.8, min_lr_factor=0.1
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(adamw, lr_lambda)
        for step, batch in enumerate(batches):
            tokens = torch.tensor(
                batch["input"].numel() * self.world_size, device="cuda"
            )
            logits = reference(batch["input"], aux_loss_denominator=tokens)
            loss = (
                torch.nn.functional.cross_entropy(
                    logits.float(), batch["labels"].flatten(), reduction="sum"
                )
                / tokens
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            train.update_expert_biases(reference)
            adamw.step()
            scheduler.step()
            adamw.zero_grad(set_to_none=True)
            loss = loss.detach()
            dist.all_reduce(loss)
            self.assertTrue(
                torch.equal(flex[step], torch.stack([loss, grad_norm.full_tensor()])),
                step,
            )

    @skip_if_lt_x_gpu(4)
    def test_expert_parallel_moves_tokens_exactly(self) -> None:
        # Each routed token reaches the rank holding its expert, grouped by
        # expert, and comes back to its place; so does its gradient.
        torch.manual_seed(self.rank)
        mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("efsdp", "ep"))
        ep_rank, num_experts, top_k = mesh["ep"].get_local_rank(), 8, 2
        num_local = num_experts // 2
        ids_TK = torch.rand(16, num_experts, device="cuda").topk(top_k).indices
        sorted_N = torch.argsort(ids_TK.view(-1), stable=True)
        expert_N = ids_TK.view(-1)[sorted_N]
        counts_E = torch.bincount(expert_N, minlength=num_experts)
        # Integer values keep every sum below exact.
        routed_ND = torch.randint(0, 100, (32, 4), device="cuda").float()
        routed_ND.requires_grad_()
        dispatcher = AllToAllDispatcher(mesh["ep"].get_group())
        routed_RD, counts_e, metadata = dispatcher.dispatch(routed_ND, counts_E)
        expert_R = ep_rank * num_local + torch.repeat_interleave(
            torch.arange(num_local, device="cuda"), counts_e
        )
        # Each "expert" adds a multiple of its id.
        out_ND = dispatcher.combine(
            routed_RD + 1000 * (expert_R + 1).unsqueeze(-1).float(), metadata
        )
        expected = routed_ND.detach() + 1000 * (expert_N + 1).unsqueeze(-1).float()
        self.assertTrue(torch.equal(out_ND, expected))
        (out_ND * (expert_N + 1).unsqueeze(-1).float()).sum().backward()
        self.assertTrue(
            torch.equal(
                routed_ND.grad, (expert_N + 1).unsqueeze(-1).float().expand(-1, 4)
            )
        )

    @skip_if_lt_x_gpu(4)
    def test_expert_parallel_trains_like_without(self) -> None:
        # EP over 2 ranks, each rank's experts FSDP-sharded over the other 2,
        # trains as sharding everything over 4 ranks does, up to the rounding
        # of differently grouped expert matmuls and gradient sums. (Against
        # torchtitan's FSDP2 with the same EP, the full model is bitwise.)
        torch.use_deterministic_algorithms(True)
        config = _config()
        weights = _full_weights(config)
        batches = _batches(config, self.rank, steps=3)
        without = self._steps(*self._sharded(config, weights, muon=False), batches)
        with_ep = self._steps(
            *self._sharded(config, weights, muon=False, ep=2), batches
        )
        with_ep, without = torch.stack(with_ep), torch.stack(without)
        # From the same weights, the first loss and gradient norm agree to
        # rounding; the updates then drift apart slowly.
        torch.testing.assert_close(with_ep[0], without[0], rtol=1e-6, atol=0)
        torch.testing.assert_close(with_ep, without, rtol=1e-4, atol=0)

    @skip_if_lt_x_gpu(4)
    def test_muon_step_is_local_and_exact(self) -> None:
        # With Muon, every matrix is whole on one rank: DistMuon's step runs no
        # NCCL kernel, and its updates equal DistMuon's on the full tensors, with
        # and without expert parallelism.
        for ep in (1, 2):
            with self.subTest(ep=ep):
                self._check_muon_step_is_local_and_exact(ep)

    def _check_muon_step_is_local_and_exact(self, ep: int) -> None:
        torch.use_deterministic_algorithms(True)
        config = _config()
        model, optimizers, _, _ = self._sharded(
            config, _full_weights(config), muon=True, ep=ep
        )
        batch = _batches(config, self.rank, steps=1)[0]
        tokens = torch.tensor(batch["input"].numel() * self.world_size, device="cuda")
        logits = model(batch["input"], aux_loss_denominator=tokens)
        torch.nn.functional.cross_entropy(
            logits.float(), batch["labels"].flatten(), reduction="sum"
        ).div(tokens).backward()
        fqns = [
            fqn for fqn, _ in model.named_parameters() if kimi_matrices(fqn, config)
        ]
        params = dict(model.named_parameters())
        layouts = get_flex_shard_global_layouts(model)

        def gather(tensors):
            # Every rank's (layout, local tensor), assembled into full tensors.
            pieces = [None] * self.world_size
            dist.all_gather_object(
                pieces,
                [(layouts[fqn], tensors[fqn].detach().cpu()) for fqn in fqns],
            )
            full = {}
            for index, fqn in enumerate(fqns):
                out = torch.empty(layouts[fqn].global_shape)
                for rank_pieces in pieces:
                    layout, local = rank_pieces[index]
                    for g_off, l_off, size in zip(
                        layout.global_offsets, layout.local_offsets, layout.local_sizes
                    ):
                        out[tuple(slice(o, o + n) for o, n in zip(g_off, size))] = (
                            local[tuple(slice(o, o + n) for o, n in zip(l_off, size))]
                        )
                full[fqn] = out.cuda()
            return full

        grads = {
            fqn: p.grad if p.grad is not None else torch.zeros_like(p)
            for fqn, p in params.items()
        }
        full_params, full_grads = gather(params), gather(grads)
        muon = optimizers["muon"]
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            muon.step()
            torch.cuda.synchronize()
        kernels = [
            e.name
            for e in profile.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        self.assertTrue(kernels)
        self.assertFalse([k for k in kernels if "nccl" in k.lower()])

        # DistMuon on the full tensors, on the same one-rank mesh.
        mesh = muon.param_groups[0]["params"][0].device_mesh
        (axis,) = mesh.mesh_dim_names
        references = {}
        for fqn in fqns:
            reference = torch.nn.Parameter(
                DTensor.from_local(full_params[fqn], mesh, (Replicate(),))
            )
            reference.grad = DTensor.from_local(full_grads[fqn], mesh, (Replicate(),))
            references[fqn] = reference
        DistMuon(
            [{"params": list(references.values()), "param_names": fqns}],
            compute_sharding_by_fqn={
                fqn: ComputeLayout(
                    shardings_by_mesh_axis={axis: kimi_matrices(fqn, config)}
                )
                for fqn in fqns
            },
            bucket_configs=[BucketConfig(name="all", patterns=("*",))],
            lr=1e-3,
            adjust_lr_fn="match_rms_adamw",
        ).step()
        updated = gather(params)
        for fqn in fqns:
            self.assertTrue(torch.equal(updated[fqn], references[fqn].to_local()), fqn)

    @skip_if_lt_x_gpu(4)
    def test_checkpoint_resume_is_bitwise(self) -> None:
        # 2 steps, a checkpoint, a fresh model loading it and 2 more steps match
        # 4 straight steps, with AdamW and with Muon, with and without expert
        # parallelism.
        torch.use_deterministic_algorithms(True)
        config = _config()
        weights = _full_weights(config)
        batches = _batches(config, self.rank, steps=4)
        path = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(path)
        try:
            for muon, ep in ((False, 1), (True, 1), (False, 2), (True, 2)):
                with self.subTest(muon=muon, ep=ep):
                    straight = self._steps(
                        *self._sharded(config, weights, muon=muon, ep=ep), batches
                    )
                    model, optimizers, schedulers, experts = self._sharded(
                        config, weights, muon=muon, ep=ep
                    )
                    first = self._steps(
                        model, optimizers, schedulers, experts, batches[:2]
                    )
                    data = SyntheticBatches(
                        vocab_size=128, batch_size=2, seq_len=32, rank=self.rank
                    )
                    checkpoint = os.path.join(path[0], f"muon{muon}_ep{ep}")
                    train.save_checkpoint(
                        checkpoint,
                        model=model,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        data=data,
                        step=2,
                    )
                    model, optimizers, schedulers, experts = self._sharded(
                        config, _full_weights(config, seed=1), muon=muon, ep=ep
                    )
                    step = train.load_checkpoint(
                        checkpoint,
                        model=model,
                        optimizers=optimizers,
                        schedulers=schedulers,
                        data=data,
                    )
                    self.assertEqual(step, 2)
                    resumed = first + self._steps(
                        model, optimizers, schedulers, experts, batches[2:]
                    )
                    for index, (a, b) in enumerate(zip(straight, resumed, strict=True)):
                        self.assertTrue(torch.equal(a, b), index)
        finally:
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path[0])


if __name__ == "__main__":
    run_tests()
