# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek V3 example (``examples/deepseek_v3``) at small sizes: its
bucket layout trains bitwise like FSDP2, its checkpoints resume bitwise, and
activation checkpointing changes nothing."""

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
from torch.distributed.tensor import Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TestCase

sys.path.insert(0, str(Path(__file__).parents[3] / "examples" / "deepseek_v3"))
import train  # noqa: E402
from data import SyntheticBatches  # noqa: E402
from model import DeepSeekV3, DeepSeekV3Config, GateUpLinear, param_init_fns  # noqa: E402

from .. import get_flex_shard_global_layouts, init_from_full  # noqa: E402


def _config(**kwargs) -> DeepSeekV3Config:
    """The 16B architecture at small sizes: one dense and two MoE blocks."""
    return DeepSeekV3Config(
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


class TestDeepSeekV3Training(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    def _sharded(self, config, weights):
        """The model as train.py builds it, from ``weights``."""
        mesh = init_device_mesh("cuda", (self.world_size,))
        with torch.device("meta"):
            model = DeepSeekV3(config)
        train.shard_model(model, mesh)
        model.to_empty(device="cuda")
        model.init_buffers(torch.device("cuda"))
        layouts = get_flex_shard_global_layouts(model)
        with torch.no_grad():
            for fqn, param in model.named_parameters():
                init_from_full(
                    param, layouts[fqn], lambda full, fqn=fqn: full.copy_(weights[fqn])
                )
        optimizers = train.build_optimizers(model, lr=1e-3, weight_decay=0.1)
        lr_lambda = train.build_lr_lambda(
            total_steps=4, warmup_steps=1, decay_ratio=0.8, min_lr_factor=0.1
        )
        schedulers = {
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            for name, optimizer in optimizers.items()
        }
        return model, optimizers, schedulers

    def _steps(self, model, optimizers, schedulers, batches):
        return [
            torch.stack(
                train.train_step(
                    model,
                    batch,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    max_norm=1.0,
                    use_positions=False,
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
        model, optimizers, schedulers = self._sharded(config, weights)
        flex = self._steps(model, optimizers, schedulers, batches)

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
    def test_checkpoint_resume_is_bitwise(self) -> None:
        # 2 steps, a checkpoint, a fresh model loading it and 2 more steps match
        # 4 straight steps.
        torch.use_deterministic_algorithms(True)
        config = _config()
        weights = _full_weights(config)
        batches = _batches(config, self.rank, steps=4)
        straight = self._steps(*self._sharded(config, weights), batches)
        model, optimizers, schedulers = self._sharded(config, weights)
        first = self._steps(model, optimizers, schedulers, batches[:2])
        data = SyntheticBatches(
            vocab_size=128, batch_size=2, seq_len=32, rank=self.rank
        )
        path = [tempfile.mkdtemp() if self.rank == 0 else None]
        dist.broadcast_object_list(path)
        try:
            train.save_checkpoint(
                path[0],
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                data=data,
                step=2,
            )
            model, optimizers, schedulers = self._sharded(
                config, _full_weights(config, seed=1)
            )
            step = train.load_checkpoint(
                path[0],
                model=model,
                optimizers=optimizers,
                schedulers=schedulers,
                data=data,
            )
        finally:
            dist.barrier()
            if self.rank == 0:
                shutil.rmtree(path[0])
        self.assertEqual(step, 2)
        resumed = first + self._steps(model, optimizers, schedulers, batches[2:])
        for index, (a, b) in enumerate(zip(straight, resumed, strict=True)):
            self.assertTrue(torch.equal(a, b), index)


if __name__ == "__main__":
    run_tests()
