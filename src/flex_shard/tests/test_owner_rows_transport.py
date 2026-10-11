# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
from collections import Counter, defaultdict
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import flex_shard
from ..custom_placements.owned import make_bucketed_owned_full_param_placement_fn
from ..custom_placements.owner_rows_transport import (
    owner_rows_transport_buffers,
    owner_rows_transport_plan,
    OwnerRowsTransportPlan,
    plan_owner_rows_transport,
)
from ..dist_muon import AssignmentGroup, assign_matrices, WholeMatrixSpec
from ..flex_shard.bucket_storage import BucketSpec, MixedPrecisionPolicy
from ..flex_shard.utils import mesh_rank_hosts
from .common import make_test_sgd

# DeepSeek V3 671B's whole Muon matrices of one MoE layer.
_MOE_LAYER = (
    7168 * 16384,
    1536 * 7168,
    2048 * 7168,
    2048 * 7168,
    7168 * 2048,
    256 * 7168,
)


def _subtract(
    start: int, numel: int, holes: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    out, position = [], start
    for hole_start, hole_numel in sorted(holes):
        if hole_start > position:
            out.append((position, hole_start - position))
        position = max(position, hole_start + hole_numel)
    if position < start + numel:
        out.append((position, start + numel - position))
    return out


def _replay_gather(
    test: TestCase, plan: OwnerRowsTransportPlan, row_numels: list[int]
) -> None:
    """Replay the plan's gather on intervals: every rank's chunk must end up
    holding exactly its positions, each written once, and every relay must
    hold a slice before it passes it on."""
    chunk = plan.chunk_numel
    starts = list(itertools.accumulate(row_numels, initial=0))
    held: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    def write(rank: int, start: int, numel: int) -> None:
        offset = plan.buffer_offset(rank, start)
        test.assertLessEqual(offset + numel, plan.buffer_numel)
        held[rank].append((offset, start, numel))

    def holds(rank: int, start: int, numel: int) -> bool:
        offset = plan.buffer_offset(rank, start)
        return any(
            o <= offset and offset + numel <= o + n and s + offset - o == start
            for o, s, n in held[rank]
        )

    def owns(rank: int, start: int) -> bool:
        return starts[rank] <= start < starts[rank] + row_numels[rank]

    sent: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for move in plan.owner_moves:
        test.assertTrue(owns(move.src, move.start), move)
        write(move.dst, move.start, move.numel)
        sent[move.src].append((move.start, move.numel))
    for move in plan.host_moves:
        if owns(move.src, move.start):
            # An owner relaying its own slice stages it in its buffer first.
            write(move.src, move.start, move.numel)
            sent[move.src].append((move.start, move.numel))
    for owner, (start, numel) in enumerate(zip(starts, row_numels)):
        for kept_start, kept_numel in _subtract(start, numel, sent[owner]):
            test.assertGreaterEqual(kept_start, owner * chunk)
            test.assertLessEqual(kept_start + kept_numel, (owner + 1) * chunk)
            write(owner, kept_start, kept_numel)
    for move in plan.host_moves:
        test.assertTrue(holds(move.src, move.start, move.numel), move)
        write(move.dst, move.start, move.numel)
    for move in plan.relay_moves:
        test.assertTrue(holds(move.src, move.start, move.numel), move)
        write(move.dst, move.start, move.numel)

    total = sum(row_numels)
    for rank in range(len(row_numels)):
        filled = 0
        for offset, start, numel in sorted(h for h in held[rank] if h[0] < chunk):
            # Contiguous, no overlap, each position at its place.
            test.assertEqual(offset, filled)
            test.assertEqual(start, rank * chunk + offset)
            filled += numel
        test.assertEqual(filled, max(0, min(chunk, total - rank * chunk)))
        end = chunk
        for offset, _start, numel in sorted(h for h in held[rank] if h[0] >= chunk):
            test.assertGreaterEqual(offset, end)
            end = offset + numel


class TestOwnerRowsTransportPlan(TestCase):
    def _check(
        self, row_numels: list[int], hosts: tuple[int, ...]
    ) -> OwnerRowsTransportPlan:
        plan = plan_owner_rows_transport(tuple(row_numels), hosts)
        self.assertGreaterEqual(plan.gathered_numel, sum(row_numels))
        self.assertEqual(plan.gathered_numel, len(row_numels) * plan.chunk_numel)
        _replay_gather(self, plan, row_numels)
        for move in plan.owner_moves + plan.relay_moves:
            self.assertEqual(hosts[move.src], hosts[move.dst], move)
        for move in plan.host_moves:
            self.assertNotEqual(hosts[move.src], hosts[move.dst], move)
        return plan

    def test_moe_stage_on_8_hosts(self) -> None:
        # Two-level owners of a 4-layer MoE stage on 64 ranks: only each heavy
        # host's bytes over its share cross hosts, split over all 8 ranks.
        hosts = mesh_rank_hosts(range(64), 8)
        group = AssignmentGroup(
            matrices=tuple(
                WholeMatrixSpec(f"layer{layer}.m{index}", numel)
                for layer in range(4)
                for index, numel in enumerate(_MOE_LAYER)
            ),
            block_groups=(),
        )
        owners = assign_matrices((group,), num_ranks=64, hosts=hosts).rank_by_matrix
        rows = [0] * 64
        for matrix in group.matrices:
            rows[owners[matrix.name]] += matrix.numel
        plan = self._check(rows, hosts)
        share = 8 * plan.chunk_numel
        loads = [sum(rows[host * 8 : (host + 1) * 8]) for host in range(8)]
        self.assertEqual(
            sum(move.numel for move in plan.host_moves),
            sum(max(0, load - share) for load in loads),
        )
        # Each heavy host's overflow leaves over its 8 ranks.
        for host in (0, 2, 4, 6):
            self.assertEqual(
                {move.src for move in plan.host_moves if hosts[move.src] == host},
                set(range(host * 8, host * 8 + 8)),
            )
        # The buffer: a chunk (21.8 MB in BF16) plus a relayed share.
        self.assertLess(plan.buffer_numel * 2, 30e6)

    def test_one_host_moves_inside_it(self) -> None:
        plan = self._check([117, 57, 117, 57, 0, 30, 0, 9], (0,) * 8)
        self.assertEqual(plan.host_moves, ())
        self.assertEqual(plan.relay_moves, ())

    def test_owner_relays_its_own_slice(self) -> None:
        # Rank 0's row spills into host 1's chunks; rank 0 is also host 0's
        # first relay, so it stages that part in its own buffer.
        plan = self._check(
            [100_000, 0, 0, 0, 1000, 1000, 1000, 1000], (0, 0, 0, 0, 1, 1, 1, 1)
        )
        self.assertTrue(any(move.src == 0 for move in plan.host_moves))
        self.assertTrue(plan.relay_moves)

    def test_hosts_of_different_sizes(self) -> None:
        # A mesh that doesn't cover whole hosts: 3 ranks on one, 2 on another.
        self._check([700, 0, 50, 0, 250], (0, 0, 0, 1, 1))
        self._check([0, 0, 50, 900, 0], (0, 0, 0, 1, 1))

    def test_row_spanning_several_hosts(self) -> None:
        self._check(
            [5000, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            mesh_rank_hosts(range(12), 4),
        )

    def test_rows_on_every_rank(self) -> None:
        torch.manual_seed(0)
        for _ in range(20):
            rows = torch.randint(0, 3000, (16,)).tolist()
            self._check(rows, mesh_rank_hosts(range(16), 4))


class TestOwnerRowsTransportBuffers(FSDPTest):
    @property
    def world_size(self) -> int:
        return 4

    @skip_if_lt_x_gpu(4)
    def test_host_buffers(self) -> None:
        # 2 hosts of 2: each rank writes into its host neighbor's buffer.
        mesh = init_device_mesh("cuda", (self.world_size,))
        device = torch.device("cuda", torch.cuda.current_device())
        hosts = (0, 0, 1, 1)
        buffers = owner_rows_transport_buffers(mesh, hosts)
        self.assertIs(owner_rows_transport_buffers(mesh, hosts), buffers)
        neighbor = self.rank ^ 1
        self.assertEqual(sorted(buffers.host_ranks), sorted((self.rank, neighbor)))
        for numel in (1024, 4096):  # the second request grows the buffer
            buffer = buffers.get("gather", torch.bfloat16, numel, device)
            self.assertGreaterEqual(buffer.tensor.numel(), numel)
            buffer.tensor.fill_(-1)
            buffer.barrier()
            buffer.peers[neighbor][:numel].fill_(self.rank)
            buffer.barrier()
            torch.cuda.synchronize()
            self.assertTrue(bool((buffer.tensor[:numel] == neighbor).all()))
        reduce_buffer = buffers.get("reduce", torch.bfloat16, 1024, device)
        self.assertNotEqual(reduce_buffer.tensor.data_ptr(), buffer.tensor.data_ptr())

    @skip_if_lt_x_gpu(4)
    def test_matches_grouped_path(self) -> None:
        # Whole-parameter owners of uneven sizes on 4 ranks, as 2 hosts of 2
        # (rank 0's row spills into host 1, through relays) and as 1 host of 4,
        # in FP32 and with BF16 params reducing in BF16 (the DeepSeek V3
        # recipes' setting), both reducing the grads in the param layout (#74),
        # and with BF16 params reducing in FP32 (grads copied into the rows),
        # resharding after forward or not.
        # Through equal chunks against the grouped broadcasts and reduces
        # (FLEX_SHARD_OWNER_ROWS_TRANSPORT=grouped):
        # - step 0's params equal the reference's cast to the param dtype, bit
        #   for bit, and so do the outputs;
        # - step 0's reduced grads come back in FP32 and match, up to the
        #   reduction's summation order, as do later steps' outputs;
        # - two runs give the same outputs bit for bit;
        # - NCCL's all-gather and reduce-scatter run, no grouped calls, and the
        #   gathered storage is padded only to whole chunks.
        mesh = init_device_mesh("cuda", (self.world_size,))
        precisions = (
            (torch.float32, None),
            (torch.bfloat16, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        )
        for local_world_size, (
            param_dtype,
            reduce_dtype,
        ), reshard_after_forward in itertools.product(
            ("2", "4"), precisions, (False, True)
        ):
            runs = {}
            for transport in ("chunks", "chunks again", "grouped"):
                with (
                    self.subTest(
                        local_world_size=local_world_size,
                        param_dtype=param_dtype,
                        reduce_dtype=reduce_dtype,
                        reshard_after_forward=reshard_after_forward,
                        transport=transport,
                    ),
                    mock.patch.dict(
                        os.environ,
                        {
                            "LOCAL_WORLD_SIZE": local_world_size,
                            "FLEX_SHARD_OWNER_ROWS_TRANSPORT": transport.split()[0],
                        },
                    ),
                ):
                    runs[transport] = self._train(
                        mesh,
                        MixedPrecisionPolicy(
                            param_dtype=param_dtype, reduce_dtype=reduce_dtype
                        ),
                        reshard_after_forward,
                        transport.split()[0],
                    )
            chunks, again, grouped = (
                runs["chunks"],
                runs["chunks again"],
                runs["grouped"],
            )
            for first, second in zip(chunks[0], again[0], strict=True):
                self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(chunks[0][0], grouped[0][0]))
            # BF16 sums in another order round differently.
            tolerance = (
                {"rtol": 2e-2, "atol": 1e-2}
                if reduce_dtype == torch.bfloat16
                else {"rtol": 1e-5, "atol": 1e-6}
            )
            for name, grad in chunks[1].items():
                self.assertEqual(grad.dtype, torch.float32)
                self.assertEqual(grad, grouped[1][name], **tolerance)
            for first, second in zip(chunks[0], grouped[0], strict=True):
                self.assertEqual(first.float(), second.float(), rtol=2e-2, atol=2e-2)

    def _train(self, mesh, mp_policy, reshard_after_forward, transport):
        """Train 3 steps; return the outputs and step 0's sharded grads."""
        param_dtype = mp_policy.param_dtype
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(64, 64), nn.Linear(64, 16), nn.Linear(16, 16), nn.Linear(16, 8)
        ).cuda()
        expected = {
            name: param.detach().to(param_dtype).clone()
            for name, param in model.named_parameters()
        }
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["*"],
                    placement_fn=make_bucketed_owned_full_param_placement_fn(),
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=reshard_after_forward,
                )
            ],
        )
        # Step 0: check each param while its module runs, unsharded.
        mismatches: list[str] = []
        storages: set[int] = set()

        def check(module_name):
            def hook(module, _args):
                for name, param in module.named_parameters(recurse=False):
                    fqn = f"{module_name}.{name}"
                    storages.add(param.untyped_storage().nbytes())
                    if param.dtype != param_dtype or not torch.equal(
                        param.detach(), expected[fqn]
                    ):
                        mismatches.append(fqn)

            return hook

        handles = [
            module.register_forward_pre_hook(check(name))
            for name, module in model.named_children()
        ]
        calls: Counter[str] = Counter()
        names = (
            "all_gather",
            "all_gather_into_tensor",
            "reduce_scatter",
            "reduce_scatter_tensor",
        )
        originals = {name: getattr(dist, name) for name in names}

        def counting(name):
            def call(*args, **kwargs):
                calls[name] += 1
                return originals[name](*args, **kwargs)

            return call

        optimizer = make_test_sgd(model.parameters(), lr=0.1)
        outputs, grads = [], {}
        torch.manual_seed(1 + self.rank)
        with mock.patch.multiple(dist, **{name: counting(name) for name in names}):
            for step in range(3):
                output = model(torch.randn(4, 64, device="cuda", dtype=param_dtype))
                if step == 0:
                    for handle in handles:
                        handle.remove()
                outputs.append(output.detach().clone())
                output.float().square().sum().backward()
                if step == 0:
                    grads = {
                        name: param.grad.detach().clone()
                        for name, param in model.named_parameters()
                        if param.grad is not None and param.grad.numel()
                    }
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        self.assertEqual(mismatches, [])
        if transport == "grouped":
            self.assertGreater(calls["all_gather"], 0, calls)
            self.assertGreater(calls["reduce_scatter"], 0, calls)
            return outputs, grads
        self.assertEqual(calls["all_gather"] + calls["reduce_scatter"], 0, calls)
        self.assertGreater(calls["all_gather_into_tensor"], 0, calls)
        self.assertGreater(calls["reduce_scatter_tensor"], 0, calls)
        # Rows of their owners' lengths: the gathered storage is padded only
        # up to whole chunks.
        row = torch.tensor(
            [model.sharded_bucket_storages[0].total_bytes // 4], device="cuda"
        )
        rows = [torch.zeros_like(row) for _ in range(self.world_size)]
        dist.all_gather(rows, row)
        plan = owner_rows_transport_plan(
            mesh, [int(r) for r in rows], torch.device("cuda")
        )
        self.assertIsNotNone(plan)
        self.assertEqual(storages, {plan[0].gathered_numel * param_dtype.itemsize})
        if os.environ["LOCAL_WORLD_SIZE"] == "2":
            self.assertTrue(plan[0].host_moves)
        return outputs, grads


if __name__ == "__main__":
    run_tests()
