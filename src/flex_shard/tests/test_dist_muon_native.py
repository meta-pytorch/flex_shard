# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import io
from contextlib import contextmanager
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests

from .. import BucketSpec, flex_shard, get_placements
from ..custom_placements.block_shard import (
    BlockShard,
    make_bucketed_block_placement_fn,
)
from ..custom_placements.mixed_bucket import MixedBucketPlacement
from ..custom_placements.owned import (
    BucketedOwned,
    make_bucketed_owned_full_param_placement_fn,
    make_bucketed_owned_full_param_segments,
)
from ..custom_placements.shard import per_param_placements
from ..dist_muon import (
    build_dist_muon,
    build_local_dist_muon,
    capture_flex_shard_muon_canonical_shards,
    DistMuon,
)
from ..dist_muon.optimizer_reshard import (
    BlockShard as ComputeBlockShard,
    BucketConfig,
    ComputeLayout,
    Owned as ComputeOwned,
)


MUON_KWARGS = dict(
    lr=0.01,
    weight_decay=0.01,
    momentum=0.95,
    nesterov=True,
    ns_coefficients=(3.4445, -4.7750, 2.0315),
    ns_steps=5,
    eps=1e-7,
    adjust_lr_fn="original",
)
ADAMW_KWARGS = dict(lr=1e-3, eps=1e-6, weight_decay=0.01, foreach=False)
_LARGE_ADAMW = {"tok_embeddings.weight", "pos_embeddings.weight", "output.weight"}


@contextmanager
def _without_dtensor_adapter():
    # Patch constructors rather than import aliases so a hidden adapter cannot
    # evade the check by importing its own DTensor or DeviceMesh symbol.
    with (
        mock.patch.object(
            DTensor, "__new__", side_effect=AssertionError("DTensor proxy")
        ),
        mock.patch.object(
            DeviceMesh, "__init__", side_effect=AssertionError("optimizer mesh")
        ),
        mock.patch.object(
            dist.distributed_c10d,
            "_new_process_group_helper",
            side_effect=AssertionError("optimizer process group"),
        ),
    ):
        yield


def _serialized_state(optimizer):
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embeddings = nn.Embedding(32, 16)
        self.pos_embeddings = nn.Embedding(8, 16)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    16,
                    4,
                    dim_feedforward=32,
                    dropout=0.0,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(2)
            ]
        )
        self.norm = nn.LayerNorm(16)
        self.output = nn.Linear(16, 32, bias=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device)
        hidden = self.tok_embeddings(tokens) + self.pos_embeddings(positions)
        mask = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(
            1
        )
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask, is_causal=True)
        return self.output(self.norm(hidden))


def _transformer_placements(named_params, mesh):
    owned = [
        (name, p)
        for name, p in named_params
        if name not in _LARGE_ADAMW and not name.endswith("self_attn.in_proj_weight")
    ]
    mixed = MixedBucketPlacement(
        make_bucketed_owned_full_param_segments(owned, mesh.size())
    )
    return {
        name: (
            mixed.block_shard(blocks_per_rank=(2, 1))
            if name.endswith("self_attn.in_proj_weight")
            else mixed.block_shard(blocks_per_rank=(1, 1))
            if name in _LARGE_ADAMW
            else mixed.bucketed_owned,
        )
        for name, _ in named_params
    }


def _transformer_buckets(mesh):
    return [
        BucketSpec(
            [pattern],
            mesh=mesh,
            placement_fn=_transformer_placements,
            gradient_reduce_op=dist.ReduceOp.AVG,
            reshard_after_forward=False,
        )
        for pattern in (
            "tok_embeddings.*",
            "pos_embeddings.*",
            "layers.0.*",
            "layers.1.*",
            "norm.*",
            "output.*",
        )
    ]


def _transformer_slice(name, full, parameter, rank):
    (placement,) = get_placements(parameter)
    if isinstance(placement, BlockShard):
        rows = full.shape[0] // sum(placement.blocks_per_rank)
        start = rows * sum(placement.blocks_per_rank[:rank])
        return full.narrow(0, start, rows * placement.blocks_per_rank[rank])
    assert isinstance(placement, BucketedOwned)
    (segment,) = placement.segments_by_fqn[name]
    return full if segment.owner_rank == rank else full.new_empty(parameter.shape)


class _Matrices(nn.Module):
    def __init__(self, device):
        super().__init__()
        for index, (name, shape) in enumerate(
            (
                ("owned", (4, 3)),
                ("packed", (12, 3)),
                ("prefix", (2, 4, 3)),
                ("experts", (4, 4, 3)),
            )
        ):
            value = torch.arange(torch.Size(shape).numel(), device=device).float()
            self.register_parameter(
                name, nn.Parameter(value.reshape(shape).div(30).add(index / 10))
            )


def _matrix_buckets(mesh):
    def packed(named, _mesh):
        return {name: (BlockShard((2, 1)),) for name, _ in named}

    return [
        BucketSpec(
            ["owned"],
            mesh=mesh,
            placement_fn=make_bucketed_owned_full_param_placement_fn(),
            reshard_after_forward=False,
        ),
        BucketSpec(
            ["packed"], mesh=mesh, placement_fn=packed, reshard_after_forward=False
        ),
        BucketSpec(
            ["prefix", "experts"],
            mesh=mesh,
            placement_fn=make_bucketed_block_placement_fn(
                dims=(0,), blocks_per_rank=(3, 3)
            ),
            reshard_after_forward=False,
        ),
    ]


def _matrix_slice(full, name, rank):
    # Independently specified expected ownership for _matrix_buckets, including
    # the experts offset after the two prefix matrices in the packed bucket.
    if name == "owned":
        return full if rank == 0 else full[:0]
    if name == "packed":
        return full[:8] if rank == 0 else full[8:]
    if name == "prefix":
        return full if rank == 0 else full[:0]
    assert name == "experts"
    return full[:1] if rank == 0 else full[1:]


def _legacy_factory(groups, layouts):
    compute = {}
    for name, layout in layouts.items():
        if layout.kind == "owned":
            placement = ComputeOwned()
        elif layout.kind == "row_blocks":
            placement = ComputeBlockShard(0, layout.block_size)
        else:
            placement = Shard(0)
        compute[name] = ComputeLayout({"local": placement})
    return build_dist_muon(
        groups,
        compute_sharding_by_fqn=compute,
        bucket_configs=(BucketConfig(patterns=("*",), name="legacy"),),
        **MUON_KWARGS,
    )


class TestDistMuonNative(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _mesh(self):
        return init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))

    @skip_if_lt_x_gpu(2)
    def test_transformer_matches_reference_without_adapter(self):
        mesh = self._mesh()
        torch.manual_seed(42)
        reference = _TinyTransformer().cuda().train()
        model = copy.deepcopy(reference)
        muon_names = {
            name
            for name, p in reference.named_parameters()
            if name.startswith("layers.") and p.ndim == 2
        }
        flex_shard(model, buckets=_transformer_buckets(mesh))
        named = dict(model.named_parameters())
        ref_named = dict(reference.named_parameters())
        parts = {
            name: [nn.Parameter(x) for x in p.detach().chunk(3)]
            if name.endswith("self_attn.in_proj_weight")
            else [p]
            for name, p in ref_named.items()
            if name in muon_names
        }
        ref_muon = torch.optim.Muon(
            [p for group in parts.values() for p in group], **MUON_KWARGS
        )
        ref_adam = torch.optim.AdamW(
            [p for name, p in ref_named.items() if name not in muon_names],
            **ADAMW_KWARGS,
        )
        adam = torch.optim.AdamW(
            [p for name, p in named.items() if name not in muon_names], **ADAMW_KWARGS
        )
        rng = torch.Generator(device="cuda").manual_seed(1234 + self.rank)
        with _without_dtensor_adapter():
            selected = [p for name, p in named.items() if name in muon_names]
            muon = DistMuon(selected, **MUON_KWARGS)
            self.assertEqual(
                [id(p) for p in selected],
                [id(p) for p in muon.param_groups[0]["params"]],
            )
            for step in range(3):
                tokens = torch.randint(32, (2, 9), device="cuda", generator=rng)
                muon.zero_grad(set_to_none=True)
                adam.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                ref_muon.zero_grad(set_to_none=True)
                ref_adam.zero_grad(set_to_none=True)
                losses = []
                for current in (reference, model):
                    logits = current(tokens[:, :-1])
                    loss = F.cross_entropy(
                        logits.flatten(0, 1), tokens[:, 1:].flatten()
                    )
                    loss.backward()
                    losses.append(loss.detach())
                    if current is reference:
                        for p in reference.parameters():
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                self.assertEqual(losses[0], losses[1], atol=3e-5, rtol=3e-5)
                for name, group in parts.items():
                    if len(group) == 3:
                        for p, grad in zip(
                            group, ref_named[name].grad.chunk(3), strict=True
                        ):
                            p.grad = grad
                for name, p in named.items():
                    if p.numel():
                        self.assertEqual(
                            p.grad,
                            _transformer_slice(
                                name, ref_named[name].grad, p, self.rank
                            ),
                            atol=3e-5,
                            rtol=3e-4,
                        )
                muon.step()
                adam.step()
                ref_muon.step()
                ref_adam.step()
                for name, p in named.items():
                    self.assertNotIsInstance(p, DTensor)
                    self.assertEqual(
                        p,
                        _transformer_slice(name, ref_named[name], p, self.rank),
                        atol=2e-4,
                        rtol=3e-4,
                    )
                    if not p.numel():
                        self.assertNotIn(p, muon.state)
                        continue
                    if name in muon_names:
                        reference_state = [
                            ref_muon.state[q]["momentum_buffer"] for q in parts[name]
                        ]
                        expected = (
                            torch.cat(reference_state)
                            if len(reference_state) == 3
                            else reference_state[0]
                        )
                        actual = muon.state[p]["momentum_buffer"]
                        self.assertNotIsInstance(actual, DTensor)
                        self.assertEqual(actual.shape, p.shape)
                        self.assertEqual(
                            actual,
                            _transformer_slice(name, expected, p, self.rank),
                            atol=3e-6,
                            rtol=5e-4,
                        )
                    else:
                        for key in ("exp_avg", "exp_avg_sq"):
                            self.assertEqual(
                                adam.state[p][key],
                                _transformer_slice(
                                    name,
                                    ref_adam.state[ref_named[name]][key],
                                    p,
                                    self.rank,
                                ),
                                atol=3e-6,
                                rtol=5e-4,
                            )

    def _check_matrix_resume(self, legacy):
        mesh = self._mesh()
        reference = _Matrices("cuda")
        model = copy.deepcopy(reference)
        if legacy:
            capture_flex_shard_muon_canonical_shards(model)
        flex_shard(model, buckets=_matrix_buckets(mesh))
        named = dict(model.named_parameters())
        ref_named = dict(reference.named_parameters())
        ref_parts = {}
        for name, p in ref_named.items():
            views = (
                p.detach().chunk(3)
                if name == "packed"
                else p.detach().unbind()
                if p.ndim == 3
                else (p,)
            )
            ref_parts[name] = [nn.Parameter(view) for view in views]
        reference_optimizer = torch.optim.Muon(
            [p for group in ref_parts.values() for p in group], **MUON_KWARGS
        )
        legacy_binding = (
            build_local_dist_muon(
                model, list(named.items()), optimizer_factory=_legacy_factory
            )
            if legacy
            else None
        )
        with _without_dtensor_adapter():
            optimizer = DistMuon(model.parameters(), **MUON_KWARGS)
        for step in range(3):
            for name, full in ref_named.items():
                gradient = torch.cos(
                    torch.arange(full.numel(), device="cuda").reshape_as(full) + step
                ).div(10)
                named[name].grad = _matrix_slice(gradient, name, self.rank).clone()
                gradients = (
                    gradient.chunk(3)
                    if name == "packed"
                    else gradient.unbind()
                    if full.ndim == 3
                    else (gradient,)
                )
                for part, grad in zip(ref_parts[name], gradients, strict=True):
                    part.grad = grad
            if legacy and step == 0:
                legacy_binding.step()
                state = _serialized_state(legacy_binding.optimizer)
            else:
                with _without_dtensor_adapter():
                    optimizer.step()
                    state = _serialized_state(optimizer)
            reference_optimizer.step()
            for name, p in named.items():
                self.assertEqual(
                    p,
                    _matrix_slice(ref_named[name], name, self.rank),
                    atol=2e-4,
                    rtol=3e-4,
                )
                if p.numel() and not (legacy and step == 0):
                    momentum = [
                        reference_optimizer.state[q]["momentum_buffer"]
                        for q in ref_parts[name]
                    ]
                    full = (
                        torch.cat(momentum)
                        if name == "packed"
                        else torch.stack(momentum)
                        if p.ndim == 3
                        else momentum[0]
                    )
                    self.assertEqual(
                        optimizer.state[p]["momentum_buffer"],
                        _matrix_slice(full, name, self.rank),
                        atol=3e-6,
                        rtol=5e-4,
                    )
            if step == 0:
                # Fresh optimizer on the same actual parameters must restore the
                # serialized state, including filtered legacy parameter groups.
                with _without_dtensor_adapter():
                    optimizer = DistMuon(model.parameters(), lr=0.123)
                    optimizer.load_state_dict(state)
                    self.assertEqual(optimizer.param_groups[0]["lr"], MUON_KWARGS["lr"])
                    for p in named.values():
                        if p.numel():
                            self.assertNotIsInstance(
                                optimizer.state[p]["momentum_buffer"], DTensor
                            )
                        else:
                            self.assertNotIn(p, optimizer.state)
            with _without_dtensor_adapter():
                optimizer.zero_grad(set_to_none=bool(step % 2))
                for p in named.values():
                    self.assertTrue(p.grad is None or not torch.count_nonzero(p.grad))

    @skip_if_lt_x_gpu(2)
    def test_matrix_layouts_checkpoint_resume(self):
        self._check_matrix_resume(legacy=False)

    @skip_if_lt_x_gpu(2)
    def test_legacy_adapter_checkpoint_resume(self):
        self._check_matrix_resume(legacy=True)

    @skip_if_lt_x_gpu(2)
    def test_empty_owner_and_invalid_gradients(self):
        mesh = self._mesh()
        model = _Matrices("cuda")
        flex_shard(model, buckets=_matrix_buckets(mesh))
        with _without_dtensor_adapter():
            optimizer = DistMuon(model.parameters(), **MUON_KWARGS)
            active = [p for p in model.parameters() if p.numel()]
            for p in active:
                p.grad = torch.ones_like(p)
            for initialized in (False, True):
                if initialized:
                    active[-1].grad = torch.ones_like(active[-1])
                    optimizer.step()
                active[-1].grad = None
                before = [p.detach().clone() for p in active]
                state_before = copy.deepcopy(optimizer.state_dict())
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    optimizer.step()
                self.assertEqual([p.detach() for p in active], before)
                self.assertEqual(optimizer.state_dict(), state_before)

            owner_only = DistMuon(
                [dict(model.named_parameters())["owned"]], **MUON_KWARGS
            )
            p = owner_only.param_groups[0]["params"][0]
            if self.rank == 0:
                p.grad = torch.ones_like(p)
            else:
                self.assertEqual(p.numel(), 0)
                p.grad = None
            owner_only.step()
            owner_only.zero_grad(set_to_none=False)
            if self.rank == 1:
                self.assertEqual(dict(owner_only.state), {})
            saved = _serialized_state(owner_only)
            owner_only.load_state_dict(saved)
            owner_only.step()

        partial = nn.Linear(3, 8, bias=False, device="cuda")
        flex_shard(
            partial,
            buckets=[
                BucketSpec(
                    ["*"],
                    mesh=mesh,
                    placement_fn=per_param_placements,
                    reshard_after_forward=False,
                )
            ],
        )
        with self.assertRaises((ValueError, NotImplementedError)):
            DistMuon(partial.parameters(), **MUON_KWARGS)

    @skip_if_lt_x_gpu(2)
    def test_outer_shard_checkpoint_coordinates(self):
        mesh = init_device_mesh("cuda", (2, 1), mesh_dim_names=("outer", "fsdp"))
        full = torch.arange(24, device="cuda").reshape(8, 3).float().div(20)
        model = nn.Module()
        model.weight = nn.Parameter(distribute_tensor(full, mesh["outer"], (Shard(0),)))
        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["weight"],
                    mesh=mesh["fsdp"],
                    placement_fn=make_bucketed_owned_full_param_placement_fn(),
                    reshard_after_forward=False,
                )
            ],
        )
        p = dict(model.named_parameters())["weight"]
        with _without_dtensor_adapter():
            optimizer = DistMuon(model.parameters(), **MUON_KWARGS)
            p.grad = torch.ones_like(p)
            optimizer.step()
            saved = _serialized_state(optimizer)
            momentum = next(iter(saved["state"].values()))["momentum_buffer"]
            self.assertEqual(momentum.global_shape, (8, 3))
            self.assertEqual(momentum.global_offsets, ((self.rank * 4, 0),))
            self.assertEqual(momentum.local_sizes, ((4, 3),))
            baseline = p.detach().clone()
            optimizer.step()
            expected = p.detach().clone()
            with torch.no_grad():
                p.copy_(baseline)
            resumed = DistMuon(model.parameters(), **MUON_KWARGS)
            resumed.load_state_dict(saved)
            resumed.step()
            self.assertEqual(p, expected)


if __name__ == "__main__":
    run_tests()
