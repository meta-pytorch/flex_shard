# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compute-ready FlexShard storage layouts used by the current Muon recipes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from ..custom_placements.block_shard import (
    BlockShard as StorageBlockShard,
    BucketedBlockShard,
)
from ..custom_placements.owned import BucketedOwned
from ..flex_shard.bucket_storage import ParamInfo
from ..flex_shard.placement_contract import get_global_layout
from ..flex_shard.sharded_param import (
    get_global_shape,
    get_placements,
    is_flex_shard_param,
)
from ..flex_shard.utils import _strip_checkpoint_wrapped_module_path as canonical_fqn


_CANONICAL_SHARDS_ATTR = "_flex_shard_muon_canonical_shards"


@dataclass(frozen=True, kw_only=True, slots=True)
class LocalMuonStateShard:
    global_shape: torch.Size
    global_offset: tuple[int, ...]
    local_shape: torch.Size


@dataclass(frozen=True, kw_only=True, slots=True)
class LocalMuonComputeLayout:
    kind: Literal["owned", "row_blocks", "matrix_batch"]
    state_checkpoint_shard: LocalMuonStateShard
    block_size: int | None = None


def capture_flex_shard_muon_canonical_shards(model: nn.Module) -> None:
    """Capture checkpoint coordinates before FlexShard replaces parameters."""
    shards: dict[str, LocalMuonStateShard] = {}
    for raw_fqn, parameter in model.named_parameters():
        if is_flex_shard_param(parameter):
            raise ValueError(
                "capture FlexShard Muon canonical shards before calling flex_shard"
            )
        shards[canonical_fqn(raw_fqn)] = _capture_canonical_shard(parameter)
    setattr(model, _CANONICAL_SHARDS_ATTR, shards)  # noqa: B010


def _capture_canonical_shard(parameter: torch.Tensor) -> LocalMuonStateShard:
    layout = get_global_layout(parameter)
    if layout is None:
        return LocalMuonStateShard(
            global_shape=parameter.shape,
            global_offset=(0,) * parameter.ndim,
            local_shape=parameter.shape,
        )
    if (
        len(layout.global_offsets) != 1
        or layout.global_shape[1:] != tuple(parameter.shape[1:])
        or layout.local_offsets[0] != (0,) * parameter.ndim
        or layout.local_sizes[0] != tuple(parameter.shape)
    ):
        raise NotImplementedError(
            "FlexShard Muon recipes only support an outer Shard(0) layout"
        )
    return LocalMuonStateShard(
        global_shape=torch.Size(layout.global_shape),
        global_offset=layout.global_offsets[0],
        local_shape=parameter.shape,
    )


def get_flex_shard_muon_compute_layouts(
    model: nn.Module,
    named_params: Sequence[tuple[str, torch.Tensor]],
) -> tuple[LocalMuonComputeLayout, ...]:
    canonical_shards = getattr(model, _CANONICAL_SHARDS_ATTR)
    bucket_param_infos = _collect_bucket_param_infos(model)
    return tuple(
        _flex_shard_muon_compute_layout(
            canonical_shards[canonical_fqn(fqn)],
            bucket_param_infos,
            fqn,
            parameter,
        )
        for fqn, parameter in named_params
    )


def _flex_shard_muon_compute_layout(
    canonical_shard: LocalMuonStateShard,
    bucket_param_infos: dict[str, ParamInfo],
    fqn: str,
    parameter: torch.Tensor,
) -> LocalMuonComputeLayout:
    info = bucket_param_infos.get(canonical_fqn(fqn))
    if info is not None:
        _validate_captured_outer_layout(fqn, canonical_shard, info)
    global_storage_shape = get_global_shape(parameter)
    placements = get_placements(parameter)
    mesh = parameter._mesh
    assert global_storage_shape is not None
    assert placements is not None and len(placements) == 1
    assert isinstance(mesh, DeviceMesh)

    placement = placements[0]
    if isinstance(placement, BucketedOwned):
        return _owned_layout(
            fqn,
            parameter,
            global_storage_shape,
            canonical_shard,
        )
    if isinstance(placement, StorageBlockShard):
        return _block_shard_layout(
            fqn,
            parameter,
            global_storage_shape,
            canonical_shard,
            placement,
            rank=mesh.get_local_rank(),
        )
    if isinstance(placement, BucketedBlockShard):
        return _bucketed_block_shard_layout(
            fqn,
            parameter,
            global_storage_shape,
            canonical_shard,
            placement,
            bucket_param_infos[canonical_fqn(fqn)],
        )
    raise NotImplementedError(
        f"FlexShard placement {placement!r} for Muon parameter {fqn!r} "
        "is not used by the supported recipes."
    )


def _validate_captured_outer_layout(
    fqn: str,
    canonical_shard: LocalMuonStateShard,
    info: ParamInfo,
) -> None:
    # flex_shard() records the declared outer layout; a shard captured before it
    # was declared recorded the outer local shard as the whole parameter.
    outer = info.outer_layout
    if outer is None:
        expected = (tuple(info.global_shape), ((0,) * len(info.global_shape),))
    else:
        expected = (outer.global_shape, outer.global_offsets)
    captured = (tuple(canonical_shard.global_shape), (canonical_shard.global_offset,))
    if captured != expected:
        raise ValueError(
            f"FlexShard Muon parameter {fqn!r} was captured with global shape and "
            f"offsets {captured}, but flex_shard() recorded {expected}; declare "
            "outer layouts before capture_flex_shard_muon_canonical_shards()."
        )


def _owned_layout(
    fqn: str,
    parameter: torch.Tensor,
    global_storage_shape: torch.Size,
    canonical_shard: LocalMuonStateShard,
) -> LocalMuonComputeLayout:
    if len(global_storage_shape) != 2 or parameter.shape != global_storage_shape:
        raise NotImplementedError(
            f"BucketedOwned Muon parameter {fqn!r} must own one complete 2D matrix."
        )
    return _local_muon_compute_layout(
        parameter,
        global_storage_shape,
        canonical_shard,
        kind="owned",
        inner_global_offset=(0, 0),
    )


def _block_shard_layout(
    fqn: str,
    parameter: torch.Tensor,
    global_storage_shape: torch.Size,
    canonical_shard: LocalMuonStateShard,
    placement: StorageBlockShard,
    *,
    rank: int,
) -> LocalMuonComputeLayout:
    if len(global_storage_shape) != 2 or placement.dim != 0:
        raise NotImplementedError(
            f"BlockShard Muon parameter {fqn!r} must be a 2D dim-0 block shard."
        )
    num_matrices = sum(placement.blocks_per_rank)
    matrix_rows = global_storage_shape[0] // num_matrices
    row_offset = sum(placement.blocks_per_rank[:rank]) * matrix_rows
    return _local_muon_compute_layout(
        parameter,
        global_storage_shape,
        canonical_shard,
        kind="row_blocks",
        block_size=matrix_rows,
        inner_global_offset=(row_offset, 0),
    )


def _bucketed_block_shard_layout(
    fqn: str,
    parameter: torch.Tensor,
    global_storage_shape: torch.Size,
    canonical_shard: LocalMuonStateShard,
    placement: BucketedBlockShard,
    param_info: ParamInfo,
) -> LocalMuonComputeLayout:
    if len(global_storage_shape) != 3 or placement.dims != (0,):
        raise NotImplementedError(
            f"BucketedBlockShard Muon parameter {fqn!r} must store a 3D "
            "matrix batch on dim 0."
        )
    matrix_numel = math.prod(global_storage_shape[1:])
    local_flat_offset = _bucketed_block_local_flat_offset(param_info)
    assert local_flat_offset % matrix_numel == 0
    assert parameter.numel() % matrix_numel == 0
    return _local_muon_compute_layout(
        parameter,
        global_storage_shape,
        canonical_shard,
        kind="matrix_batch",
        inner_global_offset=(local_flat_offset // matrix_numel, 0, 0),
    )


def _bucketed_block_local_flat_offset(param_info: ParamInfo) -> int:
    bucket_layout = param_info.bucket_layout
    assert bucket_layout is not None
    param_layout = bucket_layout.param_layouts[param_info.fqn]
    return param_layout.local_global_offset - param_layout.param_offset


def _local_muon_compute_layout(
    parameter: torch.Tensor,
    global_storage_shape: torch.Size,
    canonical_shard: LocalMuonStateShard,
    *,
    kind: Literal["owned", "row_blocks", "matrix_batch"],
    inner_global_offset: tuple[int, ...],
    block_size: int | None = None,
) -> LocalMuonComputeLayout:
    assert canonical_shard.local_shape == global_storage_shape
    global_offset = tuple(
        outer + inner
        for outer, inner in zip(
            canonical_shard.global_offset,
            inner_global_offset,
            strict=True,
        )
    )
    return LocalMuonComputeLayout(
        kind=kind,
        block_size=block_size,
        state_checkpoint_shard=LocalMuonStateShard(
            global_shape=canonical_shard.global_shape,
            global_offset=global_offset,
            local_shape=parameter.shape,
        ),
    )


def _collect_bucket_param_infos(model: nn.Module) -> dict[str, ParamInfo]:
    infos: dict[str, ParamInfo] = {}
    for module_fqn, module in model.named_modules():
        for bucket_storage in getattr(module, "sharded_bucket_storages", ()):
            for storage_fqn, param_info in bucket_storage.param_infos.items():
                qualified_fqn = (
                    f"{module_fqn}.{storage_fqn}" if module_fqn else storage_fqn
                )
                infos[canonical_fqn(qualified_fqn)] = param_info
    return infos
