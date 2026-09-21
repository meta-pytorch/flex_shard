# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard as DTensorShard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo
    from .placement_contract import Placement


# Hidden attribute names for FlexShard metadata on local parameter tensors.
_PLACEMENTS_ATTR = "_placements"
_GLOBAL_SHAPE_ATTR = "_global_shape"
_GLOBAL_STRIDE_ATTR = "_global_stride"
_MESH_ATTR = "_mesh"
_SHARD_METADATA_ATTR = "_flex_shard_metadata"


@dataclass(frozen=True)
class ShardRegion:
    """One rectangular region mapping local storage into canonical coordinates.

    Offsets are measured in elements along each dimension, relative to the
    local tensor and original global parameter respectively.
    """

    local_offset: tuple[int, ...]
    global_offset: tuple[int, ...]
    shape: torch.Size


@dataclass(frozen=True)
class ShardMetadata:
    """Snapshot of a replacement parameter's layout, independent of optimizers.

    ``storage_shape`` and ``storage_stride`` describe the pre-inner-sharding
    tensor, preserving the meaning of ``get_global_shape``. ``canonical_shape``
    describes the original parameter before any outer DTensor sharding.
    ``outer_offset`` and ``regions`` are ``None`` if the layout cannot be
    represented by known rectangular regions. Empty local shards have no
    regions. ``local_flat_offset`` is parameter-relative, never bucket-relative.

    The record and its coordinate values are immutable; placements and mesh
    identify the existing FlexShard configuration. No mutable bucket layouts
    or storage buffers are exposed.
    """

    fqn: str
    placements: tuple[Placement, ...]
    mesh: DeviceMesh
    storage_shape: torch.Size
    storage_stride: tuple[int, ...]
    canonical_shape: torch.Size
    canonical_stride: tuple[int, ...]
    outer_offset: tuple[int, ...] | None
    local_shape: torch.Size
    regions: tuple[ShardRegion, ...] | None
    local_flat_offset: int | None


@dataclass(frozen=True)
class _CanonicalShard:
    shape: torch.Size
    stride: tuple[int, ...]
    local_shape: torch.Size
    offset: tuple[int, ...] | None


def _capture_canonical_shard(parameter: torch.Tensor) -> _CanonicalShard:
    """Capture outer coordinates before a DTensor is replaced by local storage.

    Unknown DTensor layouts must not prevent otherwise supported FlexShard
    usage. Their canonical shape is retained, with unavailable coordinates.
    """
    if not isinstance(parameter, DTensor):
        return _CanonicalShard(
            parameter.shape,
            tuple(parameter.stride()),
            parameter.shape,
            (0,) * parameter.ndim,
        )
    offset = None
    local_shape = parameter.to_local().shape
    if all(type(p) in (Replicate, DTensorShard) for p in parameter.placements):
        computed_shape, computed_offset = compute_local_shape_and_global_offset(
            parameter.shape, parameter.device_mesh, parameter.placements
        )
        if torch.Size(computed_shape) == local_shape:
            offset = tuple(computed_offset)
    return _CanonicalShard(
        parameter.shape, tuple(parameter.stride()), local_shape, offset
    )


def get_shard_metadata(tensor: torch.Tensor) -> ShardMetadata | None:
    """Return the immutable layout snapshot installed by ``flex_shard``."""
    return getattr(tensor, _SHARD_METADATA_ATTR, None)


def _inner_region_offset(
    info: ParamInfo, mesh: DeviceMesh
) -> tuple[tuple[int, ...], int] | None:
    # Import built-in placement types lazily: placements also use ParamInfo.
    from ..custom_placements.block_shard import BlockShard, BucketedBlockShard
    from ..custom_placements.owned import BucketedOwned
    from ..custom_placements.shard import Shard

    placement = info.placement
    rank = mesh.get_local_rank()
    shape = info.global_shape
    offset = [0] * len(shape)
    if isinstance(placement, BlockShard):
        dim = placement.dim % len(shape)
        block_size = shape[dim] // sum(placement.blocks_per_rank)
        offset[dim] = sum(placement.blocks_per_rank[:rank]) * block_size
        return tuple(offset), offset[dim] * math.prod(shape[dim + 1 :])
    if isinstance(placement, Shard):
        dim = placement.dim % len(shape)
        chunk = (shape[dim] + mesh.size() - 1) // mesh.size()
        offset[dim] = min(shape[dim], rank * chunk)
        return tuple(offset), offset[dim] * math.prod(shape[dim + 1 :])
    if isinstance(placement, BucketedOwned):
        segments = [
            segment
            for segment in placement.segments_by_fqn[info.fqn]
            if segment.owner_rank == rank
        ]
        if not segments:
            return tuple(offset), 0
        flat_offset = segments[0].param_offset
        suffix_numel = math.prod(shape[1:])
        if not suffix_numel or flat_offset % suffix_numel:
            return None
        offset[0] = flat_offset // suffix_numel
        return tuple(offset), flat_offset
    if isinstance(placement, BucketedBlockShard):
        # Multiple flattened prefix dimensions need multiple global regions.
        # Leave these available to custom consumers, without asserting a false
        # single-rectangle mapping.
        if placement.dims != (0,) or info.bucket_layout is None:
            return None
        param_layout = info.bucket_layout.param_layouts[info.fqn]
        flat_offset = param_layout.local_global_offset - param_layout.param_offset
        suffix_numel = math.prod(shape[1:])
        if not suffix_numel or flat_offset % suffix_numel:
            return None
        offset[0] = flat_offset // suffix_numel
        return tuple(offset), flat_offset
    return None


def _build_shard_metadata(info: ParamInfo, mesh: DeviceMesh) -> ShardMetadata:
    from .utils import _strip_checkpoint_wrapped_module_path

    canonical = info.canonical_shard
    if canonical is None:
        canonical = _CanonicalShard(
            info.global_shape,
            info.global_stride,
            info.global_shape,
            (0,) * len(info.global_shape),
        )
    regions: tuple[ShardRegion, ...] | None = None
    flat_offset = None
    if info.local_numel == 0:
        regions = ()
    elif canonical.offset is not None and canonical.local_shape == info.global_shape:
        inner = _inner_region_offset(info, mesh)
        if inner is not None and len(info.local_shape) == len(canonical.shape):
            inner_offset, flat_offset = inner
            regions = (
                ShardRegion(
                    local_offset=(0,) * len(info.local_shape),
                    global_offset=tuple(
                        outer + offset
                        for outer, offset in zip(
                            canonical.offset, inner_offset, strict=True
                        )
                    ),
                    shape=info.local_shape,
                ),
            )
    return ShardMetadata(
        fqn=_strip_checkpoint_wrapped_module_path(info.fqn),
        placements=info.placements,
        mesh=mesh,
        storage_shape=info.global_shape,
        storage_stride=info.global_stride,
        canonical_shape=canonical.shape,
        canonical_stride=canonical.stride,
        outer_offset=canonical.offset,
        local_shape=info.local_shape,
        regions=regions,
        local_flat_offset=flat_offset,
    )


def get_placements(tensor: torch.Tensor) -> tuple[Placement, ...] | None:
    """Get FlexShard placements from a tensor, or None if not annotated."""
    return getattr(tensor, _PLACEMENTS_ATTR, None)


def get_global_shape(tensor: torch.Tensor) -> torch.Size | None:
    """Get the global unsharded shape from a tensor, or None if not annotated."""
    return getattr(tensor, _GLOBAL_SHAPE_ATTR, None)


def is_flex_shard_param(tensor: torch.Tensor) -> bool:
    """Return whether a tensor represents a FlexShard-managed parameter."""
    return hasattr(tensor, _PLACEMENTS_ATTR)


def set_sharding_info(
    tensor: torch.Tensor,
    placements: tuple[Placement, ...],
    global_shape: torch.Size,
    global_stride: tuple[int, ...],
    mesh: DeviceMesh,
    metadata: ShardMetadata | None = None,
) -> None:
    """Annotate a local parameter tensor with its global FlexShard metadata."""
    setattr(tensor, _PLACEMENTS_ATTR, placements)
    setattr(tensor, _GLOBAL_SHAPE_ATTR, global_shape)
    setattr(tensor, _GLOBAL_STRIDE_ATTR, global_stride)
    setattr(tensor, _MESH_ATTR, mesh)
    setattr(tensor, _SHARD_METADATA_ATTR, metadata)


__all__ = [
    "get_global_shape",
    "get_placements",
    "get_shard_metadata",
    "is_flex_shard_param",
    "ShardMetadata",
    "ShardRegion",
    "set_sharding_info",
]
