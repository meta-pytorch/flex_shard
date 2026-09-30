# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed.checkpoint import CheckpointableTensor

if TYPE_CHECKING:
    import torch.nn as nn
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo


@dataclass(frozen=True)
class LocalStorageLayout:
    """Placement-owned local storage layout for one parameter."""

    local_shape: torch.Size
    local_numel: int
    storage_nbytes: int


@dataclass(frozen=True)
class BucketParamStorageLayout:
    """Placement-owned bucket storage layout for one parameter."""

    local_shape: torch.Size
    local_numel: int
    byte_offset: int
    storage_nbytes: int
    bucket_layout: Any | None = None


@dataclass(frozen=True)
class BucketStorageLayout:
    """Placement-owned storage layout for a whole bucket."""

    param_layouts: dict[str, BucketParamStorageLayout]
    total_bytes: int


@dataclass(frozen=True)
class GlobalLayout:
    """Regions of the full parameter held by one rank's local tensor.

    Fields match DCP's ``CheckpointableTensor`` protocol.
    """

    global_shape: tuple[int, ...]
    global_offsets: tuple[tuple[int, ...], ...]
    local_offsets: tuple[tuple[int, ...], ...]
    local_sizes: tuple[tuple[int, ...], ...]


def get_global_layout(tensor: torch.Tensor) -> GlobalLayout | None:
    """Return the layout ``tensor`` declares, or ``None`` if it declares none."""
    if not isinstance(tensor, CheckpointableTensor):
        return None
    return GlobalLayout(
        global_shape=tuple(tensor.global_shape),
        global_offsets=tuple(map(tuple, tensor.global_offsets)),
        local_offsets=tuple(map(tuple, tensor.local_offsets)),
        local_sizes=tuple(map(tuple, tensor.local_sizes)),
    )


def set_global_layout(tensor: torch.Tensor, layout: GlobalLayout) -> None:
    """Declare where ``tensor`` sits in the full parameter.

    Sets the fields of DCP's ``CheckpointableTensor`` protocol. Parameters
    passed to ``flex_shard()`` that are local shards of an outer (e.g. TP/EP)
    sharding must declare their layout this way.
    """
    tensor.global_shape = layout.global_shape
    tensor.global_offsets = layout.global_offsets
    tensor.local_offsets = layout.local_offsets
    tensor.local_sizes = layout.local_sizes


def compose_global_layouts(outer: GlobalLayout, inner: GlobalLayout) -> GlobalLayout:
    """Map ``inner`` chunks into ``outer``'s global coordinates.

    ``outer`` maps an intermediate tensor ``T`` into the global tensor ``G``;
    ``inner`` maps the local tensor ``L`` into ``T``. Each overlapping pair of
    chunks in ``T`` yields one chunk mapping ``L`` directly into ``G``.
    """
    global_offsets = []
    local_offsets = []
    local_sizes = []
    for o_global, o_local, o_size in zip(
        outer.global_offsets, outer.local_offsets, outer.local_sizes, strict=True
    ):
        for i_global, i_local, i_size in zip(
            inner.global_offsets, inner.local_offsets, inner.local_sizes, strict=True
        ):
            start = tuple(map(max, i_global, o_local))
            end = tuple(
                min(ig + isz, ol + osz)
                for ig, isz, ol, osz in zip(i_global, i_size, o_local, o_size)
            )
            if any(e <= s for s, e in zip(start, end)):
                continue
            global_offsets.append(
                tuple(og + s - ol for og, s, ol in zip(o_global, start, o_local))
            )
            local_offsets.append(
                tuple(il + s - ig for il, s, ig in zip(i_local, start, i_global))
            )
            local_sizes.append(tuple(e - s for s, e in zip(start, end)))
    return GlobalLayout(
        global_shape=outer.global_shape,
        global_offsets=tuple(global_offsets),
        local_offsets=tuple(local_offsets),
        local_sizes=tuple(local_sizes),
    )


class Placement(ABC):
    """Base class for FlexShard placement strategies.

    The base class provides shared local storage helpers. Subclasses implement
    the methods below that raise NotImplementedError: per-param local payload
    selection and the bucket collective lifecycle for unshard and gradient
    reduction. For example, Shard uses all-gather and reduce-scatter, while
    Owned uses broadcast and reduce-to-owner.
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "__eq__" in cls.__dict__ and cls.__dict__.get("__hash__") is None:
            raise TypeError(
                f"{cls.__name__} must define __hash__ when overriding __eq__."
            )

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        """Return whether two placement instances have identical semantics."""
        raise NotImplementedError

    @abstractmethod
    def __hash__(self) -> int:
        """Return a hash consistent with placement equality."""
        raise NotImplementedError

    def bucket_compatibility_key(self) -> object:
        """Return a key shared by placements that can use one bucket collective."""
        return self

    def compute_local_numel(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> int:
        """How many elements this rank holds for a param with global_shape."""
        return math.prod(self.compute_local_shape(global_shape, rank, world_size))

    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        """Local shape this rank holds for a param with global_shape."""
        raise NotImplementedError

    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        """Extract this rank's local payload from the full parameter.

        The returned tensor may be a view and does not need to be contiguous.
        FlexShard makes it contiguous before copying into bucket byte storage.
        """
        raise NotImplementedError

    def local_storage_layout(
        self,
        global_shape: torch.Size,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ) -> LocalStorageLayout:
        """Return the local storage layout for one parameter."""
        local_shape = self.compute_local_shape(global_shape, rank, world_size)
        local_numel = self.compute_local_numel(global_shape, rank, world_size)
        return LocalStorageLayout(
            local_shape=local_shape,
            local_numel=local_numel,
            storage_nbytes=local_numel * dtype.itemsize,
        )

    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        """Return an optional bucket-global storage layout.

        Returning None asks ShardedBucketStorage to use the default sequential
        per-parameter layout derived from local_storage_layout().
        """
        return None

    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        """Pack one full parameter into its placement-owned local storage."""
        param_data = param.detach()
        if param_data.device.type == "meta":
            return
        shard = self.extract_local_shard(param_data, rank, world_size)
        if shard.numel() == 0:
            return
        if not shard.is_contiguous():
            shard = shard.contiguous()
        nbytes = shard.numel() * shard.element_size()
        if nbytes > info.storage_nbytes:
            raise ValueError(
                f"Placement {self!r} produced {nbytes} bytes for {info.fqn!r}, "
                f"but its storage layout only reserved {info.storage_nbytes} bytes."
            )
        byte_storage[info.byte_offset : info.byte_offset + nbytes].copy_(
            shard.view(-1).view(torch.uint8)
        )

    def make_local_storage_view(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
    ) -> torch.Tensor:
        """Return the exposed local parameter view from bucket storage."""
        nbytes = info.local_numel * info.dtype.itemsize
        byte_view = byte_storage[info.byte_offset : info.byte_offset + nbytes]
        return byte_view.view(info.dtype).view(info.local_shape)

    def global_layout(
        self, info: ParamInfo, rank: int, world_size: int
    ) -> GlobalLayout:
        """Return where this rank's local parameter sits in the full parameter."""
        raise NotImplementedError(
            f"{self!r} does not support distributed checkpointing."
        )

    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Prepare buffers for this placement's bucket unshard collective."""
        raise NotImplementedError

    def run_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> None:
        """Launch this placement's prepared unshard collective."""
        raise NotImplementedError

    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Finish the prepared unshard and return full parameters."""
        raise NotImplementedError

    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        """Prepare buffers for this placement's bucket gradient reduction."""
        raise NotImplementedError

    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        """Reduce prepared full gradients and return local gradient shards."""
        raise NotImplementedError


@dataclass
class PlacementPreparedUnshard:
    """Placement-owned prepared bucket unshard request.

    ``buffers`` have finish lifetime: the bucket runtime releases them after
    ``finish_prepared_unshard()`` queues their final use.
    ``placement_state`` is private metadata passed back to the same placement's
    run/finish methods.
    """

    placement: Placement
    buffers: list[torch.Tensor]
    placement_state: Any


@dataclass
class PlacementUnshardResult:
    """Full parameters and buffers with post-finish lifetime requirements.

    ``buffers`` preserves the legacy finish lifetime for custom
    placements. New placements should classify storage explicitly.
    ``finish_buffers`` are tensors created during finish that may be released
    after its copy-out work. ``consumer_buffers`` are the minimal storage roots
    backing full-parameter views and must remain live through their consumer.
    """

    full_params: list[torch.Tensor]
    buffers: list[torch.Tensor] = field(default_factory=list)
    finish_buffers: list[torch.Tensor] = field(default_factory=list)
    consumer_buffers: list[torch.Tensor] = field(default_factory=list)


@dataclass
class PlacementPreparedReduceGrad:
    """Placement-owned packed gradient reduction request.

    ``buffers`` are tensors whose lifetime the bucket runtime owns.
    ``placement_state`` is private metadata passed back to the same placement's
    reduce method.
    """

    placement: Placement
    buffers: list[torch.Tensor]
    placement_state: Any


@dataclass
class PlacementReduceGradResult:
    """Placement-owned reduced local gradients and temporary buffers."""

    sharded_grads: list[torch.Tensor]
    buffers: list[torch.Tensor] = field(default_factory=list)


__all__ = [
    "BucketParamStorageLayout",
    "BucketStorageLayout",
    "compose_global_layouts",
    "get_global_layout",
    "GlobalLayout",
    "LocalStorageLayout",
    "Placement",
    "PlacementPreparedUnshard",
    "PlacementPreparedReduceGrad",
    "PlacementReduceGradResult",
    "PlacementUnshardResult",
    "set_global_layout",
]
