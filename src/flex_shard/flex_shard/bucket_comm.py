# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING

import torch
from torch.distributed.device_mesh import _get_device_handle

from .placement_contract import (
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo
    from .placement_contract import Placement


class UnshardHandle:
    """Handle for a FlexShard bucket unshard operation."""

    def finish(self) -> UnshardLease:
        """Wait for the unshard and transfer its parameter lease once."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the unshard result is usable on the current stream."""
        raise NotImplementedError

    def release_buffers(self) -> None:
        """Release buffers owned by an unconsumed unshard operation."""
        raise NotImplementedError


class ReduceGradHandle:
    """Handle for a FlexShard bucket gradient reduction operation."""

    def finish(self) -> list[torch.Tensor]:
        """Wait for the reduction and return local gradient shards."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the reduce-grad result is usable on the current stream."""
        raise NotImplementedError

    def synchronize(self) -> None:
        """Block the host until the reduce-grad result is complete."""
        raise NotImplementedError

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release temporary buffers owned by the reduce-grad operation."""
        raise NotImplementedError

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        """Track sharded grads until queued work on stream is complete."""
        raise NotImplementedError


@dataclass
class UnshardLease:
    """Full parameters and view-backing storage retained through their consumer."""

    full_params: list[torch.Tensor]
    consumer_handoff: StreamHandoff | None

    def has_consumer_buffers(self) -> bool:
        """Return whether this lease owns storage needed by a consumer."""
        return self.consumer_handoff is not None

    def take_full_params(self) -> list[torch.Tensor]:
        """Transfer full-parameter references while retaining buffer ownership."""
        full_params = self.full_params
        self.full_params = []
        return full_params

    def release(self) -> None:
        """Order view-backing storage after the consumer and drop owners."""
        if self.consumer_handoff is not None:
            self.consumer_handoff.release_after_current_stream()
            self.consumer_handoff = None
        self.full_params.clear()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass


@dataclass
class PreparedReduceGrad:
    """Packed reduce-grad inputs whose collective launch can be deferred."""

    prepared: PlacementPreparedReduceGrad
    copy_in_done: torch.Event | None
    copy_in_stream: torch.Stream | None
    device_handle: ModuleType | None


def _first_tensor_device(*tensor_groups: list[torch.Tensor]) -> torch.device | None:
    for tensors in tensor_groups:
        if tensors:
            return tensors[0].device
    return None


def _get_bucket_placement(
    infos: list[ParamInfo],
    operation: str,
) -> Placement:
    placement = infos[0].placement
    compatibility_key = placement.bucket_compatibility_key()
    for info in infos[1:]:
        if info.placement.bucket_compatibility_key() != compatibility_key:
            raise ValueError(
                f"FlexShard bucket {operation} requires all parameters in a "
                "bucket to use compatible placements, but "
                f"{infos[0].fqn!r} uses {placement!r} and {info.fqn!r} uses "
                f"{info.placement!r}."
            )
    return placement


def begin_bucket_unshard(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    unshard_stream: torch.Stream,
    debug_fqn: str | None = None,
) -> UnshardHandle:
    """Begin a bucket unshard and return a handle for full params."""
    placement = _get_bucket_placement(infos, "unshard")

    if torch.compiler.is_compiling():
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, debug_fqn)
        result = _run_and_finish_unshard(prepared)
        return SyncUnshardResult(result.full_params)

    device = tensors[0].device
    device_handle = _get_device_handle(device.type)
    copy_in_done = device_handle.Event()
    copy_in_done.record(device_handle.current_stream(device))
    with device_handle.stream(unshard_stream):
        unshard_stream.wait_event(copy_in_done)
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, debug_fqn)
        prepared.placement.run_prepared_unshard(prepared)
        event = device_handle.Event()
        event.record(unshard_stream)
    return AsyncUnshardResult(
        prepared=prepared,
        event=event,
        unshard_stream=unshard_stream,
        device_handle=device_handle,
    )


def prepare_reduce_grad(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    debug_fqn: str | None = None,
) -> PreparedReduceGrad:
    """Prepare reduce-grad inputs without launching the collective."""
    placement = _get_bucket_placement(infos, "reduce-grad")
    placement_prepared = placement.prepare_reduce_grad(
        tensors,
        infos,
        mesh,
        debug_fqn,
    )

    if torch.compiler.is_compiling():
        return PreparedReduceGrad(
            prepared=placement_prepared,
            copy_in_done=None,
            copy_in_stream=None,
            device_handle=None,
        )

    device = placement_prepared.buffers[0].device
    device_handle = _get_device_handle(device.type)
    copy_in_stream = device_handle.current_stream(device)
    copy_in_done = device_handle.Event()
    copy_in_done.record(copy_in_stream)
    return PreparedReduceGrad(
        prepared=placement_prepared,
        copy_in_done=copy_in_done,
        copy_in_stream=copy_in_stream,
        device_handle=device_handle,
    )


def launch_reduce_grad(
    prepared: PreparedReduceGrad,
    reduce_grad_stream: torch.Stream,
) -> ReduceGradHandle:
    """Launch a previously packed reduce-grad request."""
    if torch.compiler.is_compiling():
        result = prepared.prepared.placement.reduce_prepared_grad(prepared.prepared)
        return SyncReduceGradResult(result.sharded_grads)

    if (
        prepared.device_handle is None
        or prepared.copy_in_done is None
        or prepared.copy_in_stream is None
    ):
        raise AssertionError("Expected eager reduce-grad launch metadata.")
    with prepared.device_handle.stream(reduce_grad_stream):
        reduce_grad_stream.wait_event(prepared.copy_in_done)
        result = prepared.prepared.placement.reduce_prepared_grad(prepared.prepared)
        event = prepared.device_handle.Event()
        event.record(reduce_grad_stream)
    return AsyncReduceGradResult(
        sharded_grads=result.sharded_grads,
        event=event,
        buffers=[*prepared.prepared.buffers, *result.buffers],
        allocation_streams=(prepared.copy_in_stream, reduce_grad_stream),
        device_handle=prepared.device_handle,
    )


def begin_reduce_grad(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    reduce_grad_stream: torch.Stream,
    debug_fqn: str | None = None,
) -> ReduceGradHandle:
    """Begin a bucket reduce-grad and return a handle for local grad shards."""
    prepared = prepare_reduce_grad(tensors, infos, mesh, debug_fqn)
    return launch_reduce_grad(prepared, reduce_grad_stream)


@dataclass
class SyncUnshardResult(UnshardHandle):
    """Already-finished unshard result used during graph capture."""

    full_params: list[torch.Tensor]
    _lease_taken: bool = field(default=False, init=False)

    def finish(self) -> UnshardLease:
        if self._lease_taken:
            raise RuntimeError("An unshard lease may only be taken once.")
        lease = UnshardLease(self.full_params, None)
        self._lease_taken = True
        return lease

    def wait(self) -> None:
        return

    def release_buffers(self) -> None:
        return


@dataclass
class AsyncUnshardResult(UnshardHandle):
    """State needed to finish an async unshard launched on a side stream."""

    prepared: PlacementPreparedUnshard
    event: torch.Event | None
    unshard_stream: torch.Stream
    device_handle: ModuleType
    _result: PlacementUnshardResult | None = field(default=None, init=False)
    _device: torch.device | None = field(default=None, init=False)
    _lease_taken: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._device = _first_tensor_device(self.prepared.buffers)

    def finish(self) -> UnshardLease:
        if self._lease_taken:
            raise RuntimeError("An unshard lease may only be taken once.")
        self.wait()
        if self._result is None:
            self._result = self.prepared.placement.finish_prepared_unshard(
                self.prepared
            )
        self._device = (
            _first_tensor_device(
                self._result.full_params,
                self.prepared.buffers,
                self._result.buffers,
                self._result.finish_buffers,
                self._result.consumer_buffers,
            )
            or self._device
        )
        results = self._result.full_params
        finish_buffers, consumer_buffers = self._take_lifetime_buffers()
        finish_handoff = self._make_buffer_handoff(finish_buffers)
        if finish_handoff is not None:
            finish_handoff.release_after_current_stream()
        consumer_handoff = self._make_buffer_handoff(consumer_buffers)
        lease = UnshardLease(results, consumer_handoff)
        self._lease_taken = True
        return lease

    def wait(self) -> None:
        if self._device is None:
            return
        if self.event is not None:
            self.device_handle.current_stream(self._device).wait_event(self.event)

    def release_buffers(self) -> None:
        """Release raw unshard buffers after current-stream work is queued."""
        finish_buffers, consumer_buffers = self._take_lifetime_buffers()
        tensors = [*finish_buffers, *consumer_buffers]
        handoff = self._make_buffer_handoff(tensors)
        if handoff is not None:
            handoff.release_after_current_stream()

    def _make_buffer_handoff(
        self,
        tensors: list[torch.Tensor],
    ) -> StreamHandoff | None:
        if not tensors:
            return None
        current_stream = self.device_handle.current_stream(tensors[0].device)
        return StreamHandoff(
            tensors,
            (self.unshard_stream, current_stream),
            self.device_handle,
        )

    def _take_lifetime_buffers(
        self,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Atomically transfer finish- and consumer-lifetime buffers."""
        finish_buffers = list(self.prepared.buffers)
        consumer_buffers: list[torch.Tensor] = []
        if self._result is not None:
            finish_buffers.extend(self._result.buffers)
            finish_buffers.extend(self._result.finish_buffers)
            consumer_buffers.extend(self._result.consumer_buffers)

        self.prepared.buffers.clear()
        if self._result is not None:
            self._result.buffers.clear()
            self._result.finish_buffers.clear()
            self._result.consumer_buffers.clear()
        return finish_buffers, consumer_buffers


@dataclass
class SyncReduceGradResult(ReduceGradHandle):
    """Already-finished reduce-grad result used during graph capture."""

    sharded_grads: list[torch.Tensor]

    def finish(self) -> list[torch.Tensor]:
        return self.sharded_grads

    def wait(self) -> None:
        return

    def synchronize(self) -> None:
        return

    def release_buffers(self, release_sharded_grads: bool) -> None:
        if release_sharded_grads:
            self.sharded_grads.clear()

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        self.sharded_grads = sharded_grads


@dataclass
class AsyncReduceGradResult(ReduceGradHandle):
    """State needed to finish an async reduce-grad launched on a side stream."""

    sharded_grads: list[torch.Tensor]
    event: torch.Event | None
    buffers: list[torch.Tensor]
    allocation_streams: tuple[torch.Stream, ...]
    device_handle: ModuleType
    _device: torch.device | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._device = _first_tensor_device(self.sharded_grads, self.buffers)

    def finish(self) -> list[torch.Tensor]:
        self.wait()
        return self.sharded_grads

    def wait(self) -> None:
        if self._device is None:
            return
        if self.event is not None:
            self.device_handle.current_stream(self._device).wait_event(self.event)

    def synchronize(self) -> None:
        if self.event is not None:
            self.event.synchronize()

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release pending reduce-grad buffers after its completion wait."""
        tensors = list(self.buffers)
        self.buffers.clear()
        if release_sharded_grads:
            tensors.extend(self.sharded_grads)
            self.sharded_grads.clear()
        if not tensors:
            return
        StreamHandoff(
            tensors,
            self.allocation_streams,
            self.device_handle,
        ).release_after_current_stream()

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        self.sharded_grads = sharded_grads
        self._device = (
            _first_tensor_device(self.sharded_grads, self.buffers) or self._device
        )
        self.event = self.device_handle.Event()
        self.event.record(stream)


def _run_and_finish_unshard(prepared: PlacementPreparedUnshard):
    prepared.placement.run_prepared_unshard(prepared)
    return prepared.placement.finish_prepared_unshard(prepared)


class StreamHandoff:
    """Keep tensors alive until their allocation streams own the final wait."""

    __slots__ = (
        "_tensors",
        "_allocation_streams",
        "_device_handle",
        "_released",
    )

    def __init__(
        self,
        tensors: list[torch.Tensor],
        allocation_streams: tuple[torch.Stream, ...],
        device_handle: ModuleType | None = None,
    ) -> None:
        if device_handle is None:
            device_handle = _get_device_handle(tensors[0].device.type)
        self._tensors = tensors
        self._allocation_streams = allocation_streams
        self._device_handle = device_handle
        self._released = False

    def release_after(self, event: torch.Event) -> None:
        """Order every allocation stream after event, then drop keepalive refs."""
        if self._released:
            return
        seen_streams: set[int] = set()
        for stream in self._allocation_streams:
            stream_id = id(stream)
            if stream_id in seen_streams:
                continue
            seen_streams.add(stream_id)
            stream.wait_event(event)
        self._released = True
        self._tensors.clear()

    def release_after_current_stream(self) -> None:
        """Record the final consumer point and release after that event."""
        if self._released:
            return
        if not self._tensors:
            self._released = True
            return
        device = self._tensors[0].device
        event = self._device_handle.Event()
        event.record(self._device_handle.current_stream(device))
        self.release_after(event)

    def __del__(self) -> None:
        try:
            self.release_after_current_stream()
        except Exception:
            pass
