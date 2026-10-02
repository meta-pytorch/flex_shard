# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.utils._pytree import tree_leaves

from .bucket_comm import (
    begin_bucket_unshard,
    begin_reduce_grad,
    launch_reduce_grad,
    prepare_reduce_grad,
    PreparedReduceGrad,
    ReduceGradHandle,
    UnshardHandle,
)
from .bucket_storage import ParamInfo, ShardedBucketStorage
from .utils import _get_bucket_storage_debug_fqn, _record_function_if_eager


_EAGER_COMM_CONTEXTS_ATTR = "_flex_shard_eager_comm_contexts"
_MAX_PENDING_REDUCE_GRADS_ATTR = "_flex_shard_max_pending_reduce_grads"


@dataclass
class ParamOwnerRef:
    """Owning module and local parameter name for a managed parameter."""

    module: nn.Module
    param_name: str

    @classmethod
    def resolve(cls, root_module: nn.Module, fqn: str) -> ParamOwnerRef:
        """Resolve a parameter FQN from root, unwrapping checkpoint wrappers."""
        parts = fqn.split(".")
        leaf_module = root_module
        for part in parts[:-1]:
            child = getattr(leaf_module, part, None)
            if child is None:
                wrapped = getattr(leaf_module, "_checkpoint_wrapped_module", None)
                if wrapped is not None:
                    leaf_module = getattr(wrapped, part)
                else:
                    leaf_module = getattr(leaf_module, part)
            else:
                leaf_module = child
        if hasattr(leaf_module, "_checkpoint_wrapped_module"):
            leaf_module = leaf_module._checkpoint_wrapped_module
        return cls(leaf_module, parts[-1])


@dataclass(frozen=True)
class BucketParam:
    """Per-parameter bucket runtime state.

    param_owner locates the module slot that holds the local shard, or the
    unsharded param while the bucket is unsharded. sharded_param is the local
    shard used for unshard input and grad writes, captured at runtime install
    (after any to_empty). param_info carries immutable bucket storage and
    placement metadata for collectives. Keeping them together preserves bucket
    order and avoids repeated FQN resolution in hooks.
    """

    param_owner: ParamOwnerRef
    param_info: ParamInfo
    sharded_param: nn.Parameter


def _in_backward() -> bool:
    """Return whether the autograd engine is executing a backward pass."""
    return torch._C._current_graph_task_id() != -1


def _alloc_storage(tensor: torch.Tensor, nbytes: int) -> None:
    storage = tensor.untyped_storage()
    if storage.size() != nbytes:
        storage.resize_(nbytes)


def _free_storage(tensor: torch.Tensor) -> None:
    storage = tensor.untyped_storage()
    if storage.size() != 0:
        storage.resize_(0)


@dataclass
class PendingUnshard:
    """One in-flight one-bucket-ahead unshard."""

    bucket: BucketRuntime
    result: UnshardHandle


@dataclass(frozen=True)
class PendingUnshardKey:
    """Key separating forward and backward prefetches for one bucket."""

    bucket_id: int
    backward: bool


@dataclass
class PendingReduceGrad:
    """One in-flight reduce-grad result."""

    result: ReduceGradHandle


@dataclass
class PendingReduceGradLaunch:
    """One packed reduce-grad request waiting for a backward unshard first."""

    bucket: BucketRuntime
    prepared: PreparedReduceGrad
    sharded_params: list[nn.Parameter]


@dataclass
class BucketCommContext:
    """Communication streams shared by buckets on one root module/device."""

    device_handle: ModuleType
    unshard_stream: torch.Stream
    reduce_grad_stream: torch.Stream
    reduce_grad_release_stream: torch.Stream
    max_pending_reduce_grads: int
    buckets: list[BucketRuntime] = field(default_factory=list)
    pending_unshards: dict[PendingUnshardKey, PendingUnshard] = field(
        default_factory=dict
    )
    pending_reduce_grad_launches: list[PendingReduceGradLaunch] = field(
        default_factory=list
    )
    reduce_grad_states: list[PendingReduceGrad] = field(default_factory=list)
    retired_reduce_grad_states: list[PendingReduceGrad] = field(default_factory=list)
    reduce_grad_callback_queued: bool = False
    _forward_bucket_indices: dict[int, int] | None = None

    def add_bucket(self, bucket: BucketRuntime) -> None:
        """Append a bucket and invalidate cached scheduling metadata."""
        self.buckets.append(bucket)
        self._forward_bucket_indices = None

    def forward_bucket_index(self, bucket: BucketRuntime) -> int | None:
        """Return bucket's index in forward execution order."""
        if self._forward_bucket_indices is None:
            self._forward_bucket_indices = {
                id(candidate): idx for idx, candidate in enumerate(self.buckets)
            }
        return self._forward_bucket_indices.get(id(bucket))

    def next_forward_unshard_bucket(
        self,
        bucket: BucketRuntime,
    ) -> BucketRuntime | None:
        """Return the bucket whose forward unshard follows ``bucket``'s."""
        idx = self.forward_bucket_index(bucket)
        if idx is None or idx + 1 >= len(self.buckets):
            return None
        return self.buckets[idx + 1]

    def next_backward_unshard_bucket(
        self,
        bucket: BucketRuntime,
    ) -> BucketRuntime | None:
        """Return the bucket whose backward re-gather follows ``bucket``'s.

        Backward visits buckets in reverse forward order. Buckets without
        reshard-after-forward stay unsharded through backward and are skipped.
        """
        idx = self.forward_bucket_index(bucket)
        if idx is None:
            return None
        for candidate in reversed(self.buckets[:idx]):
            if candidate.bucket_storage._reshard_after_forward:
                return candidate
        return None

    def should_defer_reduce_grad_for_backward_prefetch(
        self,
        bucket: BucketRuntime,
    ) -> bool:
        """Return whether reduce-grad should wait for backward prefetch."""
        next_bucket = self.next_backward_unshard_bucket(bucket)
        if (
            next_bucket is None
            or next_bucket.is_unsharded
            or not self.should_prefetch_bucket(next_bucket)
        ):
            return False
        backward_prefetch_key = next_bucket.pending_unshard_key(backward=True)
        return backward_prefetch_key not in self.pending_unshards

    def should_prefetch_bucket(self, bucket: BucketRuntime) -> bool:
        """Return whether this bucket can be unsharded from another module hook."""
        _ = bucket
        return True

    @classmethod
    def get(
        cls,
        root_module: nn.Module,
        device: torch.device,
    ) -> BucketCommContext | None:
        contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
        if contexts is None:
            return None
        return contexts.get(device)

    @classmethod
    def create(
        cls,
        root_module: nn.Module,
        device: torch.device,
    ) -> BucketCommContext:
        contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
        if contexts is None:
            contexts = {}
            setattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, contexts)
        if device in contexts:
            raise AssertionError(
                f"Communication context for device {device} already exists."
            )

        device_handle = _get_device_handle(device.type)
        context = cls(
            device_handle=device_handle,
            unshard_stream=device_handle.Stream(priority=-1),
            reduce_grad_stream=device_handle.Stream(priority=-1),
            reduce_grad_release_stream=device_handle.Stream(priority=-1),
            max_pending_reduce_grads=getattr(
                root_module, _MAX_PENDING_REDUCE_GRADS_ATTR
            ),
        )
        contexts[device] = context
        return context

    def queue_reduce_grad_wait(self) -> None:
        """Queue the end-of-backward callback (once per backward).

        It finishes buckets left unsharded or partially reduced, launches
        deferred reduce-grads, and waits on all reduce-grad work.
        """
        if self.reduce_grad_callback_queued:
            return
        self.reduce_grad_callback_queued = True

        def _wait_for_reduce_grad() -> None:
            try:
                # Buckets whose post-accumulate-grad trigger never completed
                # (params unused in forward) reduce and reshard here.
                for bucket in self.buckets:
                    if bucket.is_unsharded or bucket.grad_ready_indices:
                        bucket.post_backward()
                self.flush_pending_reduce_grad_launches(max_to_flush=None)
                self.wait_and_clear_reduce_grad_states(debug_fqn=None)
            finally:
                self.pending_reduce_grad_launches.clear()
                try:
                    self.wait_and_clear_pending_unshards(debug_fqn=None)
                finally:
                    self.reduce_grad_callback_queued = False

        torch.autograd.Variable._execution_engine.queue_callback(_wait_for_reduce_grad)

    def wait_and_clear_pending_unshards(
        self,
        debug_fqn: str | None,
    ) -> None:
        """Release unconsumed prefetches before the next training step."""
        if not self.pending_unshards:
            return
        with _record_function_if_eager(
            "FlexShard::post_backward_pending_unshard_clear",
            debug_fqn,
        ):
            pending_unshards = list(self.pending_unshards.values())
            self.pending_unshards.clear()
            for pending in pending_unshards:
                pending.result.wait()
                pending.result.release_buffers()

    def wait_and_clear_reduce_grad_states(
        self,
        debug_fqn: str | None,
    ) -> None:
        """Order reduced gradients before optimizer use and release buffers."""
        if not self.reduce_grad_states and not self.retired_reduce_grad_states:
            return
        with _record_function_if_eager(
            "FlexShard::post_backward_reduce_grad_wait",
            debug_fqn,
        ):
            for pending in self.retired_reduce_grad_states:
                pending.result.wait()
            self.retired_reduce_grad_states.clear()
            for pending in self.reduce_grad_states:
                self.wait_and_release_reduce_grad_state(
                    pending,
                    wait_current_stream=True,
                )
            self.reduce_grad_states.clear()

    def wait_and_release_reduce_grad_state(
        self,
        pending: PendingReduceGrad,
        *,
        wait_current_stream: bool,
    ) -> None:
        if wait_current_stream:
            pending.result.wait()
        with self.device_handle.stream(self.reduce_grad_release_stream):
            pending.result.wait()
            pending.result.release_buffers(
                release_sharded_grads=True,
            )

    def drain_reduce_grad_states_if_needed(
        self,
        debug_fqn: str | None,
    ) -> None:
        if self.max_pending_reduce_grads <= 0:
            return
        while len(self.reduce_grad_states) >= self.max_pending_reduce_grads:
            pending = self.reduce_grad_states.pop(0)
            with _record_function_if_eager(
                "FlexShard::post_backward_reduce_grad_retire",
                debug_fqn,
            ):
                self.wait_and_release_reduce_grad_state(
                    pending,
                    wait_current_stream=False,
                )
                self.retired_reduce_grad_states.append(pending)

    def _launch_pending_reduce_grad(
        self,
        pending: PendingReduceGradLaunch,
    ) -> None:
        bucket = pending.bucket
        self.drain_reduce_grad_states_if_needed(bucket.debug_fqn)
        result = launch_reduce_grad(
            pending.prepared,
            self.reduce_grad_stream,
        )
        with self.device_handle.stream(self.reduce_grad_stream):
            with _record_function_if_eager(
                "FlexShard::reduce_grad_accumulate",
                bucket.debug_fqn,
            ):
                sharded_grads = result.finish()
                result.record_sharded_grads(
                    _accumulate_sharded_grads(
                        pending.sharded_params,
                        sharded_grads,
                    ),
                    self.reduce_grad_stream,
                )
        self.reduce_grad_states.append(PendingReduceGrad(result))

    def queue_reduce_grad_launch(
        self,
        bucket: BucketRuntime,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        sharded_params: list[nn.Parameter],
    ) -> None:
        """Pack reduce-grad input and delay launch until after next unshard."""
        if not grads:
            return
        with torch.no_grad():
            # Keep placement-owned packed send scratch bounded. A deferred
            # reduce input can be very large, so do not prepare another one
            # before launching the previous deferred request.
            self.flush_pending_reduce_grad_launches(max_to_flush=None)
            prepared = prepare_reduce_grad(
                grads,
                infos,
                bucket.bucket_storage._mesh,
                debug_fqn=bucket.debug_fqn,
            )
        self.pending_reduce_grad_launches.append(
            PendingReduceGradLaunch(
                bucket=bucket,
                prepared=prepared,
                sharded_params=sharded_params,
            )
        )
        self.queue_reduce_grad_wait()

    def flush_pending_reduce_grad_launches(
        self,
        max_to_flush: int | None,
    ) -> None:
        """Launch deferred reduce-grads after unshard prefetch has priority."""
        num_flushed = 0
        while self.pending_reduce_grad_launches and (
            max_to_flush is None or num_flushed < max_to_flush
        ):
            pending = self.pending_reduce_grad_launches.pop(0)
            with torch.no_grad():
                self._launch_pending_reduce_grad(pending)
            num_flushed += 1
        if num_flushed:
            self.queue_reduce_grad_wait()


@dataclass
class BucketRuntime:
    """Runtime state and hooks for one FlexShard bucket.

    Eager (FSDP2-style): each managed parameter has a persistent unsharded
    ``nn.Parameter``, created from the first unshard. It is swapped into the
    owning module's ``_parameters`` while unsharded, so attribute reads and
    ``module.parameters()`` see it in forward and backward. Reshard swaps the
    local shard back and frees the placement's persistent buffers backing the
    unsharded params with ``untyped_storage().resize_(0)``; later unshards
    re-allocate and refill them in place. Grads accumulate into the unsharded
    params through autograd, and a post-accumulate-grad hook reduce-scatters
    them into the local shards.

    Compile: ``_BucketUnshard`` outputs are swapped into ``_parameters`` for the
    forward, so Dynamo traces one all-gather and one reduce-scatter per bucket.
    """

    bucket_storage: ShardedBucketStorage
    bucket_params: list[BucketParam]
    context: BucketCommContext
    debug_fqn: str | None
    # Whether the unsharded params hold data and are swapped into their
    # modules, and which params' grads have accumulated in this backward.
    is_unsharded: bool = False
    grad_ready_indices: set[int] = field(default_factory=set)
    # Persistent unsharded params, created from the first unshard, and the
    # placement's persistent buffers backing them with their allocated storage
    # size in bytes.
    unsharded_params: list[nn.Parameter] | None = None
    persistent_buffers: list[torch.Tensor] = field(default_factory=list)
    persistent_buffer_nbytes: list[int] = field(default_factory=list)

    @classmethod
    def from_bucket_storage(
        cls,
        bucket_storage: ShardedBucketStorage,
        context: BucketCommContext | None = None,
    ) -> BucketRuntime:
        """Create runtime state for one bucket storage."""
        bucket_params: list[BucketParam] = []
        for info in bucket_storage._param_infos.values():
            param_owner = ParamOwnerRef.resolve(bucket_storage._module, info.fqn)
            bucket_params.append(
                BucketParam(
                    param_owner=param_owner,
                    param_info=info,
                    sharded_param=param_owner.module._parameters[
                        param_owner.param_name
                    ],
                )
            )
        comm_device = bucket_storage.byte_storage.device
        if context is None:
            context = BucketCommContext.get(bucket_storage._module, comm_device)
            if context is None:
                context = BucketCommContext.create(bucket_storage._module, comm_device)
        return cls(
            bucket_storage=bucket_storage,
            bucket_params=bucket_params,
            context=context,
            debug_fqn=_get_bucket_storage_debug_fqn(bucket_storage),
        )

    @property
    def infos(self) -> list[ParamInfo]:
        return [bucket_param.param_info for bucket_param in self.bucket_params]

    @property
    def sharded_params(self) -> list[nn.Parameter]:
        return [bucket_param.sharded_param for bucket_param in self.bucket_params]

    @property
    def num_grad_params(self) -> int:
        if self.unsharded_params is None:
            return 0
        return sum(1 for param in self.unsharded_params if param.requires_grad)

    def _local_shards(self, *, use_autograd: bool) -> list[torch.Tensor]:
        # detach() shares the parameter version counter, which versioned
        # placement caches need for optimizer-update invalidation.
        return [
            param if use_autograd else param.detach() for param in self.sharded_params
        ]

    def bucket_module(self) -> nn.Module:
        """Find the deepest common ancestor module for this bucket's params."""
        fqns = list(self.bucket_storage._param_infos.keys())
        prefixes = [".".join(fqn.split(".")[:-1]) for fqn in fqns]
        if not prefixes:
            return self.bucket_storage._module
        common = prefixes[0]
        for prefix in prefixes[1:]:
            i = 0
            while i < len(common) and i < len(prefix) and common[i] == prefix[i]:
                i += 1
            common = common[:i]
        # Trim to the last complete component so a partial name match is ignored.
        if "." in common:
            common = common[: common.rfind(".") + 1].rstrip(".")
        elif common and common not in prefixes:
            common = ""
        if not common:
            return self.bucket_storage._module
        mod = self.bucket_storage._module
        for part in common.split("."):
            mod = getattr(mod, part)
        return mod

    def forward_hook_module(self) -> nn.Module:
        """Return the module whose forward triggers this bucket."""
        # Register hooks on the deepest common ancestor module for the bucket's
        # params so one pre-forward unshard covers their parameter accesses.
        # For example, a bucket with "layers.0.attn.weight" and
        # "layers.0.mlp.weight" hooks "layers.0".
        # TODO: Avoid registering bucket hooks on passive containers such as
        # ModuleList or ModuleDict. Catch-all buckets can resolve to those
        # containers, whose hooks may never run when forward iterates children.
        target = self.bucket_module()
        return getattr(target, "_checkpoint_wrapped_module", target)

    def begin_unshard(
        self,
        local_shards: list[torch.Tensor] | None = None,
    ) -> UnshardHandle:
        """Begin this bucket's unshard on the shared stream."""
        if local_shards is None:
            local_shards = self._local_shards(use_autograd=False)
        return begin_bucket_unshard(
            local_shards,
            self.infos,
            self.bucket_storage._mesh,
            self.context.unshard_stream,
            debug_fqn=self.debug_fqn,
        )

    def pending_unshard_key(self, *, backward: bool) -> PendingUnshardKey:
        """Return the pending-unshard key for this bucket and execution phase."""
        return PendingUnshardKey(bucket_id=id(self.bucket_storage), backward=backward)

    def take_pending(self, *, backward: bool) -> UnshardHandle | None:
        """Return this bucket's prefetched unshard, releasing stale prefetches."""
        key = self.pending_unshard_key(backward=backward)
        pending = self.context.pending_unshards.pop(key, None)
        if pending is None:
            self.context.wait_and_clear_pending_unshards(self.debug_fqn)
            return None
        return pending.result

    def prefetch(self, next_bucket: BucketRuntime | None, *, backward: bool) -> None:
        """Start ``next_bucket``'s unshard if it needs one and none is in flight."""
        if (
            next_bucket is None
            or next_bucket.is_unsharded
            or self.context.pending_unshards
            or not self.context.should_prefetch_bucket(next_bucket)
        ):
            return
        key = next_bucket.pending_unshard_key(backward=backward)
        self.context.pending_unshards[key] = PendingUnshard(
            bucket=next_bucket,
            result=next_bucket.begin_unshard(),
        )

    def reduce_grads(
        self,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        sharded_params: list[nn.Parameter],
    ) -> None:
        """Reduce full-parameter grads and accumulate local sharded grads."""
        if not grads:
            return
        with torch.no_grad():
            self.context.drain_reduce_grad_states_if_needed(self.debug_fqn)
            result = begin_reduce_grad(
                grads,
                infos,
                self.bucket_storage._mesh,
                self.context.reduce_grad_stream,
                debug_fqn=self.debug_fqn,
            )
            with self.context.device_handle.stream(self.context.reduce_grad_stream):
                with _record_function_if_eager(
                    "FlexShard::reduce_grad_accumulate",
                    self.debug_fqn,
                ):
                    sharded_grads = result.finish()
                    result.record_sharded_grads(
                        _accumulate_sharded_grads(
                            sharded_params,
                            sharded_grads,
                        ),
                        self.context.reduce_grad_stream,
                    )
            self.context.reduce_grad_states.append(PendingReduceGrad(result))
            self.context.queue_reduce_grad_wait()

    def schedule_reduce_grad_tensors(
        self,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        sharded_params: list[nn.Parameter],
    ) -> None:
        """Launch or defer a pre-packed bucket reduce-grad request."""
        if self.context.should_defer_reduce_grad_for_backward_prefetch(self):
            self.context.queue_reduce_grad_launch(
                self,
                grads,
                infos,
                sharded_params,
            )
        else:
            self.reduce_grads(grads, infos, sharded_params)

    # ------------------------------------------------------------------
    # Eager: persistent unsharded parameters
    # ------------------------------------------------------------------

    def unshard(self, *, backward: bool) -> None:
        """Make the unsharded params hold data and swap them into the modules."""
        if not self.is_unsharded:
            result = self.take_pending(backward=backward)
            if result is None:
                result = self.begin_unshard()
            self.finish_unshard(result)
        self._swap_in_params(self.unsharded_params)

    def finish_unshard(self, result: UnshardHandle) -> None:
        """Finish ``result`` into the persistent buffers.

        The first unshard creates the unsharded params from the placement's
        full params and keeps the persistent buffers backing them. Later ones
        re-allocate those buffers and the placement refills them in place,
        without bumping the params' version counters (autograd may have saved
        them in forward before a reshard).
        """
        first = self.unsharded_params is None
        if first:
            preserve_versions = contextlib.nullcontext()
        else:
            for persistent_buffer, nbytes in zip(
                self.persistent_buffers, self.persistent_buffer_nbytes, strict=True
            ):
                _alloc_storage(persistent_buffer, nbytes)
            preserve_versions = torch.autograd._unsafe_preserve_version_counter(
                tuple(self.unsharded_params)
            )
        with torch.no_grad(), preserve_versions:
            unshard_lease = result.finish(
                persistent_buffers=None if first else self.persistent_buffers
            )
        full_params = unshard_lease.take_full_params()
        persistent_buffers = unshard_lease.persistent_buffers
        unshard_lease.release()
        if first:
            self._create_unsharded_params(full_params, persistent_buffers)
        self.is_unsharded = True

    def _create_unsharded_params(
        self,
        full_params: list[torch.Tensor],
        persistent_buffers: list[torch.Tensor],
    ) -> None:
        if not persistent_buffers:
            raise AssertionError(
                f"Placement {self.infos[0].placement!r} returned no persistent "
                "buffers for a persistent unshard."
            )
        unsharded_params: list[nn.Parameter] = []
        for bucket_param, full_param in zip(
            self.bucket_params, full_params, strict=True
        ):
            info = bucket_param.param_info
            unsharded_param = nn.Parameter(full_param, requires_grad=info.requires_grad)
            for name, value in info.param_attrs.items():
                setattr(unsharded_param, name, value)
            unsharded_params.append(unsharded_param)
        self.unsharded_params = unsharded_params
        self.persistent_buffers = list(persistent_buffers)
        self.persistent_buffer_nbytes = [
            persistent_buffer.untyped_storage().size()
            for persistent_buffer in persistent_buffers
        ]
        for idx, unsharded_param in enumerate(unsharded_params):
            if not unsharded_param.requires_grad:
                continue

            def hook(_param: torch.Tensor, idx: int = idx) -> None:
                self.on_grad_accumulated(idx)

            unsharded_param.register_post_accumulate_grad_hook(hook)

    def _swap_in_params(self, params: list[torch.Tensor]) -> None:
        """Expose ``params`` through their modules' ``_parameters``."""
        for bucket_param, param in zip(self.bucket_params, params, strict=True):
            param_owner = bucket_param.param_owner
            param_owner.module._parameters[param_owner.param_name] = param

    def reshard(self) -> None:
        """Swap the local shards back in and free the persistent storage."""
        self._swap_in_params(self.sharded_params)
        # Consumers of the storage were queued on this stream, which also
        # allocated it, so the caching allocator orders the reuse.
        for persistent_buffer in self.persistent_buffers:
            _free_storage(persistent_buffer)
        self.is_unsharded = False

    def pre_backward_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Re-unshard before this bucket's module runs backward (output grad hook)."""
        self.context.queue_reduce_grad_wait()
        self.unshard(backward=True)
        self.prefetch(self.context.next_backward_unshard_bucket(self), backward=True)
        self.context.flush_pending_reduce_grad_launches(max_to_flush=1)
        return grad

    def on_grad_accumulated(self, idx: int) -> None:
        """Post-accumulate-grad hook; reduce once every grad has accumulated.

        Within one backward, autograd sums all uses of a leaf before running its
        AccumulateGrad once, so this fires once per param even if the module
        ran forward several times.
        """
        self.grad_ready_indices.add(idx)
        if len(self.grad_ready_indices) == self.num_grad_params:
            self.post_backward()

    def post_backward(self) -> None:
        """Reduce-scatter accumulated full grads into the shards and reshard."""
        grads: list[torch.Tensor] = []
        infos: list[ParamInfo] = []
        sharded_params: list[nn.Parameter] = []
        for bucket_param, unsharded_param in zip(
            self.bucket_params, self.unsharded_params or [], strict=False
        ):
            if unsharded_param.grad is None:
                continue
            grads.append(unsharded_param.grad)
            infos.append(bucket_param.param_info)
            sharded_params.append(bucket_param.sharded_param)
            unsharded_param.grad = None
        self.grad_ready_indices.clear()
        if self.is_unsharded:
            self.reshard()
        if grads:
            self.schedule_reduce_grad_tensors(grads, infos, sharded_params)

    # ------------------------------------------------------------------
    # Forward hooks
    # ------------------------------------------------------------------

    def pre_forward_hook(self, mod, args) -> None:
        if torch.compiler.is_compiling():
            self._pre_forward_compile()
            return
        self.unshard(backward=False)
        if not _in_backward():
            self.prefetch(
                self.context.next_forward_unshard_bucket(self), backward=False
            )
            self.context.flush_pending_reduce_grad_launches(max_to_flush=1)

    def post_forward_hook(self, mod, args, output) -> None:
        if torch.compiler.is_compiling():
            self._swap_in_params(self.sharded_params)
            return
        if _in_backward():
            # Activation-checkpoint recompute inside backward: this bucket's
            # backward is already running and needs the params, so keep them;
            # its post-accumulate-grad hook (or the end-of-backward callback)
            # reshards.
            return
        grad_outputs = (
            [
                value
                for value in tree_leaves(output)
                if isinstance(value, torch.Tensor) and value.requires_grad
            ]
            if output is not None and torch.is_grad_enabled()
            else []
        )
        if not grad_outputs:
            # No backward will reach these params through this forward.
            self.reshard()
            return
        for value in grad_outputs:
            value.register_hook(self.pre_backward_hook)
        if self.bucket_storage._reshard_after_forward:
            self.reshard()

    def _pre_forward_compile(self) -> None:
        """Trace the bucket unshard and expose its outputs via ``_parameters``.

        Collectives run synchronously inside ``_BucketUnshard`` so Dynamo traces
        one all-gather (forward) and one reduce-scatter (backward) per bucket;
        graph passes may reorder them. The traced graph owns buffer lifetimes,
        so the persistent buffers are not used here.
        """
        full_params = _BucketUnshard.apply(self, *self._local_shards(use_autograd=True))
        params: list[torch.Tensor] = []
        for bucket_param, full_param in zip(
            self.bucket_params, full_params, strict=True
        ):
            param_dtype = bucket_param.param_info.param_dtype
            if param_dtype is not None and full_param.dtype != param_dtype:
                full_param = _UnshardedParamCast.apply(full_param, param_dtype)
            params.append(full_param)
        self._swap_in_params(params)


class _BucketUnshard(torch.autograd.Function):
    """Traced bucket unshard for the compile branch.

    Forward all-gathers the bucket and backward reduce-scatters its gradients
    with synchronous functional collectives, returning the local-shard grads
    through autograd. Graph passes own scheduling and buffer lifetimes.
    """

    @staticmethod
    def forward(
        ctx: Any,
        bucket: BucketRuntime,
        *local_shards: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.bucket = bucket
        ctx.local_shard_dtypes = tuple(shard.dtype for shard in local_shards)
        unshard_lease = bucket.begin_unshard(
            [shard.detach() for shard in local_shards]
        ).finish()
        full_params = unshard_lease.take_full_params()
        unshard_lease.release()
        frozen_params = [
            full_param
            for full_param, info in zip(full_params, bucket.infos, strict=True)
            if not info.requires_grad
        ]
        if frozen_params:
            ctx.mark_non_differentiable(*frozen_params)
        return tuple(full_params)

    @staticmethod
    def backward(
        ctx: Any,
        *full_param_grads: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        bucket: BucketRuntime = ctx.bucket
        input_grads: list[torch.Tensor | None] = [None] * len(full_param_grads)
        grads: list[torch.Tensor] = []
        infos: list[ParamInfo] = []
        indices: list[int] = []
        for idx, (grad, info) in enumerate(
            zip(full_param_grads, bucket.infos, strict=True)
        ):
            if grad is None:
                continue
            grads.append(grad)
            infos.append(info)
            indices.append(idx)
        if grads:
            sharded_grads = begin_reduce_grad(
                grads,
                infos,
                bucket.bucket_storage._mesh,
                bucket.context.reduce_grad_stream,
                debug_fqn=bucket.debug_fqn,
            ).finish()
            for idx, sharded_grad in zip(indices, sharded_grads, strict=True):
                input_dtype = ctx.local_shard_dtypes[idx]
                if sharded_grad.dtype != input_dtype:
                    sharded_grad = sharded_grad.to(input_dtype)
                input_grads[idx] = sharded_grad
        return (None, *input_grads)


class _UnshardedParamCast(torch.autograd.Function):
    """Forward param cast with identity gradient dtype semantics."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        param_dtype: torch.dtype,
    ) -> torch.Tensor:
        return x.to(param_dtype)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None


def _match_param_grad_layout(
    grad: torch.Tensor,
    param: nn.Parameter,
) -> torch.Tensor:
    """Return a grad tensor suitable for assignment to ``param.grad``."""
    if grad.dtype != param.dtype or grad.device != param.device:
        grad = grad.to(device=param.device, dtype=param.dtype)
    if grad.layout != param.layout:
        raise RuntimeError(
            "FlexShard reduced gradient layout does not match the local "
            f"parameter layout: grad={grad.layout}, param={param.layout}"
        )
    if grad.layout == torch.strided and grad.stride() != param.stride():
        aligned = torch.empty_strided(
            tuple(param.shape),
            tuple(param.stride()),
            dtype=param.dtype,
            device=param.device,
        )
        aligned.copy_(grad)
        grad = aligned
    return grad


def _accumulate_sharded_grads(
    sharded_params: list[nn.Parameter],
    sharded_grads: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Cast sharded grads to local param dtype/layout and accumulate into .grad."""
    stored_grads: list[torch.Tensor] = []
    for param, grad in zip(sharded_params, sharded_grads, strict=True):
        grad = _match_param_grad_layout(grad, param)
        stored_grads.append(grad)
        if param.grad is None:
            param.grad = grad
        else:
            param.grad += grad
    return stored_grads


def _install_bucket_unshard_hooks(
    bucket_storages: list[ShardedBucketStorage],
) -> None:
    """Install each bucket's pre/post forward hooks (one collective per bucket)."""
    for bucket_storage in bucket_storages:
        if not bucket_storage._param_infos:
            raise AssertionError("Expected FlexShard bucket storage to own parameters.")
        if bucket_storage.byte_storage.device.type != "cuda":
            raise AssertionError("Expected FlexShard bucket storage to be on CUDA.")

        bucket_runtime = BucketRuntime.from_bucket_storage(bucket_storage)
        target = bucket_runtime.forward_hook_module()
        target.register_forward_pre_hook(bucket_runtime.pre_forward_hook)
        target.register_forward_hook(
            bucket_runtime.post_forward_hook,
            always_call=True,
        )
        bucket_runtime.context.add_bucket(bucket_runtime)
