# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from dataclasses import dataclass, field, replace
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle
from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

from .bucket_comm import (
    begin_bucket_unshard,
    begin_reduce_grad,
    ReduceGradHandle,
    UnshardHandle,
)
from .bucket_storage import ParamInfo, ShardedBucketStorage
from .utils import (
    _get_bucket_storage_debug_fqn,
    _module_path_common_prefix,
    _record_function_if_eager,
)


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
    and re-read after to_empty(). param_info carries immutable bucket storage and
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
    """The one in-flight prefetched unshard."""

    bucket: BucketRuntime
    result: UnshardHandle


@dataclass
class PendingReduceGrad:
    """One in-flight reduce-grad result."""

    result: ReduceGradHandle


@dataclass
class BucketCommContext:
    """Streams and scheduling state shared by buckets on one root module/device."""

    device_handle: ModuleType
    unshard_stream: torch.Stream
    reduce_grad_stream: torch.Stream
    reduce_grad_release_stream: torch.Stream
    max_pending_reduce_grads: int
    # Buckets in registration (BucketSpec) order.
    buckets: list[BucketRuntime] = field(default_factory=list)
    # Buckets in the order their pre- and post-forward hooks first ran, learned
    # from the first forward (FSDP2 records a post-forward order). Forward
    # prefetch follows the first; backward prefetch walks the second in reverse.
    forward_order: list[BucketRuntime] = field(default_factory=list)
    post_forward_order: list[BucketRuntime] = field(default_factory=list)
    pending_unshard: PendingUnshard | None = None
    reduce_grad_states: list[PendingReduceGrad] = field(default_factory=list)
    retired_reduce_grad_states: list[PendingReduceGrad] = field(default_factory=list)
    post_backward_callback_queued: bool = False

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

    def next_forward_bucket(self, bucket: BucketRuntime) -> BucketRuntime | None:
        """Return the bucket whose forward followed ``bucket``'s."""
        idx = bucket.forward_index
        if idx is None or idx + 1 >= len(self.forward_order):
            return None
        return self.forward_order[idx + 1]

    def next_backward_bucket(self, bucket: BucketRuntime) -> BucketRuntime | None:
        """Return the bucket whose backward re-gather follows ``bucket``'s.

        Backward visits buckets in reverse post-forward order. Buckets without
        reshard-after-forward, or already unsharded, need no re-gather.
        """
        idx = bucket.post_forward_index
        if idx is None:
            return None
        for candidate in reversed(self.post_forward_order[:idx]):
            if (
                candidate.bucket_storage._reshard_after_forward
                and not candidate.is_unsharded
            ):
                return candidate
        return None

    def prefetch(self, bucket: BucketRuntime | None) -> None:
        """Start ``bucket``'s unshard ahead of its hook, one prefetch at a time."""
        if bucket is None or bucket.is_unsharded or self.pending_unshard is not None:
            return
        self.pending_unshard = PendingUnshard(bucket, bucket.begin_unshard())

    def take_pending_unshard(
        self,
        bucket: BucketRuntime | None,
    ) -> UnshardHandle | None:
        """Return ``bucket``'s prefetched unshard and release any other one.

        A prefetch for another bucket means execution diverged from the learned
        order; releasing it bounds memory and frees the prefetch slot.
        """
        pending, self.pending_unshard = self.pending_unshard, None
        if pending is None:
            return None
        if pending.bucket is bucket:
            return pending.result
        with _record_function_if_eager(
            "FlexShard::release_unused_prefetch",
            pending.bucket.debug_fqn,
        ):
            pending.result.wait()
            pending.result.release_buffers()
        return None

    def queue_post_backward_callback(self) -> None:
        """Queue the end-of-backward callback (once per backward).

        It finishes buckets still unsharded or partially reduced (params unused
        in backward, or no post-backward trigger), waits on all reduce-grad
        work, and releases an unused prefetch.
        """
        if self.post_backward_callback_queued:
            return
        self.post_backward_callback_queued = True

        def _post_backward_callback() -> None:
            try:
                for bucket in self.buckets:
                    if bucket.is_unsharded or bucket.grad_ready_indices:
                        bucket.post_backward()
                    bucket.reset_backward_state()
                self.wait_and_clear_reduce_grad_states(debug_fqn=None)
            finally:
                try:
                    self.take_pending_unshard(None)
                finally:
                    self.post_backward_callback_queued = False

        torch.autograd.Variable._execution_engine.queue_callback(
            _post_backward_callback
        )

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
    params through autograd and are reduce-scattered into the local shards once
    the bucket's backward is done: when the hooked module's input grads are
    ready (as in FSDP2) or every grad has accumulated, whichever comes first.

    Compile: ``_BucketUnshard`` outputs are swapped into ``_parameters`` for the
    forward, so Dynamo traces one all-gather and one reduce-scatter per bucket.
    """

    bucket_storage: ShardedBucketStorage
    bucket_params: list[BucketParam]
    context: BucketCommContext
    debug_fqn: str | None
    # Position in the context's learned forward and post-forward orders.
    forward_index: int | None = None
    post_forward_index: int | None = None
    # Whether the unsharded params hold data and are swapped into their
    # modules, and which params' grads have accumulated in this backward.
    is_unsharded: bool = False
    grad_ready_indices: set[int] = field(default_factory=set)
    # Forward calls since the last backward whose outputs need backward, how
    # many carried a post-backward trigger on their inputs, and how many of
    # those triggers ran; and whether each in-progress call carries one.
    backward_calls: int = 0
    input_triggers: int = 0
    input_triggers_run: int = 0
    call_has_trigger: list[bool] = field(default_factory=list)
    # Persistent unsharded params, created from the first unshard, and the
    # placement's persistent buffers backing them with their allocated storage
    # size in bytes.
    unsharded_params: list[nn.Parameter] | None = None
    persistent_buffers: list[torch.Tensor] = field(default_factory=list)
    persistent_buffer_nbytes: list[int] = field(default_factory=list)
    # Indices of the unsharded params with a post-accumulate-grad hook.
    grad_hooked: set[int] = field(default_factory=set)

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

    def reset_sharded_params(self) -> None:
        """Re-read the local-shard params after their module slots changed.

        ``to_empty()`` re-installs them; FSDP2 re-reads them the same way in
        ``reset_sharded_param()``.
        """
        self.bucket_params = [
            replace(
                bucket_param,
                sharded_param=bucket_param.param_owner.module._parameters[
                    bucket_param.param_owner.param_name
                ],
            )
            for bucket_param in self.bucket_params
        ]

    def forward_hook_module(self) -> nn.Module:
        """Return the module whose forward triggers this bucket.

        It is the deepest common ancestor of the bucket's params, so one
        pre-forward unshard covers their accesses: a bucket with
        "layers.0.attn.wq.weight" and "layers.0.mlp.w1.weight" hooks "layers.0".
        Containers without a forward, such as ``ModuleList``, give way to their
        nearest ancestor that runs one.
        """
        path = _module_path_common_prefix(
            [".".join(fqn.split(".")[:-1]) for fqn in self.bucket_storage._param_infos]
        )
        modules = [self.bucket_storage._module]
        for part in path.split(".") if path else []:
            modules.append(getattr(modules[-1], part))
        target = next(
            (
                module
                for module in reversed(modules)
                if type(module).forward is not nn.Module.forward
            ),
            modules[0],
        )
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
            self.context.queue_post_backward_callback()

    # ------------------------------------------------------------------
    # Eager: persistent unsharded parameters
    # ------------------------------------------------------------------

    def unshard(self) -> None:
        """Make the unsharded params hold data and swap them into the modules."""
        if not self.is_unsharded:
            result = self.context.take_pending_unshard(self)
            if result is None:
                result = self.begin_unshard()
            self.finish_unshard(result)
        self._swap_in_params(self.unsharded_params)

    def finish_unshard(self, result: UnshardHandle) -> None:
        """Finish ``result`` into the persistent buffers.

        The first unshard creates the unsharded params from the placement's
        full params and keeps the persistent buffers backing them. Later ones
        re-allocate those buffers and the placement refills them in place
        without bumping their version counters, which the unsharded params
        share (autograd may have saved them in forward before a reshard), as
        FSDP2 does for its all-gather outputs. The storage is created outside
        inference mode, so a model first run under ``torch.inference_mode()``
        can still train.
        """
        with torch.inference_mode(False), torch.no_grad():
            if self.unsharded_params is None:
                unshard = result.finish()
                self._create_unsharded_params(
                    unshard.full_params, unshard.persistent_buffers
                )
            else:
                for persistent_buffer, nbytes in zip(
                    self.persistent_buffers, self.persistent_buffer_nbytes, strict=True
                ):
                    _alloc_storage(persistent_buffer, nbytes)
                with torch.autograd._unsafe_preserve_version_counter(
                    tuple(self.persistent_buffers)
                ):
                    result.finish(persistent_buffers=self.persistent_buffers)
            self._sync_requires_grad()
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
            unsharded_param = nn.Parameter(
                full_param, requires_grad=bucket_param.sharded_param.requires_grad
            )
            for name, value in bucket_param.param_info.param_attrs.items():
                setattr(unsharded_param, name, value)
            unsharded_params.append(unsharded_param)
        self.unsharded_params = unsharded_params
        self.persistent_buffers = list(persistent_buffers)
        self.persistent_buffer_nbytes = [
            persistent_buffer.untyped_storage().size()
            for persistent_buffer in persistent_buffers
        ]

    def _sync_requires_grad(self) -> None:
        """Follow the local shards' ``requires_grad``, as FSDP2 does per unshard."""
        for idx, (bucket_param, unsharded_param) in enumerate(
            zip(self.bucket_params, self.unsharded_params, strict=True)
        ):
            requires_grad = bucket_param.sharded_param.requires_grad
            if unsharded_param.requires_grad != requires_grad:
                unsharded_param.requires_grad_(requires_grad)
            if requires_grad and idx not in self.grad_hooked:

                def hook(_param: torch.Tensor, idx: int = idx) -> None:
                    self.on_grad_accumulated(idx)

                unsharded_param.register_post_accumulate_grad_hook(hook)
                self.grad_hooked.add(idx)

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

    def reset_backward_state(self) -> None:
        """Clear the post-backward trigger counts at the end of a backward."""
        self.backward_calls = self.input_triggers = self.input_triggers_run = 0

    def pre_backward_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Re-unshard before this bucket's module runs backward (output grad hook)."""
        self.context.queue_post_backward_callback()
        self.unshard()
        self.context.prefetch(self.context.next_backward_bucket(self))
        return grad

    def on_input_grads(self) -> None:
        """Post-backward trigger from the hooked module's input grads.

        Like FSDP2's ``RegisterPostBackwardFunction``, it reduces and reshards
        once the module's backward is done, even if some params got no grad
        (e.g. an unrouted expert's). It fires only when every forward call since
        the last backward carried a trigger and all of them ran, so no other
        call's backward still needs the params.
        """
        self.input_triggers_run += 1
        if self.backward_calls == self.input_triggers == self.input_triggers_run:
            self.post_backward()

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
            self.reduce_grads(grads, infos, sharded_params)

    # ------------------------------------------------------------------
    # Forward hooks
    # ------------------------------------------------------------------

    def pre_forward_hook(
        self,
        mod: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if torch.compiler.is_compiling():
            self._pre_forward_compile()
            return None
        if _in_backward():
            # Activation-checkpoint recompute: the pre-backward hook usually
            # re-gathered already; otherwise this consumes its prefetch.
            self.call_has_trigger.append(False)
            self.unshard()
            return None
        if self.forward_index is None:
            self.forward_index = len(self.context.forward_order)
            self.context.forward_order.append(self)
        self.unshard()
        self.context.prefetch(self.context.next_forward_bucket(self))
        inputs = self._register_post_backward_trigger(args, kwargs)
        self.call_has_trigger.append(inputs is not None)
        return inputs

    def _register_post_backward_trigger(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        """Route the forward's grad-requiring inputs through ``_PostBackwardTrigger``."""
        if not torch.is_grad_enabled():
            return None
        flat_inputs, spec = tree_flatten((args, kwargs))
        indices = [
            idx
            for idx, value in enumerate(flat_inputs)
            if isinstance(value, torch.Tensor) and value.requires_grad
        ]
        if not indices:
            return None
        outputs = _PostBackwardTrigger.apply(
            self, *(flat_inputs[idx] for idx in indices)
        )
        for idx, output in zip(indices, outputs, strict=True):
            flat_inputs[idx] = output
        return tree_unflatten(flat_inputs, spec)

    def post_forward_hook(self, mod: nn.Module, args: Any, output: Any) -> None:
        if torch.compiler.is_compiling():
            self._swap_in_params(self.sharded_params)
            return
        has_trigger = self.call_has_trigger.pop() if self.call_has_trigger else False
        if _in_backward():
            # Activation-checkpoint recompute inside backward: this bucket's
            # backward still needs the params; post-backward reshards.
            return
        if self.post_forward_index is None:
            self.post_forward_index = len(self.context.post_forward_order)
            self.context.post_forward_order.append(self)
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
        self.backward_calls += 1
        self.input_triggers += has_trigger
        for value in grad_outputs:
            # Hook a view output's base: an in-place op on the view replaces the
            # view's autograd node, which would drop a hook registered on it.
            base = value._base
            hooked = base if base is not None and base.requires_grad else value
            hooked.register_hook(self.pre_backward_hook)
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
        self._swap_in_params(list(full_params))


class _PostBackwardTrigger(torch.autograd.Function):
    """Identity on a bucket module's grad-requiring inputs whose backward runs
    once the module's backward produced their grads (FSDP2's
    ``RegisterPostBackwardFunction``)."""

    @staticmethod
    def forward(
        ctx: Any,
        bucket: BucketRuntime,
        *inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.bucket = bucket
        return inputs

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> tuple[Any, ...]:
        ctx.bucket.on_input_grads()
        return (None, *grads)


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
        full_params = (
            bucket.begin_unshard([shard.detach() for shard in local_shards])
            .finish()
            .full_params
        )
        frozen_params = [
            full_param
            for full_param, shard in zip(full_params, local_shards, strict=True)
            if not shard.requires_grad
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
    """Install each bucket's forward hooks (one collective per bucket), and
    state-dict hooks that reshard first, as FSDP2 does."""
    owner_buckets: dict[nn.Module, dict[int, BucketRuntime]] = {}
    for bucket_storage in bucket_storages:
        if not bucket_storage._param_infos:
            raise AssertionError("Expected FlexShard bucket storage to own parameters.")
        if bucket_storage.byte_storage.device.type != "cuda":
            raise AssertionError("Expected FlexShard bucket storage to be on CUDA.")

        bucket_runtime = BucketRuntime.from_bucket_storage(bucket_storage)
        target = bucket_runtime.forward_hook_module()
        # Prepended so earlier user pre-forward hooks see the unsharded params.
        target.register_forward_pre_hook(
            bucket_runtime.pre_forward_hook,
            prepend=True,
            with_kwargs=True,
        )
        target.register_forward_hook(
            bucket_runtime.post_forward_hook,
            always_call=True,
        )
        bucket_runtime.context.buckets.append(bucket_runtime)
        for bucket_param in bucket_runtime.bucket_params:
            owner_buckets.setdefault(bucket_param.param_owner.module, {})[
                id(bucket_runtime)
            ] = bucket_runtime

    for module, buckets in owner_buckets.items():
        reshard_hook = functools.partial(_reshard_buckets, tuple(buckets.values()))
        module.register_state_dict_pre_hook(reshard_hook)
        module._register_load_state_dict_pre_hook(reshard_hook)


def _reshard_buckets(
    buckets: tuple[BucketRuntime, ...], *args: Any, **kwargs: Any
) -> None:
    """State-dict pre-hook: swap the local shards back in first."""
    for bucket in buckets:
        if bucket.is_unsharded:
            bucket.reshard()
