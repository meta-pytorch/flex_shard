# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

import torch
import torch.nn as nn

from .bucket_runtime import (
    _EAGER_COMM_CONTEXTS_ATTR,
    _install_bucket_unshard_hooks,
    _MAX_PENDING_REDUCE_GRADS_ATTR,
)
from .bucket_storage import (
    _assign_params_to_buckets,
    BucketParamFQNsByIndex,
    BucketSpec,
    GradientReduceOp,
    ShardedBucketStorage,
)
from .sharded_param import is_flex_shard_param
from .utils import (
    _get_device_from_mesh,
    _get_managed_named_params,
    _validate_bucket_uniform_dtype_and_placement,
    _validate_eager_params,
    _validate_flex_shard_mesh,
    _validate_placements,
)

if TYPE_CHECKING:
    from .placement_contract import Placement


__all__ = [
    "flex_shard",
]


_SHARDED_BUCKET_STORAGES_ATTR = "_sharded_bucket_storages"
_EAGER_HOOKS_INSTALLED_ATTR = "_flex_shard_eager_hooks_installed"


class FlexShardModule:
    """Mixin added to modules after flex_shard()."""

    @property
    def sharded_bucket_storages(self) -> list[ShardedBucketStorage]:
        """All bucket storage objects, one per bucket."""
        return getattr(self, _SHARDED_BUCKET_STORAGES_ATTR)

    def to_empty(self, *, device, recurse: bool = True):
        """Materialize FlexShard bucket storage when meta modules are initialized."""
        result = super().to_empty(device=device, recurse=recurse)
        self._materialize_after_to_empty()
        return result

    def _materialize_after_to_empty(self) -> None:
        """Rebuild sharded param views after nn.Module.to_empty()."""
        bucket_storages = getattr(self, _SHARDED_BUCKET_STORAGES_ATTR, None)
        if bucket_storages is None:
            return

        materialized_device: torch.device | None = None
        for _, param in self.named_parameters(recurse=True):
            if param.device.type != "meta":
                materialized_device = param.device
                break
        if materialized_device is None:
            return

        for bucket_storage in bucket_storages:
            if bucket_storage.byte_storage.device.type == "meta":
                bucket_storage._byte_storage = torch.empty(
                    bucket_storage.total_bytes,
                    dtype=torch.uint8,
                    device=materialized_device,
                )
            elif bucket_storage.byte_storage.device != materialized_device:
                bucket_storage._byte_storage = torch.empty(
                    bucket_storage.total_bytes,
                    dtype=torch.uint8,
                    device=materialized_device,
                )
            bucket_storage.install_sharded_params(bucket_storage.byte_storage.device)

        # Installed runtime hooks still hold the replaced local-shard params.
        for context in getattr(self, _EAGER_COMM_CONTEXTS_ATTR, {}).values():
            for bucket in context.buckets:
                bucket.reset_sharded_params()
        self._install_runtime_if_materialized()

    def _install_runtime_if_materialized(self) -> None:
        """Install eager hooks once all bucket storage has real CUDA backing."""
        if getattr(self, _EAGER_HOOKS_INSTALLED_ATTR, False):
            return

        bucket_storages = getattr(self, _SHARDED_BUCKET_STORAGES_ATTR)
        if any(
            storage.byte_storage.device.type == "meta" for storage in bucket_storages
        ):
            return

        _install_bucket_unshard_hooks(bucket_storages)
        setattr(self, _EAGER_HOOKS_INSTALLED_ATTR, True)

    def reshard(self) -> None:
        """Reshard every unsharded bucket, like FSDP2's ``FSDPModule.reshard()``.

        Buckets with ``reshard_after_forward=False`` stay unsharded after a
        forward until its backward; call this when that backward will not run.
        Until then, ``module.parameters()`` returns their unsharded params, so
        zero grads through the optimizer rather than ``module.zero_grad()``.
        """
        for context in getattr(self, _EAGER_COMM_CONTEXTS_ATTR, {}).values():
            context.reset()

    def set_gradient_reduce_op(
        self,
        op: GradientReduceOp,
        *,
        recurse: bool = True,
    ) -> None:
        """Set gradient reduction semantics for this module's FlexShard buckets."""
        modules = self.modules() if recurse else [self]
        for module in modules:
            bucket_storages = getattr(module, _SHARDED_BUCKET_STORAGES_ATTR, None)
            if bucket_storages is None:
                continue
            for bucket_storage in bucket_storages:
                bucket_storage.set_gradient_reduce_op(op)

    def set_max_pending_reduce_grads(
        self,
        max_pending_reduce_grads: int = 2,
        *,
        recurse: bool = True,
    ) -> None:
        """Set retained reduce-grad results; 0 keeps all until post-backward."""
        if not isinstance(max_pending_reduce_grads, int) or isinstance(
            max_pending_reduce_grads, bool
        ):
            raise ValueError(
                "max_pending_reduce_grads must be a non-negative int, "
                f"got {type(max_pending_reduce_grads).__name__}."
            )
        if max_pending_reduce_grads < 0:
            raise ValueError(
                "max_pending_reduce_grads must be non-negative, "
                f"got {max_pending_reduce_grads}."
            )

        modules = self.modules() if recurse else [self]
        for module in modules:
            if not isinstance(module, FlexShardModule):
                continue
            setattr(
                module,
                _MAX_PENDING_REDUCE_GRADS_ATTR,
                max_pending_reduce_grads,
            )
            contexts = getattr(module, _EAGER_COMM_CONTEXTS_ATTR, None)
            if contexts is None:
                continue
            for context in contexts.values():
                context.max_pending_reduce_grads = max_pending_reduce_grads


def flex_shard(
    module: nn.Module,
    buckets: list[BucketSpec],
) -> FlexShardModule:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module
    2. Groups parameters into communication buckets (one per bucket, or all in one)
    3. Creates a unified byte buffer per bucket for all its parameters
    4. Replaces each parameter with a plain tensor annotated with placement metadata
    5. Installs per-bucket forward hooks that unshard into persistent unsharded
       parameters (FSDP2-style) swapped into the owning modules
    6. Stores ShardedBucketStorage objects on the module

    Each bucket gets its own byte buffer and ShardedBucketStorage, enabling
    independent unshard operations per bucket.

    Args:
        module: The module to shard. CPU modules are moved to the bucket mesh's
            CUDA device before sharding. Meta parameters keep uninitialized
            storage.
        buckets: Required list of bucket specifications. Each bucket owns its
            1D CUDA mesh and its placement function. A bucket is one collective
            over one process group, so the mesh lives on the bucket; different
            buckets may use different meshes (all sharing one device type). A
            single whole-module bucket can be expressed as
            ``[BucketSpec(["*"], placement_fn=per_param_placements, mesh=mesh)]``.

    Parameters must be plain tensors. Parameters that are local shards of an
    outer (e.g. TP/EP) sharding declare their position in the full parameter
    with ``set_global_layout``; ``flex_shard.layout_adapters.dtensor_to_global_layout``
    does this for DTensor parameters.

    Returns:
        The module (mutated in-place). Use
        module.sharded_bucket_storages to inspect bucket storage internals.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        >>> model = Transformer(args)
        >>> # Single bucket without reshard-after-forward:
        >>> flex_shard(
        ...     model,
        ...     buckets=[
        ...         BucketSpec(
        ...             ["*"],
        ...             placement_fn=per_param_placements,
        ...             mesh=mesh,
        ...             reshard_after_forward=False,
        ...         )
        ...     ],
        ... )
        >>> # Explicit buckets, each carrying its own mesh:
        >>> flex_shard(
        ...     model,
        ...     buckets=[
        ...         BucketSpec(["attn.*"], placement_fn=per_param_placements, mesh=mesh),
        ...         BucketSpec(["ffn.*"], placement_fn=per_param_placements, mesh=mesh),
        ...     ],
        ... )
    Note:
        - Each parameter must have exactly one placement.
        - Each bucket must contain one original parameter dtype. Split mixed
          dtype parameters into separate buckets.
        - Parameters on meta device will have uninitialized storage
    """
    inputs = _prepare_flex_shard_inputs(
        module,
        buckets,
    )

    bucket_storages = _materialize_bucket_storages(
        module,
        inputs,
        buckets,
    )

    flex_shard_module = _attach_flex_shard_module_state(module, bucket_storages)
    flex_shard_module.set_max_pending_reduce_grads()

    # Install bucket unshard hooks for eager mode when the storage layout
    # supports one collective per bucket. Meta modules are materialized later by
    # TorchTitan's Trainer.to_empty() path, so defer hook installation until then.
    flex_shard_module._install_runtime_if_materialized()

    return flex_shard_module


def _check_not_already_flex_sharded(module: nn.Module) -> None:
    """Raise if applying FlexShard would create nested ownership."""
    if getattr(module, _SHARDED_BUCKET_STORAGES_ATTR, None) is not None:
        raise ValueError(
            f"Module {type(module).__name__} already has ShardedBucketStorage. "
            "Cannot apply flex_shard twice to the same module."
        )
    for child_fqn, child in module.named_modules():
        if (
            child_fqn
            and getattr(child, _SHARDED_BUCKET_STORAGES_ATTR, None) is not None
        ):
            raise ValueError(
                "Nested flex_shard wrapping is not supported. "
                f"Child module {child_fqn!r} is already FlexSharded. "
                "Apply flex_shard once at the root module and express bucket "
                "boundaries with BucketSpec FQN patterns."
            )
    for param_fqn, param in module.named_parameters(remove_duplicate=False):
        if is_flex_shard_param(param):
            raise ValueError(
                "Nested flex_shard wrapping is not supported. "
                f"Parameter {param_fqn!r} is already managed by FlexShard. "
                "Apply flex_shard once at the root module and express bucket "
                "boundaries with BucketSpec FQN patterns."
            )


def _attach_flex_shard_module_state(
    module: nn.Module,
    bucket_storages: list[ShardedBucketStorage],
) -> FlexShardModule:
    """Attach FlexShard ownership state and mixin accessors to a module."""
    setattr(module, _SHARDED_BUCKET_STORAGES_ATTR, bucket_storages)
    setattr(module, _EAGER_HOOKS_INSTALLED_ATTR, False)

    cls = type(module)
    if not issubclass(cls, FlexShardModule):
        module.__class__ = type(cls.__name__, (FlexShardModule, cls), {})
    return cast(FlexShardModule, module)


@dataclass(frozen=True)
class PreparedFlexShardInputs:
    """Validated inputs and derived setup state for flex_shard()."""

    named_params: list[tuple[str, nn.Parameter]]
    device: torch.device
    param_placements: dict[str, tuple[Placement, ...]]
    bucket_assignments: BucketParamFQNsByIndex


def _materialize_bucket_storages(
    module: nn.Module,
    inputs: PreparedFlexShardInputs,
    buckets: list[BucketSpec],
) -> list[ShardedBucketStorage]:
    """Create ShardedBucketStorage objects and install sharded parameters."""
    named_params_dict = dict(inputs.named_params)
    bucket_storages: list[ShardedBucketStorage] = []

    for bucket_idx, bucket_fqns in enumerate(inputs.bucket_assignments):
        if not bucket_fqns:
            continue

        bucket_spec = buckets[bucket_idx]
        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_placements = {fqn: inputs.param_placements[fqn] for fqn in bucket_fqns}
        bucket_storages.append(
            ShardedBucketStorage.from_bucket(
                module,
                bucket_named_params,
                bucket_placements,
                bucket_spec.mesh,
                inputs.device,
                bucket_spec,
            )
        )

    return bucket_storages


def _resolve_bucket_param_placements(
    named_params: list[tuple[str, nn.Parameter]],
    bucket_assignments: BucketParamFQNsByIndex,
    buckets: list[BucketSpec],
) -> dict[str, tuple[Placement, ...]]:
    """Call each bucket's placement function for the params assigned to it."""
    named_params_dict = dict(named_params)
    param_placements: dict[str, tuple[Placement, ...]] = {}

    for bucket_idx, bucket_fqns in enumerate(bucket_assignments):
        if not bucket_fqns:
            continue
        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_param_placements = buckets[bucket_idx].placement_fn(
            bucket_named_params,
            buckets[bucket_idx].mesh,
        )
        _validate_placements(bucket_param_placements, bucket_named_params)
        param_placements.update(bucket_param_placements)

    _validate_placements(param_placements, named_params)
    return param_placements


def _prepare_flex_shard_inputs(
    module: nn.Module,
    buckets: list[BucketSpec],
) -> PreparedFlexShardInputs:
    """Validate inputs and derive setup state for flex_shard()."""
    _check_not_already_flex_sharded(module)

    if not buckets:
        raise ValueError("flex_shard requires at least one BucketSpec in buckets.")
    if not all(isinstance(bucket, BucketSpec) for bucket in buckets):
        raise TypeError("flex_shard buckets must be a list of BucketSpec objects.")
    for bucket in buckets:
        if not callable(bucket.placement_fn):
            raise TypeError("BucketSpec.placement_fn must be callable.")
        _validate_flex_shard_mesh(bucket.mesh)
        if bucket.offload_policy is not None:
            raise NotImplementedError(
                "FlexShard eager mode does not yet support BucketSpec.offload_policy."
            )
    # A bucket is one collective over one process group, so the mesh is a
    # per-bucket property. Buckets may use different meshes, but this rank has a
    # single local device, so require all bucket meshes to share a device type.
    device_types = {bucket.mesh.device_type for bucket in buckets}
    if len(device_types) > 1:
        raise ValueError(
            "All BucketSpec meshes in one flex_shard() call must share a device "
            f"type, but got {sorted(device_types)}."
        )

    named_params = _get_managed_named_params(module)
    if not named_params:
        raise ValueError(
            f"Module {type(module).__name__} has no parameters to shard. "
            "All parameters may belong to already-wrapped submodules."
        )

    all_params_meta = all(param.device.type == "meta" for _, param in named_params)
    device = (
        torch.device("meta")
        if all_params_meta
        else _get_device_from_mesh(buckets[0].mesh)
    )
    if not all_params_meta and all(
        param.device.type == "cpu" for _, param in named_params
    ):
        module.to(device)
        named_params = _get_managed_named_params(module)

    _validate_eager_params(
        named_params,
        expected_device=None if all_params_meta else device,
    )

    param_fqns = [fqn for fqn, _ in named_params]
    bucket_assignments = _assign_params_to_buckets(param_fqns, buckets)
    param_placements = _resolve_bucket_param_placements(
        named_params,
        bucket_assignments,
        buckets,
    )
    _validate_bucket_uniform_dtype_and_placement(
        bucket_assignments,
        param_placements,
        buckets,
        named_params,
    )

    return PreparedFlexShardInputs(
        named_params=named_params,
        device=device,
        param_placements=param_placements,
        bucket_assignments=bucket_assignments,
    )
