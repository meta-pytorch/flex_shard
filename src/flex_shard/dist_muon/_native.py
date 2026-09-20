# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local Muon execution on the actual ordinary tensors managed by FlexShard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

from ..custom_placements.block_shard import BlockShard, BucketedBlockShard
from ..custom_placements.owned import BucketedOwned
from ..flex_shard.sharded_param import get_shard_metadata, ShardMetadata
from ..flex_shard.utils import _strip_checkpoint_wrapped_module_path as canonical_fqn
from ._muon_math import (
    _apply_muon_update,
    _compute_muon_direction,
    _prepare_muon_input,
)
from ._optimizer_reshard_runtime import _BucketedRedistributionRuntime
from ._optimizer_reshard_schedule import _LocalBucketPlan

if TYPE_CHECKING:
    from .optimizer import DistMuon


_CHECKPOINT_KEY = "native_dist_muon"
_SHARD_ATTRS = ("global_shape", "global_offsets", "local_offsets", "local_sizes")


@dataclass(frozen=True, slots=True)
class _NativeParameter:
    fqn: str
    param: Tensor
    metadata: ShardMetadata
    kind: str
    matrix_rows: int
    matrix_columns: int
    dtype: torch.dtype

    def checkpoint_layout(self) -> dict[str, Any]:
        regions = self.metadata.regions
        assert regions is not None
        return {
            "kind": self.kind,
            "matrix_shape": (self.matrix_rows, self.matrix_columns),
            "global_shape": tuple(self.metadata.canonical_shape),
            "global_offsets": tuple(tuple(region.global_offset) for region in regions),
            "local_offsets": tuple(tuple(region.local_offset) for region in regions),
            "local_sizes": tuple(tuple(region.shape) for region in regions),
            "storage_shape": tuple(self.param.shape),
        }


def _resolve_parameter(parameter: Tensor) -> _NativeParameter:
    metadata = get_shard_metadata(parameter)
    if metadata is None:
        raise TypeError(
            "native DistMuon requires real parameters obtained after flex_shard; "
            "FlexShard shard metadata is missing"
        )
    fqn = canonical_fqn(metadata.fqn)
    if parameter.is_meta:
        raise ValueError(f"materialize Muon parameter {fqn!r} before construction")
    if parameter.layout != torch.strided or not parameter.is_floating_point():
        raise ValueError(f"Muon parameter {fqn!r} must be a dense floating tensor")
    if not parameter.is_contiguous() or parameter.shape != metadata.local_shape:
        raise ValueError(f"Muon parameter {fqn!r} has inconsistent local storage")
    if metadata.regions is None:
        raise NotImplementedError(f"unknown canonical shard coordinates for {fqn!r}")
    if len(metadata.placements) != 1:
        raise NotImplementedError(
            f"native Muon requires one storage placement: {fqn!r}"
        )

    placement = metadata.placements[0]
    shape = metadata.storage_shape
    if isinstance(placement, BlockShard):
        if len(shape) != 2 or placement.dim != 0:
            raise NotImplementedError(
                f"native Muon BlockShard parameter {fqn!r} must be 2D on dimension 0"
            )
        blocks = sum(placement.blocks_per_rank)
        if not blocks or shape[0] % blocks:
            raise ValueError(f"invalid Muon block boundaries for {fqn!r}")
        rows, columns = shape[0] // blocks, shape[1]
        kind = "row_blocks"
        if rows <= 0 or parameter.ndim != 2 or parameter.shape[0] % rows:
            raise ValueError(f"Muon parameter {fqn!r} splits a logical matrix")
        if parameter.shape[1] != columns:
            raise ValueError(f"Muon parameter {fqn!r} splits matrix columns")
    elif isinstance(placement, BucketedOwned):
        if len(shape) != 2 or (parameter.numel() and parameter.shape != shape):
            raise NotImplementedError(
                f"native Muon Owned parameter {fqn!r} must own a complete 2D matrix"
            )
        rows, columns = shape
        kind = "owned"
    elif isinstance(placement, BucketedBlockShard):
        if len(shape) != 3 or placement.dims != (0,):
            raise NotImplementedError(
                f"native Muon BucketedBlockShard parameter {fqn!r} must be a "
                "3D matrix batch sharded on dimension 0"
            )
        rows, columns = shape[1:]
        kind = "matrix_batch"
        if parameter.numel() and (
            parameter.ndim != 3 or tuple(parameter.shape[1:]) != tuple(shape[1:])
        ):
            raise ValueError(f"Muon parameter {fqn!r} splits a logical matrix")
    else:
        raise NotImplementedError(
            f"native Muon requires complete matrices through BlockShard or Owned; "
            f"parameter {fqn!r} uses unsupported placement {placement!r}. "
            "Partial-matrix Shard(0) requires redistribution."
        )
    if rows <= 0 or columns <= 0:
        raise ValueError(f"Muon parameter {fqn!r} has an empty logical matrix")
    # Outer sharding may distribute packed rows or complete matrix batches,
    # but cannot leave us with a column fragment (or a 3D matrix-row fragment).
    # The core intentionally supports more outer layouts than native Muon.
    matrix_dims = (-2, -1) if kind == "matrix_batch" else (-1,)
    if len(metadata.canonical_shape) != len(shape) or any(
        shape[dim] != metadata.canonical_shape[dim]
        or (metadata.outer_offset is not None and metadata.outer_offset[dim] != 0)
        for dim in matrix_dims
    ):
        raise NotImplementedError(
            f"native Muon parameter {fqn!r} has an outer shard that splits "
            "a logical matrix; redistribution is required"
        )
    if parameter.numel():
        if len(metadata.regions) != 1:
            raise NotImplementedError(f"Muon parameter {fqn!r} needs one local region")
        region = metadata.regions[0]
        if tuple(region.shape) != tuple(parameter.shape) or any(region.local_offset):
            raise ValueError(f"Muon parameter {fqn!r} has incomplete shard coordinates")
        if kind == "row_blocks":
            if metadata.outer_offset is None:
                raise NotImplementedError(
                    f"unknown outer shard coordinates for Muon parameter {fqn!r}"
                )
            if (region.global_offset[0] - metadata.outer_offset[0]) % rows:
                raise ValueError(f"Muon parameter {fqn!r} splits a logical matrix")
    elif metadata.regions:
        raise ValueError(
            f"empty Muon parameter {fqn!r} must not describe local regions"
        )
    return _NativeParameter(
        fqn, parameter, metadata, kind, rows, columns, parameter.dtype
    )


def _validate_momentum(item: _NativeParameter, momentum: Any) -> None:
    parameter = item.param
    if (
        not isinstance(momentum, Tensor)
        or isinstance(momentum, DTensor)
        or momentum.layout != torch.strided
        or momentum.shape != parameter.shape
        or momentum.dtype != parameter.dtype
        or momentum.device != parameter.device
        or not momentum.is_contiguous()
    ):
        raise ValueError(f"momentum storage layout changed for {item.fqn!r}")


class _NativeMuon:
    def __init__(self, optimizer: DistMuon) -> None:
        self.optimizer = optimizer
        group = optimizer.param_groups[0]
        if not group["params"]:
            raise ValueError("DistMuon requires at least one original parameter")
        items = tuple(_resolve_parameter(parameter) for parameter in group["params"])
        names = tuple(item.fqn for item in items)
        supplied = group.get("param_names")
        if supplied is not None and tuple(map(canonical_fqn, supplied)) != names:
            raise ValueError("param_names must match real FlexShard parameter metadata")
        if len(set(names)) != len(names) or len(
            {id(item.param) for item in items}
        ) != len(items):
            raise ValueError("duplicate native Muon parameter or FQN")
        if len({item.param.device for item in items}) != 1:
            raise ValueError("DistMuon requires one device per process")
        group["param_names"] = list(names)
        self.items = items
        self.rebuild()

    def rebuild(self) -> None:
        """Prepare only mesh-free local plans and scratch, including after load."""
        self.active = tuple(item for item in self.items if item.param.numel())
        self.plans = (_LocalBucketPlan(self.active),)
        self.runtime = _BucketedRedistributionRuntime(self.items[0].param.device)
        self.runtime.reserve_buffers(self.plans, local_tensor_spec=self._tensor_spec)

    @staticmethod
    def _tensor_spec(
        item: _NativeParameter,
    ) -> tuple[torch.Size, torch.dtype, torch.device]:
        return item.param.shape, item.param.dtype, item.param.device

    def _validate_current_parameters(self) -> None:
        group = self.optimizer.param_groups[0]
        if tuple(id(param) for param in group["params"]) != tuple(
            id(item.param) for item in self.items
        ) or tuple(group.get("param_names", ())) != tuple(
            item.fqn for item in self.items
        ):
            raise ValueError("native DistMuon parameters and FQNs are frozen")
        for item in self.items:
            current = _resolve_parameter(item.param)
            if current.fqn != item.fqn:
                raise ValueError(
                    f"native Muon parameter FQN changed from {item.fqn!r} "
                    f"to {current.fqn!r}; rebuild DistMuon"
                )
            if current.checkpoint_layout() != item.checkpoint_layout():
                raise ValueError(
                    f"native Muon layout changed for {item.fqn!r}; rebuild DistMuon"
                )
            # Replacing a storage allocation with the same shape/layout is safe:
            # callbacks read the actual parameter and its current gradient.
            if item.param.device != self.runtime._device:
                raise ValueError(
                    f"native Muon device changed for {item.fqn!r}; rebuild DistMuon"
                )
            if item.param.dtype != item.dtype:
                raise ValueError(
                    f"native Muon dtype changed for {item.fqn!r}; rebuild DistMuon"
                )

    def step(self) -> None:
        self.optimizer._validate_groups()
        self._validate_current_parameters()
        missing = []
        for item in self.active:
            gradient = item.param.grad
            if gradient is None:
                missing.append(item.fqn)
            elif (
                isinstance(gradient, DTensor)
                or gradient.layout != torch.strided
                or gradient.shape != item.param.shape
                or gradient.dtype != item.param.dtype
                or gradient.device != item.param.device
                or not gradient.is_contiguous()
            ):
                raise ValueError(f"gradient storage layout changed for {item.fqn!r}")
            state = self.optimizer.state.get(item.param, {})
            if "momentum_buffer" in state:
                _validate_momentum(item, state["momentum_buffer"])
        if missing:
            raise RuntimeError(
                "DistMuon requires every nonempty parameter gradient before step(); "
                f"missing gradients: {missing}"
            )
        # All deterministic validation completes before state creation or any
        # in-place update, even if a later parameter's gradient is missing.
        for item in self.active:
            state = self.optimizer.state[item.param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(item.param)
        self.runtime.run(
            self.plans,
            local_tensor_spec=self._tensor_spec,
            prepare=self._prepare,
            compute=self._compute,
            finalize=self._apply,
        )

    def _prepare(self, item: _NativeParameter, out: Tensor) -> None:
        group = self.optimizer.param_groups[0]
        _prepare_muon_input(
            item.param.grad,
            self.optimizer.state[item.param]["momentum_buffer"],
            momentum=group["momentum"],
            nesterov=group["nesterov"],
            out=out,
        )

    def _compute(self, item: _NativeParameter, compute: Tensor) -> None:
        if item.kind == "row_blocks":
            compute = compute.view(-1, item.matrix_rows, item.matrix_columns)
        group = self.optimizer.param_groups[0]
        _compute_muon_direction(
            compute,
            ns_coefficients=group["ns_coefficients"],
            ns_steps=group["ns_steps"],
            eps=group["eps"],
            out=compute,
        )

    def _apply(self, item: _NativeParameter, direction: Tensor) -> None:
        group = self.optimizer.param_groups[0]
        _apply_muon_update(
            item.param,
            direction,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            adjust_lr_fn=group["adjust_lr_fn"],
            compute_matrix_shape=(item.matrix_rows, item.matrix_columns),
        )

    def state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        self._validate_current_parameters()
        result = dict(state_dict)
        result["state"] = {
            key: dict(value) for key, value in state_dict["state"].items()
        }
        layouts = {item.fqn: item.checkpoint_layout() for item in self.items}
        result[_CHECKPOINT_KEY] = {"version": 1, "layouts": layouts}
        ids = result["param_groups"][0]["params"]
        for item, state_id in zip(self.items, ids, strict=True):
            state = result["state"].get(state_id, {})
            if "momentum_buffer" not in state:
                continue
            momentum = state["momentum_buffer"]
            _validate_momentum(item, momentum)
            checkpoint = momentum.detach()
            for name in _SHARD_ATTRS:
                setattr(checkpoint, name, layouts[item.fqn][name])
            state["momentum_buffer"] = checkpoint
        return result

    def prepare_load(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Match native or legacy adapter state by FQN, before mutating anything."""
        self._validate_current_parameters()
        groups = state_dict.get("param_groups", ())
        if len(groups) != 1:
            raise ValueError("DistMuon checkpoint requires exactly one parameter group")
        saved_group = dict(groups[0])
        self.optimizer._validate_group_values([saved_group])
        saved_ids = tuple(saved_group.get("params", ()))
        saved_raw_names = tuple(saved_group.get("param_names", ()))
        if (
            len(saved_ids) != len(saved_raw_names)
            or len(set(saved_ids)) != len(saved_ids)
            or not all(isinstance(name, str) for name in saved_raw_names)
        ):
            raise ValueError(
                "DistMuon checkpoint params and param_names must be aligned"
            )
        saved_names = tuple(map(canonical_fqn, saved_raw_names))
        if len(set(saved_names)) != len(saved_names):
            raise ValueError("duplicate DistMuon checkpoint FQNs")
        native = state_dict.get(_CHECKPOINT_KEY)
        expected_names = {item.fqn for item in self.items}
        active_names = {item.fqn for item in self.active}
        if not active_names.issubset(saved_names) or not set(saved_names).issubset(
            expected_names
        ):
            raise ValueError("DistMuon checkpoint FQNs do not match local parameters")
        if native is not None:
            expected_layouts = {
                item.fqn: item.checkpoint_layout() for item in self.items
            }
            if (
                not isinstance(native, dict)
                or native.get("version") != 1
                or native.get("layouts") != expected_layouts
                or set(saved_names) != expected_names
            ):
                raise ValueError(
                    "native DistMuon checkpoint shard metadata does not match"
                )
        saved_state = state_dict.get("state", {})
        if set(saved_state) - set(saved_ids):
            raise ValueError("DistMuon checkpoint contains unknown parameter state IDs")
        ids_by_name = dict(zip(saved_names, saved_ids, strict=True))
        result_state = {}
        for current_id, item in enumerate(self.items):
            saved_id = ids_by_name.get(item.fqn)
            state = dict(saved_state.get(saved_id, {}))
            if state and not item.param.numel():
                raise ValueError(f"empty Muon parameter {item.fqn!r} cannot have state")
            if set(state) - {"momentum_buffer"}:
                raise ValueError(f"unknown Muon checkpoint state for {item.fqn!r}")
            if "momentum_buffer" in state:
                momentum = state["momentum_buffer"]
                if (
                    not isinstance(momentum, Tensor)
                    or isinstance(momentum, DTensor)
                    or momentum.layout != torch.strided
                    or momentum.shape != item.param.shape
                    or not momentum.is_floating_point()
                    or not momentum.is_contiguous()
                ):
                    raise ValueError(f"invalid checkpoint momentum for {item.fqn!r}")
                expected = item.checkpoint_layout()
                if any(
                    getattr(momentum, name, None) != expected[name]
                    for name in _SHARD_ATTRS
                ):
                    raise ValueError(
                        f"checkpoint momentum shard coordinates differ for {item.fqn!r}"
                    )
                # Optimizer.load_state_dict casts this plain tensor onto the real
                # parameter's device/dtype, without constructing any DTensor.
                state["momentum_buffer"] = momentum
            if state:
                result_state[current_id] = state
        saved_group["params"] = list(range(len(self.items)))
        saved_group["param_names"] = [item.fqn for item in self.items]
        return {"state": result_state, "param_groups": [saved_group]}
