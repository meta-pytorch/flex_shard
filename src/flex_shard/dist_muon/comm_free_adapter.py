# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bind compute-ready FlexShard parameters through a supplied optimizer factory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.optim import Optimizer

from .storage_layout import get_flex_shard_muon_compute_layouts, LocalMuonComputeLayout


OptimizerFactory = Callable[
    [list[dict[str, Any]], dict[str, LocalMuonComputeLayout]], Optimizer
]


@dataclass(frozen=True, slots=True)
class _ParameterBinding:
    fqn: str
    real_param: Tensor
    proxy_param: DTensor
    layout: LocalMuonComputeLayout


class LocalDistMuonBinding:
    def __init__(
        self,
        optimizer: Optimizer,
        parameter_bindings: tuple[_ParameterBinding, ...],
    ) -> None:
        self.optimizer = optimizer
        self._parameter_bindings = parameter_bindings
        optimizer.register_state_dict_post_hook(self._state_dict_post_hook)
        optimizer.register_load_state_dict_pre_hook(
            self._load_state_dict_pre_hook, prepend=True
        )

    def step(self) -> None:
        missing = [
            binding.fqn
            for binding in self._parameter_bindings
            if binding.real_param.grad is None
        ]
        if missing:
            raise RuntimeError(
                "local DistMuon requires every real parameter gradient before "
                f"step(); missing gradients: {missing}"
            )
        for binding in self._parameter_bindings:
            binding.proxy_param.grad = _proxy_gradient(binding)
        self.optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)
        for binding in self._parameter_bindings:
            param = binding.real_param
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()

    def _state_dict_post_hook(
        self, _optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        return self._translate_state_dict(state_dict, _checkpoint_momentum)

    def _load_state_dict_pre_hook(
        self, _optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        return self._translate_state_dict(state_dict, _proxy_momentum)

    def _translate_state_dict(
        self,
        state_dict: dict[str, Any],
        convert: Callable[[_ParameterBinding, Any], Tensor],
    ) -> dict[str, Any]:
        result = dict(state_dict)
        result["state"] = {
            state_id: dict(state) for state_id, state in state_dict["state"].items()
        }
        state_ids = result["param_groups"][0]["params"]
        for binding, state_id in zip(
            self._parameter_bindings,
            state_ids,
            strict=True,
        ):
            state = result["state"].get(state_id)
            if state is not None and "momentum_buffer" in state:
                state["momentum_buffer"] = convert(binding, state["momentum_buffer"])
        return result


def build_local_dist_muon(
    model: nn.Module,
    named_params: Sequence[tuple[str, Tensor]],
    *,
    optimizer_factory: OptimizerFactory,
) -> LocalDistMuonBinding | None:
    """Build a binding whose factory preserves one ordered parameter group."""
    if not named_params:
        raise ValueError("local DistMuon requires at least one matched parameter")
    device = named_params[0][1].device
    if device.type != "cuda" or any(
        param.device != device for _, param in named_params
    ):
        raise ValueError("local DistMuon requires one CUDA device per process")

    # Process-group creation is SPMD, including on ranks with no local Muon work.
    world_mesh = init_device_mesh(
        device.type,
        (dist.get_world_size(), 1),
        mesh_dim_names=("world", "local"),
    )
    local_mesh = world_mesh["local"]
    local_named_params = tuple(
        (fqn, param) for fqn, param in named_params if param.numel() > 0
    )
    if not local_named_params:
        # DistMuon rejects an empty parameter set; None represents no local
        # Muon work without introducing dummy optimizer or checkpoint state.
        return None
    layouts = get_flex_shard_muon_compute_layouts(model, local_named_params)
    bindings = tuple(
        _build_parameter_binding(fqn, param, layout, local_mesh)
        for (fqn, param), layout in zip(
            local_named_params,
            layouts,
            strict=True,
        )
    )

    optimizer = optimizer_factory(
        [
            {
                "params": [binding.proxy_param for binding in bindings],
                "param_names": [binding.fqn for binding in bindings],
            }
        ],
        {binding.fqn: binding.layout for binding in bindings},
    )
    return LocalDistMuonBinding(optimizer, bindings)


def _build_parameter_binding(
    fqn: str,
    real_param: Tensor,
    layout: LocalMuonComputeLayout,
    local_mesh: DeviceMesh,
) -> _ParameterBinding:
    local = real_param.detach()
    proxy = DTensor.from_local(
        local,
        device_mesh=local_mesh,
        placements=(Replicate(),),
        run_check=False,
        shape=local.shape,
        stride=local.stride(),
    ).detach()
    proxy.requires_grad_(real_param.requires_grad)
    return _ParameterBinding(fqn, real_param, proxy, layout)


def _proxy_gradient(binding: _ParameterBinding) -> DTensor:
    grad = binding.real_param.grad
    assert grad is not None
    proxy = binding.proxy_param
    return DTensor.from_local(
        grad.detach(),
        device_mesh=proxy.device_mesh,
        placements=proxy.placements,
        run_check=False,
        shape=proxy.shape,
        stride=proxy.stride(),
    )


def _checkpoint_momentum(binding: _ParameterBinding, momentum: Any) -> Tensor:
    local = momentum.to_local().detach()
    shard = binding.layout.state_checkpoint_shard
    setattr(local, "global_shape", tuple(shard.global_shape))  # noqa: B010
    setattr(local, "global_offsets", (shard.global_offset,))  # noqa: B010
    setattr(local, "local_offsets", ((0,) * local.ndim,))  # noqa: B010
    setattr(local, "local_sizes", (tuple(shard.local_shape),))  # noqa: B010
    return local


def _proxy_momentum(binding: _ParameterBinding, momentum: Any) -> DTensor:
    proxy = binding.proxy_param
    return DTensor.from_local(
        momentum.detach(),
        device_mesh=proxy.device_mesh,
        placements=proxy.placements,
        run_check=False,
        shape=proxy.shape,
        stride=proxy.stride(),
    )
