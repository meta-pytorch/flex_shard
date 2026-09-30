# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .flex_shard import FlexShardModule
from .placement_contract import compose_global_layouts, GlobalLayout, set_global_layout
from .utils import _strip_checkpoint_wrapped_module_path


def get_flex_shard_global_layouts(module: nn.Module) -> dict[str, GlobalLayout]:
    """Return where each FlexShard parameter's local shard sits in the full parameter.

    Keys are parameter FQNs relative to ``module``. Each layout composes the
    parameter's placement layout with the outer layout the parameter declared
    before ``flex_shard()``, and is computed from the current FlexShard state on
    every call.

    Raises ``NotImplementedError`` for parameters whose placement does not
    implement ``Placement.global_layout``.
    """
    storages = [
        (prefix, storage)
        for prefix, submodule in module.named_modules()
        if isinstance(submodule, FlexShardModule)
        for storage in submodule.sharded_bucket_storages
    ]
    layouts = {}
    for prefix, storage in storages:
        rank, world_size = storage._mesh.get_local_rank(), storage._mesh.size()
        for fqn, info in storage.param_infos.items():
            name = f"{prefix}.{fqn}" if prefix else fqn
            try:
                layout = info.placement.global_layout(info, rank, world_size)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"Distributed checkpointing is not supported for {name!r}: {e}"
                ) from e
            if info.outer_layout is not None:
                layout = compose_global_layouts(info.outer_layout, layout)
            layouts[name] = layout
    return layouts


def set_state_dict_global_layouts(
    module: nn.Module, state_dict: dict[str, Any]
) -> None:
    """Declare each FlexShard parameter's layout on its ``state_dict`` tensor.

    Call on the state dict passed to both ``dcp.save`` and ``dcp.load``.
    Without it, DCP treats each local shard as the full parameter. Parameters
    missing from ``state_dict`` are skipped.
    """
    for fqn, layout in get_flex_shard_global_layouts(module).items():
        key = fqn if fqn in state_dict else _strip_checkpoint_wrapped_module_path(fqn)
        if key in state_dict:
            set_global_layout(state_dict[key], layout)


__all__ = [
    "get_flex_shard_global_layouts",
    "set_state_dict_global_layouts",
]
