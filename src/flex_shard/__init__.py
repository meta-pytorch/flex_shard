# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .flex_shard import (
    BucketParamStorageLayout,
    BucketSpec,
    BucketStorageLayout,
    GlobalLayout,
    flex_shard,
    get_global_layout,
    get_global_shape,
    get_placements,
    is_flex_shard_param,
    LocalStorageLayout,
    MixedPrecisionPolicy,
    OffloadPolicy,
    Placement,
    PlacementFn,
    set_global_layout,
)


__all__ = [
    "BucketParamStorageLayout",
    "BucketSpec",
    "BucketStorageLayout",
    "GlobalLayout",
    "flex_shard",
    "get_global_layout",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "LocalStorageLayout",
    "MixedPrecisionPolicy",
    "OffloadPolicy",
    "Placement",
    "PlacementFn",
    "set_global_layout",
]
