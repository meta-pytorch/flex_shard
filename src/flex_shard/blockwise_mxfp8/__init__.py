# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""32x32-block MXFP8 weight and linear building blocks for FlexShard."""

from .blockwise_mxfp8_weight import (
    blockwise_mxfp8_dequant_weight,
    BlockwiseMxfp8Weight,
)
from .mxfp8_blockwise_linear import (
    convert_to_flex_shard_mxfp8_blockwise_linear,
    FlexShardMxfp8BlockwiseLinear,
)


__all__ = [
    "blockwise_mxfp8_dequant_weight",
    "BlockwiseMxfp8Weight",
    "convert_to_flex_shard_mxfp8_blockwise_linear",
    "FlexShardMxfp8BlockwiseLinear",
]
