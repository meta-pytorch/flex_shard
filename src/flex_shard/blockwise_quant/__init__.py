# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared building blocks for FlexShard blockwise-quantized weight consumers."""

from .linear import (
    convert_to_flex_shard_prequantized_linear,
    FlexShardPrequantizedLinear,
    PrequantizedLinearGemms,
)
from .weight import BlockwiseQuantWeight


__all__ = [
    "BlockwiseQuantWeight",
    "convert_to_flex_shard_prequantized_linear",
    "FlexShardPrequantizedLinear",
    "PrequantizedLinearGemms",
]
