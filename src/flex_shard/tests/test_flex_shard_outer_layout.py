# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard
from torch.testing._internal.common_utils import run_tests, TestCase

from .. import get_global_layout, GlobalLayout
from ..layout_adapters import dtensor_to_global_layout
from .common import single_rank_cpu_mesh


class TestFlexShardOuterLayout(TestCase):
    def test_dtensor_adapter_declares_layout(self) -> None:
        model = nn.Module()
        with single_rank_cpu_mesh() as mesh:
            model.weight = nn.Parameter(
                DTensor.from_local(torch.ones(4, 3), mesh, [Shard(0)])
            )
            dtensor_to_global_layout(model)
        self.assertNotIsInstance(model.weight, DTensor)
        self.assertEqual(
            get_global_layout(model.weight),
            GlobalLayout((4, 3), ((0, 0),), ((0, 0),), ((4, 3),)),
        )


if __name__ == "__main__":
    run_tests()
