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
from ..custom_placements.shard import Shard as FlexShardShard
from ..flex_shard.bucket_storage import ShardedBucketStorage
from ..layout_adapters import dtensor_to_global_layout
from .common import single_rank_cpu_mesh


class TestFlexShardOuterLayout(TestCase):
    def test_dtensor_adapter_declares_layout_and_preserves_grad_dtype(self) -> None:
        model = nn.Module()
        grad_dtypes = {"fp32_grad": torch.float32, "none_grad": None}
        with single_rank_cpu_mesh() as mesh:
            for name, grad_dtype in grad_dtypes.items():
                param = nn.Parameter(
                    DTensor.from_local(
                        torch.ones(4, 3, dtype=torch.bfloat16), mesh, [Shard(0)]
                    )
                )
                param.grad_dtype = grad_dtype
                setattr(model, name, param)
            dtensor_to_global_layout(model)
            param_infos, _ = ShardedBucketStorage.create_param_infos(
                list(model.named_parameters()),
                mesh,
                {name: (FlexShardShard(0),) for name in grad_dtypes},
            )
        for name, grad_dtype in grad_dtypes.items():
            param = model._parameters[name]
            self.assertNotIsInstance(param, DTensor)
            self.assertEqual(
                get_global_layout(param),
                GlobalLayout((4, 3), ((0, 0),), ((0, 0),), ((4, 3),)),
            )
            self.assertEqual(param_infos[name].unsharded_grad_dtype, grad_dtype)


if __name__ == "__main__":
    run_tests()
