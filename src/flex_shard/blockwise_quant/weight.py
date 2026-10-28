# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base tensor subclass for blockwise-quantized FlexShard all-gather output."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing


class BlockwiseQuantWeight(torch.Tensor):
    """Prequantized weight returned by a FlexShard blockwise all-gather.

    The wrapper reports the logical high-precision shape and dtype of ``_data``.
    Subclasses name their inner tensors and metadata and rebuild themselves from
    those in ``_from_inner``. Only aliasing and detach are supported; consumers
    read the quantized operands directly or call ``dequantize()``.
    """

    _inner_tensor_names: tuple[str, ...]

    _data: torch.Tensor
    _orig_dtype: torch.dtype

    @classmethod
    def _make_wrapper(
        cls,
        data: torch.Tensor,
        orig_dtype: torch.dtype,
        requires_grad: bool,
    ) -> BlockwiseQuantWeight:
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=orig_dtype,
            layout=data.layout,
            device=data.device,
            requires_grad=requires_grad,
        )

    def _metadata(self) -> dict[str, Any]:
        """Non-tensor state passed back to ``_from_inner``."""
        raise NotImplementedError

    @classmethod
    def _from_inner(
        cls,
        inner_tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        requires_grad: bool,
    ) -> BlockwiseQuantWeight:
        raise NotImplementedError

    def dequantize(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def orig_dtype(self) -> torch.dtype:
        return self._orig_dtype

    def _map_inner(self, fn) -> BlockwiseQuantWeight:
        return type(self)._from_inner(
            {name: fn(getattr(self, name)) for name in self._inner_tensor_names},
            self._metadata(),
            requires_grad=False,
        )

    def __tensor_flatten__(self) -> tuple[list[str], dict[str, Any]]:
        return list(self._inner_tensor_names), {
            **self._metadata(),
            "requires_grad": self.requires_grad,
        }

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> BlockwiseQuantWeight:
        _ = outer_size, outer_stride
        metadata = dict(metadata)
        requires_grad = metadata.pop("requires_grad")
        return cls._from_inner(inner_tensors, metadata, requires_grad)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.alias.default:
            # Autograd attaches differentiable-view metadata after dispatch returns.
            result = args[0]._map_inner(func)
            return return_and_correct_aliasing(func, args, kwargs or {}, result)

        if func == torch.ops.aten.detach.default:
            return args[0]._map_inner(torch.Tensor.detach)

        raise NotImplementedError(
            f"{cls.__name__} does not support {func}. "
            "Call dequantize() explicitly before applying this operation."
        )

    __torch_function__ = torch._C._disabled_torch_function_impl


__all__ = [
    "BlockwiseQuantWeight",
]
