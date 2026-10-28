# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base linear consumer for weights prequantized by a FlexShard all-gather."""

from __future__ import annotations

import math
from typing import Any, Callable, ClassVar, Protocol

import torch
import torch.nn as nn

from .weight import BlockwiseQuantWeight


class PrequantizedLinearGemms(Protocol):
    """Format-specific GEMMs for a linear layer with a prequantized weight.

    A stacked ``[*stack, out, in]`` weight multiplies as its flattened
    ``[prod(stack) * out, in]`` matrix.
    """

    def forward(
        self,
        x_2d: torch.Tensor,
        weight: BlockwiseQuantWeight,
    ) -> torch.Tensor:
        """Return ``x_2d @ weight_matrix.T`` with shape ``(tokens, rows)``."""
        ...

    def backward(
        self,
        grad_output_2d: torch.Tensor,
        x_2d: torch.Tensor,
        weight: BlockwiseQuantWeight,
        needs_x_grad: bool,
        needs_weight_grad: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return ``(grad_x_2d, grad_weight)``.

        ``grad_weight`` is contiguous and has the weight's shape.
        """
        ...


def _pad_last_dim_to(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    if tensor.shape[-1] == target_dim:
        return tensor
    if tensor.shape[-1] > target_dim:
        raise ValueError(
            f"Cannot pad last dim from {tensor.shape[-1]} down to {target_dim}."
        )
    padded = torch.zeros(
        (*tensor.shape[:-1], target_dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[..., : tensor.shape[-1]].copy_(tensor)
    return padded


def _as_column_major_2d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got {tuple(tensor.shape)}")
    if tensor.stride(-2) == 1:
        return tensor
    return tensor.t().contiguous().t()


class _PrequantizedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: BlockwiseQuantWeight,
        bias: torch.Tensor | None,
        gemms: PrequantizedLinearGemms,
    ) -> torch.Tensor:
        logical_in_dim = weight.shape[-1]
        x_orig_shape = x.shape
        x_2d = x.reshape(-1, x_orig_shape[-1])
        if x_2d.shape[-1] != logical_in_dim:
            raise ValueError(
                f"input last dim {x_2d.shape[-1]} does not match weight input dim "
                f"{logical_in_dim}"
            )
        out = gemms.forward(x_2d, weight).reshape(
            *x_orig_shape[:-1],
            *weight.shape[:-1],
        )
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(x, weight)
        ctx.gemms = gemms
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        x, weight = ctx.saved_tensors
        logical_out_dim = math.prod(weight.shape[:-1])
        grad_output_2d = grad_output.reshape(-1, logical_out_dim).contiguous()
        needs_x_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        grad_x, grad_weight = ctx.gemms.backward(
            grad_output_2d,
            x.reshape(-1, x.shape[-1]),
            weight,
            needs_x_grad,
            needs_weight_grad,
        )
        if grad_x is not None:
            grad_x = grad_x.reshape(x.shape)
        grad_bias = None
        if ctx.has_bias and needs_bias_grad:
            grad_bias = grad_output_2d.sum(dim=0).view(weight.shape[:-1])
        return grad_x, grad_weight, grad_bias, None


class FlexShardPrequantizedLinear(nn.Module):
    """Mixin for ``nn.Linear`` subclasses consuming FlexShard-quantized weights.

    Subclasses set ``weight_type``, return their format's GEMMs from
    ``_prequantized_gemms``, and implement ``from_float`` to build the
    replacement for an ``nn.Linear``; list this mixin before the ``nn.Linear``
    base so its ``forward`` wins. A stacked ``[*stack, out, in]`` weight runs
    one GEMM on its flattened matrix and returns ``[..., *stack, out]``.
    """

    weight_type: ClassVar[type[BlockwiseQuantWeight]]
    from_float: ClassVar[Callable[..., nn.Module]]

    def _prequantized_gemms(self) -> PrequantizedLinearGemms:
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if not isinstance(weight, self.weight_type):
            raise TypeError(
                f"{type(self).__name__} requires a {self.weight_type.__name__} "
                f"produced by a FlexShard quantized all-gather, got "
                f"{type(weight).__name__}."
            )
        return _PrequantizedLinearFunction.apply(
            input,
            weight,
            self.bias,
            self._prequantized_gemms(),
        )


def convert_to_flex_shard_prequantized_linear(
    module: nn.Module,
    linear_cls: type[FlexShardPrequantizedLinear],
    filter_fn: Callable[[str, nn.Linear], bool] | None = None,
    **from_float_kwargs: Any,
) -> nn.Module:
    """Replace matching nn.Linear children with ``linear_cls.from_float``."""

    def should_convert(fqn: str, child: nn.Linear) -> bool:
        return filter_fn(fqn, child) if filter_fn is not None else True

    def convert_children(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            fqn = f"{prefix}.{name}" if prefix else name
            if isinstance(child, FlexShardPrequantizedLinear):
                continue
            if isinstance(child, nn.Linear) and should_convert(fqn, child):
                setattr(
                    parent,
                    name,
                    linear_cls.from_float(child, **from_float_kwargs),
                )
            else:
                convert_children(child, fqn)

    convert_children(module)
    return module


__all__ = [
    "convert_to_flex_shard_prequantized_linear",
    "FlexShardPrequantizedLinear",
    "PrequantizedLinearGemms",
]
