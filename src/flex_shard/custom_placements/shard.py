# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from typing_extensions import override

from ..flex_shard.bucket_storage import gradient_reduce_op_from_infos, GradientReduceOp
from ..flex_shard.placement_contract import (
    BucketParamStorageLayout,
    BucketStorageLayout,
    Placement,
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementReduceGradResult,
    PlacementUnshardResult,
)
from ..flex_shard.utils import (
    _record_comm_if_eager,
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
    _record_function_if_eager,
)
from .utils import copy_tensor_to_dtype

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo, PlacementFn


class Shard(Placement):
    """Symmetric sharding — parameter split along dim across all ranks."""

    @dataclass(frozen=True)
    class _ReduceGradLayout:
        padded_sizes: list[torch.Size]

    @dataclass(frozen=True)
    class _PaddedUnshardLayout:
        padded_local_numels: tuple[int, ...]

    @dataclass(frozen=True)
    class _UnshardState:
        infos: list[ParamInfo]
        world_size: int
        pg: Any
        debug_fqn: str | None
        padded_layout: Shard._PaddedUnshardLayout

    @dataclass(frozen=True)
    class _ReduceGradState:
        infos: list[ParamInfo]
        layout: Shard._ReduceGradLayout
        rank: int
        world_size: int
        pg: Any
        debug_fqn: str | None
        gradient_reduce_op: GradientReduceOp

    def __init__(self, dim: int = 0):
        self.dim = dim

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shard):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        return hash((type(self), self.dim))

    def __repr__(self) -> str:
        return f"Shard({self.dim})"

    @override
    def compute_local_shape(
        self, global_shape: torch.Size, rank: int, world_size: int
    ) -> torch.Size:
        dim_size = global_shape[self.dim]
        chunk = (dim_size + world_size - 1) // world_size
        start = chunk * rank
        local_dim = min(dim_size, start + chunk) - min(dim_size, start)
        shape = list(global_shape)
        shape[self.dim] = local_dim
        return torch.Size(shape)

    @override
    def extract_local_shard(
        self,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        chunks = list(torch.chunk(param, world_size, dim=self.dim))
        empty_shape = list(param.shape)
        empty_shape[self.dim] = 0
        while len(chunks) < world_size:
            chunks.append(param.new_empty(empty_shape))
        return chunks[rank]

    @override
    def bucket_storage_layout(
        self,
        named_params: list[tuple[str, nn.Parameter]],
        param_placements: dict[str, tuple[Placement, ...]],
        mesh: DeviceMesh,
    ) -> BucketStorageLayout | None:
        if self.dim != 0:
            raise NotImplementedError(
                "Shard bucket storage currently supports only Shard(0)."
            )

        rank = mesh.get_local_rank()
        world_size = mesh.size()
        param_layouts: dict[str, BucketParamStorageLayout] = {}
        byte_offset = 0
        for fqn, param in named_params:
            if param.ndim == 0:
                raise ValueError(
                    f"Shard(0) is invalid for parameter {fqn!r} with scalar shape."
                )
            local_shape = self.compute_local_shape(param.shape, rank, world_size)
            local_numel = local_shape.numel()
            # Rank 0 has the largest chunk; reserve its capacity on every rank.
            padded_numel = self.compute_local_numel(param.shape, 0, world_size)
            storage_nbytes = padded_numel * param.dtype.itemsize
            param_layouts[fqn] = BucketParamStorageLayout(
                local_shape=local_shape,
                local_numel=local_numel,
                byte_offset=byte_offset,
                storage_nbytes=storage_nbytes,
            )
            byte_offset += storage_nbytes

        return BucketStorageLayout(
            param_layouts=param_layouts,
            total_bytes=byte_offset,
        )

    @override
    def copy_param_to_storage(
        self,
        byte_storage: torch.Tensor,
        info: ParamInfo,
        param: torch.Tensor,
        rank: int,
        world_size: int,
    ) -> None:
        if param.device.type != "meta":
            data_nbytes = info.local_numel * info.dtype.itemsize
            padding_nbytes = info.storage_nbytes - data_nbytes
            if padding_nbytes > 0:
                byte_storage[
                    info.byte_offset + data_nbytes : info.byte_offset
                    + info.storage_nbytes
                ].zero_()
        super().copy_param_to_storage(
            byte_storage,
            info,
            param,
            rank,
            world_size,
        )

    def _padded_unshard_layout(
        self,
        infos: list[ParamInfo],
    ) -> Shard._PaddedUnshardLayout:
        itemsize = infos[0].dtype.itemsize
        padded_local_numels = tuple(info.storage_nbytes // itemsize for info in infos)
        return Shard._PaddedUnshardLayout(
            padded_local_numels=padded_local_numels,
        )

    def _get_padded_bucket_storage_view(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        layout: Shard._PaddedUnshardLayout,
    ) -> torch.Tensor:
        """Return the padded bucket-storage alias used as all-gather input."""
        itemsize = tensors[0].dtype.itemsize
        bucket_byte_offset = infos[0].byte_offset

        return torch.as_strided(
            tensors[0],
            (sum(layout.padded_local_numels),),
            (1,),
            storage_offset=bucket_byte_offset // itemsize,
        )

    def _copy_out_dim0_unshard(
        self,
        gathered: torch.Tensor,
        infos: list[ParamInfo],
        world_size: int,
        layout: Shard._PaddedUnshardLayout,
    ) -> list[torch.Tensor]:
        full_params: list[torch.Tensor] = []
        split_out: list[torch.Tensor] = []
        for info, padded_local_numel in zip(
            infos,
            layout.padded_local_numels,
            strict=True,
        ):
            padded_full_param = torch.empty(
                world_size * padded_local_numel,
                dtype=info.unsharded_dtype,
                device=gathered.device,
            )
            full_param = padded_full_param[: info.global_numel].view(info.global_shape)
            full_params.append(full_param)
            split_out.append(padded_full_param.view(world_size, padded_local_numel))

        torch.split_with_sizes_copy(
            gathered.view(world_size, -1),
            layout.padded_local_numels,
            dim=1,
            out=split_out,
        )
        return full_params

    @override
    def prepare_unshard_bucket(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        """Prepare buffers for the bucket all-gather unshard."""
        if self.dim != 0:
            raise NotImplementedError(
                "Shard bucket all-gather currently supports only Shard(0)."
            )
        ws = mesh.size()
        dtype = infos[0].unsharded_dtype
        device = tensors[0].device

        with _record_copy_in_if_eager():
            padded_layout = self._padded_unshard_layout(infos)
            send_buf = self._get_padded_bucket_storage_view(
                tensors,
                infos,
                padded_layout,
            )
            send_buf = copy_tensor_to_dtype(send_buf, dtype)
            gather_buf = torch.empty(
                ws * send_buf.numel(),
                dtype=dtype,
                device=device,
            )

        return PlacementPreparedUnshard(
            placement=self,
            buffers=[send_buf, gather_buf],
            placement_state=Shard._UnshardState(
                infos=infos,
                world_size=ws,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                padded_layout=padded_layout,
            ),
        )

    @override
    def run_prepared_unshard(self, prepared: PlacementPreparedUnshard) -> None:
        """Launch the prepared bucket all-gather."""
        if not isinstance(prepared.placement_state, Shard._UnshardState):
            raise AssertionError(
                "Expected Shard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        send_buf = prepared.buffers[0]
        with _record_comm_if_eager(
            "FlexShard::all_gather",
            prepared.placement_state.debug_fqn,
        ):
            dist.all_gather_single(
                prepared.buffers[1],
                send_buf,
                group=prepared.placement_state.pg,
            )

    @override
    def finish_prepared_unshard(
        self,
        prepared: PlacementPreparedUnshard,
    ) -> PlacementUnshardResult:
        """Finish the prepared bucket all-gather and assemble full parameters."""
        if not isinstance(prepared.placement_state, Shard._UnshardState):
            raise AssertionError(
                "Expected Shard._UnshardState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        with _record_copy_out_if_eager():
            state = prepared.placement_state
            full_params = self._copy_out_dim0_unshard(
                prepared.buffers[1],
                state.infos,
                state.world_size,
                state.padded_layout,
            )

        return PlacementUnshardResult(full_params=full_params)

    def _pack_reduce_scatter_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        world_size: int,
    ) -> tuple[torch.Tensor, Shard._ReduceGradLayout]:
        dtype = infos[0].grad_reduce_dtype
        device = tensors[0].device
        padded_sizes: list[torch.Size] = []
        for tensor in tensors:
            padded_size = list(tensor.shape)
            padded_size[self.dim] = (
                (tensor.size(self.dim) + world_size - 1) // world_size
            ) * world_size
            padded_sizes.append(torch.Size(padded_size))

        input_numel = sum(s.numel() for s in padded_sizes)
        send_buf = torch.empty(input_numel, dtype=dtype, device=device)
        send_buf_2d = send_buf.view(world_size, -1)
        torch._chunk_cat(tensors, dim=self.dim, num_chunks=world_size, out=send_buf_2d)
        return send_buf, Shard._ReduceGradLayout(padded_sizes)

    def _unpack_reduce_scatter_grad(
        self,
        recv_buf: torch.Tensor,
        infos: list[ParamInfo],
        layout: Shard._ReduceGradLayout,
        rank: int,
        world_size: int,
    ) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []
        flat_offset = 0
        for info, padded_size in zip(infos, layout.padded_sizes, strict=True):
            local_shape = info.placement.compute_local_shape(
                info.global_shape, rank, world_size
            )
            shard = (
                recv_buf.narrow(0, flat_offset, local_shape.numel())
                .view(local_shape)
                .contiguous()
            )
            results.append(shard)
            flat_offset += padded_size.numel() // world_size
        return results

    @override
    def prepare_reduce_grad(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedReduceGrad:
        """Pack full gradients for bucket reduce-scatter."""
        ws = mesh.size()
        with _record_function_if_eager("FlexShard::reduce_scatter_copy_in", debug_fqn):
            send_buf, layout = self._pack_reduce_scatter_grad(tensors, infos, ws)
            if send_buf.numel() % ws != 0:
                raise AssertionError(
                    f"Packed reduce-scatter buffer has {send_buf.numel()} elements, "
                    f"which is not divisible by world size {ws}."
                )
        return PlacementPreparedReduceGrad(
            placement=self,
            buffers=[send_buf],
            placement_state=Shard._ReduceGradState(
                infos=infos,
                layout=layout,
                rank=mesh.get_local_rank(),
                world_size=ws,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                gradient_reduce_op=gradient_reduce_op_from_infos(infos),
            ),
        )

    @override
    def reduce_prepared_grad(
        self,
        prepared: PlacementPreparedReduceGrad,
    ) -> PlacementReduceGradResult:
        """Reduce a prepared gradient request using reduce-scatter."""
        if not isinstance(prepared.placement_state, Shard._ReduceGradState):
            raise AssertionError(
                "Expected Shard._ReduceGradState, "
                f"got {type(prepared.placement_state).__name__}"
            )
        send_buf = prepared.buffers[0]
        recv_buf = torch.empty(
            send_buf.numel() // prepared.placement_state.world_size,
            dtype=send_buf.dtype,
            device=send_buf.device,
        )
        with _record_comm_if_eager(
            "FlexShard::post_backward_reduce",
            prepared.placement_state.debug_fqn,
        ):
            dist.reduce_scatter_tensor(
                output=recv_buf,
                input=send_buf,
                op=prepared.placement_state.gradient_reduce_op,
                group=prepared.placement_state.pg,
            )
        with _record_function_if_eager(
            "FlexShard::reduce_scatter_copy_out",
            prepared.placement_state.debug_fqn,
        ):
            sharded_grads = self._unpack_reduce_scatter_grad(
                recv_buf,
                prepared.placement_state.infos,
                prepared.placement_state.layout,
                prepared.placement_state.rank,
                prepared.placement_state.world_size,
            )
        return PlacementReduceGradResult(sharded_grads, [recv_buf])


def make_shard_placement_fn(dim: int = 0) -> PlacementFn:
    """Return a placement function assigning ``Shard(dim)`` per parameter."""

    def placement_fn(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        del mesh
        return {fqn: (Shard(dim),) for fqn, _ in named_params}

    return placement_fn


def per_param_placements(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
) -> dict[str, tuple[Placement, ...]]:
    """Shard(0) per parameter (FSDP2-style)."""
    return make_shard_placement_fn(0)(named_params, mesh)


__all__ = [
    "make_shard_placement_fn",
    "per_param_placements",
    "Shard",
]
