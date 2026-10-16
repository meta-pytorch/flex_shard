# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dense all-gather reference for the blockwise FP8 placement A/B tests.

Not for landing: it exists to check that FP8 all-gather trains bitwise the same
as all-gathering bf16 and quantizing afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from typing_extensions import override

from ..flex_shard.placement_contract import (
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from ..flex_shard.utils import _record_copy_in_if_eager, _record_copy_out_if_eager
from .fp8_bucketed_block_shard import (
    _align_up,
    _matrix_shape,
    _quantize_dense_weight_to_blockwise_fp8,
    _scale_shape,
    _VEC_ALIGN_BYTES,
    BlockwiseFp8WeightFactory,
    Fp8BucketedBlockShard,
)
from .utils import foreach_copy_


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from ..flex_shard.bucket_storage import ParamInfo, PlacementFn
    from ..flex_shard.placement_contract import Placement


def make_dense_all_gather_reference_placement_fn(
    *,
    weight_factory: BlockwiseFp8WeightFactory,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> PlacementFn:
    """Assign a dense reference with the FP8 placement's block-row layout."""

    def dense_all_gather_reference_placements(
        named_params: list[tuple[str, nn.Parameter]],
        mesh: DeviceMesh,
    ) -> dict[str, tuple[Placement, ...]]:
        placement = DenseAllGatherReferencePlacement(
            world_size=mesh.size(),
            weight_factory=weight_factory,
            block_size=block_size,
            fp8_dtype=fp8_dtype,
        )
        return {fqn: (placement,) for fqn, _ in named_params}

    return dense_all_gather_reference_placements


class DenseAllGatherReferencePlacement(Fp8BucketedBlockShard):
    """All-gather dense shards using the FP8 placement's exact shard layout.

    It inherits the FP8 placement's block-row planner, storage layout, and grad
    reduce-scatter, so sharding, initialization and grads are identical. Only
    the unshard differs: each rank sends its dense rows, and the full
    parameters are quantized after the all-gather into the FP8 placement's
    compact buffer, which persistent unshards keep.
    """

    def __repr__(self) -> str:
        return (
            "DenseAllGatherReferencePlacement("
            f"world_size={self.world_size}, "
            f"block_size={self.block_size})"
        )

    @override
    def _prepare_local_unshard_payload(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        rank = mesh.get_local_rank()
        metadata = self._fp8_metadata_from_infos(infos)
        pack_plan = metadata.rank_pack_plans[rank]
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        # Each rank sends its dense rows in the order its FP8 bytes would take,
        # padded to the largest rank's payload.
        local_dense = torch.empty(
            _align_up(max(metadata.rank_fp8_numels), _VEC_ALIGN_BYTES),
            dtype=infos[0].unsharded_dtype,
            device=device,
        )
        local_dense[pack_plan.fp8_numel :].zero_()
        dsts, srcs = [], []
        offset = 0
        for chunk in pack_plan.chunks:
            dsts.append(local_dense[offset : offset + chunk.fp8_numel])
            srcs.append(tensors[chunk.param_index].reshape(-1))
            offset += chunk.fp8_numel
        with _record_copy_in_if_eager():
            foreach_copy_(dsts, srcs)
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_dense],
            placement_state=Fp8BucketedBlockShard._Fp8UnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                metadata=metadata,
            ),
        )

    @override
    def _finish_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
    ) -> PlacementUnshardResult:
        state = prepared.placement_state
        if not isinstance(state, Fp8BucketedBlockShard._Fp8UnshardState):
            raise AssertionError(
                "Expected Fp8BucketedBlockShard._Fp8UnshardState, "
                f"got {type(state).__name__}"
            )
        metadata = state.metadata
        dense_dtype = state.infos[0].unsharded_dtype
        if rank_rows.dtype != dense_dtype:
            raise ValueError(
                f"Expected gathered dense rank rows of dtype {dense_dtype}, "
                f"got {rank_rows.dtype}."
            )
        # The FP8 placement's compact layout: fp8 data, then aligned scales.
        fp8_nbytes = sum(metadata.rank_fp8_numels)
        scale_nbytes = sum(metadata.rank_scale_numels) * torch.float32.itemsize
        scale_byte_offset = _align_up(fp8_nbytes, _VEC_ALIGN_BYTES)
        compact = (
            prepared.persistent_buffers[0]
            if prepared.persistent_buffers is not None
            else torch.empty(
                scale_byte_offset + scale_nbytes,
                dtype=torch.uint8,
                device=rank_rows.device,
            )
        )
        fp8_flat = compact[:fp8_nbytes].view(self.fp8_dtype)
        scale_flat = compact[scale_byte_offset:].view(torch.float32)
        with _record_copy_out_if_eager():
            # Rank cuts partition one parameter-ordered block-row stream, so
            # joining the ranks' rows restores the full parameters.
            dense_flat = rank_rows.new_empty(fp8_nbytes)
            if not self._copy_out_rank_rows_triton(
                metadata,
                rank_rows.view(torch.uint8),
                dense_flat.view(torch.uint8),
                data_elem_nbytes=rank_rows.element_size(),
                with_scales=False,
            ):
                torch.cat(
                    [
                        rank_rows[rank, :numel]
                        for rank, numel in enumerate(metadata.rank_fp8_numels)
                    ],
                    out=dense_flat,
                )
            full_params: list[torch.Tensor] = []
            offset = 0
            scale_offset = 0
            for info in state.infos:
                matrix_shape = _matrix_shape(info.global_shape)
                scale_shape = _scale_shape(matrix_shape, self.block_size)
                scale_numel = scale_shape[0] * scale_shape[1]
                fp8_data, recip_scale = _quantize_dense_weight_to_blockwise_fp8(
                    dense_flat[offset : offset + info.global_numel].view(matrix_shape),
                    self.block_size,
                    self.fp8_dtype,
                )
                fp8_view = fp8_flat[offset : offset + info.global_numel]
                scale_view = scale_flat[scale_offset : scale_offset + scale_numel]
                fp8_view.view(matrix_shape).copy_(fp8_data)
                scale_view.view(scale_shape).copy_(recip_scale)
                full_params.append(
                    self._make_blockwise_fp8_weight(
                        fp8_view.view(info.global_shape),
                        scale_view.view(scale_shape),
                        orig_dtype=info.unsharded_dtype,
                        requires_grad=info.requires_grad,
                    )
                )
                offset += info.global_numel
                scale_offset += scale_numel
        if offset != fp8_flat.numel() or scale_offset != scale_flat.numel():
            raise AssertionError("Dense reference bucket metadata is inconsistent.")
        return PlacementUnshardResult(
            full_params=full_params,
            persistent_buffers=[compact] if prepared.persistent else [],
        )


__all__ = [
    "DenseAllGatherReferencePlacement",
    "make_dense_all_gather_reference_placement_fn",
]
