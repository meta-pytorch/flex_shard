# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dense all-gather references for FlexShard blockwise-quantized placements."""

import torch
from torch.distributed.device_mesh import DeviceMesh

from ..custom_placements.blockwise_quant_bucketed_block_shard import (
    _align_up,
    _VEC_ALIGN_BYTES,
    BlockwiseQuantBucketedBlockShard,
)
from ..custom_placements.utils import foreach_copy_
from ..flex_shard.bucket_storage import ParamInfo
from ..flex_shard.placement_contract import (
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)
from ..flex_shard.utils import (
    _record_copy_in_if_eager,
    _record_copy_out_if_eager,
)


class DenseAllGatherReferenceMixin(BlockwiseQuantBucketedBlockShard):
    """All-gather dense shards over a quantized placement's shard layout.

    Each gathered dense weight is then quantized whole with the placement's own
    quantizer, giving the "all-gather bf16, then block-quantize" reference.
    List this mixin before the concrete placement class.
    """

    def _dense_send_numel_per_rank(
        self,
        metadata: BlockwiseQuantBucketedBlockShard._QuantBucketMetadata,
    ) -> int:
        return _align_up(
            max(metadata.rank_qdata_numels, default=0),
            _VEC_ALIGN_BYTES,
        )

    def _pack_local_dense_chunks(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        rank: int,
        metadata: BlockwiseQuantBucketedBlockShard._QuantBucketMetadata,
    ) -> torch.Tensor:
        dtype = infos[0].unsharded_dtype
        device = next((t.device for t in tensors if t.numel() > 0), tensors[0].device)
        local_dense = torch.empty(
            self._dense_send_numel_per_rank(metadata),
            dtype=dtype,
            device=device,
        )
        payload_end = metadata.rank_qdata_numels[rank]
        if payload_end < local_dense.numel():
            local_dense[payload_end:].zero_()
        tensors_by_fqn = {
            info.fqn: tensor.reshape(info.local_shape)
            for tensor, info in zip(tensors, infos, strict=True)
        }
        infos_by_fqn = {info.fqn: info for info in infos}
        for chunk in metadata.chunks:
            if chunk.rank != rank:
                continue
            info = infos_by_fqn[chunk.fqn]
            tensor = tensors_by_fqn[chunk.fqn]
            expected_numel = self._chunk_numel(chunk, info.global_shape[1])
            if tensor.numel() != expected_numel:
                raise AssertionError(
                    "Dense reference local shard does not match planned chunk for "
                    f"{chunk.fqn!r}: expected {expected_numel} elements, got "
                    f"{tensor.numel()}."
                )
            local_dense[chunk.qdata_offset : chunk.qdata_offset + tensor.numel()].copy_(
                tensor.reshape(-1)
            )
        return local_dense

    def _prepare_local_unshard_payload(
        self,
        tensors: list[torch.Tensor],
        infos: list[ParamInfo],
        mesh: DeviceMesh,
        debug_fqn: str | None,
    ) -> PlacementPreparedUnshard:
        rank = mesh.get_local_rank()
        metadata = self._quant_metadata_from_infos(infos)
        with _record_copy_in_if_eager():
            local_dense = self._pack_local_dense_chunks(
                tensors,
                infos,
                rank,
                metadata,
            )
        return PlacementPreparedUnshard(
            placement=self,
            buffers=[local_dense],
            placement_state=BlockwiseQuantBucketedBlockShard._QuantUnshardState(
                infos=infos,
                pg=mesh.get_group(),
                debug_fqn=debug_fqn,
                metadata=metadata,
            ),
        )

    def _finish_unshard_from_rank_rows(
        self,
        prepared: PlacementPreparedUnshard,
        rank_rows: torch.Tensor,
    ) -> PlacementUnshardResult:
        state = prepared.placement_state
        if not isinstance(state, BlockwiseQuantBucketedBlockShard._QuantUnshardState):
            raise AssertionError(
                "Expected BlockwiseQuantBucketedBlockShard._QuantUnshardState, "
                f"got {type(state).__name__}"
            )
        expected_shape = (
            len(state.metadata.rank_qdata_numels),
            self._dense_send_numel_per_rank(state.metadata),
        )
        if tuple(rank_rows.shape) != expected_shape:
            raise ValueError(
                f"Expected gathered rank rows with shape {expected_shape}, "
                f"got {tuple(rank_rows.shape)}."
            )

        infos_by_fqn = {info.fqn: info for info in state.infos}
        dense_by_fqn = {
            info.fqn: torch.empty(
                info.global_shape,
                dtype=rank_rows.dtype,
                device=rank_rows.device,
            )
            for info in state.infos
        }
        copy_dsts: list[torch.Tensor] = []
        copy_srcs: list[torch.Tensor] = []
        with _record_copy_out_if_eager():
            for chunk in state.metadata.chunks:
                info = infos_by_fqn[chunk.fqn]
                in_dim = info.global_shape[1]
                numel = self._chunk_numel(chunk, in_dim)
                copy_dsts.append(
                    dense_by_fqn[chunk.fqn][chunk.row_start : chunk.row_end, :]
                )
                copy_srcs.append(
                    rank_rows[
                        chunk.rank,
                        chunk.qdata_offset : chunk.qdata_offset + numel,
                    ].view(chunk.row_end - chunk.row_start, in_dim)
                )
            foreach_copy_(copy_dsts, copy_srcs)
        full_params = []
        persistent_buffers: list[torch.Tensor] = []
        buffers_per_param = 1 + self.num_scale_tensors
        for index, info in enumerate(state.infos):
            dense_weight = dense_by_fqn[info.fqn]
            dense_weight.requires_grad_(info.requires_grad)
            qdata, scales = self._quantize_local_chunk(dense_weight)
            quantized = (qdata, *scales)
            if prepared.persistent_buffers is not None:
                # Persistent unshard: refill the previous data and scales.
                previous = prepared.persistent_buffers[
                    index * buffers_per_param : (index + 1) * buffers_per_param
                ]
                for dst, src in zip(previous, quantized, strict=True):
                    dst.copy_(src)
                quantized = tuple(previous)
            persistent_buffers.extend(quantized)
            full_params.append(
                self._make_quantized_weight_checked(
                    quantized[0],
                    tuple(quantized[1:]),
                    orig_dtype=dense_weight.dtype,
                    requires_grad=dense_weight.requires_grad,
                )
            )
        return PlacementUnshardResult(
            full_params=full_params,
            persistent_buffers=persistent_buffers if prepared.persistent else [],
        )
