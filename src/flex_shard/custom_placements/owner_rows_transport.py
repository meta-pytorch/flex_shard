# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Owner rows through equal chunks: symmetric memory inside a host, NCCL's
native all-gather and reduce-scatter across the mesh.

A bucket whose groups each live whole on one owner gathers its owner rows back
to back (see ``mixed_bucket._owner_rows_plan``). The rows have their owners'
lengths, so they're uneven, and the grouped broadcasts and reduces that take
them as they are run well below NCCL's native collectives across hosts. Here
the rows are cut into equal chunks instead, one per rank. Before the
all-gather, each owner writes the slices of its row that lie in other ranks'
chunks straight into those ranks' buffers, which are in symmetric memory and
mapped by every GPU of the host. Bytes for another host go through relays,
split evenly over the ranks of both hosts, and NCCL carries them between
hosts. The reduce runs the native reduce-scatter, then the same moves in
reverse.
"""

from __future__ import annotations

import functools
import itertools
import os
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import DeviceMesh

from ..flex_shard.utils import mesh_rank_hosts
from .utils import foreach_copy_

# Chunk sizes, rail splits and relayed slices round up to this many elements,
# 256 bytes in BF16.
_ALIGN = 128


def _align_up(numel: int) -> int:
    return -(-numel // _ALIGN) * _ALIGN


class OwnerRowsMove(NamedTuple):
    """``numel`` elements starting at ``start`` in the gathered buffer, from
    mesh rank ``src`` to mesh rank ``dst`` (the other way for the reduce)."""

    src: int
    dst: int
    start: int
    numel: int


@dataclass(frozen=True)
class OwnerRowsTransportPlan:
    """How one bucket's owner rows reach equal chunks, in mesh ranks.

    The gathered buffer holds the rows back to back, padded to
    ``gathered_numel``; rank r holds chunk ``[r * chunk_numel, (r + 1) *
    chunk_numel)``. Each owner keeps the part of its row inside its own chunk.
    For the gather, the rest moves:
    - ``owner_moves``, inside the owner's host: to the rank holding that
      chunk, or to a relay when that rank is on another host;
    - ``host_moves``, between hosts: rank j of one host to rank j of the
      other, each host pair's bytes split evenly over their ranks;
    - ``relay_moves``, inside the holder's host: relay to holder.
    The reduce makes the same moves in reverse. Each rank's buffer holds its
    chunk, then the slices it relays (see ``buffer_offset``)."""

    chunk_numel: int
    gathered_numel: int
    owner_moves: tuple[OwnerRowsMove, ...]
    host_moves: tuple[OwnerRowsMove, ...]
    relay_moves: tuple[OwnerRowsMove, ...]
    relay_offsets: dict[tuple[int, int], int]
    buffer_numel: int

    def buffer_offset(self, rank: int, start: int) -> int:
        """Where the slice starting at ``start`` sits in ``rank``'s buffer:
        at its place inside the rank's chunk, else after the chunk."""
        if rank * self.chunk_numel <= start < (rank + 1) * self.chunk_numel:
            return start - rank * self.chunk_numel
        return self.relay_offsets[(rank, start)]


@functools.lru_cache(maxsize=None)
def plan_owner_rows_transport(
    row_numels: tuple[int, ...], hosts: tuple[int, ...]
) -> OwnerRowsTransportPlan:
    """The moves for owner rows of ``row_numels`` on mesh ranks on ``hosts``
    (see ``mesh_rank_hosts``). Every rank computes the same plan."""
    world_size = len(row_numels)
    if len(hosts) != world_size:
        raise ValueError(f"hosts must give each of {world_size} ranks a host.")
    chunk = _align_up(-(-sum(row_numels) // world_size))
    members: dict[int, list[int]] = defaultdict(list)
    for rank, host in enumerate(hosts):
        members[host].append(rank)

    owner_moves: list[OwnerRowsMove] = []
    host_moves: list[OwnerRowsMove] = []
    relay_moves: list[OwnerRowsMove] = []
    leaving: dict[tuple[int, int], list[OwnerRowsMove]] = defaultdict(list)
    for owner, (row_start, row_numel) in enumerate(
        zip(itertools.accumulate(row_numels, initial=0), row_numels)
    ):
        position, end = row_start, row_start + row_numel
        while position < end:
            holder = position // chunk
            stop = min(end, (holder + 1) * chunk)
            if holder != owner:
                move = OwnerRowsMove(owner, holder, position, stop - position)
                if hosts[owner] == hosts[holder]:
                    owner_moves.append(move)
                else:
                    leaving[(hosts[owner], hosts[holder])].append(move)
            position = stop
    for (src_host, dst_host), moves in leaving.items():
        # Rank j of each host relays the j-th part, one network card each.
        rails = min(len(members[src_host]), len(members[dst_host]))
        part = _align_up(-(-sum(move.numel for move in moves) // rails))
        sent = 0
        for owner, holder, start, numel in moves:
            while numel:
                rail = sent // part
                take = min(numel, (rail + 1) * part - sent)
                relay_out = members[src_host][rail]
                relay_in = members[dst_host][rail]
                if relay_out != owner:
                    owner_moves.append(OwnerRowsMove(owner, relay_out, start, take))
                host_moves.append(OwnerRowsMove(relay_out, relay_in, start, take))
                if relay_in != holder:
                    relay_moves.append(OwnerRowsMove(relay_in, holder, start, take))
                start, numel, sent = start + take, numel - take, sent + take

    relay_offsets: dict[tuple[int, int], int] = {}
    relay_numels: dict[int, int] = defaultdict(int)

    def hold(rank: int, start: int, numel: int) -> None:
        if rank * chunk <= start < (rank + 1) * chunk or (rank, start) in relay_offsets:
            return
        relay_offsets[(rank, start)] = chunk + relay_numels[rank]
        relay_numels[rank] += _align_up(numel)

    for move in owner_moves:
        hold(move.dst, move.start, move.numel)
    for move in host_moves:
        hold(move.src, move.start, move.numel)
        hold(move.dst, move.start, move.numel)
    for move in relay_moves:
        hold(move.src, move.start, move.numel)
    return OwnerRowsTransportPlan(
        chunk_numel=chunk,
        gathered_numel=world_size * chunk,
        owner_moves=tuple(owner_moves),
        host_moves=tuple(host_moves),
        relay_moves=tuple(relay_moves),
        relay_offsets=relay_offsets,
        buffer_numel=chunk + max(relay_numels.values(), default=0),
    )


@dataclass
class SymmetricBuffer:
    """A buffer in symmetric memory and every host rank's buffer mapped here,
    by mesh rank. ``readers`` are asynchronous collectives that may still read
    it."""

    tensor: torch.Tensor
    handle: Any
    peers: dict[int, torch.Tensor]
    readers: list[dist.Work] = field(default_factory=list)

    def barrier(self) -> None:
        """Wait on this stream until every rank of the host reaches its
        barrier: its earlier writes into others' buffers are then visible."""
        self.handle.barrier(channel=0)

    def wait_readers(self) -> None:
        """Make this stream wait for the collectives still reading it."""
        for work in self.readers:
            work.wait()
        self.readers.clear()


class OwnerRowsTransportBuffers:
    """A mesh's symmetric buffers on this rank, by use and dtype, and the gloo
    group of this rank's host that their rendezvous runs over.

    A gloo group needs no device memory; an eagerly built NCCL group would take
    about 230 MiB. Each buffer grows to the largest request. A rendezvous is
    collective over the host, so every rank of the host must request a buffer
    at the same point, as SPMD code does for each bucket."""

    def __init__(self, mesh: DeviceMesh, hosts: Sequence[int]) -> None:
        if mesh.ndim != 1:
            raise ValueError(f"Owner rows need a 1-D mesh, got {mesh.ndim} dims.")
        if len(hosts) != mesh.size():
            raise ValueError(f"hosts must give each of {mesh.size()} ranks a host.")
        self._global_ranks = mesh.mesh.flatten().tolist()
        rank = mesh.get_local_rank()
        self.host_ranks = tuple(
            other for other, host in enumerate(hosts) if host == hosts[rank]
        )
        self.group = dist.new_group(
            [self._global_ranks[other] for other in self.host_ranks],
            backend="gloo",
            use_local_synchronization=True,
        )
        self._buffers: dict[tuple[str, torch.dtype], SymmetricBuffer] = {}

    def get(
        self, use: str, dtype: torch.dtype, numel: int, device: torch.device
    ) -> SymmetricBuffer:
        """This rank's buffer for ``use`` (e.g. "gather" or "reduce") with at
        least ``numel`` elements of ``dtype``."""
        key = (use, dtype)
        buffer = self._buffers.get(key)
        if buffer is not None and buffer.tensor.numel() >= numel:
            return buffer
        if buffer is not None:
            # Collectives on any stream may still read the old buffer.
            torch.cuda.synchronize(device)
        tensor = symm_mem.empty(numel, dtype=dtype, device=device)
        handle = symm_mem.rendezvous(tensor, self.group)
        peers = {
            other: handle.get_buffer(
                dist.get_group_rank(self.group, self._global_ranks[other]),
                (numel,),
                dtype,
            )
            for other in self.host_ranks
        }
        buffer = SymmetricBuffer(tensor, handle, peers)
        self._buffers[key] = buffer
        return buffer


_buffers_by_mesh: dict[tuple[str, tuple[int, ...]], OwnerRowsTransportBuffers] = {}


def owner_rows_transport_buffers(
    mesh: DeviceMesh, hosts: Sequence[int]
) -> OwnerRowsTransportBuffers:
    """The mesh's transport buffers on this rank, created on first use."""
    key = (mesh.get_group().group_name, tuple(hosts))
    buffers = _buffers_by_mesh.get(key)
    if buffers is None:
        buffers = _buffers_by_mesh[key] = OwnerRowsTransportBuffers(mesh, hosts)
    return buffers


_TRANSPORT_ENV = "FLEX_SHARD_OWNER_ROWS_TRANSPORT"


def owner_rows_transport_plan(
    mesh: DeviceMesh, row_numels: Sequence[int], device: torch.device
) -> tuple[OwnerRowsTransportPlan, tuple[int, ...]] | None:
    """The transport plan and the mesh ranks' hosts for a bucket's owner rows,
    or None where the grouped broadcasts and reduces take them as they are:
    - under torch.compile;
    - with equal rows, which already take one native collective;
    - off CUDA, or on a mesh with no host holding two of its ranks, where
      symmetric memory has nothing to share;
    - with ``FLEX_SHARD_OWNER_ROWS_TRANSPORT=grouped``.
    """
    choice = os.environ.get(_TRANSPORT_ENV, "chunks")
    if choice not in ("chunks", "grouped"):
        raise ValueError(f"{_TRANSPORT_ENV} must be chunks or grouped, got {choice!r}.")
    if (
        choice == "grouped"
        or torch.compiler.is_compiling()
        or len(set(row_numels)) == 1
        or device.type != "cuda"
        or mesh.ndim != 1
    ):
        return None
    hosts = mesh_rank_hosts(mesh.mesh.flatten().tolist())
    if max(Counter(hosts).values()) < 2:
        return None
    return plan_owner_rows_transport(tuple(row_numels), hosts), hosts


def _subtract(
    start: int, numel: int, holes: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    out, position = [], start
    for hole_start, hole_numel in sorted(holes):
        if hole_start > position:
            out.append((position, hole_start - position))
        position = max(position, hole_start + hole_numel)
    if position < start + numel:
        out.append((position, start + numel - position))
    return out


def _host_p2p(
    plan: OwnerRowsTransportPlan,
    rank: int,
    buffer: torch.Tensor,
    pg: dist.ProcessGroup,
    *,
    reverse: bool,
) -> None:
    """The moves between hosts, buffer to buffer, in one NCCL group; the
    current stream then waits for them."""
    ops = []
    for move in plan.host_moves:
        src, dst = (move.dst, move.src) if reverse else (move.src, move.dst)
        for peer_op, me, peer in ((dist.isend, src, dst), (dist.irecv, dst, src)):
            if me == rank:
                offset = plan.buffer_offset(rank, move.start)
                ops.append(
                    dist.P2POp(
                        peer_op,
                        buffer[offset : offset + move.numel],
                        group=pg,
                        group_peer=peer,
                    )
                )
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()


def gather_owner_rows(
    plan: OwnerRowsTransportPlan,
    hosts: Sequence[int],
    buffers: OwnerRowsTransportBuffers,
    *,
    rank: int,
    row_start: int,
    row_numel: int,
    sources: Sequence[tuple[torch.Tensor, int]],
    gathered: torch.Tensor,
    pg: dist.ProcessGroup,
    async_op: bool = False,
) -> dist.Work | None:
    """All-gather owner rows into ``gathered`` (``plan.gathered_numel`` long).

    This rank's row is ``[row_start, row_start + row_numel)``; ``sources`` are
    its params with their starts in the gathered buffer, in their storage.
    Each owner writes its row's slices straight from that storage into the
    buffers of the ranks whose chunks hold them, or of relays for other
    hosts, casting to ``gathered``'s dtype on the way; its own chunk's part
    goes into its own buffer. Then NCCL carries the relayed slices between
    hosts, relays write them into their holders' buffers, and NCCL's
    all-gather reads each rank's chunk. Positions no source covers are
    padding and stay unwritten."""
    buffer = buffers.get("gather", gathered.dtype, plan.buffer_numel, gathered.device)
    buffer.wait_readers()
    writes: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])

    def write(target: int, start: int, numel: int) -> None:
        base = plan.buffer_offset(target, start)
        for tensor, tensor_start in sources:
            lo = max(start, tensor_start)
            hi = min(start + numel, tensor_start + tensor.numel())
            if lo < hi:
                writes[0].append(
                    buffer.peers[target][base + lo - start : base + hi - start]
                )
                writes[1].append(
                    tensor.reshape(-1)[lo - tensor_start : hi - tensor_start]
                )

    row_end = row_start + row_numel
    sent = []
    for move in plan.owner_moves:
        if move.src == rank:
            write(move.dst, move.start, move.numel)
            sent.append((move.start, move.numel))
    for move in plan.host_moves:
        if move.src == rank and row_start <= move.start < row_end:
            # This rank relays its own slice: staged in its own buffer.
            write(rank, move.start, move.numel)
            sent.append((move.start, move.numel))
    for start, numel in _subtract(row_start, row_numel, sent):
        write(rank, start, numel)
    # Neighbors are done with what their buffers held, then the writes.
    buffer.barrier()
    if writes[0]:
        foreach_copy_(writes[0], writes[1])
    buffer.barrier()
    _host_p2p(plan, rank, buffer.tensor, pg, reverse=False)
    relays = [
        (
            buffer.peers[move.dst][plan.buffer_offset(move.dst, move.start) :][
                : move.numel
            ],
            buffer.tensor[plan.buffer_offset(rank, move.start) :][: move.numel],
        )
        for move in plan.relay_moves
        if move.src == rank
    ]
    if relays:
        foreach_copy_([dst for dst, _ in relays], [src for _, src in relays])
    if any(hosts[move.dst] == hosts[rank] for move in plan.relay_moves):
        buffer.barrier()
    work = dist.all_gather_into_tensor(
        gathered, buffer.tensor[: plan.chunk_numel], group=pg, async_op=async_op
    )
    if work is not None:
        buffer.readers.append(work)
    return work
