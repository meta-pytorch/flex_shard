# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Draw the README's figures: ``python docs/images/make_figures.py``."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
INK, MUTED = "#1f2328", "#59636e"
FSDP2_EDGE, FSDP2_FILL = "#d4651f", "#fbe3d4"
FLEX_EDGE, FLEX_FILL = "#1f8a5b", "#d3f0e2"
TILE_EDGE, TILE_FILL = "#4a7fd4", "#dce8fa"
CUT = "#1f2328"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path"})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.svg"), facecolor="white")
    plt.close(fig)


def banner(ax, x, y, w, text, edge, fill):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            0.5,
            boxstyle="round,pad=0,rounding_size=0.12",
            ec=edge,
            fc=fill,
            lw=1.6,
        )
    )
    ax.text(x + w / 2, y + 0.25, text, ha="center", va="center", fontsize=11, color=INK)


def muon_whole_matrices():
    """FSDP2's Shard(0) scatters a matrix over all ranks; Owned keeps it on one."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    banner(
        ax,
        0.3,
        8.4,
        9.2,
        "FSDP2: Shard(0), a slice on every rank",
        FSDP2_EDGE,
        FSDP2_FILL,
    )
    banner(
        ax,
        10.5,
        8.4,
        9.2,
        "FlexShard: Owned, one rank holds it all",
        FLEX_EDGE,
        FLEX_FILL,
    )
    for rank in range(8):
        y = 7.3 - rank * 0.9
        ax.add_patch(Rectangle((2.9, y), 4, 0.8, ec=FSDP2_EDGE, fc=FSDP2_FILL, lw=1.2))
        ax.text(
            4.9,
            y + 0.4,
            f"rank {rank}   896 x 2048",
            ha="center",
            va="center",
            fontsize=9.5,
            color=INK,
        )
    ax.add_patch(Rectangle((12.9, 0.5), 4.4, 7.6, ec=FLEX_EDGE, fc=FLEX_FILL, lw=1.6))
    ax.text(15.1, 4.6, "rank 3", ha="center", va="center", fontsize=11, color=INK)
    ax.text(
        15.1,
        3.9,
        "the whole [7168, 2048]",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
    )
    ax.text(
        4.9,
        0.05,
        "Muon needs the whole matrix: gather it, then send the update back",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )
    ax.text(
        15.1,
        0.05,
        "rank 3 runs Muon on it locally",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "muon_whole_matrices")


def tile_grid(ax, x0, y0, cuts, split_rows):
    rows, cols, size = 6, 4, 0.9
    for r in range(rows):
        for c in range(cols):
            fill = FSDP2_FILL if r in split_rows else TILE_FILL
            edge = FSDP2_EDGE if r in split_rows else TILE_EDGE
            ax.add_patch(
                Rectangle(
                    (x0 + c * size, y0 + (rows - 1 - r) * size),
                    size,
                    size,
                    ec=edge,
                    fc=fill,
                    lw=1,
                )
            )
    for cut in cuts:  # in tile rows from the top
        y = y0 + (rows - cut) * size
        ax.plot([x0 - 0.25, x0 + cols * size + 0.25], [y, y], color=CUT, lw=2.6)


def block_shard_tiles():
    """Shard(0) cuts quantization tiles across ranks; BlockShard cuts between them."""
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.6)
    ax.axis("off")
    banner(
        ax, 0.3, 7.8, 9.2, "Shard(0): equal shards cut tiles", FSDP2_EDGE, FSDP2_FILL
    )
    banner(ax, 10.5, 7.8, 9.2, "BlockShard: whole tiles per rank", FLEX_EDGE, FLEX_FILL)
    tile_grid(ax, 3.1, 1.6, cuts=[1.5, 3, 4.5], split_rows={1, 4})
    tile_grid(ax, 13.3, 1.6, cuts=[2, 4, 5], split_rows=set())
    ax.text(
        4.9,
        0.6,
        "a cut tile has no rank with all of it: its scale needs an amax all-reduce",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )
    ax.text(
        15.1,
        0.6,
        "each rank quantizes its own tiles",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, "block_shard_tiles")


def node(ax, x, y, w, text, kind):
    styles = {
        "bf16": ("round,pad=0,rounding_size=0.15", "#2f6fd0", "#cfe1fb"),
        "fp8": ("round,pad=0,rounding_size=0.15", "#1f8a5b", "#cdeedd"),
        "comm": ("round,pad=0,rounding_size=0.45", "#c99a06", "#fde9a6"),
        "compute": ("round,pad=0,rounding_size=0.45", "#6e7781", "#e8eaed"),
    }
    style, edge, fill = styles[kind]
    ax.add_patch(
        FancyBboxPatch((x, y), w, 0.9, boxstyle=style, ec=edge, fc=fill, lw=1.4)
    )
    ax.text(
        x + w / 2, y + 0.45, text, ha="center", va="center", fontsize=9.5, color=INK
    )
    return x + w


def arrow(ax, x0, x1, y):
    ax.annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3),
    )


def chain(ax, y, steps, gap=0.45):
    x = 0.3
    for i, (text, kind, w) in enumerate(steps):
        if i:
            arrow(ax, x, x + gap, y + 0.45)
            x += gap
        x = node(ax, x, y, w, text, kind)


def fp8_all_gather():
    """Three ways to all-gather a weight for an FP8 GEMM."""
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.set_xlim(0, 18.6)
    ax.set_ylim(0, 10.4)
    ax.axis("off")
    rows = [
        (
            "BF16 all-gather, then quantize",
            [
                ("BF16 Shard(0)", "bf16", 2.6),
                ("BF16 all-gather", "comm", 2.9),
                ("BF16 full weight", "bf16", 2.9),
                ("quantize", "compute", 2.3),
                ("FP8 full weight", "fp8", 2.8),
                ("GEMM", "compute", 2.0),
            ],
        ),
        (
            "FSDP2: all-reduce the amax, all-gather FP8",
            [
                ("BF16 Shard(0)", "bf16", 2.6),
                ("all-reduce amax", "comm", 2.9),
                ("quantize", "compute", 2.3),
                ("FP8 all-gather", "comm", 2.8),
                ("FP8 full weight", "fp8", 2.8),
                ("GEMM", "compute", 2.0),
            ],
        ),
        (
            "FlexShard: quantize whole tiles locally, all-gather FP8",
            [
                ("BF16 BlockShard", "bf16", 2.9),
                ("quantize", "compute", 2.3),
                ("FP8 all-gather", "comm", 2.8),
                ("FP8 full weight", "fp8", 2.8),
                ("GEMM", "compute", 2.0),
            ],
        ),
    ]
    for i, (title, steps) in enumerate(rows):
        y = 8.0 - i * 3.4
        ax.text(0.3, y + 1.35, title, fontsize=11.5, color=INK, fontweight="bold")
        chain(ax, y, steps)
    # Under the FlexShard row's quantize and FP8 all-gather nodes.
    ax.text(
        3.9,
        0.75,
        "no amax all-reduce",
        ha="center",
        va="center",
        fontsize=9.5,
        color=FLEX_EDGE,
        fontweight="bold",
    )
    ax.text(
        8.9,
        0.6,
        "half the bytes of BF16,\nplus one scale per tile",
        ha="center",
        va="center",
        fontsize=9.5,
        color=FLEX_EDGE,
        fontweight="bold",
    )
    save(fig, "fp8_all_gather")


# A made-up example: w[r][b] is rank r's Newton-Schulz work in bucket b.
WORK = [[6, 1, 1], [1, 5, 0], [1, 2, 1], [0, 1, 4]]
BUCKET_COLORS = ["#2f6fd0", "#2a9d8f", "#7b5cd6"]
EXCHANGE, WAIT = "#e07b39", "#e5534b"


def dist_muon_timeline():
    """FSDP2 + DistMuon waits for each bucket's slowest owner; FlexShard + DistMuon
    runs each rank's own matrices."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    exchange = 0.5
    bucket_max = [max(rank[b] for rank in WORK) for b in range(3)]
    for ax in (top, bottom):
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"rank {r}" for r in range(4)], fontsize=10, color=INK)
        ax.set_ylim(3.6, -0.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    t = 0.0
    for b in range(3):
        for r in range(4):
            top.barh(r, exchange, left=t, color=EXCHANGE, height=0.6)
            top.barh(
                r,
                exchange,
                left=t + exchange + bucket_max[b],
                color=EXCHANGE,
                height=0.6,
            )
            if WORK[r][b]:
                top.barh(
                    r, WORK[r][b], left=t + exchange, color=BUCKET_COLORS[b], height=0.6
                )
            idle = bucket_max[b] - WORK[r][b]
            if idle:
                top.barh(
                    r,
                    idle,
                    left=t + exchange + WORK[r][b],
                    color="white",
                    height=0.6,
                    edgecolor=WAIT,
                    hatch="////",
                    linewidth=0,
                )
        t += 2 * exchange + bucket_max[b]
    top.axvline(t, color=WAIT, ls="--", lw=1.4)
    top.text(
        t + 0.2,
        1.5,
        "sum over buckets of\nthe slowest owner:\n6 + 5 + 4 = 15,\nplus 6 exchanges",
        fontsize=9.5,
        color=INK,
        va="center",
    )
    top.set_title(
        "FSDP2 + DistMuon: each bucket's exchange waits for its slowest owner",
        fontsize=11.5,
        color=INK,
        loc="left",
        fontweight="bold",
    )
    totals = [sum(rank) for rank in WORK]
    for r in range(4):
        left = 0.0
        for b in range(3):
            if WORK[r][b]:
                bottom.barh(
                    r, WORK[r][b], left=left, color=BUCKET_COLORS[b], height=0.6
                )
                left += WORK[r][b]
        bottom.text(
            left + 0.15, r, str(totals[r]), va="center", fontsize=9.5, color=INK
        )
    bottom.axvline(max(totals), color=MUTED, ls="--", lw=1.4)
    bottom.text(
        max(totals) + 0.7,
        1.5,
        "the slowest rank's\nown total: 8",
        fontsize=9.5,
        color=INK,
        va="center",
    )
    bottom.set_title(
        "FlexShard + DistMuon: each rank runs only the matrices it stores",
        fontsize=11.5,
        color=INK,
        loc="left",
        fontweight="bold",
    )
    bottom.set_xlabel(
        "time, in units of Newton-Schulz work (a made-up example)",
        fontsize=10,
        color=INK,
    )
    bottom.set_xlim(0, 22)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BUCKET_COLORS]
    handles += [
        plt.Rectangle((0, 0), 1, 1, color=EXCHANGE),
        plt.Rectangle((0, 0), 1, 1, fc="white", ec=WAIT, hatch="////"),
    ]
    fig.legend(
        handles,
        ["bucket 1", "bucket 2", "bucket 3", "all-to-all exchange", "waiting"],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "dist_muon_timeline")


if __name__ == "__main__":
    muon_whole_matrices()
    block_shard_tiles()
    fp8_all_gather()
    dist_muon_timeline()
