#!/usr/bin/env python3
"""Make the report figures from results/results.csv. Needs matplotlib.

Writes three plots to results/:
  speedup_ladder.png  speedup of each stage over the naive stage 1
  scaling.png         kernel time vs library size N (log-log)
  gcups.png           throughput per stage

All figures share one TU-Berlin theme: the Mulish typeface when it is
installed (otherwise a default sans), TU red for the headline series, and greys
for the rest.
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- TU Berlin palette -------------------------------------------------------
TU_RED = "#C40D1E"
TU_DARK = "#222222"
TU_GREY = "#6E6E6E"
TU_LINE = "#D2D2D2"

# Stage colours: baselines in grey, the coalescing win in TU red, then a small
# distinct set (amber, blue) so no two stages read alike, ending near-black.
STAGE_COLORS = {1: "#B4B4B4", 2: "#6E6E6E", 3: TU_RED,
                4: "#E58A00", 5: "#2E7EB0", 6: TU_DARK}


def setup_theme():
    """Set shared rcParams; use Mulish if it is installed, else a default sans."""
    have_mulish = any("Mulish" in f.name for f in font_manager.fontManager.ttflist)
    family = "Mulish" if have_mulish else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family,
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "axes.labelcolor": TU_DARK,
        "axes.edgecolor": TU_DARK,
        "axes.linewidth": 1.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TU_DARK,
        "ytick.color": TU_DARK,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.frameon": False,
        "lines.linewidth": 2.4,
        "lines.markersize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "text.color": TU_DARK,
        "grid.color": TU_LINE,
        "grid.linewidth": 0.9,
    })


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"n": int(r["n"]), "nbits": int(r["nbits"]),
                         "stage": int(r["stage"]), "name": r["name"],
                         "min_ms": float(r["min_ms"]), "gcups": float(r["gcups"])})
    return rows


def speedup_ladder(rows, nbits, out):
    biggest = max(r["n"] for r in rows if r["nbits"] == nbits)
    sub = sorted([r for r in rows if r["nbits"] == nbits and r["n"] == biggest],
                 key=lambda r: r["stage"])
    base = next(r["min_ms"] for r in sub if r["stage"] == 1)
    stages = [r["stage"] for r in sub]
    speedups = [base / r["min_ms"] for r in sub]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = [STAGE_COLORS[s] for s in stages]
    bars = ax.bar([str(s) for s in stages], speedups, color=colors, width=0.72)
    peak = max(speedups)
    for b, s in zip(bars, speedups):
        ax.text(b.get_x() + b.get_width() / 2, s + peak * 0.02, f"{s:.1f}×",
                ha="center", va="bottom", fontsize=15,
                fontweight="bold" if s == peak else "normal",
                color=TU_RED if s == peak else TU_DARK)
    ax.axhline(1.0, color=TU_GREY, linewidth=1.0, linestyle=(0, (4, 4)), zorder=0)
    ax.set_xlabel("optimization stage")
    ax.set_ylabel(f"speedup vs stage 1 (naive)\nN={biggest}, {nbits}-bit")
    ax.set_ylim(0, peak * 1.15)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def scaling(rows, nbits, out):
    by_stage = defaultdict(list)
    for r in rows:
        if r["nbits"] == nbits:
            by_stage[r["stage"]].append((r["n"], r["min_ms"]))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for stage in sorted(by_stage):
        pts = sorted(by_stage[stage])
        lw = 3.2 if stage == 3 else 2.2
        z = 5 if stage == 3 else 3
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                color=STAGE_COLORS[stage], linewidth=lw, zorder=z,
                label=f"stage {stage}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("library size N")
    ax.set_ylabel("kernel time (ms)")
    ax.grid(True, which="major", axis="both")
    ax.legend(ncol=2, columnspacing=1.2, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def gcups(rows, out):
    biggest = max(r["n"] for r in rows)
    # Two fingerprint widths, coloured red / grey (not a fresh rainbow).
    width_color = {min({r["nbits"] for r in rows}): TU_RED,
                   max({r["nbits"] for r in rows}): TU_GREY}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for nbits in sorted({r["nbits"] for r in rows}):
        sub = sorted([r for r in rows if r["nbits"] == nbits and r["n"] == biggest],
                     key=lambda r: r["stage"])
        ax.plot([r["stage"] for r in sub], [r["gcups"] for r in sub],
                marker="s", color=width_color[nbits], label=f"{nbits}-bit")
    ax.set_xlabel("optimization stage")
    ax.set_ylabel("throughput (GCUPS)")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--nbits", type=int, default=2048)
    args = ap.parse_args()

    setup_theme()
    rows = load(args.csv)
    speedup_ladder(rows, args.nbits, "results/speedup_ladder.png")
    scaling(rows, args.nbits, "results/scaling.png")
    gcups(rows, "results/gcups.png")


if __name__ == "__main__":
    main()
