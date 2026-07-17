#!/usr/bin/env python3
"""Make the report figures from results/results.csv. Needs matplotlib.

Writes three plots to results/:
  speedup_ladder.png  speedup of each stage over the naive stage 1
  scaling.png         kernel time vs library size N (log-log)
  gcups.png           throughput per stage
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

    plt.figure(figsize=(7, 4))
    bars = plt.bar([str(s) for s in stages], speedups, color="#c0111f")
    for b, s in zip(bars, speedups):
        plt.text(b.get_x() + b.get_width() / 2, s, f"{s:.1f}×",
                 ha="center", va="bottom", fontsize=9)
    plt.xlabel("optimization stage")
    plt.ylabel(f"speedup vs Stage 1 (naive)\nN={biggest}, {nbits}-bit")
    plt.title("ChemSim-GPU optimization ladder")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print("wrote", out)


def scaling(rows, nbits, out):
    by_stage = defaultdict(list)
    for r in rows:
        if r["nbits"] == nbits:
            by_stage[r["stage"]].append((r["n"], r["min_ms"]))
    plt.figure(figsize=(7, 4))
    for stage in sorted(by_stage):
        pts = sorted(by_stage[stage])
        plt.plot([p[0] for p in pts], [p[1] for p in pts],
                 marker="o", label=f"stage {stage}")
    plt.xscale("log", base=2); plt.yscale("log")
    plt.xlabel("library size N"); plt.ylabel("kernel time (ms)")
    plt.title(f"Scaling ({nbits}-bit fingerprints)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out, dpi=140)
    print("wrote", out)


def gcups(rows, out):
    biggest = max(r["n"] for r in rows)
    plt.figure(figsize=(7, 4))
    for nbits in sorted({r["nbits"] for r in rows}):
        sub = sorted([r for r in rows if r["nbits"] == nbits and r["n"] == biggest],
                     key=lambda r: r["stage"])
        plt.plot([r["stage"] for r in sub], [r["gcups"] for r in sub],
                 marker="s", label=f"{nbits}-bit")
    plt.xlabel("optimization stage"); plt.ylabel("GCUPS")
    plt.title(f"Throughput by stage (N={biggest})")
    plt.legend(); plt.tight_layout()
    plt.savefig(out, dpi=140)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/results.csv")
    ap.add_argument("--nbits", type=int, default=2048)
    args = ap.parse_args()

    rows = load(args.csv)
    speedup_ladder(rows, args.nbits, "results/speedup_ladder.png")
    scaling(rows, args.nbits, "results/scaling.png")
    gcups(rows, "results/gcups.png")


if __name__ == "__main__":
    main()
