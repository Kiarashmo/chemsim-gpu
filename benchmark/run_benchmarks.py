#!/usr/bin/env python3
"""Run the engine over several sizes and stages and write results/results.csv.

It parses the timing lines the engine prints and saves columns:
    n, nbits, stage, name, min_ms, avg_ms, gcups
plot_results.py then turns that CSV into the figures.
"""
import argparse
import csv
import os
import re
import subprocess
import sys

LINE = re.compile(
    r"stage\s+(\d+)\s+(.+?)\s+min\s+([\d.]+)\s+ms\s+avg\s+([\d.]+)\s+ms\s+([\d.]+)\s+GCUPS")


def gen_dataset(gen, path, n, nbits, seed):
    subprocess.check_call([sys.executable, gen, "--synthetic", str(n),
                           "--nbits", str(nbits), "--seed", str(seed),
                           "--out", path], stdout=subprocess.DEVNULL)


def run_one(engine, fps, nbits, repeat):
    out = subprocess.check_output(
        [engine, "--lib", fps, "--stage", "all", "--repeat", str(repeat)],
        text=True)
    rows = []
    for line in out.splitlines():
        m = LINE.search(line)
        if m:
            rows.append({
                "stage": int(m.group(1)), "name": m.group(2).strip(),
                "min_ms": float(m.group(3)), "avg_ms": float(m.group(4)),
                "gcups": float(m.group(5)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="./build/chemsim")
    ap.add_argument("--gen", default="preprocessing/generate_fingerprints.py")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--nbits", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--fps-tmp", default="/tmp/bench.fps")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_rows = []
    for nbits in args.nbits:
        for n in args.sizes:
            print(f"== n={n} nbits={nbits} ==")
            gen_dataset(args.gen, args.fps_tmp, n, nbits, seed=1234)
            for r in run_one(args.engine, args.fps_tmp, nbits, args.repeat):
                r.update(n=n, nbits=nbits)
                all_rows.append(r)
                print(f"   stage {r['stage']} {r['name']}: {r['min_ms']:.3f} ms "
                      f"{r['gcups']:.2f} GCUPS")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "nbits", "stage", "name",
                                          "min_ms", "avg_ms", "gcups"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {args.out} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
