# ChemSim-GPU: GPU Molecular Similarity Search

This is a CUDA program that computes Tanimoto similarity between molecular
fingerprints. It is built as a step by step optimization study: start with a
simple, correct GPU kernel, then improve it one technique at a time and measure
the speedup at each step.

## Folder layout

```
include/        headers (fingerprint, cpu reference, kernels)
src/
  fingerprint_io.cpp   .fps file format, popcounts, AoS/SoA conversion
  tanimoto_cpu.cpp     stage 0, the CPU reference we check against
  kernels.cu           stages 1 to 6, the GPU kernels
  main.cu              the driver: run, time, validate, streams, top-k
tests/          CPU-only tests (no GPU needed)
preprocessing/  make .fps files from SMILES (RDKit) or random data
benchmark/      run_benchmarks.py, plot_results.py, validate.py
results/        the plots and results.csv
data/           inputs (example.smi is kept, big files are ignored)
Makefile  setup.sh
```

## The optimization ladder

Each stage is a separate kernel you can pick at run time.

| Stage | What it does | Layout |
|------|--------------|--------|
| 0 | CPU reference (one thread) | AoS |
| 1 | Naive GPU baseline, software popcount | AoS |
| 2 | Bit packing plus the `__popcll` instruction | AoS |
| 3 | Memory coalescing with SoA layout | SoA |
| 4 | Shared memory tiling | SoA |
| 5 | Thread coarsening / register blocking | SoA |
| 6 | Warp shuffle reduction (`__shfl_down_sync`) | AoS |
| 7 | CUDA streams, overlap copy and compute (`--stream N`) | AoS |
| 8 | Result handling: top-k or threshold (`--topk`, `--threshold`) | |

## Build

On a Linux box with an NVIDIA GPU:

```bash
# set SM to your GPU: T4=75, RTX 3080/3090=86, RTX 4090=89, A100=80, L4=89
make SM=86
```

Not sure which GPU you have? Run `nvidia-smi --query-gpu=name,compute_cap --format=csv`.

On any machine with no GPU you can still run the tests:

```bash
make test
```

This checks the Tanimoto math, the edge cases, the `.fps` file I/O, and the
AoS/SoA layout using only a normal C++ compiler.

## Run

```bash
# 1. make some data (random, no RDKit needed):
python3 preprocessing/generate_fingerprints.py --synthetic 8192 --nbits 2048 --out data/lib.fps

#    or from real molecules (needs: pip install rdkit):
python3 preprocessing/generate_fingerprints.py --smiles data/lib.smi --nbits 2048 --out data/lib.fps

# 2. run every stage and check each one against the CPU:
./build/chemsim --lib data/lib.fps --stage all --validate

# 3. time one stage more times:
./build/chemsim --lib data/lib.fps --stage 4 --repeat 50

# 4. query vs library, top 10 hits per query:
./build/chemsim --query data/q.fps --lib data/lib.fps --stage 3 --topk 10 --out hits.tsv

# 5. stream overlap (stage 7), 4 streams:
./build/chemsim --lib data/lib.fps --stage 3 --stream 4

# quick smoke test on a small subset:
./build/chemsim --lib data/lib.fps --stage all --nq 512 --nl 512 --validate
```

`--validate` builds the CPU matrix and prints the largest difference per stage.
Anything at or below 1e-6 counts as OK. The counts are exact integers, so the
GPU and CPU agree to floating point rounding.

## Benchmark and plots

```bash
python3 benchmark/run_benchmarks.py --sizes 1024 2048 4096 8192 --nbits 1024 2048
python3 benchmark/plot_results.py
```

This writes `results/results.csv` and three plots: the speedup ladder, the
scaling with size N, and the throughput (GCUPS) per stage.

## Check against RDKit

```bash
python3 benchmark/validate.py --engine ./build/chemsim
```

This builds fingerprints for a few real molecules with RDKit, gets the
reference from `BulkTanimotoSimilarity`, and compares it to the engine output.

## Profiling

```bash
ncu --set full -o stage4 ./build/chemsim --lib data/lib.fps --stage 4 --repeat 1
nsys profile -o streams ./build/chemsim --lib data/lib.fps --stage 3 --stream 4
```

Nsight Compute (`ncu`) gives occupancy and memory vs compute throughput per
stage. Note: some shared or managed hosts restrict the GPU performance
counters, so `ncu` may fail there. Nsight Systems (`nsys`) shows the timeline
and does not need those counters.

## Notes

- The tiled kernel (stage 4) handles up to 64 words per fingerprint. A 2048-bit
  fingerprint is 32 words, so it fits. For longer fingerprints raise `MAXW` in
  `kernels.cu` and watch the shared memory limit.
- Stage 8 does the top-k and threshold selection on the CPU after the matrix is
  computed. For libraries bigger than GPU memory you would combine `--stream`
  with per-row selection on the GPU.
- `.fps` files are little-endian. The format is defined in `include/fingerprint.hpp`.
