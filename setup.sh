#!/usr/bin/env bash
# One-shot setup for a fresh Linux + CUDA box.
# Generates fingerprints from data/lib.smi, builds the engine, and validates.
#
#   bash setup.sh [SM] [NBITS]
#     SM     GPU arch (default 86 = Ampere/RTX 30xx). T4=75 RTX40xx=89 A100=80
#     NBITS  fingerprint length (default 2048)
set -euo pipefail

SM="${1:-86}"
NBITS="${2:-2048}"
cd "$(dirname "$0")"

echo "== [1/4] install RDKit (for fingerprint generation) =="
python3 -c "import rdkit" 2>/dev/null || pip install --quiet rdkit

echo "== [2/4] generate fingerprints from data/lib.smi (${NBITS}-bit) =="
if [ ! -f data/lib.fps ]; then
  python3 preprocessing/generate_fingerprints.py --smiles data/lib.smi --nbits "$NBITS" --out data/lib.fps
else
  echo "   data/lib.fps already exists, skipping"
fi

echo "== [3/4] build engine (SM=${SM}) =="
make clean >/dev/null 2>&1 || true
make SM="$SM"

echo "== [4/4] validate correctness + per-stage timing (2000x2000 subset) =="
./build/chemsim --lib data/lib.fps --stage all --validate --nq 2000 --nl 2000

echo
echo "Done. Full benchmark next:  python3 benchmark/run_benchmarks.py && python3 benchmark/plot_results.py"
