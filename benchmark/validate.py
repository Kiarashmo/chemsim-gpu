#!/usr/bin/env python3
"""Check the engine against RDKit as an outside reference. Needs rdkit.

Build a small set with RDKit, get its Tanimoto matrix, run the engine in
all-pairs mode with --threshold 0, and confirm every pair matches.
"""
import argparse
import struct
import subprocess
import sys


def build_fps_with_rdkit(path, smiles, nbits, radius):
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            sys.exit(f"unparseable SMILES: {smi}")
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))

    nwords = nbits // 64
    with open(path, "wb") as f:
        f.write(b"FPS1")
        f.write(struct.pack("<III", len(fps), nwords, nbits))
        f.write(struct.pack("<I", 0))
        for fp in fps:
            words = [0] * nwords
            for b in fp.GetOnBits():
                words[b // 64] |= (1 << (b % 64))
            f.write(struct.pack("<%dQ" % nwords, *words))
        for fp in fps:
            f.write(struct.pack("<I", fp.GetNumOnBits()))
    return fps, DataStructs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="./build/chemsim")
    ap.add_argument("--fps", default="/tmp/validate.fps")
    ap.add_argument("--nbits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    # A handful of small, real molecules (aspirin, caffeine, benzene, ethanol, ...).
    smiles = [
        "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "c1ccccc1", "CCO", "CC(=O)O", "C1CCCCC1", "CCN(CC)CC",
        "O=C(O)c1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    ]
    fps, DataStructs = build_fps_with_rdkit(args.fps, smiles, args.nbits, args.radius)

    # Oracle matrix from RDKit.
    n = len(fps)
    oracle = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        oracle[i] = list(sims)

    # Engine: all-pairs, emit every pair.
    out = subprocess.check_output(
        [args.engine, "--lib", args.fps, "--stage", "3",
         "--threshold", "0", "--out", "/dev/stdout"],
        text=True)

    worst = 0.0
    checked = 0
    for line in out.splitlines():
        if line.startswith("query") or "\t" not in line:
            continue
        qi, lj, t = line.split("\t")
        try:
            qi, lj, t = int(qi), int(lj), float(t)
        except ValueError:
            continue
        worst = max(worst, abs(t - oracle[qi][lj]))
        checked += 1

    print(f"checked {checked} pairs; max diff vs RDKit = {worst:.3e}")
    if worst <= args.tol:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
