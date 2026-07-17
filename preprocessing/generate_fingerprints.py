#!/usr/bin/env python3
"""Write a packed .fps file.

Two modes:
  --smiles FILE   Morgan/ECFP fingerprints from SMILES with RDKit (needs rdkit).
  --synthetic N   N random fingerprints, standard library only, runs anywhere.

The format matches include/fingerprint.hpp: a small header, then n*nwords
little-endian uint64 words (AoS), then n uint32 popcounts. Bit b sits in
word b//64 at position b%64.
"""
import argparse
import random
import struct
import sys


def pack_bits(bit_indices, nwords):
    """Pack a collection of set-bit indices into `nwords` uint64 words."""
    words = [0] * nwords
    for b in bit_indices:
        words[b // 64] |= (1 << (b % 64))
    return words


def popcount_words(words):
    return sum(bin(w).count("1") for w in words)


def write_fps(path, rows, nwords, nbits):
    """rows: iterable of word-lists (each length nwords). Streams to disk."""
    rows = list(rows)
    n = len(rows)
    with open(path, "wb") as f:
        f.write(b"FPS1")
        f.write(struct.pack("<III", n, nwords, nbits))
        f.write(struct.pack("<I", 0))  # reserved
        for w in rows:
            f.write(struct.pack("<%dQ" % nwords, *w))
        for w in rows:
            f.write(struct.pack("<I", popcount_words(w)))
    print(f"wrote {path}: {n} fingerprints x {nbits} bits ({nwords} words)")


def gen_synthetic(n, nbits, density, seed):
    rng = random.Random(seed)
    nwords = nbits // 64
    on = max(1, int(nbits * density))
    for _ in range(n):
        bits = rng.sample(range(nbits), on)
        yield pack_bits(bits, nwords)


def gen_from_smiles(path, nbits, radius):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        sys.exit("rdkit not installed; use --synthetic or `pip install rdkit`")
    nwords = nbits // 64
    with open(path) as fh:
        for line in fh:
            smi = line.split()[0] if line.split() else ""
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"skip unparseable SMILES: {smi}", file=sys.stderr)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            yield pack_bits(fp.GetOnBits(), nwords)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nbits", type=int, default=2048, help="must be a multiple of 64")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="SMILES input file (RDKit path)")
    src.add_argument("--synthetic", type=int, metavar="N", help="generate N random fps")
    ap.add_argument("--radius", type=int, default=2, help="Morgan radius (ECFP=2*radius)")
    ap.add_argument("--density", type=float, default=0.03, help="synthetic on-bit fraction")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.nbits % 64 != 0:
        sys.exit("--nbits must be a multiple of 64")
    nwords = args.nbits // 64

    if args.synthetic is not None:
        rows = gen_synthetic(args.synthetic, args.nbits, args.density, args.seed)
    else:
        rows = gen_from_smiles(args.smiles, args.nbits, args.radius)
    write_fps(args.out, rows, nwords, args.nbits)


if __name__ == "__main__":
    main()
