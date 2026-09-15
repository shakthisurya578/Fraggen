"""Sanity-check processed FragmentSets: shapes, label coverage, and a magic-byte
leak test.

The leak test matters for FragGen's premise. GovDocs fragments are meant to be
*middle-of-file* chunks, so a fragment should NOT start with its format's magic
signature -- if many do, the classifier can cheat on headers and the reported
cross-corpus gap is meaningless.

Usage:
  python scripts/verify_processed.py data/processed_v4/*.npz
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.fragments import FragmentSet

MAGIC = {
    "jpg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "gif": [b"GIF87a", b"GIF89a"],
    "pdf": [b"%PDF-"],
    "zip": [b"PK\x03\x04", b"\x1f\x8b", b"BZh", b"7z\xbc\xaf\x27\x1c", b"Rar!"],
    "html": [b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML"],
    "xml": [b"<?xml"],
}


def report(path):
    fs = FragmentSet.load(path)
    classes = [str(c) for c in fs.classes]
    counts = np.bincount(fs.y, minlength=len(classes))
    print(f"\n=== {Path(path).name} ===")
    print(f"  corpus={fs.corpus}  n={len(fs):,}  block_size={fs.block_size}  dtype={fs.X.dtype}")

    empty = [c for c, n in zip(classes, counts) if n == 0]
    print(f"  classes ({len(classes)}):")
    for c, n in zip(classes, counts):
        bar = "#" * int(40 * n / max(counts.max(), 1))
        flag = "  <-- EMPTY" if n == 0 else ""
        print(f"    {c:10s} {n:>8,} {bar}{flag}")
    if empty:
        print(f"  !! empty classes: {empty}")

    # all-zero / degenerate fragments
    allsame = int((fs.X == fs.X[:, :1]).all(axis=1).sum())
    if allsame:
        print(f"  !! {allsame:,} constant fragments ({100*allsame/len(fs):.2f}%) "
              f"- padding/sparse regions carry no type signal")

    # magic-byte leak: does fragment start with its own format's signature?
    print("  header-leak check (fragments starting with own magic bytes):")
    any_leak = False
    for ci, c in enumerate(classes):
        sigs = MAGIC.get(c)
        if not sigs:
            continue
        rows = fs.X[fs.y == ci]
        if len(rows) == 0:
            continue
        hit = np.zeros(len(rows), dtype=bool)
        for s in sigs:
            arr = np.frombuffer(s, dtype=np.uint8)
            if len(arr) <= rows.shape[1]:
                hit |= (rows[:, :len(arr)] == arr).all(axis=1)
        n_hit = int(hit.sum())
        if n_hit:
            any_leak = True
            print(f"    {c:10s} {n_hit:>6,} / {len(rows):,} ({100*n_hit/len(rows):.2f}%)")
    if not any_leak:
        print("    none detected - good, no header giveaway")


def main():
    args = sys.argv[1:]
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        print("no files matched", args)
        return 1
    for p in paths:
        report(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
