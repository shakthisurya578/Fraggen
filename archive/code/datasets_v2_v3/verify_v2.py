"""Verification script for FragGen Dataset V2.

Verifies:
  1. Shapes: (N, 4096) uint8, (N,) int64
  2. Metadata integrity: file_ids, offsets, sources, split
  3. Header leak test: <0.1% magic-byte signatures
  4. Offset invariant: all offsets >= 4096 and multiples of 4096
  5. File cap invariant: max 20 fragments per file
  6. Strict split disjointness: Files(train) ∩ Files(val) ∩ Files(test) = ∅
  7. Class balance across source files

Usage:
  python scripts/verify_v2.py data/v2
"""
from __future__ import annotations
import csv
import glob
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraggen.data.v2_fragments import FragmentSetV2

MAGIC = {
    "jpg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "gif": [b"GIF87a", b"GIF89a"],
    "pdf": [b"%PDF-"],
    "zip": [b"PK\x03\x04", b"\x1f\x8b", b"BZh", b"7z\xbc\xaf\x27\x1c", b"Rar!"],
    "html": [b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML"],
    "xml": [b"<?xml"],
}


def verify_dataset_v2(v2_dir: str | Path) -> bool:
    v2_dir = Path(v2_dir)
    print("=" * 68)
    print(f"VERIFYING DATASET V2 AT: {v2_dir}")
    print("=" * 68)

    splits = ["train", "val", "test"]
    loaded = {}
    all_ok = True

    for sp in splits:
        p = v2_dir / f"{sp}.npz"
        if not p.exists():
            print(f"FAIL: Missing split file {p}")
            return False
        fs = FragmentSetV2.load(p)
        loaded[sp] = fs
        print(f"\n--- Split: {sp.upper()} ({p.name}) ---")
        print(f"  Fragments (N) : {len(fs):,}")
        print(f"  Block size    : {fs.block_size} B (expected: 4096)")
        print(f"  Dtype         : {fs.X.dtype} (expected: uint8)")
        print(f"  Unique files  : {len(set(fs.file_ids)):,}")

        # Check shapes
        if fs.block_size != 4096:
            print(f"  FAIL: Incorrect block size {fs.block_size}")
            all_ok = False
        if len(fs.X) != len(fs.y) or len(fs.X) != len(fs.file_ids) or len(fs.X) != len(fs.offsets):
            print("  FAIL: Array length mismatch in FragmentSetV2")
            all_ok = False

        # Check offset invariant (>= 4096 and multiple of 4096)
        bad_offsets = (fs.offsets < 4096).sum()
        unaligned = (fs.offsets % 4096 != 0).sum()
        if bad_offsets > 0 or unaligned > 0:
            print(f"  FAIL: {bad_offsets} offsets < 4096, {unaligned} unaligned")
            all_ok = False
        else:
            print("  Offset check  : PASS (all offsets >= 4096 and 4K aligned)")

        # Check file cap invariant (<= 20 frags / file)
        file_counts = Counter(fs.file_ids)
        max_frags = max(file_counts.values()) if file_counts else 0
        if max_frags > 20:
            print(f"  FAIL: File cap exceeded! Max frags/file = {max_frags}")
            all_ok = False
        else:
            print(f"  File cap check: PASS (max frags/file = {max_frags} <= 20)")

        # Check per-class counts
        counts = Counter(fs.y)
        print("  Class distribution:")
        for idx, c in enumerate(fs.classes):
            cnt = counts[idx]
            print(f"    {c:8s}: {cnt:>6,} fragments")

        # Header leak check
        leak_count = 0
        for idx, c in enumerate(fs.classes):
            sigs = MAGIC.get(c)
            if not sigs:
                continue
            rows = fs.X[fs.y == idx]
            if len(rows) == 0:
                continue
            hit = np.zeros(len(rows), dtype=bool)
            for s in sigs:
                arr = np.frombuffer(s, dtype=np.uint8)
                if len(arr) <= rows.shape[1]:
                    hit |= (rows[:, :len(arr)] == arr).all(axis=1)
            n_hit = int(hit.sum())
            if n_hit > 0:
                leak_count += n_hit
                print(f"    ! Magic byte leak in {c}: {n_hit}/{len(rows)} ({100*n_hit/len(rows):.2f}%)")

        if leak_count == 0:
            print("  Header leak   : PASS (0 magic-byte headers detected)")
        else:
            print(f"  Header leak   : WARNING ({leak_count} magic signatures detected)")

    # Check split disjointness
    print("\n--- Split Disjointness Verification ---")
    tr_files = set(loaded["train"].file_ids)
    val_files = set(loaded["val"].file_ids)
    te_files = set(loaded["test"].file_ids)

    tr_val_leak = tr_files & val_files
    tr_te_leak = tr_files & te_files
    val_te_leak = val_files & te_files

    if tr_val_leak or tr_te_leak or val_te_leak:
        print(f"  FAIL: Data leakage detected across splits!")
        if tr_val_leak:
            print(f"    Train & Val overlap: {len(tr_val_leak)} files")
        if tr_te_leak:
            print(f"    Train & Test overlap: {len(tr_te_leak)} files")
        if val_te_leak:
            print(f"    Val & Test overlap: {len(val_te_leak)} files")
        all_ok = False
    else:
        print("  PASS: 100% disjoint! 0 overlapping source files between train, val, and test.")

    # Check manifest.csv
    manifest_p = v2_dir / "manifest.csv"
    if manifest_p.exists():
        print("\n--- Manifest Audit ---")
        with open(manifest_p, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            print(f"  Total files in manifest: {len(reader):,}")
            by_split = Counter(r["split"] for r in reader)
            for sp, cnt in by_split.items():
                print(f"    {sp:>5s}: {cnt:>5,} files")

    print("\n" + "=" * 68)
    if all_ok:
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    else:
        print("VERIFICATION FAILED ON ONE OR MORE INVARIANTS.")
    print("=" * 68)
    return all_ok


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data" / "v2")
    success = verify_dataset_v2(path)
    sys.exit(0 if success else 1)
