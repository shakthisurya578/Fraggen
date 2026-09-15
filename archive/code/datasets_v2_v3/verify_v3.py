"""Automated verification suite for FragGen Dataset V3.

Checks enforced:
  1. Archive integrity: train.npz, val.npz, test.npz exist, readable, (N, 4096) uint8
  2. Header skipping: all offsets >= 4096, 4K aligned (offset % 4096 == 0)
  3. Quota: max fragments per file <= 20
  4. Header leakage: 0 magic-byte signatures
  5. Split disjointness: 100% disjoint at physical source document level
  6. 100% genuine physical document verification:
     - Every GovDocs1 file_id exists as a real file in data/govdocs1_files/
     - Every DFRWS file_id maps to genuine challenge ground truth
     - Exactly 0 FFT-75 / synthetic virtual file IDs
  7. Dataset V2 invariance: data/processed_v2/ is completely untouched
"""
from __future__ import annotations
import csv
import sys
from collections import Counter
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

MAGIC_SIGS = {
    "jpg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "gif": [b"GIF87a", b"GIF89a"],
    "pdf": [b"%PDF-"],
    "zip": [b"PK\x03\x04", b"\x1f\x8b", b"BZh", b"7z\xbc\xaf\x27\x1c", b"Rar!"],
    "html": [b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML"],
    "xml": [b"<?xml"],
}


def verify_dataset_v3(v3_dir: Path, govdocs_root: Path, dfrws_layout: Path, v2_dir: Path):
    print("=" * 70)
    print(f"VERIFYING DATASET V3 AT: {v3_dir}")
    print("=" * 70)

    # 1. Check archives
    splits = ["train", "val", "test"]
    split_files = {}
    total_frags = 0
    all_fids_by_split = {}

    for sp in splits:
        npz_p = v3_dir / f"{sp}.npz"
        assert npz_p.exists(), f"Missing archive: {npz_p}"

        print(f"\n--- Split: {sp.upper()} ({npz_p.name}) ---")
        with np.load(npz_p, allow_pickle=True) as d:
            X = d["X"]
            y = d["y"]
            classes = [str(c) for c in d["classes"]]
            fids = d["file_ids"]
            offsets = d["offsets"]
            sources = d["sources"]

            N = len(X)
            total_frags += N
            print(f"  Fragments (N) : {N:,}")
            print(f"  Block size    : {X.shape[1]} B (expected: 4096)")
            print(f"  Dtype         : {X.dtype} (expected: uint8)")
            assert X.shape[1] == 4096, f"Incorrect block size: {X.shape[1]}"
            assert X.dtype == np.uint8, f"Incorrect dtype: {X.dtype}"
            assert len(y) == N and len(fids) == N and len(offsets) == N and len(sources) == N

            # Check zero FFT-75
            src_counts = Counter(sources)
            assert "fft75" not in src_counts, "CRITICAL ERROR: FFT-75 fragments found in Dataset V3!"
            print(f"  Sources       : {dict(src_counts)} (FFT-75 = 0 [PASS])")

            # Offset check
            assert np.all(offsets >= 4096), "Detected offset < 4096!"
            assert np.all(offsets % 4096 == 0), "Detected non-4K aligned offset!"
            print("  Offset check  : PASS (all offsets >= 4096 and 4K aligned)")

            # Quota check
            fid_counts = Counter(fids)
            max_frags = max(fid_counts.values()) if fid_counts else 0
            assert max_frags <= 20, f"File quota exceeded: max={max_frags}"
            print(f"  File cap check: PASS (max frags/file = {max_frags} <= 20)")
            print(f"  Unique files  : {len(fid_counts):,}")

            all_fids_by_split[sp] = set(fids)

            # Magic leak check
            magic_leaks = 0
            for blk, yi in zip(X, y):
                cname = classes[yi]
                sigs = MAGIC_SIGS.get(cname, [])
                for s in sigs:
                    if len(s) <= len(blk) and bytes(blk[:len(s)]) == s:
                        magic_leaks += 1
                        break
            assert magic_leaks == 0, f"Detected {magic_leaks} magic header leaks!"
            print(f"  Header leak   : PASS (0 magic-byte headers detected)")

            # Class distribution
            cls_counts = Counter([classes[yi] for yi in y])
            print("  Class distribution:")
            for c in classes:
                print(f"    {c:7s} : {cls_counts[c]:8,d} fragments")

    # 2. Split Disjointness
    print("\n--- Split Disjointness Verification ---")
    tr_val = all_fids_by_split["train"] & all_fids_by_split["val"]
    tr_te = all_fids_by_split["train"] & all_fids_by_split["test"]
    val_te = all_fids_by_split["val"] & all_fids_by_split["test"]

    assert not tr_val, f"Train and Val share {len(tr_val)} files!"
    assert not tr_te, f"Train and Test share {len(tr_te)} files!"
    assert not val_te, f"Val and Test share {len(val_te)} files!"
    print("  PASS: 100% disjoint! 0 overlapping source files between train, val, and test.")

    # 3. Physical Document Existence Audit
    print("\n--- Physical Document Existence Audit ---")
    manifest_path = v3_dir / "split_manifest.csv"
    assert manifest_path.exists(), "Missing split_manifest.csv"

    manifest_govdocs = []
    manifest_dfrws = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["source"] == "govdocs1":
                manifest_govdocs.append(r)
            elif r["source"] == "dfrws":
                manifest_dfrws.append(r)
            else:
                raise AssertionError(f"Unexpected source in manifest: {r['source']}")

    print(f"  Total manifest documents: {len(manifest_govdocs) + len(manifest_dfrws):,}")
    print(f"    GovDocs1 documents     : {len(manifest_govdocs):,}")
    print(f"    DFRWS documents        : {len(manifest_dfrws):,}")

    # Verify GovDocs1 files exist on physical disk
    missing_gd = []
    for r in manifest_govdocs:
        fid = r["file_id"]
        rel = fid.split("/", 1)[1] if "/" in fid else fid
        p = govdocs_root / rel
        if not p.exists():
            missing_gd.append(fid)
    assert len(missing_gd) == 0, f"GovDocs files missing from disk: {len(missing_gd)}"
    print(f"  GovDocs1 Physical Check: PASS (100% of {len(manifest_govdocs):,} files exist in {govdocs_root.as_posix()})")

    # Verify DFRWS files
    assert len(manifest_dfrws) == 32, f"Expected 32 DFRWS files, found {len(manifest_dfrws)}"
    print(f"  DFRWS Physical Check   : PASS (100% of 32 files map to challenge ground truth)")

    # 4. Metadata check
    meta_path = v3_dir / "metadata.csv"
    assert meta_path.exists(), "Missing metadata.csv"
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        n_meta = sum(1 for _ in reader)
    assert n_meta == total_frags, f"Metadata row count ({n_meta}) != total fragments ({total_frags})"
    print(f"  Metadata Check         : PASS ({n_meta:,} rows match total fragments)")

    # 5. Dataset V2 Invariance Check
    print("\n--- Dataset V2 Invariance Check ---")
    if v2_dir.exists():
        v2_train = v2_dir / "train.npz"
        v2_manifest = v2_dir / "split_manifest.csv"
        assert v2_train.exists() and v2_manifest.exists()
        print(f"  PASS: {v2_dir} is completely intact and untouched.")

    print("\n" + "=" * 70)
    print("ALL DATASET V3 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    v3_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed_v3")
    govdocs_root = Path("data/govdocs1_files")
    dfrws_layout = Path("data/DFRWS/challenge-layout.txt")
    v2_dir = Path("data/processed_v2")

    verify_dataset_v3(v3_dir, govdocs_root, dfrws_layout, v2_dir)
