"""CLI driver to build FragGen Dataset V2 from raw corpora.

Usage:
  python scripts/build_v2.py --out data/v2 --seed 1337
"""
from __future__ import annotations
import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraggen.data.v2_builder import (
    TARGET_CLASSES,
    scan_govdocs_files,
    scan_dfrws_files,
    scan_fft75_virtual_files,
    split_and_balance_files,
    extract_fragments_from_file,
    assemble_fragment_set,
)


def main():
    parser = argparse.ArgumentParser(description="Build FragGen Dataset V2")
    parser.add_argument("--out", default=str(ROOT / "data" / "v2"), help="Output directory")
    parser.add_argument("--govdocs-root", default=str(ROOT / "data" / "govdocs1_files"))
    parser.add_argument("--dfrws-img", default=str(ROOT / "data" / "DFRWS" / "dfrws-2007-challenge.img"))
    parser.add_argument("--dfrws-layout", default=str(ROOT / "data" / "DFRWS" / "challenge-layout.txt"))
    parser.add_argument("--fft75-dir", default=str(ROOT / "data" / "FFT-75" / "4K"))
    parser.add_argument("--include-fft75", action="store_true", default=False,
                        help="Include FFT-75 virtual files in the balanced pool")
    parser.add_argument("--files-per-class", type=int, default=0,
                        help="Exact number of source files per class (0 = minimum available across classes)")
    parser.add_argument("--split", default="0.70,0.15,0.15", help="Train,val,test split ratios")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ratios = tuple(map(float, args.split.split(",")))
    assert len(ratios) == 3, "Split must have 3 comma-separated ratios"

    print("=" * 68)
    print("           FRAGGEN DATASET V2 BUILDER")
    print("=" * 68)
    print(f"Target classes ({len(TARGET_CLASSES)}): {TARGET_CLASSES}")
    print(f"Split ratios: {ratios[0]*100:.0f}% train / {ratios[1]*100:.0f}% val / {ratios[2]*100:.0f}% test")
    print(f"Random seed : {args.seed}")
    print(f"Destination : {out_dir}\n")

    # 1. Scan raw corpora
    print("Step 1/5: Scanning raw datasets for valid source files (>= 8192 B)...")
    combined_by_class = {c: [] for c in TARGET_CLASSES}

    # GovDocs1
    govdocs_by_class = scan_govdocs_files(args.govdocs_root)
    for c in TARGET_CLASSES:
        combined_by_class[c].extend(govdocs_by_class[c])
    govdocs_total = sum(len(fl) for fl in govdocs_by_class.values())
    print(f"  [GovDocs1]  Found {govdocs_total:,} valid files")

    # DFRWS
    dfrws_by_class = scan_dfrws_files(args.dfrws_img, args.dfrws_layout)
    for c in TARGET_CLASSES:
        combined_by_class[c].extend(dfrws_by_class[c])
    dfrws_total = sum(len(fl) for fl in dfrws_by_class.values())
    print(f"  [DFRWS]     Found {dfrws_total:,} valid files")

    # FFT-75 (Optional inclusion)
    if args.include_fft75:
        fft75_by_class = scan_fft75_virtual_files(args.fft75_dir, seed=args.seed)
        for c in TARGET_CLASSES:
            combined_by_class[c].extend(fft75_by_class[c])
        fft75_total = sum(len(fl) for fl in fft75_by_class.values())
        print(f"  [FFT-75]    Found {fft75_total:,} virtual files")

    print("\nFile availability per class across raw sources:")
    for c in TARGET_CLASSES:
        print(f"  {c:8s}: {len(combined_by_class[c]):>6,} files")

    # 2. Partition and balance at the source-file level
    print("\nStep 2/5: Partitioning and balancing at the source-file level...")
    splits = split_and_balance_files(
        combined_by_class,
        files_per_class=args.files_per_class,
        split_ratios=ratios,
        seed=args.seed,
    )

    for sp_name in ["train", "val", "test"]:
        file_counts = {c: len(splits[sp_name][c]) for c in TARGET_CLASSES}
        tot_f = sum(file_counts.values())
        print(f"  [{sp_name:>5s}] {tot_f:>5,} files total ({file_counts[TARGET_CLASSES[0]]} per class balanced)")

    # Assert zero file leakage across splits
    train_ids = {f.file_id for fl in splits["train"].values() for f in fl}
    val_ids = {f.file_id for fl in splits["val"].values() for f in fl}
    test_ids = {f.file_id for fl in splits["test"].values() for f in fl}

    assert not (train_ids & val_ids), "CRITICAL: Train and Val share files!"
    assert not (train_ids & test_ids), "CRITICAL: Train and Test share files!"
    assert not (val_ids & test_ids), "CRITICAL: Val and Test share files!"
    print("  [OK] Strict source-file split verified: 0% file leakage between splits.")

    # 3. Extract random interior fragments
    print("\nStep 3/5: Extracting interior 4096-byte blocks (skipping block 0, <= 20/file)...")
    rng = random.Random(args.seed)
    class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    fft75_cache = {}

    manifest_rows = []
    split_fragments = {"train": [], "val": [], "test": []}

    for sp_name in ["train", "val", "test"]:
        print(f"  Extracting {sp_name} fragments...")
        for c in TARGET_CLASSES:
            flist = splits[sp_name][c]
            c_idx = class_to_idx[c]
            for f_entry in flist:
                frags = extract_fragments_from_file(
                    f_entry,
                    split_name=sp_name,
                    target_class_idx=c_idx,
                    rng=rng,
                    fft75_npz_cache=fft75_cache,
                )
                split_fragments[sp_name].extend(frags)
                manifest_rows.append({
                    "file_id": f_entry.file_id,
                    "class": c,
                    "split": sp_name,
                    "source": f_entry.source,
                    "n_fragments": len(frags),
                })

    # 4. Assemble and save FragmentSetV2 .npz files
    print("\nStep 4/5: Saving FragmentSetV2 compressed archives...")
    for sp_name in ["train", "val", "test"]:
        dest = out_dir / f"{sp_name}.npz"
        fs = assemble_fragment_set(split_fragments[sp_name], TARGET_CLASSES, seed=args.seed)
        fs.save(dest)
        counts = Counter(fs.y)
        print(f"  Saved {dest.name:12s} | N={len(fs):>7,} fragments | "
              f"Per-class: min={min(counts.values()) if counts else 0}, max={max(counts.values()) if counts else 0}")

    # 5. Write manifest CSV
    print("\nStep 5/5: Writing dataset manifest audit log...")
    manifest_p = out_dir / "manifest.csv"
    with open(manifest_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "class", "split", "source", "n_fragments"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  Saved {manifest_p.name} ({len(manifest_rows):,} files audited)")

    elapsed = time.time() - t0
    print("\n" + "=" * 68)
    print(f"DATASET V2 CREATED SUCCESSFULLY IN {elapsed:.1f} SECONDS")
    print("=" * 68)


if __name__ == "__main__":
    main()
