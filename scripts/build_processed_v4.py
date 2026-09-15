"""Production script to generate FragGen Dataset V4.

Writes fragments to data/processed_v4/{train,val,test}.npz and provenance to
data/metadata/v4_{split_manifest,fragment_metadata}.csv + v4_build_report.md.

V4 = V3 (genuine documents only, no FFT-75) + newly-ingested real-world
sources for the classes the data audit found starved
(reports/DATA_AUDIT_2026-09-14.md sections 3.2/3.3/3.5), PLUS a genuine
per-class fragment cap so no single class can dominate the way pdf (49%)
did in V3 or zip (32%) did in V2.

Sources:
  govdocs1   -> all 9 classes (existing, real physical documents)
  dfrws      -> pdf/jpg/zip, evaluation only (existing)
  rico_png   -> png   (RICO mobile-UI screenshot corpus)
  tgif_gif   -> gif   (TGIF animated-gif corpus)
  uci_csv    -> csv   (real CSVs scraped from GitHub repos)
  github_zip -> zip   (real .zip archives scraped from GitHub repos)
  wiki_pages -> xml   (full-length Simple Wikipedia articles, extracted by
                       scripts/extract_wiki_pages.py -- fixes the previous
                       wikipedia_xml staging, whose median file was 822 B,
                       far below the 8192 B usability floor)

Balance mechanism (the actual fix, not present in V2/V3):
  A PER-CLASS, PER-SPLIT fragment cap. Candidate files for a class are
  pooled across ALL sources providing that class, shuffled, split 70/15/15
  at the FILE level (so leakage-safety is identical to V3), then fragments
  are drawn file-by-file (up to 20/file, for document diversity) until the
  split's class quota is filled or files run out. Classes that cannot reach
  the quota (most likely csv, and possibly xml) simply contribute what they
  have -- reported explicitly, not silently padded.

Same schema, same manifests, same leakage/magic-byte-leak safeguards as V3,
so it can be audited with the identical methodology used on V2/V3.
"""
from __future__ import annotations
import csv
from datetime import datetime
import os
from pathlib import Path
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple, Tuple
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.preprocessing.fft75 import ALIAS

TARGET_CLASSES = ["txt", "csv", "html", "xml", "pdf", "jpg", "png", "gif", "zip"]
BLOCK_SIZE = 4096
HEADER_SKIP = 4096
MIN_FILE_SIZE = HEADER_SKIP + BLOCK_SIZE  # 8192 B
MAX_FRAGS_PER_FILE = 20
PER_CLASS_FRAG_CAP = 10_000          # total across train+val+test
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

MAGIC = {
    "jpg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "gif": [b"GIF87a", b"GIF89a"],
    "pdf": [b"%PDF-"],
    "zip": [b"PK\x03\x04", b"\x1f\x8b", b"BZh", b"7z\xbc\xaf\x27\x1c", b"Rar!"],
    "html": [b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML"],
    "xml": [b"<?xml"],
}

# extra flat-folder sources: folder -> (glob, class, source_tag)
EXTRA_SOURCES = [
    ("data/raw/rico_png", "*.png", "png", "rico"),
    ("data/raw/tgif_gif", "*.gif", "gif", "tgif"),
    # NOTE: no separate "*.GIF" entry -- on Windows, glob() is case-insensitive,
    # so "*.gif" already matches .GIF files too. Adding both patterns produced
    # duplicate SourceFileInfo entries for the same physical file, which then
    # landed in different splits and caused real train/val leakage (caught by
    # the smoke test's disjointness assertion). scan_extra() also deduplicates
    # by resolved path as a portability safety net for case-sensitive filesystems.
    ("data/raw/uci_csv", "*.csv", "csv", "uci_csv"),
    ("data/raw/github_zip", "*.zip", "zip", "github_zip"),
    ("data/raw/wiki_pages", "*.xml", "xml", "wiki_pages"),
]


class SourceFileInfo(NamedTuple):
    file_id: str
    class_name: str
    source: str
    path: Path
    size_bytes: int


def has_magic_leak(block: np.ndarray, class_name: str) -> bool:
    sigs = MAGIC.get(class_name)
    if not sigs:
        return False
    for s in sigs:
        if len(s) <= len(block) and bytes(block[: len(s)]) == s:
            return True
    return False


def scan_govdocs(root: Path, ignored: Counter) -> Dict[str, List[SourceFileInfo]]:
    by_class: Dict[str, List[SourceFileInfo]] = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            c = ALIAS.get(ext)
            if not c or c not in TARGET_CLASSES:
                continue
            p = Path(dirpath) / fname
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if sz < MIN_FILE_SIZE:
                ignored[f"govdocs: < 8192 B ({c})"] += 1
                continue
            rel = f"govdocs1/{p.relative_to(root).as_posix()}"
            by_class[c].append(SourceFileInfo(rel, c, "govdocs1", p, sz))
    return by_class


def scan_extra(folder: str, glob: str, cls: str, source_tag: str, ignored: Counter) -> List[SourceFileInfo]:
    root = Path(folder)
    if not root.exists():
        return []
    out = []
    for p in root.glob(glob):
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz < MIN_FILE_SIZE:
            ignored[f"{source_tag}: < 8192 B ({cls})"] += 1
            continue
        rel = f"{source_tag}/{p.name}"
        out.append(SourceFileInfo(rel, cls, source_tag, p, sz))
    return out


def scan_dfrws(truth_csv: Path, ignored: Counter):
    """DFRWS stays 100% test, evaluation-only.

    Uses the already-byte-aligned truth CSV (start,end,type,file_id) produced
    by scripts/dfrws_layout_to_truth.py --block-size 4096 in August, which
    already accounts for header-skipping -- this is the SAME format and
    the SAME logic as src/preprocessing/dfrws.py::build(), not the
    sector-layout format build_processed_v3.py's scan_dfrws() assumed. (That
    function was copied from a script that actually reads challenge-layout.txt,
    a different, sector-based file; dfrws2007_gt_4096.csv is simpler and
    already correct in byte terms, so this reimplements the intended contract.)
    """
    manifest = []
    blocks: Dict[str, List[Tuple[int, int]]] = {}
    by_file: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    if not truth_csv.exists():
        return manifest, blocks
    with open(truth_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                start, end = int(r["start"]), int(r["end"])
            except (KeyError, ValueError):
                continue
            by_file[r["file_id"]].append((start, end, r["type"].strip().lower()))

    for fid_raw, spans in by_file.items():
        raw_type = spans[0][2]
        c = ALIAS.get(raw_type)
        if not c or c not in TARGET_CLASSES:
            ignored["dfrws: unmapped type"] += 1
            continue
        interior = []
        for start, end, _ in spans:
            off = start
            while off + BLOCK_SIZE <= end:
                interior.append((off, off))  # (file-relative off, image byte off) -- identical here
                off += BLOCK_SIZE
        if not interior:
            ignored[f"dfrws: no interior blocks ({c})"] += 1
            continue
        fid = f"dfrws/{fid_raw}"
        blocks[fid] = interior
        manifest.append(SourceFileInfo(fid, c, "dfrws", None,
                                       max(o for o, _ in interior) + BLOCK_SIZE))
    return manifest, blocks


def main():
    global PER_CLASS_FRAG_CAP
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=PER_CLASS_FRAG_CAP,
                    help="per-class fragment cap (use a small value to smoke-test)")
    ap.add_argument("--out", default="data/processed_v4")
    ap.add_argument("--metadata-dir", default="data/metadata",
                    help="where v4_split_manifest.csv, v4_fragment_metadata.csv and "
                         "v4_build_report.md are written")
    args = ap.parse_args()
    PER_CLASS_FRAG_CAP = args.cap

    t_start = time.time()
    seed = 42
    rng = random.Random(seed)
    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = repo_root / args.metadata_dir
    meta_dir.mkdir(parents=True, exist_ok=True)
    ignored = Counter()

    print("=" * 74)
    print("  FRAGGEN DATASET V4 BUILDER (genuine docs + balanced ingestion)")
    print("=" * 74)
    print(f"Per-class fragment cap: {PER_CLASS_FRAG_CAP:,} (train/val/test = "
          f"{SPLIT_RATIOS['train']:.0%}/{SPLIT_RATIOS['val']:.0%}/{SPLIT_RATIOS['test']:.0%} of that)\n")

    # ---------------- Step 1: gather candidate files per class ----------------
    print("Step 1/4: Scanning all sources...")
    govdocs_root = repo_root / "data" / "raw" / "govdocs1_files"
    by_class = scan_govdocs(govdocs_root, ignored)
    seen_resolved = {c: set() for c in TARGET_CLASSES}  # dedup safety net (see note above)
    for folder, glob, cls, tag in EXTRA_SOURCES:
        extra = scan_extra(str(repo_root / folder), glob, cls, tag, ignored)
        deduped = []
        for info in extra:
            key = str(info.path.resolve())
            if key in seen_resolved[cls]:
                ignored[f"{tag}: duplicate path (case-insensitive glob)"] += 1
                continue
            seen_resolved[cls].add(key)
            deduped.append(info)
        by_class[cls].extend(deduped)
        print(f"  {tag:12s} -> class {cls:5s}: +{len(deduped):,} usable files "
              f"({len(extra)-len(deduped)} deduped)")

    dfrws_layout = repo_root / "data" / "raw" / "DFRWS" / "dfrws2007_gt_4096.csv"
    dfrws_manifest, dfrws_blocks = scan_dfrws(dfrws_layout, ignored)
    print(f"  dfrws        -> {len(dfrws_manifest):,} usable files (test-only)")

    for c in TARGET_CLASSES:
        print(f"  class {c:5s}: {len(by_class[c]):,} total candidate files "
              f"(cap: {PER_CLASS_FRAG_CAP:,} fragments)")

    # ---------------- Step 2: file-level 70/15/15 split per class ----------------
    print("\nStep 2/4: File-level split (70/15/15) per class...")
    split_files: Dict[str, Dict[str, List[SourceFileInfo]]] = {c: {} for c in TARGET_CLASSES}
    for c in TARGET_CLASSES:
        flist = list(by_class[c])
        rng.shuffle(flist)
        n = len(flist)
        n_tr = int(SPLIT_RATIOS["train"] * n)
        n_val = int(SPLIT_RATIOS["val"] * n)
        split_files[c]["train"] = flist[:n_tr]
        split_files[c]["val"] = flist[n_tr:n_tr + n_val]
        split_files[c]["test"] = flist[n_tr + n_val:]

    # sanity: file-level disjointness (guaranteed by slicing, verify anyway)
    for c in TARGET_CLASSES:
        ids = {"train": {f.file_id for f in split_files[c]["train"]},
              "val": {f.file_id for f in split_files[c]["val"]},
              "test": {f.file_id for f in split_files[c]["test"]}}
        assert not (ids["train"] & ids["val"]), f"{c}: train/val overlap"
        assert not (ids["train"] & ids["test"]), f"{c}: train/test overlap"
        assert not (ids["val"] & ids["test"]), f"{c}: val/test overlap"
    print("  [OK] file-level split verified disjoint for every class")

    # ---------------- Step 3: extract fragments, respecting the per-class cap ----------------
    print("\nStep 3/4: Extracting fragments (capped for balance)...")
    class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    split_manifest_rows, metadata_rows = [], []
    frag_global_idx = 0
    class_frag_counts = Counter()
    split_sizes_mb = {}
    dfrws_img = repo_root / "data" / "raw" / "DFRWS" / "dfrws-2007-challenge.img"
    dfrws_fh = open(dfrws_img, "rb") if dfrws_img.exists() else None

    for sp_name in ["val", "test", "train"]:
        t_sp = time.time()
        X_list, y_list, fid_list, off_list, src_list = [], [], [], [], []
        split_cap = int(round(PER_CLASS_FRAG_CAP * SPLIT_RATIOS[sp_name]))

        for c in TARGET_CLASSES:
            files_here = list(split_files[c][sp_name])
            rng.shuffle(files_here)
            kept_class_split = 0

            for f_info in files_here:
                if kept_class_split >= split_cap:
                    break
                try:
                    with open(f_info.path, "rb") as fh:
                        data = fh.read()
                except OSError:
                    continue
                cand_offsets = list(range(HEADER_SKIP, len(data) - BLOCK_SIZE + 1, BLOCK_SIZE))
                if not cand_offsets:
                    continue
                n_take = min(MAX_FRAGS_PER_FILE, len(cand_offsets), split_cap - kept_class_split)
                sampled = rng.sample(cand_offsets, n_take)
                kept_for_file = 0
                for off in sampled:
                    block = np.frombuffer(data[off:off + BLOCK_SIZE], dtype=np.uint8).copy()
                    if has_magic_leak(block, c):
                        ignored[f"{f_info.source}: magic leak dropped ({c})"] += 1
                        continue
                    X_list.append(block); y_list.append(class_to_idx[c])
                    fid_list.append(f_info.file_id); off_list.append(off)
                    src_list.append(f_info.source)
                    metadata_rows.append({"fragment_idx": frag_global_idx, "file_id": f_info.file_id,
                                          "class": c, "split": sp_name, "source": f_info.source,
                                          "offset": off})
                    frag_global_idx += 1
                    kept_for_file += 1
                    kept_class_split += 1
                    class_frag_counts[c] += 1
                split_manifest_rows.append({"file_id": f_info.file_id, "class": c, "split": sp_name,
                                            "source": f_info.source, "size_bytes": f_info.size_bytes,
                                            "n_fragments": kept_for_file})

            if kept_class_split < split_cap:
                print(f"    [{sp_name}] class {c}: only reached {kept_class_split:,}/{split_cap:,} "
                      f"(ran out of usable files)")

            # DFRWS: unconditionally added to test, on top of the cap (small, fixed volume)
            if sp_name == "test" and c in ("pdf", "jpg", "zip") and dfrws_fh:
                for f_info in [m for m in dfrws_manifest if m.class_name == c]:
                    cand = dfrws_blocks.get(f_info.file_id, [])
                    n_take = min(MAX_FRAGS_PER_FILE, len(cand))
                    for file_off, img_off in rng.sample(cand, n_take):
                        dfrws_fh.seek(img_off)
                        raw = dfrws_fh.read(BLOCK_SIZE)
                        if len(raw) != BLOCK_SIZE:
                            continue
                        block = np.frombuffer(raw, dtype=np.uint8).copy()
                        if has_magic_leak(block, c):
                            ignored[f"dfrws: magic leak dropped ({c})"] += 1
                            continue
                        X_list.append(block); y_list.append(class_to_idx[c])
                        fid_list.append(f_info.file_id); off_list.append(file_off)
                        src_list.append("dfrws")
                        metadata_rows.append({"fragment_idx": frag_global_idx, "file_id": f_info.file_id,
                                              "class": c, "split": sp_name, "source": "dfrws",
                                              "offset": file_off})
                        frag_global_idx += 1
                        class_frag_counts[c] += 1

        print(f"  [{sp_name}] extracted {len(X_list):,} fragments, saving...")
        X_arr = np.array(X_list, dtype=np.uint8)
        np.savez_compressed(
            out_dir / f"{sp_name}.npz",
            X=X_arr, y=np.array(y_list, dtype=np.int64),
            classes=np.array(TARGET_CLASSES, dtype=object),
            file_ids=np.array(fid_list, dtype=object),
            offsets=np.array(off_list, dtype=np.int64),
            sources=np.array(src_list, dtype=object),
            split=np.array([sp_name] * len(X_list), dtype=object),
        )
        sz_mb = (out_dir / f"{sp_name}.npz").stat().st_size / 1e6
        split_sizes_mb[sp_name] = sz_mb
        print(f"    saved {sp_name}.npz: {sz_mb:.1f} MB in {time.time()-t_sp:.1f}s")

    if dfrws_fh:
        dfrws_fh.close()

    # ---------------- Step 4: manifests + report ----------------
    print("\nStep 4/4: Writing manifests and report...")
    with open(meta_dir / "v4_split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_id", "class", "split", "source", "size_bytes", "n_fragments"])
        w.writeheader(); w.writerows(split_manifest_rows)
    with open(meta_dir / "v4_fragment_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fragment_idx", "file_id", "class", "split", "source", "offset"])
        w.writeheader(); w.writerows(metadata_rows)

    total_frags = sum(class_frag_counts.values())
    with open(meta_dir / "v4_build_report.md", "w", encoding="utf-8") as f:
        f.write(f"# FragGen Dataset V4: Balanced Ingestion Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"Per-class fragment cap: {PER_CLASS_FRAG_CAP:,} (train "
                f"{SPLIT_RATIOS['train']:.0%} / val {SPLIT_RATIOS['val']:.0%} / "
                f"test {SPLIT_RATIOS['test']:.0%})\n\n")
        f.write(f"Total fragments: {total_frags:,}\n\n")
        f.write("## Class distribution\n\n| class | fragments | share | vs cap |\n|---|---|---|---|\n")
        for c in TARGET_CLASSES:
            n = class_frag_counts[c]
            f.write(f"| {c} | {n:,} | {n/max(total_frags,1):.2%} | "
                    f"{'FULL' if n >= PER_CLASS_FRAG_CAP*0.98 else f'{n/PER_CLASS_FRAG_CAP:.0%} of cap'} |\n")
        f.write("\n## Split sizes\n\n| split | file size |\n|---|---|\n")
        for sp, sz in split_sizes_mb.items():
            f.write(f"| {sp} | {sz:.1f} MB |\n")
        f.write("\n## Ignored / filtered\n\n| reason | count |\n|---|---|\n")
        for k, v in sorted(ignored.items(), key=lambda kv: -kv[1]):
            f.write(f"| {k} | {v:,} |\n")

    print(f"\nDone in {time.time()-t_start:.1f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()
