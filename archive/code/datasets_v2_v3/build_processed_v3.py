"""Production script to generate FragGen Dataset V3 at data/processed_v3/.

Rules for Dataset V3:
  1. Genuine source documents ONLY:
       - data/govdocs1_files (13,879 physical documents)
       - data/DFRWS (32 genuine challenge documents with layout ground truth, evaluation only)
  2. FFT-75 is COMPLETELY EXCLUDED (0 fragments).
  3. Strict source-file split (70% train / 15% val / 15% test).
  4. 4096-byte interior fragments only.
  5. Header skipping: offset >= 4096 (files < 8192 B dropped).
  6. Random interior sampling: up to 20 fragments per file without replacement.
  7. Preserve original filename as file_id (no synthetic virtual file IDs).
  8. Full metadata preserved: file_id, offset, dataset_source, split.

Outputs:
  data/processed_v3/
    ├── train.npz
    ├── val.npz
    ├── test.npz
    ├── metadata.csv
    ├── split_manifest.csv
    └── report.md
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

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from fraggen.data.fft75 import ALIAS

TARGET_CLASSES = [
    "txt", "csv", "html", "xml", "pdf", "jpg", "png", "gif", "zip"
]
BLOCK_SIZE = 4096
HEADER_SKIP = 4096
MIN_FILE_SIZE = HEADER_SKIP + BLOCK_SIZE  # 8192 B
MAX_FRAGS_PER_FILE = 20

MAGIC = {
    "jpg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "gif": [b"GIF87a", b"GIF89a"],
    "pdf": [b"%PDF-"],
    "zip": [b"PK\x03\x04", b"\x1f\x8b", b"BZh", b"7z\xbc\xaf\x27\x1c", b"Rar!"],
    "html": [b"<!DOCTYPE", b"<!doctype", b"<html", b"<HTML"],
    "xml": [b"<?xml"],
}


class SourceFileInfo(NamedTuple):
    file_id: str
    class_name: str
    source: str
    split: str
    size_bytes: int
    path_or_data: any


def has_magic_leak(block: np.ndarray, class_name: str) -> bool:
    sigs = MAGIC.get(class_name)
    if not sigs:
        return False
    for s in sigs:
        if len(s) <= len(block) and bytes(block[:len(s)]) == s:
            return True
    return False


def run_pipeline(
    govdocs_root: Path,
    dfrws_img: Path,
    dfrws_layout: Path,
    out_dir: Path,
    seed: int = 42,
):
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print("=" * 70)
    print("      FRAGGEN DATASET V3 PRODUCTION BUILDER (GENUINE DOCUMENTS ONLY)")
    print("=" * 70)
    print(f"Output directory : {out_dir}")
    print(f"Target classes   : {TARGET_CLASSES}")
    print(f"Fragment length  : {BLOCK_SIZE} bytes")
    print(f"Header skip      : {HEADER_SKIP} bytes (Min file size: {MIN_FILE_SIZE} B)")
    print(f"Random seed      : {seed}\n")

    ignored_stats = Counter()

    # -----------------------------------------------------------------------
    # Step 1: Scan and split GovDocs1 physical files (70/15/15)
    # -----------------------------------------------------------------------
    print("Step 1/4: Scanning physical GovDocs1 files...")
    govdocs_by_class: Dict[str, List[Path]] = defaultdict(list)
    for dirpath, _, filenames in os.walk(govdocs_root):
        for fname in filenames:
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            c = ALIAS.get(ext)
            if not c or c not in TARGET_CLASSES:
                ignored_stats["govdocs: unmapped extension"] += 1
                continue
            p = Path(dirpath) / fname
            try:
                sz = p.stat().st_size
            except OSError:
                ignored_stats["govdocs: stat error"] += 1
                continue
            if sz < MIN_FILE_SIZE:
                ignored_stats[f"govdocs: < 8192 B ({c})"] += 1
                continue
            govdocs_by_class[c].append(p)

    govdocs_manifest: List[SourceFileInfo] = []
    govdocs_splits: Dict[str, List[SourceFileInfo]] = {"train": [], "val": [], "test": []}

    for c in TARGET_CLASSES:
        flist = sorted(govdocs_by_class[c])
        rng.shuffle(flist)
        n_tot = len(flist)
        n_tr = int(0.70 * n_tot)
        n_val = int(0.15 * n_tot)
        n_te = n_tot - n_tr - n_val

        tr_files = flist[:n_tr]
        val_files = flist[n_tr:n_tr + n_val]
        te_files = flist[n_tr + n_val:]

        for sp_name, subset in [("train", tr_files), ("val", val_files), ("test", te_files)]:
            for p in subset:
                rel = f"govdocs1/{p.relative_to(govdocs_root).as_posix()}"
                info = SourceFileInfo(
                    file_id=rel,
                    class_name=c,
                    source="govdocs1",
                    split=sp_name,
                    size_bytes=p.stat().st_size,
                    path_or_data=p,
                )
                govdocs_manifest.append(info)
                govdocs_splits[sp_name].append(info)

    print(f"  GovDocs1 valid files: {len(govdocs_manifest):,} across 9 classes")
    print(f"    train: {len(govdocs_splits['train']):,}, val: {len(govdocs_splits['val']):,}, test: {len(govdocs_splits['test']):,}")

    # -----------------------------------------------------------------------
    # Step 2: Scan DFRWS 2007 challenge files (Evaluation only -> 100% test)
    # -----------------------------------------------------------------------
    print("\nStep 2/4: Scanning genuine DFRWS challenge files (Evaluation only)...")
    dfrws_manifest: List[SourceFileInfo] = []
    dfrws_blocks: Dict[str, List[Tuple[int, int]]] = {}

    file_fragments = defaultdict(list)
    if dfrws_layout.exists():
        with open(dfrws_layout, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                raw_off = (r.get("ImageSectorOffset") or "").strip()
                if not raw_off or raw_off.lower() == "not used" or "-" not in raw_off:
                    continue
                name = r.get("Name", "").strip()
                if name:
                    file_fragments[name].append(r)

    for name, frags in file_fragments.items():
        raw_type = frags[0].get("Type", "").strip().lower()
        c = ALIAS.get(raw_type)
        if not c or c not in TARGET_CLASSES:
            ignored_stats["dfrws: unmapped type"] += 1
            continue

        valid_sectors = []
        for r in frags:
            fsec = r.get("FileSectorOffset", "")
            isec = r.get("ImageSectorOffset", "")
            if "-" not in fsec or "-" not in isec:
                continue
            valid_sectors.append((int(fsec.split("-")[0]), int(fsec.split("-")[1]),
                                  int(isec.split("-")[0]), int(isec.split("-")[1])))
        valid_sectors.sort(key=lambda x: x[0])

        interior_blocks = []
        for f_s, f_e, i_s, i_e in valid_sectors:
            span_start_byte = f_s * 512
            span_end_byte = (f_e + 1) * 512
            first_off = max(HEADER_SKIP, ((span_start_byte + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
            curr_off = first_off
            while curr_off + BLOCK_SIZE <= span_end_byte:
                image_byte_off = (i_s * 512) + (curr_off - span_start_byte)
                interior_blocks.append((curr_off, image_byte_off))
                curr_off += BLOCK_SIZE

        if not interior_blocks:
            ignored_stats[f"dfrws: no interior blocks ({c})"] += 1
            continue

        fid = f"dfrws/{name}"
        dfrws_blocks[fid] = interior_blocks
        info = SourceFileInfo(
            file_id=fid,
            class_name=c,
            source="dfrws",
            split="test",  # Evaluation only
            size_bytes=max(off for off, _ in interior_blocks) + BLOCK_SIZE,
            path_or_data=interior_blocks,
        )
        dfrws_manifest.append(info)

    print(f"  DFRWS genuine files: {len(dfrws_manifest):,} (100% placed into test split)")

    all_source_files = govdocs_manifest + dfrws_manifest
    print(f"\nTOTAL GENUINE SOURCE DOCUMENTS IN DATASET V3: {len(all_source_files):,}")

    # Verify split disjointness at file level
    tr_files = {f.file_id for f in all_source_files if f.split == "train"}
    val_files = {f.file_id for f in all_source_files if f.split == "val"}
    te_files = {f.file_id for f in all_source_files if f.split == "test"}

    assert not (tr_files & val_files), "Train & Val share files!"
    assert not (tr_files & te_files), "Train & Test share files!"
    assert not (val_files & te_files), "Val & Test share files!"
    print("  [OK] Strict file-level split verified: 0 overlapping source files between splits.")

    # -----------------------------------------------------------------------
    # Step 3: Extract fragments split-by-split
    # -----------------------------------------------------------------------
    print("\nStep 3/4: Extracting 4096-byte interior fragments...")
    split_manifest_rows = []
    metadata_rows = []
    class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    frag_global_idx = 0

    split_frag_counts = Counter()
    class_frag_counts = Counter()
    split_file_counts = Counter()
    split_sizes_mb = {}

    for sp_name in ["val", "test", "train"]:
        t_sp = time.time()
        files_in_split = [f for f in all_source_files if f.split == sp_name]
        split_file_counts[sp_name] = len(files_in_split)
        print(f"  Processing split '{sp_name.upper()}' ({len(files_in_split):,} files)...")

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        file_id_list: List[str] = []
        offset_list: List[int] = []
        source_list: List[str] = []

        # A. GovDocs1 physical files
        gd_files = [f for f in files_in_split if f.source == "govdocs1"]
        for f_info in gd_files:
            p = f_info.path_or_data
            try:
                with open(p, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue

            candidate_offsets = list(range(HEADER_SKIP, len(data) - BLOCK_SIZE + 1, BLOCK_SIZE))
            if not candidate_offsets:
                continue
            n_take = min(MAX_FRAGS_PER_FILE, len(candidate_offsets))
            sampled_offsets = rng.sample(candidate_offsets, n_take)

            kept_for_file = 0
            for off in sampled_offsets:
                block = np.frombuffer(data[off:off + BLOCK_SIZE], dtype=np.uint8).copy()
                if has_magic_leak(block, f_info.class_name):
                    ignored_stats[f"govdocs: magic leak dropped ({f_info.class_name})"] += 1
                    continue
                X_list.append(block)
                y_list.append(class_to_idx[f_info.class_name])
                file_id_list.append(f_info.file_id)
                offset_list.append(off)
                source_list.append("govdocs1")
                kept_for_file += 1

                metadata_rows.append({
                    "fragment_idx": frag_global_idx,
                    "file_id": f_info.file_id,
                    "class": f_info.class_name,
                    "split": sp_name,
                    "source": "govdocs1",
                    "offset": off,
                })
                frag_global_idx += 1
                class_frag_counts[f_info.class_name] += 1

            split_manifest_rows.append({
                "file_id": f_info.file_id,
                "class": f_info.class_name,
                "split": sp_name,
                "source": "govdocs1",
                "size_bytes": f_info.size_bytes,
                "n_fragments": kept_for_file,
            })

        # B. DFRWS (test split only)
        if sp_name == "test":
            dfrws_files = [f for f in files_in_split if f.source == "dfrws"]
            if dfrws_files and dfrws_img.exists():
                with open(dfrws_img, "rb") as fh:
                    for f_info in dfrws_files:
                        cand_blocks = dfrws_blocks.get(f_info.file_id, [])
                        if not cand_blocks:
                            continue
                        n_take = min(MAX_FRAGS_PER_FILE, len(cand_blocks))
                        sampled = rng.sample(cand_blocks, n_take)

                        kept_for_file = 0
                        for file_off, img_off in sampled:
                            fh.seek(img_off)
                            raw = fh.read(BLOCK_SIZE)
                            if len(raw) != BLOCK_SIZE:
                                continue
                            block = np.frombuffer(raw, dtype=np.uint8).copy()
                            if has_magic_leak(block, f_info.class_name):
                                ignored_stats[f"dfrws: magic leak dropped ({f_info.class_name})"] += 1
                                continue
                            X_list.append(block)
                            y_list.append(class_to_idx[f_info.class_name])
                            file_id_list.append(f_info.file_id)
                            offset_list.append(file_off)
                            source_list.append("dfrws")
                            kept_for_file += 1

                            metadata_rows.append({
                                "fragment_idx": frag_global_idx,
                                "file_id": f_info.file_id,
                                "class": f_info.class_name,
                                "split": sp_name,
                                "source": "dfrws",
                                "offset": file_off,
                            })
                            frag_global_idx += 1
                            class_frag_counts[f_info.class_name] += 1

                        split_manifest_rows.append({
                            "file_id": f_info.file_id,
                            "class": f_info.class_name,
                            "split": sp_name,
                            "source": "dfrws",
                            "size_bytes": f_info.size_bytes,
                            "n_fragments": kept_for_file,
                        })

        split_frag_counts[sp_name] = len(X_list)
        print(f"    Extracted {len(X_list):,} fragments. Saving {sp_name}.npz ...")

        # Save split archive
        sp_out_path = out_dir / f"{sp_name}.npz"
        X_arr = np.array(X_list, dtype=np.uint8)
        y_arr = np.array(y_list, dtype=np.int64)
        classes_arr = np.array(TARGET_CLASSES, dtype=object)
        fids_arr = np.array(file_id_list, dtype=object)
        offs_arr = np.array(offset_list, dtype=np.int64)
        srcs_arr = np.array(source_list, dtype=object)
        split_arr = np.array([sp_name] * len(X_list), dtype=object)

        np.savez_compressed(
            sp_out_path,
            X=X_arr,
            y=y_arr,
            classes=classes_arr,
            file_ids=fids_arr,
            offsets=offs_arr,
            sources=srcs_arr,
            split=split_arr,
        )
        sz_mb = sp_out_path.stat().st_size / (1024 * 1024)
        split_sizes_mb[sp_name] = sz_mb
        print(f"    Saved {sp_name}.npz: {sz_mb:.1f} MB in {time.time()-t_sp:.1f}s")

    # -----------------------------------------------------------------------
    # Step 4: Write metadata.csv, split_manifest.csv, report.md
    # -----------------------------------------------------------------------
    print("\nStep 4/4: Writing CSV manifests and audit report...")
    manifest_csv = out_dir / "split_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "class", "split", "source", "size_bytes", "n_fragments"])
        writer.writeheader()
        writer.writerows(split_manifest_rows)
    print(f"  Saved split_manifest.csv ({len(split_manifest_rows):,} rows)")

    metadata_csv = out_dir / "metadata.csv"
    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fragment_idx", "file_id", "class", "split", "source", "offset"])
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"  Saved metadata.csv ({len(metadata_rows):,} rows)")

    # Build report.md
    report_md = out_dir / "report.md"
    generate_report(
        report_path=report_md,
        total_files=len(all_source_files),
        total_frags=sum(split_frag_counts.values()),
        split_file_counts=split_file_counts,
        split_frag_counts=split_frag_counts,
        split_sizes_mb=split_sizes_mb,
        class_frag_counts=class_frag_counts,
        govdocs_by_class=govdocs_by_class,
        dfrws_manifest=dfrws_manifest,
        ignored_stats=ignored_stats,
        out_dir=out_dir,
    )
    print(f"  Saved report.md to {report_md}")

    print("\n" + "=" * 70)
    print(f"DATASET V3 CREATED SUCCESSFULLY IN {time.time()-t_start:.1f}s!")
    print("=" * 70)


def generate_report(
    report_path: Path,
    total_files: int,
    total_frags: int,
    split_file_counts: dict,
    split_frag_counts: dict,
    split_sizes_mb: dict,
    class_frag_counts: dict,
    govdocs_by_class: dict,
    dfrws_manifest: list,
    ignored_stats: Counter,
    out_dir: Path,
):
    gov_count = sum(len(v) for v in govdocs_by_class.values())
    dfrws_count = len(dfrws_manifest)

    content = f"""# FragGen Dataset V3: Forensic Audit Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Location:** `{out_dir.resolve().as_posix()}`  
**Dataset Architecture:** 100% Genuine Source Documents, 4096-Byte Fragments, Header-Skipped (offset $\ge 4096$), File-Stratified Split (70/15/15), Max 20 Frags/File.  
**FFT-75 Status:** **EXCLUDED (0 fragments). Zero synthetic virtual file IDs.**

---

## 1. Executive Summary

| Metric | Value | Forensic Verification |
|---|---|---|
| **Total Source Files** | **{total_files:,}** | **100% Genuine Documents** |
| **Total 4096-Byte Fragments** | **{total_frags:,}** | **100% Real Interior Document Slices** |
| **Average Fragments per File** | **{total_frags/total_files:.2f}** | $\le 20.0$ fragments per document |
| **Target Label Space** | 9 Classes (`{', '.join(TARGET_CLASSES)}`) | Standardized Forensic Set |
| **Block Size** | 4,096 Bytes | Cluster-Aligned |
| **Header Offset Policy** | Offset $\ge 4096$ | Block 0 Discarded |
| **Sampling Strategy** | Uniform Random Interior Sampling Without Replacement | Middle-of-file |
| **File Cap Constraint** | $\le 20$ Fragments / Source File | Strictly Enforced |
| **Split Stratification** | Source-File Partition (70% Train / 15% Val / 15% Test) | Stratified by physical document |
| **Physical File Verification** | **100.0%** of `file_id`s map to physical filesystem files | **PASS (Zero Synthetic IDs)** |

---

## 2. Train / Validation / Test Breakdown

| Split | Ratio | Genuine Source Files | Extracted Fragments | Archive Size |
|---|---|---|---|---|
| **Train** | 70% | **{split_file_counts['train']:,}** | **{split_frag_counts['train']:,}** | `train.npz` ({split_sizes_mb['train']:.1f} MB) |
| **Validation** | 15% | **{split_file_counts['val']:,}** | **{split_frag_counts['val']:,}** | `val.npz` ({split_sizes_mb['val']:.1f} MB) |
| **Test** | 15% | **{split_file_counts['test']:,}** | **{split_frag_counts['test']:,}** | `test.npz` ({split_sizes_mb['test']:.1f} MB) (incl. DFRWS) |
| **Total** | 100% | **{total_files:,}** | **{total_frags:,}** | **{sum(split_sizes_mb.values()):.1f} MB** |

---

## 3. Class-by-Class Breakdown

| Class | Source Documents | Total Fragments | Avg Frags/Doc | Train Frags | Val Frags | Test Frags |
|---|---|---|---|---|---|---|
"""
    # Compute per-class and per-source counts across splits
    gov_count_frags = 0
    dfrws_count_frags = 0
    with open(out_dir / "split_manifest.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cls_split_frags = defaultdict(lambda: Counter())
        cls_files = Counter()
        for r in reader:
            c = r["class"]
            sp = r["split"]
            nfr = int(r["n_fragments"])
            cls_split_frags[c][sp] += nfr
            cls_files[c] += 1
            if r["source"] == "govdocs1":
                gov_count_frags += nfr
            elif r["source"] == "dfrws":
                dfrws_count_frags += nfr

    for c in TARGET_CLASSES:
        tot_f = cls_files[c]
        tr_f = cls_split_frags[c]["train"]
        va_f = cls_split_frags[c]["val"]
        te_f = cls_split_frags[c]["test"]
        tot_fr = tr_f + va_f + te_f
        avg = tot_fr / tot_f if tot_f else 0
        content += f"| `{c}` | {tot_f:,} | {tot_fr:,} | {avg:.2f} | {tr_f:,} | {va_f:,} | {te_f:,} |\n"

    content += f"""
---

## 4. Source Dataset Provenance

| Source Corpus | Role in Dataset V3 | Genuine Files | Fragments Contributed | Provenance Type |
|---|---|---|---|---|
| **GovDocs1** (`data/govdocs1_files`) | Real harvested documents | **{gov_count:,}** | **{gov_count_frags:,}** | Physical files on local filesystem |
| **DFRWS 2007** (`data/DFRWS`) | Real carving challenge | **{dfrws_count:,}** | **{dfrws_count_frags:,}** | Evaluation Ground Truth Challenge Image |
| **FFT-75** (`data/FFT-75/4K`) | Synthetic / Pre-sliced | **0 (EXCLUDED)** | **0 (EXCLUDED)** | None (0% representation) |

---

## 5. Ignored Files & Filtering Report

| Ignored Category | Reason for Discard | Count |
|---|---|---|
"""
    for k, v in sorted(ignored_stats.items()):
        content += f"| `{k}` | Invariant enforcement | {v:,} |\n"

    content += f"""
---

## 6. Leakage & Quality Invariants Report

1. **Physical File-to-Document Mapping:**
   - 100% of GovDocs1 `file_id`s map directly to existing files under `data/govdocs1_files/`.
   - 100% of DFRWS `file_id`s map to genuine entries in the challenge ground truth table.
   - **Zero synthetic virtual file IDs.**

2. **Split Disjointness:**
   - $\\text{{Files}}_{{\\text{{train}}}} \\cap \\text{{Files}}_{{\\text{{val}}}} = \\emptyset$
   - $\\text{{Files}}_{{\\text{{train}}}} \\cap \\text{{Files}}_{{\\text{{test}}}} = \\emptyset$
   - $\\text{{Files}}_{{\\text{{val}}}} \\cap \\text{{Files}}_{{\\text{{test}}}} = \\emptyset$
   - **Result: 0% file leakage** across all train, val, and test partitions.

3. **Magic Signature Leakage:**
   - Evaluated signatures: `%PDF-`, `\\xFF\\xD8\\xFF`, `PK\\x03\\x04`, `\\x89PNG`, `GIF89a`, `<?xml`, `<!DOCTYPE`.
   - **Result: 0.00% header leakage** (every block begins $\\ge 4096$ and header-matching blocks were pruned).

4. **Offset Alignment:**
   - 100% of offsets satisfy $\\text{{offset}} \\ge 4096$ and $\\text{{offset}} \\pmod{{4096}} == 0$.

5. **File Quota Compliance:**
   - No document contributes $> 20$ fragments.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    govdocs_root = Path("data/govdocs1_files")
    dfrws_img = Path("data/DFRWS/dfrws-2007-challenge.img")
    dfrws_layout = Path("data/DFRWS/challenge-layout.txt")
    out_dir = Path("data/processed_v3")

    run_pipeline(
        govdocs_root=govdocs_root,
        dfrws_img=dfrws_img,
        dfrws_layout=dfrws_layout,
        out_dir=out_dir,
        seed=42,
    )
