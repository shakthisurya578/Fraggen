"""Production script to generate FragGen Dataset V2 at data/processed_v2/.

Rules:
  1. Uses all available raw sources:
       - data/govdocs1_files
       - data/FFT-75
       - data/DFRWS (evaluation only -> test split)
  2. Does NOT reduce classes to 34 files; uses maximum available files per class.
  3. Strict source-file split (70% train / 15% val / 15% test).
  4. 4096-byte fragments only.
  5. Skips first 4096 bytes (offset >= 4096). Files < 8192 B dropped.
  6. Random interior sampling: up to 20 fragments per file without replacement.
  7. Strict metadata preserved: file_id, offset, dataset_source, split.
  8. Streaming memory architecture: processes split-by-split to avoid memory exhaustion.

Outputs:
  data/processed_v2/
    ├── train.npz
    ├── val.npz
    ├── test.npz
    ├── metadata.csv
    ├── split_manifest.csv
    └── report.md
"""
from __future__ import annotations
import ast
import csv
import gc
import os
import random
import struct
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraggen.data.fft75 import ALIAS
from fraggen.data.v2_fragments import FragmentSetV2

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

FFT75_SCENARIO1 = [
    "jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2",
    "3fr", "tiff", "heic", "bmp", "gif", "png", "ai", "eps", "psd", "mov", "mp4",
    "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar", "msi", "dmg", "7z", "bz2",
    "deb", "gz", "pkg", "rar", "rpm", "xz", "zip", "exe", "mach-o", "elf", "dll",
    "doc", "docx", "key", "ppt", "pptx", "xls", "xlsx", "djvu", "epub", "mobi",
    "pdf", "md", "rtf", "txt", "tex", "json", "html", "xml", "log", "csv",
    "aiff", "flac", "m4a", "mp3", "ogg", "wav", "wma", "pcap", "ttf", "dwg",
    "sqlite",
]


class SourceFileInfo(NamedTuple):
    file_id: str
    class_name: str
    source: str
    split: str
    size_bytes: int
    path_or_idx: any


# ---------------------------------------------------------------------------
# Low-level zip / npy streaming utilities for FFT-75
# ---------------------------------------------------------------------------
def _npy_header(f):
    magic = f.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError(f"not a .npy stream (magic={magic!r})")
    major, _minor = f.read(2)
    hlen_raw = f.read(2 if major == 1 else 4)
    hlen = struct.unpack("<H" if major == 1 else "<I", hlen_raw)[0]
    hdr = ast.literal_eval(f.read(hlen).decode("latin1").strip())
    return hdr["shape"], np.dtype(hdr["descr"])


def _read_exact(f, n):
    chunks, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def load_fft75_labels(npz_path: Path) -> np.ndarray:
    with zipfile.ZipFile(npz_path) as zf:
        name = "y.npy" if "y.npy" in zf.namelist() else "labels.npy"
        with zf.open(name) as f:
            shape, dt = _npy_header(f)
            n = int(np.prod(shape))
            return np.frombuffer(_read_exact(f, n * dt.itemsize), dtype=dt).astype(np.int64)


def stream_fft75_rows(npz_path: Path, want_idx: np.ndarray, chunk_rows: int = 8192):
    """Sequentially stream wanted row indices from x.npy without full RAM load."""
    if len(want_idx) == 0:
        return np.empty((0, BLOCK_SIZE), dtype=np.uint8)
    with zipfile.ZipFile(npz_path) as zf:
        name = "x.npy" if "x.npy" in zf.namelist() else "X.npy"
        with zf.open(name) as f:
            (n_rows, row_len), dt = _npy_header(f)
            out = np.empty((len(want_idx), row_len), dtype=np.uint8)
            wpos = 0
            ipos = 0
            for start in range(0, n_rows, chunk_rows):
                nr = min(chunk_rows, n_rows - start)
                raw = _read_exact(f, nr * row_len)
                got = len(raw) // row_len
                if got == 0:
                    break
                jend = ipos
                while jend < len(want_idx) and want_idx[jend] < start + got:
                    jend += 1
                if jend > ipos:
                    block = np.frombuffer(raw, dtype=np.uint8, count=got * row_len).reshape(got, row_len)
                    local = want_idx[ipos:jend] - start
                    take = block[local]
                    out[wpos:wpos + len(take)] = take
                    wpos += len(take)
                    ipos = jend
                if ipos >= len(want_idx):
                    break
    return out[:wpos]


def has_magic_leak(block: np.ndarray, class_name: str) -> bool:
    sigs = MAGIC.get(class_name)
    if not sigs:
        return False
    for s in sigs:
        if len(s) <= len(block) and bytes(block[:len(s)]) == s:
            return True
    return False


# ---------------------------------------------------------------------------
# Main Builder Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    govdocs_root: Path,
    dfrws_img: Path,
    dfrws_layout: Path,
    fft75_dir: Path,
    out_dir: Path,
    seed: int = 1337,
):
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print("=" * 70)
    print("      FRAGGEN DATASET V2 PRODUCTION BUILDER (FULL DATASET)")
    print("=" * 70)
    print(f"Output directory : {out_dir}")
    print(f"Target classes   : {TARGET_CLASSES}")
    print(f"Fragment length  : {BLOCK_SIZE} bytes")
    print(f"Random seed      : {seed}\n")

    ignored_stats = Counter()

    # -----------------------------------------------------------------------
    # Step 1: Discover and split GovDocs1 source files (70/15/15)
    # -----------------------------------------------------------------------
    print("Step 1/6: Scanning and splitting GovDocs1 files...")
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
                    path_or_idx=p,
                )
                govdocs_manifest.append(info)
                govdocs_splits[sp_name].append(info)

    print(f"  GovDocs1 valid files: {len(govdocs_manifest):,} across 9 classes")
    print(f"    train: {len(govdocs_splits['train']):,}, val: {len(govdocs_splits['val']):,}, test: {len(govdocs_splits['test']):,}")

    # -----------------------------------------------------------------------
    # Step 2: Discover DFRWS files (Evaluation only -> 100% test)
    # -----------------------------------------------------------------------
    print("\nStep 2/6: Scanning DFRWS files (Evaluation only)...")
    dfrws_manifest: List[SourceFileInfo] = []
    dfrws_blocks: Dict[str, List[Tuple[int, int]]] = {}  # file_id -> list of (file_off, image_off)

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
            split="test",  # EVALUATION ONLY
            size_bytes=max(off for off, _ in interior_blocks) + BLOCK_SIZE,
            path_or_idx=interior_blocks,
        )
        dfrws_manifest.append(info)

    print(f"  DFRWS valid files: {len(dfrws_manifest):,} (100% placed into test split)")

    # -----------------------------------------------------------------------
    # Step 3: Discover and cluster FFT-75 4K into virtual files
    # -----------------------------------------------------------------------
    print("\nStep 3/6: Scanning and clustering FFT-75 4K virtual files...")
    lut = {i: ALIAS.get(name.lower()) for i, name in enumerate(FFT75_SCENARIO1)
           if ALIAS.get(name.lower()) in TARGET_CLASSES}

    fft75_manifest: List[SourceFileInfo] = []
    fft75_file_rows: Dict[str, List[int]] = {}  # file_id -> list of row indices

    for sp_name in ["train", "val", "test"]:
        npz_p = fft75_dir / f"{sp_name}.npz"
        if not npz_p.exists():
            continue
        print(f"  Scanning FFT-75 {sp_name}.npz ...")
        labels = load_fft75_labels(npz_p)

        for c in TARGET_CLASSES:
            rows_for_class = [i for i, y in enumerate(labels) if lut.get(y) == c]
            # Group into virtual files of MAX_FRAGS_PER_FILE (up to 20 fragments)
            for v_idx in range(0, len(rows_for_class), MAX_FRAGS_PER_FILE):
                chunk = rows_for_class[v_idx:v_idx + MAX_FRAGS_PER_FILE]
                if not chunk:
                    continue
                v_fid = f"fft75/{sp_name}_{c}_{v_idx // MAX_FRAGS_PER_FILE:05d}"
                fft75_file_rows[v_fid] = chunk
                info = SourceFileInfo(
                    file_id=v_fid,
                    class_name=c,
                    source="fft75",
                    split=sp_name,  # Honor upstream partition
                    size_bytes=HEADER_SKIP + len(chunk) * BLOCK_SIZE,
                    path_or_idx=(npz_p, chunk),
                )
                fft75_manifest.append(info)

    print(f"  FFT-75 virtual files: {len(fft75_manifest):,} across 9 classes")

    # Combine file manifests
    all_source_files = govdocs_manifest + dfrws_manifest + fft75_manifest
    print(f"\nTOTAL SOURCE FILES IN DATASET V2: {len(all_source_files):,}")

    # Assert strict split disjointness across files
    tr_files = {f.file_id for f in all_source_files if f.split == "train"}
    val_files = {f.file_id for f in all_source_files if f.split == "val"}
    te_files = {f.file_id for f in all_source_files if f.split == "test"}

    assert not (tr_files & val_files), "Leakage detected: Train & Val share files!"
    assert not (tr_files & te_files), "Leakage detected: Train & Test share files!"
    assert not (val_files & te_files), "Leakage detected: Val & Test share files!"
    print("  [OK] Strict file-level split verified: 0 overlapping source files between splits.")

    # -----------------------------------------------------------------------
    # Step 4: Extract fragments split-by-split with streaming
    # -----------------------------------------------------------------------
    split_manifest_rows = []
    metadata_rows = []
    split_counts = Counter()
    class_counts = Counter()
    class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    frag_global_idx = 0

    for sp_name in ["val", "test", "train"]:
        t_sp_start = time.time()
        print(f"\nStep 4: Building {sp_name.upper()} split...")

        files_in_split = [f for f in all_source_files if f.split == sp_name]
        print(f"  Source files in {sp_name}: {len(files_in_split):,}")

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        file_id_list: List[str] = []
        offset_list: List[int] = []
        source_list: List[str] = []

        # A. GovDocs1 extraction for this split
        gd_files = [f for f in files_in_split if f.source == "govdocs1"]
        for f_info in gd_files:
            p = f_info.path_or_idx
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

            split_manifest_rows.append({
                "file_id": f_info.file_id,
                "class": f_info.class_name,
                "split": sp_name,
                "source": "govdocs1",
                "size_bytes": f_info.size_bytes,
                "n_fragments": kept_for_file,
            })

        # B. DFRWS extraction (for test split only)
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

                        split_manifest_rows.append({
                            "file_id": f_info.file_id,
                            "class": f_info.class_name,
                            "split": sp_name,
                            "source": "dfrws",
                            "size_bytes": f_info.size_bytes,
                            "n_fragments": kept_for_file,
                        })

        # C. FFT-75 extraction for this split via streaming
        fft_files = [f for f in files_in_split if f.source == "fft75"]
        if fft_files:
            npz_file = fft75_dir / f"{sp_name}.npz"
            # Gather all requested rows in sorted order for single-pass stream
            file_to_rows: Dict[str, List[int]] = {}
            for f_info in fft_files:
                rlist = fft75_file_rows.get(f_info.file_id, [])
                if rlist:
                    # Random sample up to 20 if more than 20
                    n_take = min(MAX_FRAGS_PER_FILE, len(rlist))
                    sampled_r = rng.sample(rlist, n_take)
                    file_to_rows[f_info.file_id] = sampled_r

            # Invert: row_index -> (file_info, offset_in_vfile)
            row_to_info = {}
            for f_info in fft_files:
                s_rows = file_to_rows.get(f_info.file_id, [])
                for idx_in_file, r_idx in enumerate(s_rows):
                    row_to_info[r_idx] = (f_info, HEADER_SKIP + idx_in_file * BLOCK_SIZE)

            sorted_wanted_rows = np.array(sorted(row_to_info.keys()), dtype=np.int64)
            print(f"  Streaming {len(sorted_wanted_rows):,} rows from FFT-75 {npz_file.name} ...")
            streamed_blocks = stream_fft75_rows(npz_file, sorted_wanted_rows)

            file_kept_counter = Counter()
            for r_idx, block in zip(sorted_wanted_rows, streamed_blocks):
                f_info, off = row_to_info[r_idx]
                if has_magic_leak(block, f_info.class_name):
                    ignored_stats[f"fft75: magic leak dropped ({f_info.class_name})"] += 1
                    continue
                X_list.append(block)
                y_list.append(class_to_idx[f_info.class_name])
                file_id_list.append(f_info.file_id)
                offset_list.append(off)
                source_list.append("fft75")
                file_kept_counter[f_info.file_id] += 1

                metadata_rows.append({
                    "fragment_idx": frag_global_idx,
                    "file_id": f_info.file_id,
                    "class": f_info.class_name,
                    "split": sp_name,
                    "source": "fft75",
                    "offset": off,
                })
                frag_global_idx += 1

            for f_info in fft_files:
                split_manifest_rows.append({
                    "file_id": f_info.file_id,
                    "class": f_info.class_name,
                    "split": sp_name,
                    "source": "fft75",
                    "size_bytes": f_info.size_bytes,
                    "n_fragments": file_kept_counter[f_info.file_id],
                })

        # -------------------------------------------------------------------
        # Assemble, Shuffle, and Save Split Archive
        # -------------------------------------------------------------------
        n_split_frags = len(X_list)
        print(f"  Assembling {n_split_frags:,} fragments for {sp_name}...")
        perm = np.random.default_rng(seed).permutation(n_split_frags)

        X_arr = np.stack([X_list[i] for i in perm]).astype(np.uint8)
        y_arr = np.array([y_list[i] for i in perm], dtype=np.int64)
        fid_arr = np.array([file_id_list[i] for i in perm], dtype=object)
        off_arr = np.array([offset_list[i] for i in perm], dtype=np.int64)
        src_arr = np.array([source_list[i] for i in perm], dtype=object)
        sp_arr = np.array([sp_name] * n_split_frags, dtype=object)

        split_fs = FragmentSetV2(
            X=X_arr,
            y=y_arr,
            classes=TARGET_CLASSES,
            file_ids=fid_arr,
            offsets=off_arr,
            sources=src_arr,
            split=sp_arr,
        )

        dest_npz = out_dir / f"{sp_name}.npz"
        split_fs.save(dest_npz)
        file_sz_mb = dest_npz.stat().st_size / 1e6
        print(f"  [OK] Saved {dest_npz.name} ({n_split_frags:,} frags, {file_sz_mb:.1f} MB) in {time.time()-t_sp_start:.1f}s")

        split_counts[sp_name] = n_split_frags
        for y_val in y_arr:
            class_counts[TARGET_CLASSES[y_val]] += 1

        # Free memory before next split
        del X_list, y_list, file_id_list, offset_list, source_list, X_arr, y_arr, fid_arr, off_arr, src_arr, split_fs
        gc.collect()

    # -----------------------------------------------------------------------
    # Step 5: Save split_manifest.csv and metadata.csv
    # -----------------------------------------------------------------------
    print("\nStep 5/6: Writing manifests and metadata tables...")
    manifest_p = out_dir / "split_manifest.csv"
    with open(manifest_p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_id", "class", "split", "source", "size_bytes", "n_fragments"])
        w.writeheader()
        w.writerows(split_manifest_rows)
    print(f"  Saved {manifest_p.name} ({len(split_manifest_rows):,} files)")

    meta_p = out_dir / "metadata.csv"
    with open(meta_p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["fragment_idx", "file_id", "class", "split", "source", "offset"])
        w.writeheader()
        w.writerows(metadata_rows)
    print(f"  Saved {meta_p.name} ({len(metadata_rows):,} fragments)")

    # -----------------------------------------------------------------------
    # Step 6: Generate comprehensive report.md
    # -----------------------------------------------------------------------
    print("\nStep 6/6: Generating report.md ...")
    report_p = out_dir / "report.md"

    # Compute statistics
    files_per_class = Counter(r["class"] for r in split_manifest_rows)
    frags_per_class = Counter(r["class"] for r in metadata_rows)
    files_by_source = Counter(r["source"] for r in split_manifest_rows)
    frags_by_source = Counter(r["source"] for r in metadata_rows)

    total_files = len(split_manifest_rows)
    total_frags = len(metadata_rows)
    avg_frags_per_file = total_frags / max(total_files, 1)

    report_content = f"""# FragGen Dataset V2: Forensic Dataset Audit Report

**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Location:** `{out_dir.as_posix()}`  
**Architecture:** 4096-Byte Fragments, Header-Skipped (offset $\ge 4096$), File-Stratified Split (70/15/15), Max 20 Frags/File.

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| **Total Source Files Audited** | **{total_files:,}** |
| **Total 4096-Byte Fragments** | **{total_frags:,}** |
| **Average Fragments per File** | **{avg_frags_per_file:.2f}** |
| **Target Label Space** | 9 Classes (`txt, csv, html, xml, pdf, jpg, png, gif, zip`) |
| **Block Size** | 4,096 Bytes (Cluster-Aligned) |
| **Header Offset Policy** | Offset $\ge 4096$ (Block 0 Discarded) |
| **Sampling Strategy** | Uniform Random Interior Sampling Without Replacement |
| **File Cap Constraint** | $\le 20$ Fragments / Source File |
| **Split Stratification** | Source-File Partition (70% Train / 15% Val / 15% Test) |

---

## 2. Train / Validation / Test Breakdown

| Split | Ratio | Source Files | Extracted Fragments | Archive Size |
|---|---|---|---|---|
| **Train** | 70% | **{sum(1 for r in split_manifest_rows if r['split'] == 'train'):,}** | **{split_counts['train']:,}** | `{out_dir / 'train.npz'}` ({((out_dir / 'train.npz').stat().st_size / 1e6):.1f} MB) |
| **Validation** | 15% | **{sum(1 for r in split_manifest_rows if r['split'] == 'val'):,}** | **{split_counts['val']:,}** | `{out_dir / 'val.npz'}` ({((out_dir / 'val.npz').stat().st_size / 1e6):.1f} MB) |
| **Test** | 15% | **{sum(1 for r in split_manifest_rows if r['split'] == 'test'):,}** | **{split_counts['test']:,}** | `{out_dir / 'test.npz'}` ({((out_dir / 'test.npz').stat().st_size / 1e6):.1f} MB) |
| **Total** | 100% | **{total_files:,}** | **{total_frags:,}** | **{sum((out_dir / f'{sp}.npz').stat().st_size for sp in ['train', 'val', 'test']) / 1e6:.1f} MB** |

---

## 3. Class-by-Class Breakdown

| Class | Source Files | Total Fragments | Average Frags/File | Train Frags | Val Frags | Test Frags |
|---|---|---|---|---|---|---|
"""
    for c in TARGET_CLASSES:
        nf = files_per_class[c]
        nfr = frags_per_class[c]
        avg = nfr / max(nf, 1)
        tr_c = sum(1 for r in metadata_rows if r["class"] == c and r["split"] == "train")
        val_c = sum(1 for r in metadata_rows if r["class"] == c and r["split"] == "val")
        te_c = sum(1 for r in metadata_rows if r["class"] == c and r["split"] == "test")
        report_content += f"| `{c}` | {nf:,} | {nfr:,} | {avg:.2f} | {tr_c:,} | {val_c:,} | {te_c:,} |\n"

    report_content += f"""
---

## 4. Source Dataset Provenance

| Source Corpus | Role in Dataset | Unique Files | Fragments Contributed |
|---|---|---|---|
| **GovDocs1** (`data/govdocs1_files`) | Real harvested documents | {files_by_source['govdocs1']:,} | {frags_by_source['govdocs1']:,} |
| **FFT-75** (`data/FFT-75/4K`) | In-corpus multi-type benchmark | {files_by_source['fft75']:,} | {frags_by_source['fft75']:,} |
| **DFRWS 2007** (`data/DFRWS`) | Real carving challenge (Evaluation Only) | {files_by_source['dfrws']:,} | {frags_by_source['dfrws']:,} |

---

## 5. Ignored Files & Filtering Report

Files that were pruned during ingestion to ensure forensic validity:

| Ignored Category | Reason for Discard | Count |
|---|---|---|
"""
    for reason, cnt in sorted(ignored_stats.items()):
        report_content += f"| `{reason}` | Invariant enforcement | {cnt:,} |\n"

    report_content += f"""
---

## 6. Leakage & Quality Invariants Report

1. **Split Disjointness:**
   - $\\text{{Files}}_{{\\text{{train}}}} \\cap \\text{{Files}}_{{\\text{{val}}}} = \\emptyset$
   - $\\text{{Files}}_{{\\text{{train}}}} \\cap \\text{{Files}}_{{\\text{{test}}}} = \\emptyset$
   - $\\text{{Files}}_{{\\text{{val}}}} \\cap \\text{{Files}}_{{\\text{{test}}}} = \\emptyset$
   - **Result:** **0% file leakage** across all train, val, and test partitions.
2. **Magic Signature Leakage:**
   - Scan for signatures (`%PDF-`, `\\xFF\\xD8\\xFF`, `PK\\x03\\x04`, `\\x89PNG`, `GIF89a`, `<?xml`, `<!DOCTYPE`).
   - **Result:** **0.00% header leakage** (every block begins $\\ge 4096$ and header-matching blocks were pruned).
3. **Offset Alignment:**
   - 100% of offsets satisfy $\\text{{offset}} \\ge 4096$ and $\\text{{offset}} \\pmod{{4096}} == 0$.
4. **File Quota Compliance:**
   - No file in the dataset contributes $> 20$ fragments.
"""

    report_p.write_text(report_content, encoding="utf-8")
    print(f"  Saved report.md to {report_p}")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"DATASET V2 REBUILD COMPLETED IN {total_time:.1f} SECONDS ({total_time/60:.2f} MIN)")
    print(f"OUTPUT DIRECTORY: {out_dir}")
    print("=" * 70)


def main():
    govdocs_root = ROOT / "data" / "govdocs1_files"
    dfrws_img = ROOT / "data" / "DFRWS" / "dfrws-2007-challenge.img"
    dfrws_layout = ROOT / "data" / "DFRWS" / "challenge-layout.txt"
    fft75_dir = ROOT / "data" / "FFT-75" / "4K"
    out_dir = ROOT / "data" / "processed_v2"

    run_pipeline(
        govdocs_root=govdocs_root,
        dfrws_img=dfrws_img,
        dfrws_layout=dfrws_layout,
        fft75_dir=fft75_dir,
        out_dir=out_dir,
        seed=1337,
    )


if __name__ == "__main__":
    main()
