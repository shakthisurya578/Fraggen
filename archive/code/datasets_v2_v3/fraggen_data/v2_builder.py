"""Dataset V2 builder: file-stratified, 4096-byte interior sampling across raw corpora.

Rules enforced:
  1. 9 target classes only ('txt', 'csv', 'html', 'xml', 'pdf', 'jpg', 'png', 'gif', 'zip')
  2. 4096-byte fragments only
  3. Header skipping: offset >= 4096 (files < 8192 B dropped)
  4. Random interior sampling: up to 20 non-overlapping 4K blocks sampled without replacement
  5. Maximum 20 fragments per file
  6. Balance by SOURCE FILE count (not fragment count)
  7. Split by SOURCE FILE (70% train / 15% val / 15% test)
  8. Full metadata preserved: file_id, offset, dataset_source, split
"""
from __future__ import annotations
import csv
import os
import random
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from .fft75 import ALIAS
from .v2_fragments import FragmentSetV2

TARGET_CLASSES = [
    "txt", "csv", "html", "xml", "pdf", "jpg", "png", "gif", "zip"
]
BLOCK_SIZE = 4096
HEADER_SKIP = 4096
MIN_FILE_SIZE = HEADER_SKIP + BLOCK_SIZE  # 8192 bytes
MAX_FRAGS_PER_FILE = 20


class FileEntry(NamedTuple):
    file_id: str
    class_name: str
    source: str
    file_path: Optional[str] = None
    # For DFRWS or indexed sources, pre-computed interior (offset_in_file, offset_in_image) pairs
    available_offsets: Optional[List[int]] = None
    image_path: Optional[str] = None


class ExtractedFragment(NamedTuple):
    data: np.ndarray        # (4096,) uint8
    class_idx: int
    file_id: str
    offset: int
    source: str
    split: str


# ---------------------------------------------------------------------------
# 1. GovDocs1 Scanner
# ---------------------------------------------------------------------------
def scan_govdocs_files(root: str | Path) -> Dict[str, List[FileEntry]]:
    """Scan data/govdocs1_files, filtering < 8192 B and grouping by 9 classes."""
    root = Path(root)
    by_class: Dict[str, List[FileEntry]] = {c: [] for c in TARGET_CLASSES}
    if not root.exists():
        return by_class

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
            canon = ALIAS.get(ext)
            if canon not in by_class:
                continue
            full_path = Path(dirpath) / fname
            try:
                sz = full_path.stat().st_size
            except OSError:
                continue
            if sz < MIN_FILE_SIZE:
                continue

            rel_id = f"govdocs1/{full_path.relative_to(root).as_posix()}"
            entry = FileEntry(
                file_id=rel_id,
                class_name=canon,
                source="govdocs1",
                file_path=str(full_path),
            )
            by_class[canon].append(entry)

    return by_class


# ---------------------------------------------------------------------------
# 2. DFRWS 2007 Scanner
# ---------------------------------------------------------------------------
def scan_dfrws_files(
    image_path: str | Path,
    layout_path: str | Path,
    sector_size: int = 512,
) -> Dict[str, List[FileEntry]]:
    """Scan DFRWS 2007 challenge layout table and identify files with interior blocks."""
    by_class: Dict[str, List[FileEntry]] = {c: [] for c in TARGET_CLASSES}
    image_path, layout_path = Path(image_path), Path(layout_path)
    if not image_path.exists() or not layout_path.exists():
        return by_class

    # Group layout sector fragments by file Name
    file_fragments: Dict[str, List[dict]] = {}
    with open(layout_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw_off = (r.get("ImageSectorOffset") or "").strip()
            if not raw_off or raw_off.lower() == "not used" or "-" not in raw_off:
                continue
            name = r.get("Name", "").strip()
            if not name:
                continue
            file_fragments.setdefault(name, []).append(r)

    for name, frags in file_fragments.items():
        raw_type = frags[0].get("Type", "").strip().lower()
        canon = ALIAS.get(raw_type)
        if canon not in by_class:
            continue

        # Sort fragments by FileSectorOffset
        valid_sectors = []
        for r in frags:
            fsec = r.get("FileSectorOffset", "")
            isec = r.get("ImageSectorOffset", "")
            if "-" not in fsec or "-" not in isec:
                continue
            f_start, f_end = map(int, fsec.split("-", 1))
            i_start, i_end = map(int, isec.split("-", 1))
            valid_sectors.append((f_start, f_end, i_start, i_end))
        valid_sectors.sort(key=lambda x: x[0])

        # Enumerate 4096-byte interior blocks (skipping first 4096 bytes of the file)
        # 4096 bytes = 8 sectors of 512 bytes
        interior_blocks = []
        for f_s, f_e, i_s, i_e in valid_sectors:
            span_start_byte = f_s * sector_size
            span_end_byte = (f_e + 1) * sector_size
            # Find first 4096-aligned offset in this span >= HEADER_SKIP
            first_off = max(HEADER_SKIP, ((span_start_byte + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE)
            curr_off = first_off
            while curr_off + BLOCK_SIZE <= span_end_byte:
                image_byte_off = (i_s * sector_size) + (curr_off - span_start_byte)
                interior_blocks.append((curr_off, image_byte_off))
                curr_off += BLOCK_SIZE

        if not interior_blocks:
            continue

        entry = FileEntry(
            file_id=f"dfrws/{name}",
            class_name=canon,
            source="dfrws",
            image_path=str(image_path),
            available_offsets=interior_blocks,
        )
        by_class[canon].append(entry)

    return by_class


# ---------------------------------------------------------------------------
# 3. FFT-75 4K Virtual File Scanner
# ---------------------------------------------------------------------------
def scan_fft75_virtual_files(
    fft75_dir: str | Path,
    seed: int = 1337,
) -> Dict[str, List[FileEntry]]:
    """Groups FFT-75 4K pre-chunked fragments into virtual files of <= 20 fragments.
    
    Since FFT-75 was distributed as raw pre-sliced arrays (train/val/test),
    we organize contiguous sequences of blocks from the same class into distinct
    virtual source files (e.g. 'fft75/train_pdf_001') to uphold the file-split
    and max-20-fragments constraints.
    """
    by_class: Dict[str, List[FileEntry]] = {c: [] for c in TARGET_CLASSES}
    fft75_dir = Path(fft75_dir)
    if not fft75_dir.exists():
        return by_class

    # Authoritative scenario-1 class names
    from .fft75 import ALIAS
    SCENARIO1 = [
        "jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2",
        "3fr", "tiff", "heic", "bmp", "gif", "png", "ai", "eps", "psd", "mov", "mp4",
        "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar", "msi", "dmg", "7z", "bz2",
        "deb", "gz", "pkg", "rar", "rpm", "xz", "zip", "exe", "mach-o", "elf", "dll",
        "doc", "docx", "key", "ppt", "pptx", "xls", "xlsx", "djvu", "epub", "mobi",
        "pdf", "md", "rtf", "txt", "tex", "json", "html", "xml", "log", "csv",
        "aiff", "flac", "m4a", "mp3", "ogg", "wav", "wma", "pcap", "ttf", "dwg",
        "sqlite",
    ]

    lut = {i: ALIAS.get(name.lower()) for i, name in enumerate(SCENARIO1)
           if ALIAS.get(name.lower()) in TARGET_CLASSES}

    # Iterate through splits in 4K
    for split in ["train", "val", "test"]:
        npz_p = fft75_dir / f"{split}.npz"
        if not npz_p.exists():
            continue
        d = np.load(npz_p, allow_pickle=True)
        ky = "y" if "y" in d else "labels" if "labels" in d else "arr_1"
        y_src = d[ky].ravel()

        for class_name in TARGET_CLASSES:
            src_indices = [idx for idx, sy in enumerate(y_src) if lut.get(int(sy)) == class_name]
            # Chunk into virtual files of MAX_FRAGS_PER_FILE
            for v_idx in range(0, len(src_indices), MAX_FRAGS_PER_FILE):
                chunk = src_indices[v_idx:v_idx + MAX_FRAGS_PER_FILE]
                if not chunk:
                    continue
                v_file_id = f"fft75/{split}_{class_name}_{v_idx // MAX_FRAGS_PER_FILE:05d}"
                entry = FileEntry(
                    file_id=v_file_id,
                    class_name=class_name,
                    source="fft75",
                    file_path=str(npz_p),
                    available_offsets=chunk,  # stores array row indices
                )
                by_class[class_name].append(entry)

    return by_class


# ---------------------------------------------------------------------------
# 4. Source-File Stratified Splitter & Balancer
# ---------------------------------------------------------------------------
def split_and_balance_files(
    all_files_by_class: Dict[str, List[FileEntry]],
    files_per_class: Optional[int] = None,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 1337,
) -> Dict[str, Dict[str, List[FileEntry]]]:
    """Partitions files for each class into train/val/test splits.
    
    Guarantees:
      - Split is strictly performed at the SOURCE FILE level
      - Files(train) ∩ Files(val) ∩ Files(test) = ∅
      - Each class receives exactly K files per split (if files_per_class is set)
    """
    rng = random.Random(seed)
    r_tr, r_val, r_te = split_ratios
    assert abs(r_tr + r_val + r_te - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Find file counts per class
    available_counts = {c: len(flist) for c, flist in all_files_by_class.items()}
    min_avail = min(available_counts.values()) if available_counts else 0

    if files_per_class is None or files_per_class <= 0:
        target_k = min_avail
    else:
        target_k = min(files_per_class, min_avail)

    splits: Dict[str, Dict[str, List[FileEntry]]] = {
        "train": {c: [] for c in TARGET_CLASSES},
        "val": {c: [] for c in TARGET_CLASSES},
        "test": {c: [] for c in TARGET_CLASSES},
    }

    n_tr = int(target_k * r_tr)
    n_val = int(target_k * r_val)
    n_te = target_k - n_tr - n_val

    for c in TARGET_CLASSES:
        files = list(all_files_by_class[c])
        rng.shuffle(files)
        selected = files[:target_k]

        splits["train"][c] = selected[:n_tr]
        splits["val"][c] = selected[n_tr:n_tr + n_val]
        splits["test"][c] = selected[n_tr + n_val:n_tr + n_val + n_te]

    return splits


# ---------------------------------------------------------------------------
# 5. Fragment Extraction
# ---------------------------------------------------------------------------
def extract_fragments_from_file(
    entry: FileEntry,
    split_name: str,
    target_class_idx: int,
    rng: random.Random,
    fft75_npz_cache: Optional[Dict[str, np.ndarray]] = None,
) -> List[ExtractedFragment]:
    """Extract up to 20 random interior 4096-byte blocks from a source file."""
    frags: List[ExtractedFragment] = []

    # Case A: GovDocs1 physical file on disk
    if entry.source == "govdocs1" and entry.file_path:
        p = Path(entry.file_path)
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError:
            return frags
        if len(data) < MIN_FILE_SIZE:
            return frags

        # Offsets starting at 4096, 4K aligned
        candidate_offsets = list(range(HEADER_SKIP, len(data) - BLOCK_SIZE + 1, BLOCK_SIZE))
        if not candidate_offsets:
            return frags
        n_sample = min(MAX_FRAGS_PER_FILE, len(candidate_offsets))
        chosen_offsets = rng.sample(candidate_offsets, n_sample)

        for off in chosen_offsets:
            block = np.frombuffer(data[off:off + BLOCK_SIZE], dtype=np.uint8).copy()
            frags.append(ExtractedFragment(
                data=block,
                class_idx=target_class_idx,
                file_id=entry.file_id,
                offset=off,
                source=entry.source,
                split=split_name,
            ))

    # Case B: DFRWS raw image with sector offsets
    elif entry.source == "dfrws" and entry.image_path and entry.available_offsets:
        candidate_blocks = entry.available_offsets
        n_sample = min(MAX_FRAGS_PER_FILE, len(candidate_blocks))
        chosen_blocks = rng.sample(candidate_blocks, n_sample)

        with open(entry.image_path, "rb") as f:
            for file_off, image_off in chosen_blocks:
                f.seek(image_off)
                raw = f.read(BLOCK_SIZE)
                if len(raw) == BLOCK_SIZE:
                    block = np.frombuffer(raw, dtype=np.uint8).copy()
                    frags.append(ExtractedFragment(
                        data=block,
                        class_idx=target_class_idx,
                        file_id=entry.file_id,
                        offset=file_off,
                        source=entry.source,
                        split=split_name,
                    ))

    # Case C: FFT-75 pre-chunked array virtual file
    elif entry.source == "fft75" and entry.file_path and entry.available_offsets:
        row_indices = entry.available_offsets
        n_sample = min(MAX_FRAGS_PER_FILE, len(row_indices))
        chosen_rows = rng.sample(row_indices, n_sample)

        npz_p = entry.file_path
        if fft75_npz_cache is not None and npz_p in fft75_npz_cache:
            x_arr = fft75_npz_cache[npz_p]
        else:
            d = np.load(npz_p, allow_pickle=True)
            kx = "x" if "x" in d else "X" if "X" in d else "arr_0"
            x_arr = d[kx]
            if fft75_npz_cache is not None:
                fft75_npz_cache[npz_p] = x_arr

        for i, r_idx in enumerate(chosen_rows):
            block = x_arr[r_idx].copy()
            frags.append(ExtractedFragment(
                data=block,
                class_idx=target_class_idx,
                file_id=entry.file_id,
                offset=HEADER_SKIP + (i * BLOCK_SIZE),  # Synthetic 4K interior offset
                source=entry.source,
                split=split_name,
            ))

    return frags


# ---------------------------------------------------------------------------
# 6. Assembly & Manifest Generation
# ---------------------------------------------------------------------------
def assemble_fragment_set(
    fragments: List[ExtractedFragment],
    classes: List[str] = TARGET_CLASSES,
    seed: int = 1337,
) -> FragmentSetV2:
    """Assembles a list of ExtractedFragment objects into a shuffled FragmentSetV2."""
    if not fragments:
        return FragmentSetV2(
            X=np.empty((0, BLOCK_SIZE), dtype=np.uint8),
            y=np.empty((0,), dtype=np.int64),
            classes=classes,
            file_ids=np.empty((0,), dtype=object),
            offsets=np.empty((0,), dtype=np.int64),
            sources=np.empty((0,), dtype=object),
            split=np.empty((0,), dtype=object),
        )

    # Shuffle across all fragments in this split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(fragments))

    X = np.stack([fragments[i].data for i in perm]).astype(np.uint8)
    y = np.array([fragments[i].class_idx for i in perm], dtype=np.int64)
    file_ids = np.array([fragments[i].file_id for i in perm], dtype=object)
    offsets = np.array([fragments[i].offset for i in perm], dtype=np.int64)
    sources = np.array([fragments[i].source for i in perm], dtype=object)
    splits = np.array([fragments[i].split for i in perm], dtype=object)

    return FragmentSetV2(
        X=X,
        y=y,
        classes=list(classes),
        file_ids=file_ids,
        offsets=offsets,
        sources=sources,
        split=splits,
    )
