# FragGen Dataset V2: True Source File Identity Forensic Audit Report

**Date:** 2026-09-11  
**Audit Target:** Dataset V2 (`data/processed_v2/`)  
**Audited Files:**
- `data/processed_v2/split_manifest.csv`
- `data/processed_v2/metadata.csv`
- `data/processed_v2/train.npz`
- `data/processed_v2/val.npz`
- `data/processed_v2/test.npz`
- `fraggen/data/v2_builder.py`
- `scripts/build_processed_v2.py`

---

## 1. Trace `file_id` Origin

In Dataset V2, `file_id` generation diverges sharply depending on the underlying data corpus. Below is the precise provenance analysis and code trace for each source:

### A. GovDocs1 (`data/govdocs1_files`)
- **Derived from original filename/path:** **YES**.
- **Generated from a fragment index:** **NO**.
- **Generated per NPZ row:** **NO**.
- **Stable across fragments from the same document:** **YES**. All fragments sampled from the same physical document share the exact same canonical `file_id`.
- **Code Trace (`fraggen/data/v2_builder.py` lines 76–82; `scripts/build_processed_v2.py` lines 226–234):
```python
rel = f"govdocs1/{p.relative_to(govdocs_root).as_posix()}"
info = SourceFileInfo(
    file_id=rel,
    class_name=c,
    source="govdocs1",
    split=sp_name,
    size_bytes=p.stat().st_size,
    path_or_idx=p,
)
```
Each GovDocs1 file is an actual physical document on disk (e.g. `govdocs1/003/003442.pdf`). All interior 4KB blocks extracted from that document retain this `file_id`.

### B. DFRWS 2007 (`data/DFRWS`)
- **Derived from original filename/path:** **YES** (from challenge carving ground truth layout table `dfrws2007_layout.csv`).
- **Generated from a fragment index:** **NO**.
- **Generated per NPZ row:** **NO**.
- **Stable across fragments from the same document:** **YES**.
- **Code Trace (`scripts/build_processed_v2.py` lines 291–300):
```python
fid = f"dfrws/{name}"
info = SourceFileInfo(
    file_id=fid,
    class_name=c,
    source="dfrws",
    split="test",  # Evaluation only
    size_bytes=max(off for off, _ in interior_blocks) + BLOCK_SIZE,
    path_or_idx=interior_blocks,
)
```

### C. FFT-75 (`data/FFT-75/4K/`)
- **Derived from original filename/path:** **NO**. Upstream FFT-75 contains no filenames or paths.
- **Generated from a fragment index:** **YES**.
- **Generated per NPZ row:** **NO** (generated per chunk of 20 consecutive NPZ rows).
- **Stable across fragments from the same document:** **NO**. It does not track original documents.
- **Code Trace (`fraggen/data/v2_builder.py` lines 206–220; `scripts/build_processed_v2.py` lines 324–339):
```python
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
```

**Summary of Origin:**  
For **83.75%** of the files in `split_manifest.csv` (71,693 of 85,604), `file_id` is an artificial virtual grouping identifier synthesized by bucketing consecutive pre-extracted fragments from FFT-75 into blocks of 20.

---

## 2. FFT-75 Identity Audit

The raw FFT-75 benchmark dataset (`data/FFT-75/4K/{train,val,test}.npz`) is distributed exclusively as pre-sliced numpy arrays containing only `x.npy` (4096-byte blocks) and `y.npy` (integer class labels). The original upstream creators of FiFTy-75 did not package file boundaries, document names, or source file IDs.

As a result:
- **Can multiple fragments from the same original document be verified to share one `file_id`?**  
  **NO.** Because upstream document identities do not exist in the FFT-75 archives, consecutive rows in `x.npy` might come from the same document or from entirely different documents. The grouping into 20-fragment chunks is synthetic.
- **Total FFT-75 `file_id`s reported:** **71,693** virtual files
- **Unique original documents:** **Unknown / 0 verifiable original documents** (None recorded in raw upstream FFT-75)
- **Average fragments per "document" (virtual file):** **19.98**
- **Maximum fragments per "document" (virtual file):** **20**

Fragment count distribution across all 71,693 FFT-75 virtual files:
- Exactly 20 fragments: **70,234 files** (97.96%)
- Exactly 19 fragments: **1,409 files** (1.97%)
- Exactly 18 fragments: **26 files**
- 1 to 17 fragments (tail chunks / magic leak pruned): **24 files**

---

## 3. GovDocs1 Identity Audit

GovDocs1 `file_id`s correspond to real, physical files on the local filesystem in `data/govdocs1_files`.

### Verification Metrics
- **Total GovDocs1 files in `split_manifest.csv`:** **13,879**
- **Files verified existing on disk:** **13,879 / 13,879 (100.0%)**
- **Missing files:** **0**

### Random Sample Audit (20 Entries)

| # | `file_id` | Original Path | Class | Fragment Count | File Size (Bytes) | Exists on Disk? |
|---|---|---|---|---|---|---|
| 1 | `govdocs1/003/003804.jpg` | `data/govdocs1_files/003/003804.jpg` | `jpg` | 20 | 159,914 | **Yes** |
| 2 | `govdocs1/011/011814.log` | `data/govdocs1_files/011/011814.log` | `txt` | 20 | 294,639 | **Yes** |
| 3 | `govdocs1/017/017459.html` | `data/govdocs1_files/017/017459.html` | `html` | 14 | 62,423 | **Yes** |
| 4 | `govdocs1/007/007117.html` | `data/govdocs1_files/007/007117.html` | `html` | 20 | 129,159 | **Yes** |
| 5 | `govdocs1/012/012410.pdf` | `data/govdocs1_files/012/012410.pdf` | `pdf` | 20 | 485,451 | **Yes** |
| 6 | `govdocs1/014/014819.pdf` | `data/govdocs1_files/014/014819.pdf` | `pdf` | 17 | 77,212 | **Yes** |
| 7 | `govdocs1/015/015450.pdf` | `data/govdocs1_files/015/015450.pdf` | `pdf` | 11 | 51,635 | **Yes** |
| 8 | `govdocs1/014/014936.pdf` | `data/govdocs1_files/014/014936.pdf` | `pdf` | 20 | 305,306 | **Yes** |
| 9 | `govdocs1/007/007088.pdf` | `data/govdocs1_files/007/007088.pdf` | `pdf` | 20 | 893,191 | **Yes** |
| 10 | `govdocs1/010/010498.pdf` | `data/govdocs1_files/010/010498.pdf` | `pdf` | 3 | 19,733 | **Yes** |
| 11 | `govdocs1/000/000918.html` | `data/govdocs1_files/000/000918.html` | `html` | 13 | 60,120 | **Yes** |
| 12 | `govdocs1/000/000509.jpg` | `data/govdocs1_files/000/000509.jpg` | `jpg` | 20 | 1,688,344 | **Yes** |
| 13 | `govdocs1/001/001380.pdf` | `data/govdocs1_files/001/001380.pdf` | `pdf` | 20 | 1,745,867 | **Yes** |
| 14 | `govdocs1/013/013794.xml` | `data/govdocs1_files/013/013794.xml` | `xml` | 7 | 32,868 | **Yes** |
| 15 | `govdocs1/002/002858.gz` | `data/govdocs1_files/002/002858.gz` | `zip` | 20 | 987,987 | **Yes** |
| 16 | `govdocs1/006/006262.pdf` | `data/govdocs1_files/006/006262.pdf` | `pdf` | 18 | 80,833 | **Yes** |
| 17 | `govdocs1/012/012080.jpg` | `data/govdocs1_files/012/012080.jpg` | `jpg` | 8 | 39,092 | **Yes** |
| 18 | `govdocs1/012/012366.pdf` | `data/govdocs1_files/012/012366.pdf` | `pdf` | 15 | 65,677 | **Yes** |
| 19 | `govdocs1/019/019709.html` | `data/govdocs1_files/019/019709.html` | `html` | 8 | 39,040 | **Yes** |
| 20 | `govdocs1/013/013812.pdf` | `data/govdocs1_files/013/013812.pdf` | `pdf` | 17 | 75,336 | **Yes** |

Every GovDocs1 `file_id` traces directly to an authentic document on disk whose size is $\ge 8,192$ bytes.

---

## 4. Split Integrity: Original Document Identity

To audit split integrity, we test whether genuine original documents appear in more than one partition (Train, Val, Test):

### A. GovDocs1 (Genuine Physical Files)
- **Train Split Files:** 9,711 unique physical files
- **Val Split Files:** 2,078 unique physical files
- **Test Split Files:** 2,090 unique physical files
- **Cross-Split Overlaps:**
  - $\text{Train} \cap \text{Val} = \emptyset$ (**0 overlapping files**)
  - $\text{Train} \cap \text{Test} = \emptyset$ (**0 overlapping files**)
  - $\text{Val} \cap \text{Test} = \emptyset$ (**0 overlapping files**)
- **GovDocs1 Result:** **PASS (100% disjoint at the original document level).**

### B. DFRWS (Genuine Carved Files)
- **Train Split Files:** 0
- **Val Split Files:** 0
- **Test Split Files:** 32
- **DFRWS Result:** **PASS (100% disjoint; strictly isolated to Test).**

### C. FFT-75 (Pre-sliced Upstream Benchmark)
- In Dataset V2, FFT-75 virtual files were partitioned by honoring the upstream split filenames (`train.npz` $\to$ Train, `val.npz` $\to$ Val, `test.npz` $\to$ Test).
- At the level of synthetic virtual IDs (`fft75/train_*`, `fft75/val_*`, `fft75/test_*`), there is 0 overlap across splits.
- **However, at the original document level:** Because raw FFT-75 does not publish document identities, **it is impossible to mathematically guarantee or audit whether the upstream authors included fragments from the same original document across their `train.npz`, `val.npz`, and `test.npz` files.**

---

## 5. Final Verdict

### **FAIL — file_id is actually a fragment identifier.**

*(More specifically: while GovDocs1 (13,879 files) and DFRWS (32 files) use genuine source documents, **83.75% of the reported source files** (71,693 / 85,604) are synthetic fragment-batch identifiers manufactured from FFT-75 array slices).*

### Explanation of Where the Issue Exists (Without Code Changes)

1. **Location in Codebase:**
   - [`fraggen/data/v2_builder.py`](file:///f:/sem%207/CSE4074%20-%20Forensics%20Analytics/Project/fraggen/fraggen/data/v2_builder.py) — function `scan_fft75_virtual_files` (lines 164–222).
   - [`scripts/build_processed_v2.py`](file:///f:/sem%207/CSE4074%20-%20Forensics%20Analytics/Project/fraggen/scripts/build_processed_v2.py) — lines 324–339.

2. **Root Cause:**
   The raw dataset `data/FFT-75/4K/` was downloaded as pre-sliced arrays (`x.npy`, `y.npy`) without the original underlying files or document boundaries. To accommodate the user requirements ("split by source file" and "max 20 fragments per file"), the builder synthesised "virtual files" by grouping contiguous blocks of 20 fragments from `x.npy` into artificial IDs:
   ```python
   v_fid = f"fft75/{sp_name}_{c}_{v_idx // MAX_FRAGS_PER_FILE:05d}"
   ```
   Each such entry was reported as a "source file" in `split_manifest.csv` with a synthesized size `size_bytes = 4096 + len(chunk) * 4096`.

3. **Impact on Reported Metrics:**
   - Reported: **85,604 source files**.
   - Genuine Source Documents: **13,911** (13,879 GovDocs1 + 32 DFRWS).
   - Synthetic Virtual 20-Fragment Batches: **71,693** (all FFT-75).
