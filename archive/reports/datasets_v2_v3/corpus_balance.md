# FragGen Dataset V2: Training Split Corpus Rebalance Audit Report

**Generated on:** 2026-09-11 00:08:47  
**Target Index:** [`data\processed_v2\train_index.npy`](data/processed_v2/train_index.npy)  
**Sampling Strategy:** `stratified` (RNG Seed = `42`)  
**Corpus Rebalance Target:** FFT-75 $\approx$ 57.5%, GovDocs1 $\approx$ 42.5%, DFRWS = 0.0% (Evaluation Only)  
**Cap Constraint:** No single corpus contributes $> 60.0\%$ of training fragments  

---

## 1. Executive Summary

In the initial unconstrained extraction of FragGen Dataset V2, **FFT-75** contributed **90.14%** (1,145,857 fragments) of the training set, while **GovDocs1** contributed only **9.86%** (125,334 fragments). Because FFT-75 is an in-corpus benchmark, this heavy imbalance creates a severe risk of model over-specialization to synthetic/benchmark file structures at the expense of real-world document generalization.

To eliminate this bias **without altering file split assignments or regenerating fragments**, we constructed a balanced training index:
- **GovDocs1 Contribution:** Retains **100.0%** of available training fragments (125,334 fragments across all 9,711 files).
- **FFT-75 Contribution:** Subsampled without replacement to **169,570** fragments (54,976 unique files represented).
- **DFRWS Contribution:** Strictly **0%** (DFRWS remains 100% reserved for evaluation in the Test split).
- **No Duplicate NPZ Files:** Balanced training is achieved via index-based sampling using [`data\processed_v2\train_index.npy`](data/processed_v2/train_index.npy).

---

## 2. Corpus Distribution: Before vs. After Rebalancing

| Corpus | Role in Training | Before Count | Before % | After Count | After % | Target Range | Status |
|---|---|---|---|---|---|---|---|
| **FFT-75** | Multi-type benchmark | 1,145,857 | 90.14% | **169,570** | **57.50%** | 55.0% – 60.0% | **[BALANCED]** |
| **GovDocs1** | Harvested real documents | 125,334 |  9.86% | **125,334** | **42.50%** | 40.0% – 45.0% | **[BALANCED]** |
| **DFRWS** | Real disk challenge | 0 | 0.00% | **0** | **0.00%** | Evaluation Only (0%) | **[EXCLUDED]** |
| **Total** | Full Training Set | 1,271,191 | 100.00% | **294,904** | **100.00%** | — | — |

> [!IMPORTANT]
> **Corpus Cap Compliance:** The maximum contribution from any single corpus is **57.50%**, strictly satisfying the $\le 60.0\%$ requirement.

---

## 3. Class Distribution: Before vs. After Rebalancing

Because GovDocs1 has a high concentration of real-world document types (`pdf`, `txt`, `html`), rebalancing the corpus shift naturally enhances the document representation while reining in synthetic archive dominance (`zip` dropped from 32.33% to 21.02%):

| Class | Before Count | Before % | After Count | After % | Delta (pp) | Relative Ratio |
|---|---|---|---|---|---|---|
| `txt` |    188,374 |  14.82% | **    48,770** | ** 16.54%** |  +1.72% |  25.9% retained |
| `csv` |     82,625 |   6.50% | **    12,947** | **  4.39%** |  -2.11% |  15.7% retained |
| `html` |    105,262 |   8.28% | **    35,563** | ** 12.06%** |  +3.78% |  33.8% retained |
| `xml` |     82,933 |   6.52% | **    12,881** | **  4.37%** |  -2.16% |  15.5% retained |
| `pdf` |    143,603 |  11.30% | **    73,987** | ** 25.09%** | +13.79% |  51.5% retained |
| `jpg` |     91,428 |   7.19% | **    21,808** | **  7.39%** |  +0.20% |  23.9% retained |
| `png` |     81,867 |   6.44% | **    12,364** | **  4.19%** |  -2.25% |  15.1% retained |
| `gif` |     84,100 |   6.62% | **    14,583** | **  4.94%** |  -1.67% |  17.3% retained |
| `zip` |    410,999 |  32.33% | **    62,001** | ** 21.02%** | -11.31% |  15.1% retained |

---

## 4. Source File Representation & Diversity

Although total training fragments were scaled down from 1,271,191 to 294,904, source file diversity remains exceptionally high:

| Corpus | Original Train Files | Balanced Train Files Retained | File Retention Rate | Mean Frags / File |
|---|---|---|---|---|
| **GovDocs1** | 9,711 | **9,711** | **100.00%** | 12.91 |
| **FFT-75** | 57,356 | **54,976** | **95.85%** | 3.08 |
| **Total** | 67,067 | **64,687** | **96.45%** | 4.56 |

---

## 5. Forensic Leakage & Quality Invariants Confirmation

1. **Zero File Leakage:**
   - $\text{Balanced Train Files} \cap \text{Validation Files} = \emptyset$ (**0 overlapping files**)
   - $\text{Balanced Train Files} \cap \text{Test Files} = \emptyset$ (**0 overlapping files**)
   - **Result: PASS (100% disjoint, zero data leakage)**.

2. **Evaluation Purity:**
   - DFRWS fragments in balanced training set: **0 (0.00%)**.
   - DFRWS remains 100% confined to evaluation (`test.npz`).

3. **Validation & Test Integrity:**
   - `data/processed_v2/val.npz`: **Untouched** (170,036 fragments across 9,240 source files).
   - `data/processed_v2/test.npz`: **Untouched** (171,818 fragments across 9,297 source files).
   - `data/processed_v2/split_manifest.csv`: **Untouched** (85,604 files, zero file assignment changes).
   - `data/processed_v2/metadata.csv`: **Untouched** (1,613,045 rows, zero metadata drift).

4. **Index Validity:**
   - Target file: `data/processed_v2/train_index.npy`
   - Array length: **294,904** elements
   - Dtype: `int64`
   - Array properties: Monotonically strictly increasing, zero duplicates, $0 \le \text{idx} < 1,271,191$.

---

## 6. PyTorch Integration Guide

To use the balanced index during model training without altering dataset files:

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from fraggen.data.v2_fragments import FragmentSetV2

# 1. Load the original train archive
train_set = FragmentSetV2.load("data/processed_v2/train.npz")

# 2. Load the balanced index (294,904 fragments)
train_indices = np.load("data/processed_v2/train_index.npy")

# 3. Create a Subset or use SubsetRandomSampler
train_subset = train_set.subset(train_indices)
# Alternatively, with torch.utils.data.DataLoader:
sampler = SubsetRandomSampler(train_indices)
# loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.batch_size)
```
