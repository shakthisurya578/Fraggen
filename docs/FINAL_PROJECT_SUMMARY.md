# FragGen V4 — Final Project Summary

**File-fragment type classification for digital forensics**
CSE4074 Forensics Analytics · repository cleanup and research completion · 2026-09-15

This is the closing record of the repository-wide refactor: what the audit found, what
moved where, what the completed ablation study shows, and what is needed to treat this
as a finished, presentable, reproducible project.

---

## 1. What changed

| phase | outcome |
|---|---|
| 1. Audit | 106,708 files / 51.3 GiB inventoried before any change. `docs/REPOSITORY_AUDIT.md` |
| 2. Reorganisation | Moved to the target layout below. Every code/report/checkpoint move verified by SHA-1; every bulk-data folder verified by file count and total bytes |
| 3. Cleanup | 65 files deleted (caches, byte-duplicate JSON, one 0-byte failed-run log, superseded demo outputs); 2,194 files archived, byte-identical, to `archive/`. Nothing deleted outside those categories |
| 4. Ablation | The B_aug → E_full ladder completed on Dataset V4 (12 runs, 3 seeds × 4 configs), joining the existing A_plain runs |
| 5. Figures | 8 publication figures, matplotlib only |
| 6. Explainability | One inference module (`src/inference`); JSON/CSV reports and PDF stream mapping in `src/explainability` |
| 7. Demo | CLI, Streamlit app and viva script, all calling that one module; a real header-exclusion bug found and fixed during consolidation |
| 8. Documentation | Demo guide, viva script, 15-question examiner Q&A |
| 9. Report | `docs/thesis/RESULTS_AND_DISCUSSION.md` (+ PDF) |
| 10. This check | Tests, links, demo, figures and this summary |

---

## 2. Final folder tree

```
FragGen/
├── README.md · LICENSE · requirements.txt · .gitignore
├── src/            models · training · inference · preprocessing · explainability · utils
├── configs/        default.yaml, real.yaml
├── data/
│   ├── raw/            corpora (git-ignored; data/raw/README.md)
│   ├── processed_v4/   train.npz · val.npz · test.npz
│   └── metadata/       v4_split_manifest.csv · v4_fragment_metadata.csv · v4_build_report.md
├── checkpoints/
│   ├── v4/             15 checkpoints (A_plain, B_aug, C_aux, D_adv, E_full × 3 seeds)
│   └── calibration/     A_plain_s1337.calib.json
├── reports/
│   ├── figures/        8 figures (+ cross_corpus/, 5 more from the August study)
│   ├── tables/         8 CSV/Markdown result tables
│   ├── confusion_matrices/  21 pooled + long-format confusion-matrix files
│   ├── ablations/       v4/ (15 runs) and cross_corpus_fft75_govdocs1/ (validated August results)
│   └── datasets/        DATA_AUDIT_2026-09-14.md, DATASET_V4_AND_CLEANUP_2026-09-14.md
├── docs/
│   ├── thesis/          RESULTS_AND_DISCUSSION (.md + .pdf), FINDINGS.md, REVIEW1.md
│   ├── viva/            DEMO_GUIDE.md, VIVA_SCRIPT.md, EXAMINER_QA.md
│   ├── architecture/    ARCHITECTURE.md
│   ├── REPOSITORY_AUDIT.md (+ inventory.csv), REMOVED_FILES.md, DATASETS.md
│   └── FINAL_PROJECT_SUMMARY.md (this file, + .pdf)
├── demo/           cli/ (demo.py, viva_demo.py) · streamlit/app.py · samples/ (11 files) · outputs/
├── scripts/        build, train-driver, analysis, calibration, figures, PDF rendering, link check
├── tests/          26 unit + integration tests
└── archive/        2,194 files, byte-identical, superseded datasets/checkpoints/code (archive/README.md)
```

---

## 3. Files removed and archived

Full itemised record: [`REMOVED_FILES.md`](REMOVED_FILES.md).

| outcome | files | size |
|---|---:|---:|
| deleted (caches, duplicates, a failed-run log, an empty marker) | 52 | 282.4 KiB |
| deleted (superseded demo outputs, once their replacements existed) | 13 | 393.3 KiB |
| **archived**, byte-identical | 2,194 | 6.3 GiB |
| relocated inside the active tree, nothing removed | 104,422+ | 44.8 GiB |
| empty folders removed | 14 | — |

Nothing under `data/`, `checkpoints/` or any results folder was deleted — only moved to
`archive/` or renamed within the active tree. The two datasets superseded by V4 (V2, V3)
and the FragmentSets behind the cross-corpus study are archived, not deleted, because
they back a validated result and remain reproducible.

---

## 4. Research completed

### 4.1 Ablation study (`reports/ablations/v4/ABLATION_RESULTS.md`)

The cumulative ladder — A_plain, B_aug, C_aux, D_adv, E_full — was trained on Dataset V4
under an identical protocol (6 epochs, seeds 1337/1338/1339, 4096 B, every training
fragment), the same driver and settings as the existing A_plain runs.

| config | test accuracy | Δ vs A_plain | Holm p | McNemar seeds significant |
|---|---|---|---|---|
| A_plain | 0.8938 ± 0.0036 | — | — | — |
| B_aug | 0.8958 ± 0.0032 | +0.0019 | 1.000 | 2/3 |
| C_aux | 0.8938 ± 0.0086 | −0.0001 | 1.000 | 2/3 |
| D_adv | 0.8603 ± 0.0152 | **−0.0336** | 0.313 | **3/3** |
| E_full | 0.8760 ± 0.0159 | −0.0179 | 0.759 | 2/3 |

No mechanism improves test accuracy at a level distinguishable from seed-to-seed
variation at n = 3. The entropy adversary (D_adv) shows the same behaviour it showed in
the earlier 12-seed cross-corpus study on an unrelated dataset: its McNemar test is
significant in every seed, and in none of them does it fix more fragments than it
breaks — a consistent, if statistically unconfirmed at this sample size, cost. E_full
also costs roughly 7× A_plain's training time for a directionally worse result. Full
statistics, per-class breakdowns and the run provenance (including a GPU
memory-contention issue mid-run and its allocator-only fix) are in that folder.

**Consistency checks passed:** all 15 checkpoints were re-scored independently on the
GPU and matched their original evaluation reports exactly (max |Δ| = 0); accuracy and
macro-F1 recomputed from each confusion matrix matched the reported values to
floating-point precision.

### 4.2 Figures (`reports/figures/`, matplotlib only)

| figure | shows |
|---|---|
| fig1_confusion_matrix | A_plain, V4 test, 3 seeds pooled |
| fig2_precision_recall | per-class precision/recall and F1 |
| fig3_fragment_confidence_timeline | live inference on 3 samples |
| fig4_dataset_class_distribution | V4 balance, by split and by source |
| fig5_training_curves | selection-split accuracy per epoch, all 5 configs; E_full pre-training loss |
| fig6_ablation_comparison | val/test accuracy and macro-F1 per config, with significance annotations |
| fig7_pdf_container_analysis | one file's stream map + all 1,500 test PDF fragments by stream type |
| fig8_pipeline_architecture | pipeline + ByteCNN + ablation mechanisms (also SVG) |

### 4.3 Reports and tables

`reports/ablations/v4/ABLATION_RESULTS.md`, `reports/tables/v4_*.{csv,md}` (8 files),
`reports/confusion_matrices/v4_*.csv` (21 files, pooled and per-run), and
`reports/tables/v4_test_pdf_fragments_by_stream_region.{csv,json}` — the byte-verified
PDF stream-type breakdown behind Section 9 of the results report.

---

## 5. Reproducibility checklist

- [x] No broken imports — `tests/test_repository.py` imports every `src` module and compiles all active code
- [x] No import of the pre-refactor `fraggen` package anywhere in `src/`, `scripts/`, `demo/`, `tests/`
- [x] No duplicate files — the 5 byte-identical duplicates found in the audit were removed; the rest are documented as intentional (demo samples) or genuine corpus duplicates
- [x] All reports linked correctly — `python scripts/check_links.py` finds 0 broken links in 28 active documents (1 legacy pre-refactor path noted, harmless)
- [x] All figures render — 8/8 regenerated from current data, visually inspected
- [x] Streamlit launches — confirmed live in-browser: sample selection, verdict banner, vote chart, confidence chart, fragment timeline with PDF stream map, per-fragment table and downloads all work
- [x] CLI works — `demo/cli/demo.py` tested on multiple samples including the tricky PDF, with `--map`
- [x] Viva script runs — full run including the stream-map explanation, `--no-pause` mode verified
- [x] Checkpoints load — all 15 V4 checkpoints load with the correct 9-class label space (`tests/test_checkpoints.py`)
- [x] README is accurate — quick start, layout, reproduction commands and ablation table match the actual repository and results
- [x] Test suite passes — 26/26 (`python -m unittest discover -s tests -t .`)
- [x] Refactor equivalence — the new `src` package verified bit-identical to the pre-refactor `fraggen` package (logits, evaluation output, one-epoch training weights) before the old package was retired

---

## 6. GitHub readiness checklist

- [x] `.gitignore` excludes raw data, `archive/data`, `archive/checkpoints`, `demo/outputs`, and Python/OS junk
- [x] `LICENSE` (MIT) covers code and documentation; `docs/DATASETS.md` and the README state that the datasets are not covered and carry their own third-party terms
- [x] `requirements.txt` is current (adds `streamlit`, notes tested versions)
- [ ] **Repository is not yet a git repository.** `git init`, then an initial commit, is needed before pushing
- [ ] `data/processed_v4/train.npz` is 132 MiB, over GitHub's 100 MB per-file limit — track with Git LFS, or keep it out of the pushed repository and rely on `data/metadata/` + `scripts/build_processed_v4.py` to rebuild it
- [ ] Decide whether `checkpoints/` (204 MiB across 15 files) goes into the repository, a release asset, or Git LFS
- [x] No secrets or machine-specific absolute paths in tracked files (`.claude/` and `powershell.cmd` are local environment files and are git-ignored)

---

## 7. What is intentionally out of scope

Per the cleanup brief's own rules, this pass did not: retrain any deployed model, alter
any trained checkpoint's weights, overwrite the validated cross-corpus result, or invent
any metric not read from an experiment output file. Every number in
`docs/thesis/RESULTS_AND_DISCUSSION.md` traces to a named file in `reports/` or
`checkpoints/`.

---

## 8. One remaining decision for the user

`git init` was not run — creating a git repository is an environment change outside
this cleanup's remit. Once initialised, the two size issues above (`train.npz` and the
checkpoints) should be resolved before the first push.
