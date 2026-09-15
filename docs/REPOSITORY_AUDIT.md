# FragGen V4 — Repository Audit

**Phase 1 of the repository cleanup. Written before any file was moved or deleted.**

| | |
|---|---|
| Audit date | 2026-09-14 (snapshot taken 22:40 IST) |
| Repository root | `fraggen/` |
| Method | Full recursive walk (`os.walk`) of every file. All files that share a byte size with another file were hashed (full SHA-1 up to 64 MiB; first MiB + last MiB + size for larger files) to find duplicates. |
| Per-file inventory | [`REPOSITORY_AUDIT_inventory.csv`](REPOSITORY_AUDIT_inventory.csv): 592 rows covering every non-raw-data file, plus V4 and DFRWS metadata, each with its category, planned action, destination and reason |
| Scope note | The ablation runs started during this audit (B_aug/C_aux/D_adv/E_full) write to a separate staging folder outside the repository, so they are not counted here |

---

## 1. Totals

| metric | value |
|---|---|
| **Total files** | **106,708** |
| **Total folders** | **86** (excluding the root) |
| **Storage used** | **51.3 GiB** (55,130,706,975 bytes) |
| Byte-duplicate groups | 33 groups, 66 files, 1.5 MiB redundant |
| Python bytecode caches | 45 `.pyc` files in 6 `__pycache__` folders (236 KiB) |
| Zero-byte files | 4 (three empty `__init__.py`, one aborted-run log) |
| Empty folders | 2 (`data/v2_scaled/`, `reports/overnight/`) |
| OS junk (`.DS_Store`, `Thumbs.db`, `desktop.ini`) | 0 |
| Notebooks / `.ipynb_checkpoints` / `.pytest_cache` / `.mypy_cache` | 0 |

### Storage by top-level folder

| folder | files | size | note |
|---|---:|---:|---|
| `data/` | 106,131 | 51.1 GiB | 99.6% of the repository |
| `checkpoints/` | 105 | 204.0 MiB | 92 `.pt` + 13 calibration JSON |
| `reports/` | 315 | 1.4 MiB | results, logs, write-ups |
| `demo_v4/` | 32 | 993.9 KiB | demo layer |
| `fraggen/` | 62 | 273.5 KiB | Python package (40 of these files are `.pyc` bytecode) |
| `scripts/` | 29 | 231.6 KiB | experiment tooling |
| `tools/` | 22 | 88.1 KiB | one-off ingestion scripts |
| root files | 4 | 14.9 KiB | `README.md`, `DATASETS.md`, `requirements.txt`, `powershell.cmd` |
| `reports_v2/`, `reports_v3/` | 3 | 11.9 KiB | V2/V3 dataset notes |
| `configs/` | 2 | 3.2 KiB | YAML |
| `.claude/` | 3 | 501 B | local editor/agent configuration |

### Storage inside `data/`

| folder | files | size | role |
|---|---:|---:|---|
| `FFT-75/` | 6 | 27.4 GiB | FiFTy scenario-1 npz (512 B + 4 K). Train corpus for the cross-corpus study |
| `govdocs1_files/` | 16,318 | 6.0 GiB | extracted GovDocs1 files (threads 000–020, label-set types) |
| `Govdoc1/` | 21 | 5.7 GiB | original GovDocs1 thread zips `000.zip`–`020.zip` |
| `processed_v2/` | 7 | 4.7 GiB | Dataset V2 (superseded) |
| `tgif_gif/` | 1,824 | 1.8 GiB | TGIF animated GIFs (V4 gif source) |
| `rico_png/` | 55,912 | 1022.8 MiB | RICO screenshots (V4 png source) |
| `processed/` | 10 | 928.9 MiB | FFT-75 / GovDocs1 / DFRWS FragmentSets behind the **cross-corpus results** |
| `uci_csv/` | 2,000 | 885.9 MiB | real CSVs (V4 csv source) |
| `DFRWS/` | 9 | 671.0 MiB | DFRWS 2007 image + ground truth (+ 2006 zip) |
| `wiki_pages/` | 26,628 | 534.3 MiB | full Simple-Wikipedia articles (V4 xml source) |
| `github_zip/` | 1,375 | 523.0 MiB | real `.zip` archives (V4 zip source) |
| `processed_v3/` | 6 | 456.8 MiB | Dataset V3 (superseded) |
| `raw_downloads/` | 1 | 339.7 MiB | `simplewiki-latest-pages-articles.xml.bz2` |
| `processed_v4/` | 6 | 195.9 MiB | **Dataset V4**: train 61,940 / val 13,111 / test 13,754 fragments |
| `semantic_annotations.zip` | 1 | 150.5 MiB | RICO archive that `rico_png/` was extracted from |
| `v2/` | 4 | 10.9 MiB | small V2 prototype build |
| `wikipedia_xml/` | 2,000 | 8.1 MiB | superseded xml staging (median file 822 B, too small to use) |
| `synthetic/` | 3 | 6.8 MiB | synthetic A/B corpora (can be regenerated) |
| `v2_scaled/` | 0 | 0 | empty folder |

---

## 2. Classification

Every file was put in one of the seven requested categories. Research, Data, Models and Demo are split further so that validated experiment outputs are never mixed up with generated demo files.

| category | files | size |
|---|---:|---:|
| **Source Code** (Python modules, training, inference, utilities, configs) | 74 | 372.9 KiB |
| **Data**: raw corpora | 106,089 | 44.8 GiB |
| **Data**: processed, current (V4 npz) | 3 | 190.1 MiB |
| **Data**: processed, superseded (cross-corpus, V2, V3, v2 prototype, synthetic) | 24 | 6.0 GiB |
| **Data**: metadata (manifests, build reports, DFRWS ground truth) | 15 | 100.9 MiB |
| **Models**: checkpoints | 92 | 204.0 MiB |
| **Models**: calibration files | 13 | 7.9 KiB |
| **Research**: result files (evaluation JSON, significance text) | 104 | 941.8 KiB |
| **Research**: training/eval logs | 184 | 100.0 KiB |
| **Research**: documents (Markdown) | 19 | 135.9 KiB |
| **Research**: figures | 5 | 272.8 KiB |
| **Demo**: code, docs, config | 8 | 68.6 KiB |
| **Demo**: sample files | 11 | 520.2 KiB |
| **Generated** (demo PNG/JSON/CSV outputs, synthetic eval, smoke output) | 15 | 404.0 KiB |
| **Unwanted** (bytecode, duplicate JSON, aborted-run log, empty marker) | 52 | 282.4 KiB |
| **total** | **106,708** | **51.3 GiB** |

> **Not unwanted, despite matching a pattern.** `data/govdocs1_files/` holds 110 `.log` files (344 MiB, the largest is 250 MiB). These are real GovDocs1 corpus documents whose extension happens to be `.log`, not project logs, so they are kept. Likewise, the 184 training and eval logs under `reports/` are the only record of per-epoch validation accuracy (used by the training-curve figure) and are kept as research evidence.

---

## 3. Duplicate files

33 groups of byte-identical files were found. Each group has a different explanation:

| group type | groups | resolution |
|---|---:|---|
| `reports/ablations/{A_plain,B_aug,C_aux,D_adv,E_full}.json` identical to the matching `*_s1337.json` | 5 | **remove** the copy without the `_s1337` suffix (the seed-tagged name is the one the driver and the significance scripts use). SHA-1 is re-checked right before deletion. |
| `demo_v4/samples/*` identical to a raw-corpus file | 11 | **keep**. These copies are intentional, so the demo works on a machine without the multi-GB raw corpora. |
| inside `data/govdocs1_files/` | 16 | **keep**. They are genuine duplicates within the public GovDocs1 corpus. |
| inside `data/uci_csv/` | 1 | **keep**. It is a genuine duplicate within the scraped source. |

### Research-integrity check on the raw-corpus duplicates

Each raw duplicate was checked against the V4 split manifest (`data/processed_v4/split_manifest.csv`):

| finding | fragments involved | impact |
|---|---|---|
| `govdocs1/003/003458.txt` (**test**, 20 fragments) is byte-identical to `govdocs1/005/005888.txt` (**val**, 20 fragments) | 20 test fragments (0.15% of 13,754) | **No train↔test leakage.** `val.npz` is used only for reporting. Best-epoch selection in `train.py` uses a 10% split carved from `train.npz`, so this val↔test overlap cannot affect the model or the test number. It is recorded here for completeness. |
| `govdocs1/012/012900.html` (**train**, 7 fragments, label *html*) is byte-identical to `govdocs1/012/012946.txt` (**train**, 7 fragments, label *txt*) | 14 train fragments (0.02% of 61,940) | Minor label noise inherited from GovDocs1's extension-based labelling. Too small to matter; recorded for completeness. |
| `govdocs1/003/003937.html` and `govdocs1/013/013764.html` (both **train**) | 4 train fragments | Harmless: both copies are in the same split. |
| the remaining 14 groups | 0 | At most one member of each group was sampled into V4, or none. |

No train↔test pair of identical source files exists in Dataset V4.

---

## 4. Removable files (Phase 3)

Only the categories the cleanup brief allows are removed:

| item | files | size | reason |
|---|---:|---:|---|
| `__pycache__/` in `fraggen/`, `scripts/`, `demo_v4/` | 45 | 236.0 KiB | bytecode cache |
| `reports/ablations/{A_plain,B_aug,C_aux,D_adv,E_full}.json` | 5 | 46.4 KiB | byte-identical duplicate JSON outputs |
| `reports/ablations/logs/A_plain_s1339.train.log` | 1 | 0 B | aborted run that left no checkpoint or report (a failed run) |
| `fraggen/features/__init__.py` | 1 | 0 B | empty marker of a package that is merged into `src/preprocessing/` |
| `demo_v4/figures/*.png` | 5 | 344.8 KiB | earlier figure drafts, superseded by the Phase-5 publication figures. They are held aside and only deleted once their replacements exist. |
| `demo_v4/outputs/**` | 8 | 48.5 KiB | demo JSON/CSV exports, regenerated by the Phase-10 CLI/viva runs |
| `data/v2_scaled/`, `reports/overnight/` | 0 | 0 | empty folders |
| **total** | **65 files + 2 folders** | **675.7 KiB** | |

Nothing under `data/`, `checkpoints/` or any experiment-results folder is deleted.

---

## 5. Preserved files

| what | where it goes | why it is preserved |
|---|---|---|
| **Cross-corpus results** (`reports/ablations_gpu`, `blocksize_512`, `blocksize_4096`, `fulldata_512`, `blocksize_comparison.txt`, `forensic_*.json`, `real_baseline.json`, `eval_dfrws_4096.json`) | `reports/ablations/cross_corpus_fft75_govdocs1/` | These validated results include the **0.6812 cross-corpus accuracy** (FFT-75 → GovDocs1, 4096 B). The content is untouched: every file is SHA-1-verified after the move. |
| `reports/FINDINGS.md`, `reports/REVIEW1.md` | `docs/thesis/` | the validated research write-ups |
| V4 A_plain results (`reports/v4_results/`) | `reports/ablations/v4/` | 3-seed V4 results: val 0.9082 ± 0.0049, test 0.8938 ± 0.0036. The driver's `summary.{json,md}` files are renamed `driver_summary_A_plain.*` so that the B–E driver cannot overwrite them. |
| V4 checkpoints `checkpoints/v4_results/A_plain_s133{7,8,9}.pt` | `checkpoints/v4/` | the trained models (byte-identical) |
| V4 calibration `A_plain_s1337.calib.json` | `checkpoints/calibration/` | temperature T = 1.0214 |
| Dataset V4 (`data/processed_v4/*.npz`) | stays in place | current dataset |
| V4 metadata (`metadata.csv`, `split_manifest.csv`, `report.md`) | `data/metadata/v4_*` | provenance for every fragment |
| raw corpora (GovDocs1, DFRWS, FFT-75, RICO, TGIF, CSV, ZIP, Wikipedia) | `data/raw/` | source of record for every dataset |
| demo samples | `demo/samples/` | 9 correctly-classified samples + 2 documented tricky cases |
| `configs/`, `requirements.txt` | stay in place | configuration |
| `powershell.cmd`, `.claude/` | stay in place (git-ignored) | local environment/editor configuration, not part of the project |

---

## 6. Archive candidates (moved to `archive/`, byte-identical, not deleted)

| item | files | size | reason for archiving |
|---|---:|---:|---|
| `data/processed_v2/`, `data/processed_v3/`, `data/v2/` | 17 | 5.2 GiB | superseded datasets: V2 zip-overweighted, V3 severely imbalanced (see `DATA_AUDIT_2026-09-14.md`) |
| `data/processed/` | 10 | 928.9 MiB | FragmentSets behind the cross-corpus experiment. Not needed by V4 or the demo, but required to reproduce FINDINGS.md. |
| `data/synthetic/`, `data/wikipedia_xml/` | 2,003 | 14.9 MiB | synthetic smoke-test corpora; superseded xml staging |
| `checkpoints/ablations_gpu/`, `blocksize_512/`, `blocksize_4096/`, `fulldata_512/` | 95 | 184.1 MiB | cross-corpus checkpoints (back the preserved cross-corpus reports) |
| `checkpoints/{A_plain,B_aug,C_aux,D_adv,E_full,model}.pt` | 6 | 13.4 MiB | first single-seed 512 B runs, plus one undocumented checkpoint |
| `reports/ablations/` (early single-seed CPU matrix, minus duplicates) | 21 | ~64 KiB | superseded by the 12-seed GPU matrix |
| `reports/PROJECT_STATUS.md` | 1 | 17.6 KiB | 2026-09-10 status snapshot from before V4 |
| `reports/corpus_balance.md`, `reports_v2/`, `reports_v3/` | 4 | 17.9 KiB | Dataset V2/V3 notes |
| `reports/eval.json`, `reports/smoke_report.txt` | 2 | 10.7 KiB | synthetic-corpus outputs |
| `scripts/build_processed_v2.py`, `build_processed_v3.py`, `build_v2.py`, `build_train_index.py`, `verify_v2.py`, `verify_v3.py` and `fraggen/data/v2_*.py` | 9 | 114.6 KiB | Dataset V2/V3 tooling |
| `scripts/demo.py`, `scripts/classify.py`, `demo_v4/make_visuals.py` | 3 | 29.1 KiB | August 512 B demo and duplicate inference logic; superseded by `src/inference/` and `demo/` |
| `scripts/run_overnight.sh` | 1 | 3.1 KiB | one-off orchestration hard-coded to pre-refactor paths |
| `tools/ingestion/` | 22 | 88.1 KiB | one-off download/scrape scripts; several reference temp folders deleted on 2026-09-14 |

Archived code keeps its original `fraggen.*` imports and is not maintained. It is preserved for provenance only, as `archive/README.md` explains.

### Optional space recovery (flagged, **not** acted on; the user decides)

| item | size | observation |
|---|---:|---|
| `data/raw/FFT-75/` | 27.4 GiB | only needed to rebuild the archived cross-corpus FragmentSets |
| `archive/data/processed_v2/` | 4.7 GiB | superseded dataset |
| `data/raw/Govdoc1/*.zip` | 5.7 GiB | original thread archives; the label-set files are already extracted into `govdocs1_files/` |
| `data/raw/DFRWS/dfrws-2007-challenge.zip` | 314.0 MiB | compressed copy of the `.img` next to it; no script references it |
| `data/raw/DFRWS/(Copy) dfrws-2006-challenge.zip` | 42.6 MiB | 2006 challenge; no script references it |

---

## 7. Other observations

1. **Mislabelled driver summary.** `reports/v4_results/summary.md` was generated by `scripts/run_ablations.py`, whose table header is hard-coded as *"Train: FFT-75 … Cross-corpus test: GovDocs1"* and which appends the FFT-75 logistic-regression baseline row. For the V4 run both the header and the baseline row are wrong labels, although the A_plain numbers themselves are correct. The file is preserved unchanged; the new V4 ablation report states the correct setup, and the driver's header is made setup-aware going forward.
2. **`scripts/classify.py` and `demo_v4/fragcore.py` both implement fragmenting, temperature scaling and voting.** This is duplicated inference logic, consolidated into `src/inference/`.
3. **Naming in the brief vs the code.** The brief asks for `C_contrastive`. The repository's cumulative ablation ladder names its third rung `C_aux` (+ entropy auxiliary features); contrastive pre-training is added at `E_full`. The code ladder is kept so V4 stays directly comparable with every earlier ablation. The contrastive contribution is measured as E_full − D_adv.
4. **Streamlit uploads leak temp files.** `demo_v4/app.py` writes each upload with `NamedTemporaryFile(delete=False)` and never removes it. This is fixed in the refactor by classifying bytes in memory.
5. **GitHub size limit.** `data/processed_v4/train.npz` is 131.9 MiB, over GitHub's 100 MB per-file limit, so it must be git-ignored or tracked with Git LFS. Raw corpora must not be pushed at all: third-party licences apply, and the total is 44.8 GiB.
6. **No test suite exists** (`tests/` is absent and pytest is not installed). A stdlib `unittest` suite is added so it runs without extra dependencies.
