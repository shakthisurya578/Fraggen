# Removed and archived files

Phase 3 record of the 2026-09-14 repository cleanup. It is generated from the migration executor's own log, so every row here is an operation that was actually performed.

**Verification.** Every file moved individually was SHA-1-hashed before and after the move. Every whole directory that was renamed was re-walked afterwards: code, report and checkpoint trees were compared file by file (size and SHA-1), and bulk-data trees by file count and total bytes. All checks passed. Deletions were limited to the categories the cleanup brief allows, and duplicate JSONs were re-hashed at deletion time.

## Summary

| outcome | files | size |
|---|---:|---:|
| deleted: caches, duplicates, failed-run log, empty marker | 52 | 282.4 KiB |
| deleted: superseded generated demo outputs | 13 | 393.3 KiB |
| **archived** (moved to `archive/`, byte-identical) | 2,194 | 6.3 GiB |
| relocated inside the active tree (not removed) | 104,422 | 44.8 GiB |
| empty folders removed | 14 | 0 B |

## 1. Deleted

| filename | original path | reason | status |
|---|---|---|---|
| `A_plain.json` | `reports/ablations/A_plain.json` | byte-identical duplicate of reports/ablations/A_plain_s1337.json (sha1 verified at deletion time) | deleted |
| `B_aug.json` | `reports/ablations/B_aug.json` | byte-identical duplicate of reports/ablations/B_aug_s1337.json (sha1 verified at deletion time) | deleted |
| `C_aux.json` | `reports/ablations/C_aux.json` | byte-identical duplicate of reports/ablations/C_aux_s1337.json (sha1 verified at deletion time) | deleted |
| `D_adv.json` | `reports/ablations/D_adv.json` | byte-identical duplicate of reports/ablations/D_adv_s1337.json (sha1 verified at deletion time) | deleted |
| `E_full.json` | `reports/ablations/E_full.json` | byte-identical duplicate of reports/ablations/E_full_s1337.json (sha1 verified at deletion time) | deleted |
| `A_plain_s1339.train.log` | `reports/ablations/logs/A_plain_s1339.train.log` | 0-byte log of an aborted run: no checkpoint, no report (failed run) | deleted |
| `__init__.py` | `fraggen/features/__init__.py` | 0-byte package marker; fraggen.features merged into src/preprocessing | deleted |
| 1 × `*.pyc` | `demo_v4/__pycache__/` | Python bytecode cache (regenerated automatically) | deleted |
| 11 × `*.pyc` | `fraggen/__pycache__/` | Python bytecode cache (regenerated automatically) | deleted |
| 17 × `*.pyc` | `fraggen/data/__pycache__/` | Python bytecode cache (regenerated automatically) | deleted |
| 4 × `*.pyc` | `fraggen/features/__pycache__/` | Python bytecode cache (regenerated automatically) | deleted |
| 8 × `*.pyc` | `fraggen/models/__pycache__/` | Python bytecode cache (regenerated automatically) | deleted |
| 4 × `*.pyc` | `scripts/__pycache__/` | Python bytecode cache (regenerated automatically) | deleted |

<details><summary>Every deleted bytecode file</summary>

| filename | original path |
|---|---|
| `fragcore.cpython-311.pyc` | `demo_v4/__pycache__/fragcore.cpython-311.pyc` |
| `__init__.cpython-310.pyc` | `fraggen/__pycache__/__init__.cpython-310.pyc` |
| `__init__.cpython-311.pyc` | `fraggen/__pycache__/__init__.cpython-311.pyc` |
| `calibration.cpython-311.pyc` | `fraggen/__pycache__/calibration.cpython-311.pyc` |
| `config.cpython-310.pyc` | `fraggen/__pycache__/config.cpython-310.pyc` |
| `config.cpython-311.pyc` | `fraggen/__pycache__/config.cpython-311.pyc` |
| `eval.cpython-310.pyc` | `fraggen/__pycache__/eval.cpython-310.pyc` |
| `eval.cpython-311.pyc` | `fraggen/__pycache__/eval.cpython-311.pyc` |
| `metrics.cpython-310.pyc` | `fraggen/__pycache__/metrics.cpython-310.pyc` |
| `metrics.cpython-311.pyc` | `fraggen/__pycache__/metrics.cpython-311.pyc` |
| `train.cpython-310.pyc` | `fraggen/__pycache__/train.cpython-310.pyc` |
| `train.cpython-311.pyc` | `fraggen/__pycache__/train.cpython-311.pyc` |
| `__init__.cpython-310.pyc` | `fraggen/data/__pycache__/__init__.cpython-310.pyc` |
| `__init__.cpython-311.pyc` | `fraggen/data/__pycache__/__init__.cpython-311.pyc` |
| `augment.cpython-310.pyc` | `fraggen/data/__pycache__/augment.cpython-310.pyc` |
| `augment.cpython-311.pyc` | `fraggen/data/__pycache__/augment.cpython-311.pyc` |
| `dfrws.cpython-310.pyc` | `fraggen/data/__pycache__/dfrws.cpython-310.pyc` |
| `dfrws.cpython-311.pyc` | `fraggen/data/__pycache__/dfrws.cpython-311.pyc` |
| `fft75.cpython-310.pyc` | `fraggen/data/__pycache__/fft75.cpython-310.pyc` |
| `fft75.cpython-311.pyc` | `fraggen/data/__pycache__/fft75.cpython-311.pyc` |
| `fragments.cpython-310.pyc` | `fraggen/data/__pycache__/fragments.cpython-310.pyc` |
| `fragments.cpython-311.pyc` | `fraggen/data/__pycache__/fragments.cpython-311.pyc` |
| `govdocs.cpython-310.pyc` | `fraggen/data/__pycache__/govdocs.cpython-310.pyc` |
| `govdocs.cpython-311.pyc` | `fraggen/data/__pycache__/govdocs.cpython-311.pyc` |
| `synthetic.cpython-310.pyc` | `fraggen/data/__pycache__/synthetic.cpython-310.pyc` |
| `synthetic.cpython-311.pyc` | `fraggen/data/__pycache__/synthetic.cpython-311.pyc` |
| `v2_builder.cpython-311.pyc` | `fraggen/data/__pycache__/v2_builder.cpython-311.pyc` |
| `v2_fragments.cpython-311.pyc` | `fraggen/data/__pycache__/v2_fragments.cpython-311.pyc` |
| `v2_sampler.cpython-311.pyc` | `fraggen/data/__pycache__/v2_sampler.cpython-311.pyc` |
| `__init__.cpython-310.pyc` | `fraggen/features/__pycache__/__init__.cpython-310.pyc` |
| `__init__.cpython-311.pyc` | `fraggen/features/__pycache__/__init__.cpython-311.pyc` |
| `entropy.cpython-310.pyc` | `fraggen/features/__pycache__/entropy.cpython-310.pyc` |
| `entropy.cpython-311.pyc` | `fraggen/features/__pycache__/entropy.cpython-311.pyc` |
| `__init__.cpython-310.pyc` | `fraggen/models/__pycache__/__init__.cpython-310.pyc` |
| `__init__.cpython-311.pyc` | `fraggen/models/__pycache__/__init__.cpython-311.pyc` |
| `baseline.cpython-310.pyc` | `fraggen/models/__pycache__/baseline.cpython-310.pyc` |
| `baseline.cpython-311.pyc` | `fraggen/models/__pycache__/baseline.cpython-311.pyc` |
| `cnn.cpython-310.pyc` | `fraggen/models/__pycache__/cnn.cpython-310.pyc` |
| `cnn.cpython-311.pyc` | `fraggen/models/__pycache__/cnn.cpython-311.pyc` |
| `contrastive.cpython-310.pyc` | `fraggen/models/__pycache__/contrastive.cpython-310.pyc` |
| `contrastive.cpython-311.pyc` | `fraggen/models/__pycache__/contrastive.cpython-311.pyc` |
| `classify.cpython-311.pyc` | `scripts/__pycache__/classify.cpython-311.pyc` |
| `make_synthetic.cpython-310.pyc` | `scripts/__pycache__/make_synthetic.cpython-310.pyc` |
| `prep_fft75.cpython-311.pyc` | `scripts/__pycache__/prep_fft75.cpython-311.pyc` |
| `smoke_baseline.cpython-310.pyc` | `scripts/__pycache__/smoke_baseline.cpython-310.pyc` |

</details>

## 2. Deleted generated demo outputs

Replaced by `reports/figures/` (publication figures) and `demo/outputs/` (regenerated by the Phase-10 CLI and viva runs). The files were set aside rather than deleted immediately, and only deleted once their replacements existed.

| filename | original path | reason | status |
|---|---|---|---|
| `sample.csv` | `demo_v4/outputs/sample.csv` | demo JSON/CSV exports; regenerated by the Phase-10 CLI/viva runs into demo/outputs/ | deleted |
| `sample.json` | `demo_v4/outputs/sample.json` | demo JSON/CSV exports; regenerated by the Phase-10 CLI/viva runs into demo/outputs/ | deleted |
| `csv_sample.csv.json` | `demo_v4/outputs/viva/csv_sample.csv.json` | demo JSON/CSV exports; regenerated by the Phase-10 CLI/viva runs into demo/outputs/ | deleted |
| `pdf_sample.pdf.json` | `demo_v4/outputs/viva/pdf_sample.pdf.json` | demo JSON/CSV exports; regenerated by the Phase-10 CLI/viva runs into demo/outputs/ | deleted |
| `pdf_tricky_pdf_container_example.pdf.json` | `demo_v4/outputs/viva/pdf_tricky_pdf_container_example.pdf.json` | demo JSON/CSV exports; regenerated by the Phase-10 CLI/viva runs into demo/outputs/ | deleted |
| `png_sample.png.json` | `demo_v4/outputs/viva/png_sample.png.json` | demo JSON/CSV exports; regenerated by the Phase-10 CLI/viva runs into demo/outputs/ | deleted |
| `xml_tricky_xml_html_boundary_example.xml.json` | `demo_v4/outputs/viva/xml_tricky_xml_html_boundary_example.xml.json` | demo JSON/CSV exports; regenerated by the Phase-10 CLI/viva runs into demo/outputs/ | deleted |
| `zip_sample.zip.json` | `demo_v4/outputs/viva/zip_sample.zip.json` | demo JSON/CSV exports; regenerated by the Phase-10 CLI/viva runs into demo/outputs/ | deleted |
| `fig1_confusion_matrix.png` | `demo_v4/figures/fig1_confusion_matrix.png` | earlier figure drafts; superseded by the publication figures in reports/figures/ (duplicate generated images) | deleted |
| `fig2_precision_recall.png` | `demo_v4/figures/fig2_precision_recall.png` | earlier figure drafts; superseded by the publication figures in reports/figures/ (duplicate generated images) | deleted |
| `fig3_fragment_timeline.png` | `demo_v4/figures/fig3_fragment_timeline.png` | earlier figure drafts; superseded by the publication figures in reports/figures/ (duplicate generated images) | deleted |
| `fig4_class_distribution.png` | `demo_v4/figures/fig4_class_distribution.png` | earlier figure drafts; superseded by the publication figures in reports/figures/ (duplicate generated images) | deleted |
| `fig5_training_curves.png` | `demo_v4/figures/fig5_training_curves.png` | earlier figure drafts; superseded by the publication figures in reports/figures/ (duplicate generated images) | deleted |

## 4. Archived (moved to `archive/`, byte-identical)

### Whole folders

| folder | original path | new path | files | size | reason | status |
|---|---|---|---:|---:|---|---|
| `wikipedia_xml/` | `data/wikipedia_xml/` | `archive/data/wikipedia_xml_staging/` | 2,000 | 8.1 MiB | superseded xml staging (median 822 B, unusable); replaced by data/raw/wiki_pages | archived |
| `processed/` | `data/processed/` | `archive/data/processed_cross_corpus_fft75_govdocs1/` | 10 | 928.9 MiB | FFT-75 / GovDocs1 / DFRWS FragmentSets behind the cross-corpus results | archived |
| `processed_v2/` | `data/processed_v2/` | `archive/data/processed_v2/` | 7 | 4.7 GiB | Dataset V2 (superseded, zip-overweighted) | archived |
| `processed_v3/` | `data/processed_v3/` | `archive/data/processed_v3/` | 6 | 456.8 MiB | Dataset V3 (superseded, severely imbalanced) | archived |
| `v2/` | `data/v2/` | `archive/data/v2_small/` | 4 | 10.9 MiB | small V2 prototype build | archived |
| `synthetic/` | `data/synthetic/` | `archive/data/synthetic/` | 3 | 6.8 MiB | synthetic A/B corpora (regenerable with scripts/make_synthetic.py) | archived |
| `ingestion/` | `tools/ingestion/` | `archive/code/ingestion/` | 22 | 88.1 KiB | one-off raw-data acquisition scripts (several reference deleted temp folders) | archived |
| `ablations_gpu/` | `checkpoints/ablations_gpu/` | `archive/checkpoints/cross_corpus_fft75_govdocs1/ablations_512_12seed/` | 62 | 133.2 MiB | cross-corpus 512B x 12-seed checkpoints (back FINDINGS.md sec. 3) | archived |
| `blocksize_512/` | `checkpoints/blocksize_512/` | `archive/checkpoints/cross_corpus_fft75_govdocs1/blocksize_512/` | 10 | 22.2 MiB | cross-corpus block-size study, 512B arm | archived |
| `blocksize_4096/` | `checkpoints/blocksize_4096/` | `archive/checkpoints/cross_corpus_fft75_govdocs1/blocksize_4096/` | 20 | 22.2 MiB | cross-corpus block-size study, 4096B arm (backs the 0.6812 result) | archived |
| `fulldata_512/` | `checkpoints/fulldata_512/` | `archive/checkpoints/cross_corpus_fft75_govdocs1/fulldata_512/` | 3 | 6.5 MiB | cross-corpus full-data 512B runs | archived |

### Individual files

| filename | original path | new path | reason | status |
|---|---|---|---|---|
| `A_plain_s1337.json` | `reports/ablations/A_plain_s1337.json` | `archive/reports/early_single_seed_ablations_512/A_plain_s1337.json` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `A_plain_s1338.json` | `reports/ablations/A_plain_s1338.json` | `archive/reports/early_single_seed_ablations_512/A_plain_s1338.json` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `B_aug_s1337.json` | `reports/ablations/B_aug_s1337.json` | `archive/reports/early_single_seed_ablations_512/B_aug_s1337.json` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `C_aux_s1337.json` | `reports/ablations/C_aux_s1337.json` | `archive/reports/early_single_seed_ablations_512/C_aux_s1337.json` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `D_adv_s1337.json` | `reports/ablations/D_adv_s1337.json` | `archive/reports/early_single_seed_ablations_512/D_adv_s1337.json` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `E_full_s1337.json` | `reports/ablations/E_full_s1337.json` | `archive/reports/early_single_seed_ablations_512/E_full_s1337.json` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `A_plain.eval.log` | `reports/ablations/logs/A_plain.eval.log` | `archive/reports/early_single_seed_ablations_512/logs/A_plain.eval.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `A_plain.train.log` | `reports/ablations/logs/A_plain.train.log` | `archive/reports/early_single_seed_ablations_512/logs/A_plain.train.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `A_plain_s1338.eval.log` | `reports/ablations/logs/A_plain_s1338.eval.log` | `archive/reports/early_single_seed_ablations_512/logs/A_plain_s1338.eval.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `A_plain_s1338.train.log` | `reports/ablations/logs/A_plain_s1338.train.log` | `archive/reports/early_single_seed_ablations_512/logs/A_plain_s1338.train.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `B_aug.eval.log` | `reports/ablations/logs/B_aug.eval.log` | `archive/reports/early_single_seed_ablations_512/logs/B_aug.eval.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `B_aug.train.log` | `reports/ablations/logs/B_aug.train.log` | `archive/reports/early_single_seed_ablations_512/logs/B_aug.train.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `C_aux.eval.log` | `reports/ablations/logs/C_aux.eval.log` | `archive/reports/early_single_seed_ablations_512/logs/C_aux.eval.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `C_aux.train.log` | `reports/ablations/logs/C_aux.train.log` | `archive/reports/early_single_seed_ablations_512/logs/C_aux.train.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `D_adv.eval.log` | `reports/ablations/logs/D_adv.eval.log` | `archive/reports/early_single_seed_ablations_512/logs/D_adv.eval.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `D_adv.train.log` | `reports/ablations/logs/D_adv.train.log` | `archive/reports/early_single_seed_ablations_512/logs/D_adv.train.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `E_full.eval.log` | `reports/ablations/logs/E_full.eval.log` | `archive/reports/early_single_seed_ablations_512/logs/E_full.eval.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `E_full.train.log` | `reports/ablations/logs/E_full.train.log` | `archive/reports/early_single_seed_ablations_512/logs/E_full.train.log` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `summary.json` | `reports/ablations/summary.json` | `archive/reports/early_single_seed_ablations_512/summary.json` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `summary.md` | `reports/ablations/summary.md` | `archive/reports/early_single_seed_ablations_512/summary.md` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `summary_5config_singleseed.md` | `reports/ablations/summary_5config_singleseed.md` | `archive/reports/early_single_seed_ablations_512/summary_5config_singleseed.md` | first single-seed CPU ablation (2026-08-21), superseded by the 12-seed matrix | archived |
| `A_plain.pt` | `checkpoints/A_plain.pt` | `archive/checkpoints/early_single_seed_512/A_plain.pt` | first single-seed 512B ablation checkpoints (superseded by 12-seed matrix) | archived |
| `B_aug.pt` | `checkpoints/B_aug.pt` | `archive/checkpoints/early_single_seed_512/B_aug.pt` | first single-seed 512B ablation checkpoints (superseded by 12-seed matrix) | archived |
| `C_aux.pt` | `checkpoints/C_aux.pt` | `archive/checkpoints/early_single_seed_512/C_aux.pt` | first single-seed 512B ablation checkpoints (superseded by 12-seed matrix) | archived |
| `D_adv.pt` | `checkpoints/D_adv.pt` | `archive/checkpoints/early_single_seed_512/D_adv.pt` | first single-seed 512B ablation checkpoints (superseded by 12-seed matrix) | archived |
| `E_full.pt` | `checkpoints/E_full.pt` | `archive/checkpoints/early_single_seed_512/E_full.pt` | first single-seed 512B ablation checkpoints (superseded by 12-seed matrix) | archived |
| `model.pt` | `checkpoints/model.pt` | `archive/checkpoints/misc/model.pt` | undocumented single checkpoint (2026-08-24) | archived |
| `make_visuals.py` | `demo_v4/make_visuals.py` | `archive/code/legacy_demo/make_visuals.py` | superseded by scripts/make_publication_figures.py | archived |
| `v2_builder.py` | `fraggen/data/v2_builder.py` | `archive/code/datasets_v2_v3/fraggen_data/v2_builder.py` | Dataset V2 builder module; V2 superseded by V4 | archived |
| `v2_fragments.py` | `fraggen/data/v2_fragments.py` | `archive/code/datasets_v2_v3/fraggen_data/v2_fragments.py` | Dataset V2 container; V2 superseded by V4 | archived |
| `v2_sampler.py` | `fraggen/data/v2_sampler.py` | `archive/code/datasets_v2_v3/fraggen_data/v2_sampler.py` | Dataset V2 sampler; V2 superseded by V4 | archived |
| `PROJECT_STATUS.md` | `reports/PROJECT_STATUS.md` | `archive/docs/PROJECT_STATUS_2026-09-10.md` | status snapshot from before V4; superseded by docs/FINAL_PROJECT_SUMMARY.md | archived |
| `corpus_balance.md` | `reports/corpus_balance.md` | `archive/reports/datasets_v2_v3/corpus_balance.md` | Dataset V2 balancing report | archived |
| `eval.json` | `reports/eval.json` | `archive/reports/synthetic/eval.json` | synthetic A/B demo evaluation | archived |
| `smoke_report.txt` | `reports/smoke_report.txt` | `archive/reports/synthetic/smoke_report.txt` | torch-free smoke-test output | archived |
| `file_identity_audit.md` | `reports_v2/file_identity_audit.md` | `archive/reports/datasets_v2_v3/file_identity_audit.md` | Dataset V2 identity audit | archived |
| `download_report.md` | `reports_v3/download_report.md` | `archive/reports/datasets_v2_v3/download_report.md` | Dataset V3 download/balance reports | archived |
| `raw_dataset_balance.md` | `reports_v3/raw_dataset_balance.md` | `archive/reports/datasets_v2_v3/raw_dataset_balance.md` | Dataset V3 download/balance reports | archived |
| `build_processed_v2.py` | `scripts/build_processed_v2.py` | `archive/code/datasets_v2_v3/build_processed_v2.py` | Dataset V2 build; superseded by V4 | archived |
| `build_processed_v3.py` | `scripts/build_processed_v3.py` | `archive/code/datasets_v2_v3/build_processed_v3.py` | Dataset V3 build; superseded by V4 | archived |
| `build_train_index.py` | `scripts/build_train_index.py` | `archive/code/datasets_v2_v3/build_train_index.py` | Dataset V2 balancing index; superseded by V4 build-time cap | archived |
| `build_v2.py` | `scripts/build_v2.py` | `archive/code/datasets_v2_v3/build_v2.py` | Dataset V2 driver; superseded by V4 | archived |
| `classify.py` | `scripts/classify.py` | `archive/code/legacy_demo/classify.py` | duplicate inference logic; superseded by src/inference + demo/cli/demo.py --map | archived |
| `demo.py` | `scripts/demo.py` | `archive/code/legacy_demo/demo.py` | August 512-byte live demo; superseded by demo/ | archived |
| `run_overnight.sh` | `scripts/run_overnight.sh` | `archive/code/cross_corpus/run_overnight.sh` | one-off orchestration of the August cross-corpus matrix (hard-coded pre-refactor paths) | archived |
| `verify_v2.py` | `scripts/verify_v2.py` | `archive/code/datasets_v2_v3/verify_v2.py` | Dataset V2 verifier | archived |
| `verify_v3.py` | `scripts/verify_v3.py` | `archive/code/datasets_v2_v3/verify_v3.py` | Dataset V3 verifier | archived |

## 5. Path migration map (active tree, nothing removed)

This section is for following old links in earlier documents such as `docs/thesis/FINDINGS.md` and `docs/thesis/REVIEW1.md`, which still use the pre-refactor paths.

### Folders

| old path | new path | files |
|---|---|---:|
| `data/DFRWS/` | `data/raw/DFRWS/` | 9 |
| `data/FFT-75/` | `data/raw/FFT-75/` | 6 |
| `data/Govdoc1/` | `data/raw/Govdoc1/` | 21 |
| `data/govdocs1_files/` | `data/raw/govdocs1_files/` | 16,318 |
| `data/github_zip/` | `data/raw/github_zip/` | 1,375 |
| `data/rico_png/` | `data/raw/rico_png/` | 55,912 |
| `data/tgif_gif/` | `data/raw/tgif_gif/` | 1,824 |
| `data/uci_csv/` | `data/raw/uci_csv/` | 2,000 |
| `data/wiki_pages/` | `data/raw/wiki_pages/` | 26,628 |
| `data/raw_downloads/` | `data/raw/downloads/` | 1 |
| `demo_v4/samples/` | `demo/samples/` | 11 |
| `reports/ablations_gpu/` | `reports/ablations/cross_corpus_fft75_govdocs1/ablations_512_12seed/` | 183 |
| `reports/blocksize_512/` | `reports/ablations/cross_corpus_fft75_govdocs1/blocksize_512/` | 32 |
| `reports/blocksize_4096/` | `reports/ablations/cross_corpus_fft75_govdocs1/blocksize_4096/` | 32 |
| `reports/fulldata_512/` | `reports/ablations/cross_corpus_fft75_govdocs1/fulldata_512/` | 11 |

### Files

| old path | new path | change |
|---|---|---|
| `DATASETS.md` | `docs/DATASETS.md` | moved (byte-identical) |
| `checkpoints/v4_results/A_plain_s1337.calib.json` | `checkpoints/calibration/A_plain_s1337.calib.json` | moved (byte-identical) |
| `checkpoints/v4_results/A_plain_s1337.pt` | `checkpoints/v4/A_plain_s1337.pt` | moved (byte-identical) |
| `checkpoints/v4_results/A_plain_s1338.pt` | `checkpoints/v4/A_plain_s1338.pt` | moved (byte-identical) |
| `checkpoints/v4_results/A_plain_s1339.pt` | `checkpoints/v4/A_plain_s1339.pt` | moved (byte-identical) |
| `data/processed_v4/metadata.csv` | `data/metadata/v4_fragment_metadata.csv` | moved (byte-identical) |
| `data/processed_v4/report.md` | `data/metadata/v4_build_report.md` | moved (byte-identical) |
| `data/processed_v4/split_manifest.csv` | `data/metadata/v4_split_manifest.csv` | moved (byte-identical) |
| `data/semantic_annotations.zip` | `data/raw/semantic_annotations.zip` | moved (byte-identical) |
| `demo_v4/.streamlit/config.toml` | `.streamlit/config.toml` | moved (byte-identical) |
| `demo_v4/DEMO_GUIDE.md` | `docs/viva/DEMO_GUIDE.md` | moved, then refactored (imports/paths) |
| `demo_v4/app.py` | `demo/streamlit/app.py` | moved, then refactored (imports/paths) |
| `demo_v4/demo.py` | `demo/cli/demo.py` | moved, then refactored (imports/paths) |
| `demo_v4/fragcore.py` | `src/inference/classifier.py` | moved, then refactored (imports/paths) |
| `demo_v4/viva_demo.py` | `demo/cli/viva_demo.py` | moved, then refactored (imports/paths) |
| `fraggen/__init__.py` | `src/__init__.py` | moved, then refactored (imports/paths) |
| `fraggen/calibration.py` | `src/utils/calibration.py` | moved (byte-identical) |
| `fraggen/config.py` | `src/utils/config.py` | moved, then refactored (imports/paths) |
| `fraggen/data/__init__.py` | `src/preprocessing/__init__.py` | moved (byte-identical) |
| `fraggen/data/augment.py` | `src/preprocessing/augment.py` | moved (byte-identical) |
| `fraggen/data/dfrws.py` | `src/preprocessing/dfrws.py` | moved (byte-identical) |
| `fraggen/data/fft75.py` | `src/preprocessing/fft75.py` | moved (byte-identical) |
| `fraggen/data/fragments.py` | `src/preprocessing/fragments.py` | moved (byte-identical) |
| `fraggen/data/govdocs.py` | `src/preprocessing/govdocs.py` | moved (byte-identical) |
| `fraggen/data/synthetic.py` | `src/preprocessing/synthetic.py` | moved (byte-identical) |
| `fraggen/eval.py` | `src/training/eval.py` | moved, then refactored (imports/paths) |
| `fraggen/features/entropy.py` | `src/preprocessing/entropy.py` | moved (byte-identical) |
| `fraggen/metrics.py` | `src/utils/metrics.py` | moved (byte-identical) |
| `fraggen/models/__init__.py` | `src/models/__init__.py` | moved (byte-identical) |
| `fraggen/models/baseline.py` | `src/models/baseline.py` | moved, then refactored (imports/paths) |
| `fraggen/models/cnn.py` | `src/models/cnn.py` | moved (byte-identical) |
| `fraggen/models/contrastive.py` | `src/models/contrastive.py` | moved (byte-identical) |
| `fraggen/train.py` | `src/training/train.py` | moved, then refactored (imports/paths) |
| `reports/DATASET_V4_AND_CLEANUP_2026-09-14.md` | `reports/datasets/DATASET_V4_AND_CLEANUP_2026-09-14.md` | moved (byte-identical) |
| `reports/DATA_AUDIT_2026-09-14.md` | `reports/datasets/DATA_AUDIT_2026-09-14.md` | moved (byte-identical) |
| `reports/FINDINGS.md` | `docs/thesis/FINDINGS.md` | moved (byte-identical) |
| `reports/REVIEW1.md` | `docs/thesis/REVIEW1.md` | moved (byte-identical) |
| `reports/V4_TRAINING_RESULTS_2026-09-14.md` | `reports/ablations/v4/V4_TRAINING_RESULTS_2026-09-14.md` | moved (byte-identical) |
| `reports/blocksize_comparison.txt` | `reports/ablations/cross_corpus_fft75_govdocs1/blocksize_comparison.txt` | moved (byte-identical) |
| `reports/eval_dfrws_4096.json` | `reports/ablations/cross_corpus_fft75_govdocs1/eval_dfrws_4096.json` | moved (byte-identical) |
| `reports/figures/fig1_gap.png` | `reports/figures/cross_corpus/fig1_gap.png` | moved (byte-identical) |
| `reports/figures/fig2_perclass.png` | `reports/figures/cross_corpus/fig2_perclass.png` | moved (byte-identical) |
| `reports/figures/fig3_confusion.png` | `reports/figures/cross_corpus/fig3_confusion.png` | moved (byte-identical) |
| `reports/figures/fig4_reject.png` | `reports/figures/cross_corpus/fig4_reject.png` | moved (byte-identical) |
| `reports/figures/fig5_blocksize.png` | `reports/figures/cross_corpus/fig5_blocksize.png` | moved (byte-identical) |
| `reports/forensic_4096.json` | `reports/ablations/cross_corpus_fft75_govdocs1/forensic_4096.json` | moved (byte-identical) |
| `reports/forensic_512.json` | `reports/ablations/cross_corpus_fft75_govdocs1/forensic_512.json` | moved (byte-identical) |
| `reports/real_baseline.json` | `reports/ablations/cross_corpus_fft75_govdocs1/real_baseline.json` | moved (byte-identical) |
| `reports/v4_results/A_plain_s1337.json` | `reports/ablations/v4/A_plain_s1337.json` | moved (byte-identical) |
| `reports/v4_results/A_plain_s1338.json` | `reports/ablations/v4/A_plain_s1338.json` | moved (byte-identical) |
| `reports/v4_results/A_plain_s1339.json` | `reports/ablations/v4/A_plain_s1339.json` | moved (byte-identical) |
| `reports/v4_results/logs/A_plain_s1337.eval.log` | `reports/ablations/v4/logs/A_plain_s1337.eval.log` | moved (byte-identical) |
| `reports/v4_results/logs/A_plain_s1337.train.log` | `reports/ablations/v4/logs/A_plain_s1337.train.log` | moved (byte-identical) |
| `reports/v4_results/logs/A_plain_s1338.eval.log` | `reports/ablations/v4/logs/A_plain_s1338.eval.log` | moved (byte-identical) |
| `reports/v4_results/logs/A_plain_s1338.train.log` | `reports/ablations/v4/logs/A_plain_s1338.train.log` | moved (byte-identical) |
| `reports/v4_results/logs/A_plain_s1339.eval.log` | `reports/ablations/v4/logs/A_plain_s1339.eval.log` | moved (byte-identical) |
| `reports/v4_results/logs/A_plain_s1339.train.log` | `reports/ablations/v4/logs/A_plain_s1339.train.log` | moved (byte-identical) |
| `reports/v4_results/summary.json` | `reports/ablations/v4/driver_summary_A_plain.json` | moved (byte-identical) |
| `reports/v4_results/summary.md` | `reports/ablations/v4/driver_summary_A_plain.md` | moved (byte-identical) |

### Module names

| old import | new import |
|---|---|
| `fraggen.train` | `src.training.train` |
| `fraggen.eval` | `src.training.eval` (`load_model` → `src.models.checkpoint`) |
| `fraggen.config` | `src.utils.config` |
| `fraggen.metrics` | `src.utils.metrics` |
| `fraggen.calibration` | `src.utils.calibration` |
| `fraggen.models.*` | `src.models.*` |
| `fraggen.data.*` | `src.preprocessing.*` |
| `fraggen.features.entropy` | `src.preprocessing.entropy` |
| `demo_v4/fragcore.py` | `src.inference` + `src.explainability` |

Empty folders removed: `data/v2_scaled`, `reports/overnight`, `fraggen/features`, `fraggen/models`, `fraggen/data`, `fraggen`, `demo_v4/.streamlit`, `demo_v4`, `reports/v4_results/logs`, `reports/v4_results`, `reports_v2`, `reports_v3`, `tools`, `checkpoints/v4_results`.
