# archive/

Everything in this folder was **moved here byte-for-byte** during the 2026-09-14
repository cleanup. Nothing here is used by the active code (`src/`, `scripts/`,
`demo/`, `tests/`). It is kept for provenance and so that earlier experiments
stay reproducible.

Every move was verified by SHA-1 (code, reports, checkpoints) or by file count
plus total bytes (bulk data). The item-by-item list, with the reason for each,
is in [`docs/REMOVED_FILES.md`](../docs/REMOVED_FILES.md).

| folder | contents | why it is archived |
|---|---|---|
| `checkpoints/cross_corpus_fft75_govdocs1/` | 95 checkpoints from the August cross-corpus study: 512 B × 12 seeds, the 512-vs-4096 block-size study, and full-data 512 B runs | They back the preserved cross-corpus results in `reports/ablations/cross_corpus_fft75_govdocs1/`, but nothing in the V4 system or the demo uses them |
| `checkpoints/early_single_seed_512/`, `checkpoints/misc/` | the first single-seed runs; one undocumented checkpoint | superseded by the 12-seed matrix |
| `data/processed_cross_corpus_fft75_govdocs1/` | FFT-75, GovDocs1 and DFRWS FragmentSets (512 B and 4096 B) | inputs to the cross-corpus study; rebuild them with `scripts/prep_fft75.py` and `python -m src.preprocessing.govdocs` |
| `data/processed_v2/`, `data/processed_v3/`, `data/v2_small/` | Dataset V2 and V3 | superseded by V4: V2 was zip-overweighted and V3 severely imbalanced (`reports/datasets/DATA_AUDIT_2026-09-14.md`) |
| `data/synthetic/` | synthetic A/B corpora | regenerate with `scripts/make_synthetic.py --out <folder>` |
| `data/wikipedia_xml_staging/` | 2,000 small Wikipedia page exports | unusable (median file 822 B); replaced by `data/raw/wiki_pages/` |
| `reports/early_single_seed_ablations_512/` | the first CPU ablation matrix | superseded by the 12-seed GPU matrix |
| `reports/datasets_v2_v3/`, `reports/synthetic/` | V2/V3 dataset notes; synthetic smoke-test outputs | superseded |
| `docs/PROJECT_STATUS_2026-09-10.md` | project status snapshot from before V4 | superseded by `docs/FINAL_PROJECT_SUMMARY.md` |
| `code/datasets_v2_v3/` | V2/V3 build and verify scripts, plus the `fraggen.data.v2_*` modules | V2/V3 retired |
| `code/legacy_demo/` | the August 512 B demo, `classify.py`, the first figure script | replaced by `demo/` and `src/inference/` (one inference implementation instead of three) |
| `code/cross_corpus/run_overnight.sh` | unattended orchestration of the August matrix | hard-coded to pre-refactor paths; the individual commands are in `docs/thesis/FINDINGS.md` |
| `code/ingestion/` | one-off download and scrape scripts for the V4 raw sources | not re-runnable as-is (several reference deleted temp folders); kept as a record of how `data/raw/` was acquired |

**Archived code is not maintained.** It still imports the pre-refactor
`fraggen.*` package. The same modules now live in `src/` (see the path map in
`docs/REMOVED_FILES.md`), so to run an archived script, update its imports first.

`archive/data/` and `archive/checkpoints/` are git-ignored because of their size
and third-party data licences.
