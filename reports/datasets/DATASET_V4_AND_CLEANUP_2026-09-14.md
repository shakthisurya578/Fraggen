# Dataset V4 (balanced) + folder cleanup — 2026-09-14

> **Path note (repository cleanup, later on 2026-09-14).** The source folders this
> report says were "deliberately not moved" (`data/rico_png/`, `data/govdocs1_files/`, …)
> were moved to `data/raw/` in the full reorganisation, and
> `scripts/build_processed_v4.py` was updated to match. The V4 manifests are now in
> `data/metadata/v4_*`. The content is unchanged. For the path map see
> [`docs/REMOVED_FILES.md`](../../docs/REMOVED_FILES.md).

Follow-up to `reports/DATA_AUDIT_2026-09-14.md`, which found V3 severely
imbalanced (pdf 49%, png 0.2%) and V2 zip-heavy (32%), and found that the
newly-downloaded raw corpora were staged but not yet wired into any build.

This session: (1) checked whether ingesting the staged data actually fixes
the imbalance, (2) downloaded one additional source where it didn't,
(3) built and independently audited a new balanced dataset (V4), and
(4) cleaned up ~7.5 GB of confirmed-dead files and reorganized the project
root.

---

## 1. Feasibility check — does the staged data fix V3's imbalance?

Estimated usable fragment yield per source (files ≥8192 B, capped at 20
fragments/file — the same rule the pipeline already used):

| source | usable files | est. fragments | target class | verdict |
|---|---|---|---|---|
| `rico_png` | 55,912 | 168,304 | png | ✅ far more than enough |
| `tgif_gif` | 1,745 | 33,624 | gif | ✅ more than enough |
| `github_zip` | 1,315 | 20,420 | zip | ✅ more than enough |
| `uci_csv` | 650 / 2,000 | 6,940 | csv | ⚠️ helps, but below a 10k target |
| `wikipedia_xml` | 43 / 1,999 usable | ~192 | xml | 🔴 **does not fix it** — median file 822 B |

**png/gif/zip: yes, ingestion fixes them outright.** **csv: helps but stays
the natural ceiling.** **xml: the staged download does not work** — the
individual Wikipedia page exports have a median size of 822 bytes (p99 is
only 16 KB), and the one bundled dump (`chrwiki`, a small wiki) averages
~1.9 KB/article. Neither clears the 8,192 B usability floor for more than a
couple dozen files.

## 2. Additional download: a real fix for xml

Per your instruction to download more data if the staged material fell
short, I fetched Simple English Wikipedia's full article dump (a bigger wiki
with substantial article lengths, not the small `chrwiki` already staged):

```
https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
356,186,307 bytes, verified against Content-Length
```

Wrote `scripts/extract_wiki_pages.py` to slice it into individual real
articles at the dump's own `<page>...</page>` document boundaries (not
synthetic — this is a real per-article document, the same granularity
MediaWiki's own Special:Export produces). Filtered to main-namespace,
non-redirect, ≥8192 B:

```
pages seen         : 560,605
non-article (ns!=0): 160,875
redirects skipped  : 114,981
too small (<8192B) : 258,121
KEPT                : 26,628  ->  data/wiki_pages/
```

Size check on the result: p0=8,192 B, p50=12,961 B, p90=36,032 B,
p100=666,934 B — estimated ~87,077 fragments available, comfortably above
what's needed.

## 3. Built Dataset V4

`scripts/build_processed_v4.py` — same genuine-documents-only architecture
as V3 (govdocs1 + dfrws for existing classes), plus the newly-usable sources
for the starved classes, plus **a genuine per-class-per-split fragment cap
(10,000/class)** — the actual mechanism V2/V3 lacked.

Two bugs caught and fixed by the pipeline's own smoke test before the real
run (a `--cap` flag was added specifically to make this cheap to check
first):

1. **Windows `glob()` is case-insensitive.** An initial version scanned both
   `*.gif` and `*.GIF` patterns for the same folder, which on Windows match
   the *same* files twice. That produced duplicate file entries assigned to
   different splits — genuine train/val leakage, caught by the script's own
   disjointness assertion (`gif: train/val overlap`) rather than by silent
   corruption. Fixed by removing the redundant pattern and adding a
   resolved-path dedup as a portability safety net.
2. My first `scan_dfrws()` was copied from `build_processed_v3.py` under the
   wrong assumption about which ground-truth file it reads. V3 actually
   parses a sector-layout file (`challenge-layout.txt`); I had it pointed at
   `dfrws2007_gt_4096.csv`, which uses a simpler, already-byte-aligned
   `start,end,type,file_id` schema (the output of August's
   `dfrws_layout_to_truth.py --block-size 4096`). Rewrote `scan_dfrws()` to
   match that schema before running anything for real.

### Result

```
Total fragments: 88,805
```

| class | fragments | share | status |
|---|---|---|---|
| txt | 10,000 | 11.26% | full cap |
| html | 10,000 | 11.26% | full cap |
| xml | 10,000 | 11.26% | full cap |
| pdf | 10,201 | 11.49% | full cap (+dfrws) |
| jpg | 10,360 | 11.67% | full cap (+dfrws) |
| png | 10,000 | 11.26% | full cap |
| gif | 10,000 | 11.26% | full cap |
| zip | 10,040 | 11.31% | full cap (+dfrws) |
| **csv** | **8,204** | **9.24%** | 82% of cap — natural ceiling, documented, not silently padded |

Compare to before:

| | V3 (old) | V4 (new) |
|---|---|---|
| pdf share | 49.3% | 11.5% |
| png share | 0.22% (278 fragments) | 11.3% (10,000 fragments) |
| xml share | 0.56% (949 fragments) | 11.3% (10,000 fragments) |
| gif share | 2.0% | 11.3% |
| zip share | 1.0% | 11.3% |
| csv share | 0.7% (1,264 fragments) | 9.2% (8,204 fragments) |

Every class within 2 points of the 11.1% equal-share target, csv included.

## 4. Independent audit of V4 (same methodology as the earlier audit)

Not just re-reading the pipeline's own `report.md` — re-verified from the
raw `.npz` bytes:

- **File-level leakage**: train (8,316 files) / val (1,785) / test (1,801) —
  **0 / 0 / 0 overlap**, confirmed by set-intersection of `file_ids`.
- **Magic-byte header leakage**: full byte-level scan (not sampled) of every
  fragment in all three splits. **0.0000%** for every class with a defined
  signature, in every split.
- **Offset alignment**: 0% misaligned in train/val. **Test shows 3.54%
  "misaligned"** — traced to the 601 DFRWS fragments, whose offsets are real
  disk-image byte positions on the original forensic image's own layout
  (512-byte sector grid), not the 4096-byte file-relative grid the other
  sources use. This is expected and benign, not a leak: the same byte-level
  scan confirms 0% header leakage on those DFRWS fragments specifically.
- **Degenerate (constant-byte) rows**: 0.000%–0.007% — negligible, matches
  the low rate observed in every other dataset audited so far.

**V4 passes every hygiene check V2/V3 passed, with dramatically better
balance.**

### Still true regardless of V2/V3/V4

Per the earlier audit's §3.1: govdocs1 fragments still appear in both train
and test in V4 (same corpus, held-out files), and DFRWS remains a thin
601-fragment, 3-class (pdf/jpg/zip) slice. **V4 is not a cross-corpus test
either** — same caveat as V3, now with much better in-corpus class balance.
It should not be compared directly to the FFT-75→GovDocs1 GAP numbers in
`FINDINGS.md`.

---

## 5. Folder cleanup

### Deleted (confirmed dead weight, ~7.5 GB freed)

| item | size | why |
|---|---|---|
| `data/unique_uis.tar.gz` | 6.1 GB | Contains `.jpg`/`.json` (RICO's raw screenshot format), **not** the source of `rico_png` — that's `semantic_annotations.zip`. No script references it. |
| `data/temp_tidytuesday.zip` | 521 MB | Valid zip magic bytes but Python's `zipfile` rejects it (`BadZipFile`) — truncated/corrupted download. Unreferenced by any script. |
| `data/temp_csv_repos/` | 848 MB | Two **incomplete** git clones (`COVID-19`, `data`) — `.git/shallow.lock` and `.git/index.lock` present, meaning the clone was interrupted and abandoned. `uci_csv` already supplies the real CSV data separately. |
| `data/TGIF-Release-master.zip` | 12 MB | The TGIF GitHub repo's *source code* (README, `.gitmodules`, tooling) — not the actual GIF images. `tgif_gif/` already holds the real files, fetched separately. |
| `data/iris.zip` | 4 KB | The classic UCI Iris dataset — unrelated to this project, unreferenced anywhere. |
| `8`, `wiki_gifs.txt` (project root) | 0 B each | Empty stub files, unreferenced. |

Verified before deleting: each item's actual contents (not just its name),
whether any script references it, and — for the two large archives whose
extracted counterparts already exist — which archive was the *real* source
before removing the other as redundant.

**Disk free space: 27 GB → 34 GB.**

### Reorganized

Moved 22 loose one-off download/scrape scripts that were sitting at the
**project root** (`download_*.py`, `scrape_*.py`, `finish_datasets*.py`,
`test_*.py`, `audit.py`, `debug.py`, `count_pages.py`, `expand_datasets.py`,
`fast_gif.py`) into `tools/ingestion/`. Verified first that nothing imports
them as modules (`grep` for `import <name>` across the project came back
empty) — they're standalone CLI entry points, so moving them doesn't break
anything.

Left `powershell.cmd` (a small PATH-setup shim) in place — harmless, and
possibly still in personal use.

**Deliberately not moved**, to avoid invalidating the just-verified,
reproducible V4 build without a compensating benefit:

- `data/rico_png/`, `data/tgif_gif/`, `data/uci_csv/`, `data/github_zip/`,
  `data/wiki_pages/`, `data/raw_downloads/` — `build_processed_v4.py` has
  these paths hardcoded; moving them would require re-verifying the whole
  pipeline for a purely cosmetic gain.
- `data/govdocs1_files/`, `data/FFT-75/`, `data/DFRWS/`, `data/Govdoc1/` —
  the original, foundational corpora. Referenced by `build_processed_v2.py`,
  `v3.py`, `v4.py`, and the original August pipeline (`prep_fft75.py`,
  `fraggen/data/govdocs.py`). Untouched.
- `data/wikipedia_xml/` (the old, too-small staging) — kept for provenance
  rather than deleted; it's small and superseded by `wiki_pages/`, not
  actively harmful to leave.

---

## 6. Bottom line

| dataset | balance | cross-corpus? | recommendation |
|---|---|---|---|
| V2 | zip over-weighted (32%) | ~0.35% of test is truly out-of-corpus | usable, label results "held-out-file, mixed-corpus" |
| V3 | severe (pdf 49%, png 0.2%) | ~2.1% of test is truly out-of-corpus | **do not train on as-is** |
| **V4** | **within 2 pts of equal share, every class** | same caveat as V3 — ~2.1% out-of-corpus | **usable for a held-out-file, genuine-documents benchmark**; still not a substitute for the FFT-75→GovDocs1 cross-corpus results |

V4 is now the dataset to use for any "does this generalize to unseen
documents from the same real-world sources" question. It is not, and should
not be reported as, a cross-corpus generalization result — that claim still
rests solely on the FFT-75→GovDocs1 experiments already in `FINDINGS.md`.
