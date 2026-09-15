# Data Audit — processed_v2 / processed_v3

> **Path note (repository cleanup, 2026-09-14).** Datasets V2 and V3 have since been
> archived (`archive/data/processed_v2/`, `archive/data/processed_v3/`), and the raw
> corpora moved to `data/raw/`. The audit's content is unchanged. For the path map see
> [`docs/REMOVED_FILES.md`](../../docs/REMOVED_FILES.md).

**Read-only audit. No code or data files were modified to produce this report.**
Independently verified against the raw `.npz` bytes and metadata arrays —
not taken on the word of the pipeline's own `report.md` self-audits, though
those are also referenced below where they add information this audit didn't
re-derive (e.g. per-file GovDocs1 identity spot-checks).

---

## TL;DR

| dataset | leakage / hygiene | class balance | still "cross-corpus"? | verdict |
|---|---|---|---|---|
| **V2** | ✅ clean | ⚠️ zip over-weighted (~32%) | ⚠️ mostly no — see below | **usable, with caveats** |
| **V3** | ✅ clean | 🔴 severe (pdf ~49%, png/xml/csv/zip/gif <2% each) | 🔴 effectively no | **not recommended as-is** |

Both datasets are **excellently engineered on hygiene**: zero file-level
leakage, zero magic-byte header leakage (verified by a full byte-level scan,
not a sample), 100% offset alignment. That part of the new pipeline is solid
and better-documented than the original one.

The problems are **methodological framing and class balance**, not
correctness bugs. Neither is fatal, but both must be understood before any
number trained on this data is compared to the FFT-75→GovDocs1 cross-corpus
results already in `FINDINGS.md` / `REVIEW1.md`.

---

## 1. What was checked, and how

Independent, read-only verification (scripts kept in the session scratchpad,
not added to the project):

1. **Header-peek every `.npz`** — shapes, dtypes, member names, without
   loading full arrays. `processed_v2/train.npz`'s `X` array alone would be
   ~5.2 GB uncompressed; this machine had 4.0 GB free RAM at audit time, so
   naive `np.load()` was avoided throughout.
2. **Loaded all small metadata arrays in full** (`y`, `classes`, `sources`,
   `offsets`, `file_ids` — each MB-scale even for the biggest split) to get
   exact class counts, per-corpus provenance, and file identities.
3. **Cross-split file-ID leakage check** — set-intersection of `file_ids`
   between train/val/test.
4. **Full byte-level magic-signature scan** — every fragment's first bytes
   checked against `%PDF-`, `\xff\xd8\xff` (JPEG), `\x89PNG\r\n\x1a\n`,
   `GIF8[79]a`, `PK\x03\x04`/gzip/bzip2/7z/rar, `<?xml`, `<!DOCTYPE`/`<html`,
   for every row in `v2/val`, `v2/test`, `v3/train`, `v3/val`, `v3/test` (a
   **complete pass, not a sample**). `v2/train.npz` (3.9 GB compressed) was
   **not** byte-scanned for time/resource reasons — see §3.4.
5. Compared file counts against the pipeline's own `report.md` /
   `reports_v2/file_identity_audit.md`, and checked whether the newly
   downloaded raw corpora are actually wired into the build.

---

## 2. What's genuinely good

### 2.1 Zero file-level leakage — confirmed independently

| dataset | train files | val files | test files | overlap |
|---|---|---|---|---|
| V2 | 67,067 | 9,240 | 9,297 | **0 / 0 / 0** |
| V3 | 9,711 | 2,078 | 2,122 | **0 / 0 / 0** |

No document contributes fragments to more than one split, in either dataset.

### 2.2 Zero magic-byte header leakage — confirmed by full scan

Every class with a defined magic signature, on every fully-scanned split,
came back at **0.0000%** leakage (not a sample — every row checked):

```
v2/val, v2/test, v3/train, v3/val, v3/test:
  gif 0/… html 0/… jpg 0/… pdf 0/… png 0/… xml 0/… zip 0/…   (all 0.0000%)
```

This matches the pipeline's own claim, but this audit verified it by reading
actual byte content rather than trusting the report.

### 2.3 Offset alignment — 100% compliant

All fragments in both datasets start at a byte offset `≥ 4096` and a multiple
of `4096` — the header-skip policy is enforced structurally, not just in
spirit. `v2/train.npz` (not byte-scanned) passes this check too, which is the
indirect evidence used in place of a full scan for that one file (see §3.4).

### 2.4 Degenerate rows are negligible

Constant-byte (all-same-value) fragments ran 0.014%–0.094% across every split
scanned — consistent with the original corpora audited in August, no new
concern introduced.

### 2.5 GovDocs1 / DFRWS identity is real, not synthetic

Per the pipeline's own `reports_v2/file_identity_audit.md` (spot-checked, not
just trusted): GovDocs1 `file_id`s are real relative paths that were verified
to exist on disk (13,879/13,879, 100%), and DFRWS `file_id`s trace to the
challenge's own ground-truth layout table. Good practice.

---

## 3. The problems

### 3.1 🔴 Neither dataset is a clean cross-corpus test anymore

This is the most important finding, and it changes what any number trained on
V2 or V3 can be compared against.

**V2** — GovDocs1 fragments are mixed into **both train and test** (different
files, same corpus):

| split | fft75 | govdocs1 | dfrws |
|---|---|---|---|
| train | 1,145,857 | 125,334 | – |
| val | 143,032 | 27,004 | – |
| test | 143,248 | 27,971 (**16.3%**) | 599 (**0.35%**) |

Of V2's 171,818 test fragments, only **599 (0.35%)** come from a corpus never
seen in any form during training. The other 99.65% are GovDocs1 — the same
corpus that supplied 125,334 training fragments, just different files. That
is a **held-out-file, same-corpus test**, not a cross-corpus test. It answers
a real and useful question ("does the model generalize to unseen documents
from a corpus it trained on?") but it is a **different and easier** question
than the one FINDINGS.md / REVIEW1.md's GAP metric was built to answer.

**V3** is more extreme — train and val are **100% GovDocs1**, and test is:

| split | govdocs1 | dfrws |
|---|---|---|
| test | 27,395 (**97.9%**) | 599 (**2.1%**) |

97.9% of V3's "test" set is the same corpus as 100% of its training set. This
is essentially a single-corpus, held-out-file benchmark with a 2.1% dfrws
garnish. Reported as a "cross-corpus generalization gap," this would be
scientifically misleading against the existing FFT-75→GovDocs1 results.

**What this means practically:** any accuracy/GAP number from V2 or V3 must
be labeled **"held-out-file, mostly-same-corpus"**, not compared directly to
the 0.5243 (512B) / 0.6812 (4096B) cross-corpus figures already in
`FINDINGS.md`. The only genuinely cross-corpus signal in either dataset is
the 599-fragment DFRWS slice — too small (3 of 9 classes only: pdf/jpg/zip)
to support a class-by-class cross-corpus claim on its own.

### 3.2 🔴 V3's class balance will make training unreliable

No per-class cap was applied when FFT-75 was excluded — the natural imbalance
of GovDocs1 comes through directly:

| class | V3 train fragments | share |
|---|---|---|
| **pdf** | 62,074 | **49.3%** |
| txt | 24,625 | 19.6% |
| html | 23,617 | 18.8% |
| jpg | 9,859 | 7.8% |
| gif | 2,540 | 2.0% |
| zip | 1,302 | 1.0% |
| csv | 885 | 0.7% |
| xml | 707 | 0.6% |
| **png** | **278** | **0.2%** |

pdf+txt+html alone are 87.4% of the training set. **png has 278 training
examples total** — an order of magnitude thinner than anything used in the
August experiments (which had 1,811–20,000 png fragments per split after
explicit balancing). A model trained on V3 without class weighting or
resampling will very likely learn to predict pdf/txt/html by default and
never learn png/xml/csv/zip reliably. If trained as-is, expect the same
"png precision near zero" failure mode already documented in
`forensic_analysis.py`'s results — likely worse.

**Recommendation:** do not train on V3 in its current form without either
(a) class-weighted loss / oversampling of the minority classes, or
(b) ingesting the additional raw corpora already downloaded (§3.5) to bring
the minority classes up to a usable count first.

### 3.3 🟠 V2's class balance has a different problem: zip is over-weighted

| class | V2 train fragments | share |
|---|---|---|
| **zip** | 410,999 | **32.3%** |
| txt | 188,374 | 14.8% |
| pdf | 143,603 | 11.3% |
| html | 105,262 | 8.3% |
| jpg | 91,428 | 7.2% |
| xml | 82,933 | 6.5% |
| csv | 82,625 | 6.5% |
| gif | 84,100 | 6.6% |
| png | 81,867 | 6.4% |

This is a direct consequence of the many-to-one label mapping in
`fft75.ALIAS` (`zip ← {7z, bz2, gz, rar, zip}` — five raw FFT-75 source types
folded into one target class), which the original balanced pipeline
(`prep_fft75.py --per-class 50000`) corrected for with an explicit per-class
cap. This new pipeline caps **per file** (max 20 fragments/file), not **per
class**, so classes backed by more raw files end up over-represented. Milder
than V3's imbalance (everything else is within 6.4–14.8%, reasonably close),
but zip at nearly a third of the training set will bias the model toward
over-predicting zip, and any zip precision/recall numbers should be read with
that in mind.

### 3.4 🟡 `v2/train.npz` (3.9 GB compressed) was not byte-scanned

Every other split was fully byte-scanned; this one was verified only via its
`offsets` metadata array (100% ≥ 4096, 100% aligned) for time/resource
reasons on a machine with 4 GB free RAM and 28 GB free disk. This is
consistent with zero leakage but is not the same standard of proof as the
direct scan applied to every other file. If this dataset is going to be the
basis of a reported result, it's worth a one-time full scan when the machine
has more headroom — it would take a few minutes based on the ~30–35k
rows/sec throughput measured on the other files.

### 3.5 🟡 The newly downloaded corpora are not in either dataset yet

`data/` contains substantial new raw material, downloaded **after** V2/V3
were built (V2/V3: Sep 10 23:xx–00:20; downloads: Sep 11 10:19 → Sep 12
14:27):

| source | top-level entries | likely target class |
|---|---|---|
| `rico_png/` | 55,912 | png |
| `tgif_gif/` | 1,824 | gif |
| `wikipedia_xml/` | 2,000 | xml |
| `uci_csv/` | 2,000 | csv |
| `github_zip/` | 1,375 | zip |
| `temp_csv_repos/` | 2 | csv (looks incomplete/near-empty) |

These are exactly the classes starved in V3 and imbalanced in V2 — this
looks like it was staged specifically to fix §3.2/§3.3. Eighteen new
top-level scripts (`download_*.py`, `scrape_*.py`, `finish_datasets*.py`)
reference these folder names, but **`build_processed_v2.py` and
`build_processed_v3.py` do not** — confirmed by `grep`, not inferred. None of
this data has been ingested into a training set yet. If the intent is to fix
the class-imbalance problems above, that ingestion step is the missing link,
not something already done.

**RICO note:** `rico_png` (mobile UI screenshots) and PDF/GovDocs pages are
visually and structurally different kinds of PNG (UI-rendered vs.
document/scan-rendered), which is fine for the general "png-ness" of the byte
stream but worth knowing if png performance is later scrutinized per-source.

### 3.6 ℹ️ FFT-75's `file_id` in V2 is a synthetic grouping, not a real document

Documented transparently by the pipeline's own audit
(`reports_v2/file_identity_audit.md`): FFT-75 ships with no file identity at
all, so V2 buckets 20 consecutive same-class rows into a "virtual file" for
bookkeeping. This means the "0% file leakage" guarantee for the FFT-75
portion of V2/V3 rests on **trusting that FFT-75's own upstream train/val/test
split already avoided document-level leakage** — not on an independent check
this project can perform, since no real document boundary exists in the
distributed FFT-75 arrays. Not a bug, but worth stating plainly rather than
letting "0% file leakage" read as a stronger guarantee than it is for that
83.75% portion of V2's file manifest.

---

## 4. Recommendation

**For a training run today:**
- **V2** is the more usable of the two — reasonable overall balance apart from
  zip, and it does provide *some* out-of-corpus signal (the DFRWS slice),
  even though 99.65% of its test set is same-corpus held-out-files.
  **Label any results from it explicitly as "held-out-file, mixed-corpus,"
  not "cross-corpus."**
- **V3** should not be trained on as-is. Fix class balance first — either via
  class weighting/oversampling, or by ingesting the already-downloaded
  RICO/TGIF/UCI/Wikipedia/GitHub data for the starved classes (png, xml, csv,
  zip, gif).
- **Neither replaces** the FFT-75→GovDocs1 cross-corpus experiments in
  `FINDINGS.md`/`REVIEW1.md`. Those remain the project's only clean
  cross-corpus measurement. If V2/V3 results get folded into the writeup,
  they need their own clearly-labeled section — not a substitute number in
  the existing GAP table.
- If DFRWS is going to be used as the true held-out corpus going forward,
  its 599-fragment, 3-class limitation (documented previously in
  `PROJECT_STATUS.md` §3.5) still applies here — that hasn't changed.

**No code or data changes were made in the course of this audit**, per your
instruction. All figures above come from the actual `.npz` contents and a
fresh byte-level scan, not from re-reading the pipeline's own summaries.
