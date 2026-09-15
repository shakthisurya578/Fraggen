# FragGen V4 — Examiner Questions and Answers

Fifteen questions an examiner is likely to ask, with answers grounded in the
project's result files. Every number cites its source, so it can be shown on
screen if challenged.

| source key | file |
|---|---|
| **[V4]** | `reports/ablations/v4/A_plain_s133{7,8,9}.json` (mean ± std over 3 seeds) |
| **[ABL]** | `reports/ablations/v4/ABLATION_RESULTS.md`, `ablation_summary.json` |
| **[CC]** | `reports/ablations/cross_corpus_fft75_govdocs1/` and `docs/thesis/FINDINGS.md` |
| **[CAL]** | `checkpoints/calibration/A_plain_s1337.calib.json`; `docs/thesis/REVIEW1.md` §10 |
| **[PDF]** | `reports/tables/v4_test_pdf_fragments_by_stream_region.json` (A_plain s1337) |
| **[DATA]** | `data/metadata/v4_build_report.md`, `docs/REPOSITORY_AUDIT.md` |

---

### 1. What problem does FragGen solve, and why does it matter forensically?

It identifies a file's type from a **raw 4096-byte block with no name, no
filesystem metadata and no header**. This is the situation in unallocated space,
a damaged disk image, or memory: the directory entries are gone, and file carvers
recover blocks, not files. Knowing that a block is JPEG, ZIP or PDF data tells an
investigator which carver to run and which blocks belong together. Header-based
tools such as `file` or magic-byte signatures fail here by construction, because
the block has no header.

### 2. Why a byte-level CNN, and what does the architecture look like?

Byte order carries format structure (JPEG markers, DEFLATE block patterns, CSV
delimiters, markup nesting) that a byte histogram discards. ByteCNN learns it
directly:

`Embedding(256→32)` → 3 × [`Conv1d` k=9 → BatchNorm → ReLU → MaxPool ÷2 → Dropout 0.1]
with 64/128/256 channels → global average pooling → `Linear 256→256` → `Linear 256→9`.
A 4096-byte input becomes (N, 256, 512) feature maps, is pooled to 256 numbers,
and ends as 9 logits. The inference path has **464,713 parameters**, read from the
checkpoint (`docs/architecture/ARCHITECTURE.md`).

The evidence that order matters: on the V4 test split the byte-histogram reference
(best accuracy from byte frequencies alone) is **0.7715**, while the CNN reaches
**0.8938** **[V4]**. In the cross-corpus study the CNN beat a logistic-regression
feature baseline (0.4877 cross-corpus) at both block sizes **[CC]**.

### 3. Why 4096-byte fragments?

4096 B is the common filesystem cluster size, so it is the unit a carver actually
recovers, and it gives the model more context. The project measured this directly:
in the cross-corpus study, going from 512 B to 4096 B raised cross-corpus accuracy
by **+0.128** pooled over all configurations (**p = 0.00009**), the largest single
effect found **[CC]** (`blocksize_comparison.txt`).

### 4. Why exclude the header block, and how do you know nothing leaks?

The first block holds magic signatures (`%PDF-`, `\x89PNG`, `PK\x03\x04`). A model
that sees them learns a lookup table, not format structure, and the result would not
transfer to mid-file fragments, which are the real forensic case. Three controls:

1. **Builder:** fragments are only sampled from offset 4096 onward (a minimum file size
   of 8,192 B guarantees at least one).
2. **Filter:** any fragment that still begins with its class's magic bytes is dropped.
3. **Inference:** `FragGenClassifier.fragment_bytes` always discards block 0. An input
   with nothing after it yields *no* prediction, never a header-based guess. A unit test
   (`tests/test_inference.py`) pins this behaviour.

The V2/V3 audit and the V4 audit both scanned every stored fragment for magic
signatures and found none **[DATA]**.

### 5. How is a whole file classified?

Each fragment is classified independently. The file verdict is the **majority vote**
(ties go to the earlier class in the model's class order), reported with the vote share
and the mean confidence of the winning fragments. The per-fragment trace is always
exported (JSON and CSV), because a mixed vote is itself evidence. For example,
`sample.pdf` votes PDF 9, ZIP 3, and the three ZIP fragments lie entirely inside
FlateDecode streams.

### 6. Your test accuracy is 89%. Why do you insist that is not a cross-corpus result?

Because V4's validation and test splits are **unseen files from the same source
pool** (GovDocs1, DFRWS, RICO, TGIF, GitHub, Wikipedia) as the training split. The
split is disjoint at the file level, but the corpora are shared. That measures
generalisation to new documents, not to new sources.

| question | result |
|---|---|
| unseen *files*, same sources (V4) | test accuracy **0.8938 ± 0.0036**, macro-F1 **0.8976 ± 0.0041** **[V4]** |
| unseen *corpus* (train FFT-75 → test GovDocs1, 4096 B) | **0.6812 ± 0.0141** (2 seeds) **[CC]** |

Two internal signals confirm that V4 has no distribution shift: validation and test
accuracy differ by only 0.0144 ± 0.0024, and the model is already calibrated without
correction (Q13). A genuine corpus shift shows neither.

### 7. What did the cross-corpus study find?

The headline metric was the gap between in-corpus and cross-corpus accuracy **[CC]**:

* **Fragment length dominates:** 512 → 4096 B gave +0.128 cross-corpus accuracy (p = 0.00009).
* **More training data helps:** at 512 B, A_plain reached 0.6325 ± 0.0205 with all 50k
  fragments per class against 0.5243 ± 0.0280 with 8k, a difference of +0.108
  (`fulldata_512/summary.md` vs `ablations_512_12seed/summary.md`).
* **None of the three FragGen mechanisms improves cross-corpus accuracy.** At 12 seeds
  the full model differs from the plain CNN by −0.0003 (p = 0.975). The entropy adversary
  "improves" the gap (p = 0.014) only by *lowering in-corpus accuracy* (−0.036, p < 0.001).

Thesis: *evidence (longer fragments, more data), not training-time regularisation,
governs cross-corpus generalisation.* Practical advice: carve bigger blocks.

### 8. Why is PDF confused with ZIP and JPG?

PDF is a **container**. Most of its bytes sit in stream objects encoded with
`/FlateDecode`, which is zlib/DEFLATE, the same compression ZIP uses, or with
`/DCTDecode`, which is a JPEG stored verbatim. A 4096-byte window inside such a stream
*is* DEFLATE or JPEG data at the byte level.

This was measured, not assumed **[PDF]**. All 1,500 GovDocs1 PDF fragments in the V4
test split were re-read from their source files (1,500/1,500 byte-identical), and each
was assigned to the stream type holding most of its bytes:

| where the fragment lies | n | predicted PDF | predicted ZIP | predicted JPG |
|---|---:|---:|---:|---:|
| mostly FlateDecode data | 893 | 542 | 308 | 41 |
| mostly DCTDecode (JPEG) data | 235 | 53 | 26 | 156 |
| other streams | 250 | 201 | 35 | 7 |
| outside streams (PDF object syntax) | 122 | **122** | 0 | 0 |

Every fragment made of PDF syntax is called PDF. JPG calls concentrate in JPEG
streams, and ZIP calls in DEFLATE streams. Pooled over seeds, 24% of PDF fragments
are predicted ZIP and 13% JPG (`fig1_confusion_matrix.png`).

### 9. So is PDF recall of 0.62 a model failure?

Mostly not. At the byte level, much of it is correct behaviour scored against a
file-extension label. The evaluation's container-aware view reports test accuracy
**0.9424** when pdf/zip fragments are set aside, against 0.8938 strict **[V4]**. The
remainder is real room for improvement: even inside FlateDecode streams, 61% of
fragments are still recognised as PDF, so there is PDF-specific signal (object
syntax at stream boundaries) that a model could exploit further.

### 10. How did you balance the dataset, and why was it necessary?

Earlier builds were badly skewed: V3 was 49% PDF with PNG at 0.2%, and V2 was 32% ZIP
(`reports/datasets/DATA_AUDIT_2026-09-14.md`). V4 added real sources for the starved
classes (RICO screenshots for PNG, TGIF for GIF, GitHub CSV and ZIP files, full
Wikipedia articles for XML) and applied a **per-class cap of 10,000 fragments**, with at
most 20 fragments per source file for diversity. Result: 88,805 fragments with every
class near an equal share. CSV is the one shortfall at 8,204, because its sources ran
out of usable files, yet it still reaches F1 0.955 **[DATA]**
(`fig4_dataset_class_distribution.png`).

### 11. How do you prevent leakage between training and test data?

* The split is at the **file** level (70/15/15, seed 42), with a disjointness assertion
  for every class.
* Checkpoint selection uses a 10% split carved from `train.npz`, so `val.npz` and
  `test.npz` never influence training.
* The repository audit hashed every raw file and found **no byte-identical file shared
  between train and test**. One identical `.txt` pair spans val and test (20 fragments,
  0.15% of test), which cannot affect training or model selection **[DATA]**.

### 12. What did the ablation study show?

<!-- V4_ABLATION_ANSWER -->
On the cross-corpus task, the 12-seed ladder shows that no mechanism improves
cross-corpus accuracy (Q7) **[CC]**. The same cumulative ladder was re-run on Dataset V4
with the identical protocol (6 epochs, 3 seeds, 4096 B), compared with A_plain by
seed-paired t-tests (Holm-corrected) and exact McNemar tests on the 13,754 test
fragments **[ABL]**:

| config | test accuracy | Δ vs A_plain | Holm p | McNemar seeds significant |
|---|---|---|---|---|
| A_plain | 0.8938 ± 0.0036 | — | — | — |
| B_aug | 0.8958 ± 0.0032 | +0.0019 | 1.000 | 2/3 |
| C_aux | 0.8938 ± 0.0086 | −0.0001 | 1.000 | 2/3 |
| D_adv | 0.8603 ± 0.0152 | **−0.0336** | 0.313 | **3/3** |
| E_full | 0.8760 ± 0.0159 | −0.0179 | 0.759 | 2/3 |

Same conclusion, different dataset: **no mechanism improves test accuracy** at a level
distinguishable from seed noise. D_adv's drop is the largest and does not survive Holm
correction at n = 3, but its McNemar test is significant in all three seeds, and in
none of them does it fix more fragments than it breaks — it consistently makes
*different, net-worse* predictions, the same behaviour the entropy adversary showed in
the cross-corpus study. E_full also costs about 7× A_plain's training time for a
directionally worse result. On both datasets, the practical recommendation is the plain
CNN.
<!-- /V4_ABLATION_ANSWER -->

### 13. Is the confidence score a real probability?

On V4-like data, yes. Temperature scaling was fitted on `val.npz` and gave
**T = 1.0214**. Test-split ECE is 0.0217 before and 0.0219 after, so the model is
calibrated without correction **[CAL]**.

Across corpora, no. For the August cross-corpus model, T = 1.2903 made in-corpus
confidence quotable (ECE 0.0575 → 0.0279), but GovDocs1 confidence stayed overstated
(ECE 0.1844 → 0.1393) **[CAL]**. On evidence from an unknown source, confidence should be
used as a *ranking* of which blocks to examine first. When a probability must be
quoted, quote the empirical accuracy at a threshold: on V4 test, keeping the 90% most
confident fragments gives 0.9342 accuracy **[V4]**.

### 14. What are the main limitations?

1. **V4 is not a cross-corpus benchmark.** The only true cross-corpus evidence is the
   FFT-75 → GovDocs1 study, which was run on the older data pipeline.
2. **Labels come from file extensions**, so container formats (PDF, ZIP) produce
   "errors" that are correct at the byte level. There is also a little genuine label
   noise: one identical HTML/TXT pair in train.
3. **Nine classes only.** There are no Office documents, executables, audio/video or
   encrypted data, and high-entropy look-alikes such as encrypted blocks vs ZIP remain
   untested.
4. **Three seeds** give a paired t-test with only 2 degrees of freedom, so small effects
   cannot be resolved (see [ABL] on power).
5. **Independent fragments.** The model ignores context from neighbouring blocks, which a
   carver usually has. An exploratory contiguous-run aggregation suggested a gain, but its
   output was not saved, so it is not quoted as a result.
6. **One GPU and one machine.** Timing figures are indicative only.

### 15. What would you do next?

* **Container-aware labels:** score a PDF fragment inside a DEFLATE stream as
  "compressed data inside a PDF" rather than as a PDF error, or learn a hierarchical
  label (container → payload).
* **Sequence models over adjacent blocks** (run aggregation, or a small transformer over
  block embeddings) to use the context carvers actually have.
* **A true cross-corpus V4:** hold out an entire source (e.g. train without RICO and test
  on another screenshot corpus) so the balanced dataset also answers the corpus-shift question.
* **Calibration under shift:** temperature per source, or conformal prediction sets whose
  coverage guarantees survive a change of corpus.
* **More types**, including encrypted and random data, as the hardest high-entropy negatives.
