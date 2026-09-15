# FragGen — experimental findings

> **Path note (repository cleanup, 2026-09-14).** This document predates the
> reorganisation. Paths such as `reports/ablations_gpu/`, `checkpoints/…`,
> `data/processed/` and `fraggen/…` refer to the old layout; the findings and
> numbers are unchanged. For the new locations see
> [`docs/REMOVED_FILES.md`](../REMOVED_FILES.md) §5. The results now live in
> `reports/ablations/cross_corpus_fft75_govdocs1/`.

Cross-corpus file-fragment type classification: train on FFT-75, test on
GovDocs1, and measure how much accuracy survives the corpus change.

All results below are from a single RTX 4050. 60 training runs at 512 bytes
(5 configs x 12 seeds) plus 20 at the block-size comparison.

---

## 1. Headline

**Fragment length dominates everything else.** Going from 512- to 4096-byte
fragments buys **+0.128 cross-corpus accuracy (p = 0.00009)**. None of FragGen's
three mechanisms produces *any* statistically significant cross-corpus
improvement over a plain byte-CNN, at 12 seeds.

| | cross-corpus accuracy | |
|---|---|---|
| feature baseline (logreg) | 0.4877 | |
| plain byte-CNN @ 512 B | 0.5243 ±0.0280 | +0.037 over baseline |
| full FragGen @ 512 B | 0.5239 ±0.0258 | **+0.000 over plain CNN** (p=0.975) |
| plain byte-CNN @ 4096 B | **0.6812** ±0.0141 | **+0.181 over 512 B** |

The practical recommendation for fragment classification is therefore
**"carve bigger blocks"**, not "train more cleverly".

---

## 2. Setup

- **Train**: FFT-75 (FiFTy) scenario 1. Label order taken from the FiFTy repo's
  `labels.json` — the `.npz` files carry only integer indices, so a wrong
  ordering would silently mislabel all 6.14M fragments.
- **In-corpus test**: FFT-75 held-out split.
- **Cross-corpus test**: GovDocs1, fragmented from real .gov files
  (113,884 fragments @ 512 B; 53,929 @ 4096 B).
- **Labels**: 9 classes — txt, csv, html, xml, pdf, jpg, png, gif, zip.
  `encrypted` was dropped: *neither* corpus contains an encrypted type, so it
  would be a head with zero train and zero test examples (`configs/real.yaml`).
- **Middle-of-file only**: the first block of every file is discarded, so no
  magic signature is visible. Verified at <0.1% leakage on both block sizes.

| run | augment | aux features | entropy adversary | contrastive |
|---|---|---|---|---|
| A_plain | – | – | – | – |
| B_aug | ✓ | – | – | – |
| C_aux | ✓ | ✓ | – | – |
| D_adv | ✓ | ✓ | ✓ | – |
| E_full | ✓ | ✓ | ✓ | ✓ |

---

## 3. Do the mechanisms work? (512 B, 12 seeds)

| run | in-corpus acc | cross acc | cross macro-F1 | GAP |
|---|---|---|---|---|
| baseline (logreg) | 0.7578 | 0.4877 | – | +0.2701 |
| A_plain | 0.8007 ±0.0119 | 0.5243 ±0.0280 | 0.4437 ±0.0255 | 0.2764 ±0.0269 |
| B_aug | 0.8145 ±0.0080 | 0.5293 ±0.0190 | 0.4500 ±0.0158 | 0.2851 ±0.0178 |
| C_aux | 0.8129 ±0.0070 | 0.5402 ±0.0207 | 0.4556 ±0.0181 | 0.2728 ±0.0201 |
| D_adv | 0.7643 ±0.0215 | 0.5152 ±0.0171 | 0.4201 ±0.0222 | **0.2492** ±0.0228 |
| E_full | 0.7927 ±0.0122 | 0.5239 ±0.0258 | 0.4380 ±0.0222 | 0.2687 ±0.0255 |

### Welch's t-test vs the A_plain control

**Cross-corpus accuracy — nothing is significant:**

| run | diff | t | p |
|---|---|---|---|
| B_aug | +0.0051 | +0.52 | 0.610 |
| C_aux | +0.0159 | +1.58 | 0.130 |
| D_adv | −0.0091 | −0.96 | 0.348 |
| **E_full** | **−0.0003** | −0.03 | **0.975** |

The complete FragGen stack is **statistically indistinguishable from a plain
byte-CNN** on the metric the project exists to improve.

**What *is* significant tells the story:**

| effect | diff | p | reading |
|---|---|---|---|
| B_aug, **in-corpus** acc | +0.0138 | 0.004 | augmentation helps fit FFT-75… |
| C_aux, **in-corpus** acc | +0.0122 | 0.007 | …and so do aux features… |
| (both, cross-corpus) | ~0 | n.s. | …but neither transfers |
| D_adv, GAP | −0.0273 | 0.014 | adversary shrinks the gap… |
| D_adv, **in-corpus** acc | −0.0364 | 0.000 | …by getting worse at home |
| D_adv, cross macro-F1 | −0.0236 | 0.025 | …and worse away too |

Augmentation and aux features buy **in-corpus fit that does not transfer** —
textbook corpus overfitting, and precisely the failure FragGen set out to
prevent. The entropy adversary is the only mechanism that moves GAP, and it does
so by degrading the model everywhere; its cross-corpus macro-F1 is significantly
*worse* than the plain CNN's.

### On the GAP metric

GAP = in-corpus − cross-corpus, so a model that simply gets worse at home scores
well. `D_adv` is the demonstration: best GAP in the matrix, significantly worse
on both accuracies that matter. **Never report GAP alone.** Cross-corpus
macro-F1 is the more honest single number, especially given the imbalanced
GovDocs test set (20,000 txt vs 1,811 png).

### Statistical power

Typical per-config s.d. is 0.0221, so at 12 seeds the **minimum detectable
difference is ~0.019**. `C_aux`'s +0.0159 sits just under that: it is the one
mechanism whose trend is positive and might become significant with more seeds.
Everything else is flat or negative.

> An earlier 3-seed pass suggested augmentation *hurt* transfer (−0.0228). At 12
> seeds that reversed to +0.0051 (p=0.610). At n=3 the minimum detectable
> difference was ~0.05 and every effect was smaller, so the 3-seed matrix could
> not have supported any per-mechanism claim. This is the main methodological
> lesson of the project.

---

## 4. Does more context help? (512 vs 4096 B, matched budget)

Both arms trained at 4,000 fragments/class, 6 epochs, 2 seeds — the *only*
variable is fragment length.

| config | cross acc @512 | cross acc @4096 | change | p |
|---|---|---|---|---|
| **A_plain** | 0.5007 ±0.0151 | **0.6812** ±0.0141 | **+0.1805** | **0.007** |
| B_aug | 0.5510 ±0.0029 | 0.6705 ±0.0640 | +0.1196 | 0.230 |
| C_aux | 0.4988 ±0.0292 | 0.6444 ±0.0103 | +0.1456 | 0.064 |
| D_adv | 0.4588 ±0.0122 | 0.5447 ±0.0545 | +0.0859 | 0.256 |
| E_full | 0.4755 ±0.0321 | 0.5856 ±0.0612 | +0.1101 | 0.193 |
| **pooled** | **0.4970** | **0.6253** | **+0.1283** | **0.00009** |

Per-config tests are weak at n=2, but the direction is consistent across all
five and the pooled test is overwhelming. The plain CNN's GAP nearly halves
(0.2580 → 0.1647).

**Two observations worth the writeup:**

1. **+0.128 from block size vs ~0.000 from three mechanisms.** An order of
   magnitude apart. Effort spent on architecture and training tricks was
   misdirected relative to simply reading more bytes.
2. **At 4096 the plain CNN is again the best config** (0.6812, ahead of every
   mechanism variant), and `D_adv` is the worst (0.5447). The mechanisms do not
   become useful with more context — the adversary actively costs more.

---

## 5. Results and discussion — the analyst's reading

Accuracy answers *"how often is the model right"*. An examiner working a disk
image needs four different questions answered, and they give a much less
flattering picture than the accuracy figures above.

Numbers below are the plain CNN on GovDocs1 (the cross-corpus / "away" case),
reproducible with `scripts/forensic_analysis.py`.

### 5.1 If the tool reports type X, how often is it X?

Recall (what §3 reports) is *"of all real JPEGs, how many were found"*. An
analyst reading a carver's output is asking the reverse — **precision**.

| reported | precision | recall | n called | operational reading |
|---|---|---|---|---|
| jpg | 0.888 | 0.836 | 9,414 | **reliable** — can support a finding |
| xml | 0.790 | 0.048 | 81 | trustworthy when claimed, but almost never claimed |
| html | 0.768 | 0.734 | 9,549 | indicative — corroborate |
| txt | 0.744 | 0.587 | 7,888 | indicative — corroborate |
| gif | 0.701 | 0.906 | 7,111 | indicative — corroborate |
| pdf | 0.678 | 0.463 | 6,833 | indicative — corroborate |
| csv | 0.422 | 0.858 | 4,464 | **weak** — over-reports; lead only |
| zip | 0.377 | 0.706 | 7,876 | **unreliable** — do not report |
| png | 0.140 | 0.146 | 713 | **unreliable** — do not report |

*(4096-byte fragments. At 512 bytes `png` precision is **0.037** — when the tool
says PNG it is wrong 96% of the time.)*

Three things follow that "accuracy = 0.67" completely hides:

1. **The tool is competent on some formats and useless on others.** A single
   headline number averages a defensible JPEG identifier together with a PNG
   detector that is worse than guessing. Any finding must be reported per type,
   with its own reliability attached.
2. **`xml` shows the classic forensic asymmetry.** Precision 0.79, recall 0.048:
   when it claims XML it is usually right, but it claims it 81 times out of
   1,339 actual XML fragments. **Absence of evidence is not evidence of
   absence** — "no XML found on this disk" is an unsupportable conclusion from
   this tool.
3. **`csv` and `zip` over-report.** High recall, low precision: they act as
   catch-alls, absorbing fragments from other classes. Their positives are
   leads to triage, not findings to state.

### 5.2 Can the confidence value be quoted?

The classifier prints a softmax score. If it is to appear in a report — "the
tool identified this fragment as JPEG with 0.95 confidence" — that number must
mean what it says. It does not.

| stated confidence | actual accuracy | gap |
|---|---|---|
| 0.990 | 0.823 | **+0.167** |
| 0.927 | 0.681 | +0.247 |
| 0.845 | 0.575 | **+0.271** |
| 0.755 | 0.529 | +0.226 |
| 0.652 | 0.509 | +0.143 |
| 0.550 | 0.465 | +0.086 |

**Expected calibration error: 0.185 at 4096 B, 0.242 at 512 B.** The model is
systematically overconfident — at every level, and worst in the middle of the
range where an analyst is most likely to be making a judgement call.

**Implication:** the raw softmax value must not be presented as a probability or
a likelihood. Reporting "95% confident" when the empirical hit rate at that
score is 57–82% would misstate the strength of the evidence. Two defensible
alternatives: (a) quote the *empirical* accuracy at the chosen threshold from
the table in §5.3 rather than the model's own number, or (b) calibrate the
scores (Platt scaling / temperature scaling on a held-out split) before any
score is reported. Neither is done in the current pipeline — this is the single
most important gap between what FragGen produces and what an examiner could use.

### 5.3 What reliability can be bought by abstaining?

A carver does not have to answer on every block. Raising the confidence
threshold discards ambiguous fragments and raises accuracy on what remains.

| threshold | coverage | accuracy | operational use |
|---|---|---|---|
| 0.00 | 100.0% | 0.671 | sweep everything |
| 0.70 | 80.9% | 0.721 | broad triage |
| 0.90 | 54.5% | 0.801 | prioritisation |
| 0.95 | 46.0% | 0.823 | prioritisation |
| 0.99 | 30.8% | 0.833 | best available |

**90% reliability is not achievable at any threshold** at 4096 bytes; at 512
bytes it is reached only on the top 0.5% of blocks. There is no operating point
at which this classifier identifies single fragments to a standard that would
survive challenge.

**This is the central operational conclusion: FragGen is a triage instrument,
not an identification instrument.** It is well suited to ranking which regions
of an image an examiner should look at first, and to producing a coarse
composition estimate of unallocated space. It is not suited to establishing what
any individual fragment is. Stated that way it is genuinely useful — a 0.83
accuracy filter over 30% of a terabyte image saves substantial examiner time —
but the claim has to match the evidence.

### 5.4 When is an "error" actually a finding?

14,206 of the test fragments are labelled `pdf` or `zip`, and strict accuracy on
them is 0.535 against 0.720 on everything else. But a PDF's interior is
FlateDecode streams and embedded DCTDecode images: **a block from inside a PDF
that the model calls `jpg` may be correctly identifying an embedded object.**

Forensically this inverts. The label says "pdf" because the *file* was a PDF;
the model says "jpg" because *those bytes* are JPEG. For a carver reconstructing
content, the model's answer is the more useful one — it locates image data at a
specific offset inside a container. Scored as a classification error, it is
counted against the tool; read as an investigative result, it is exactly what
the examiner wanted to know.

The practical recommendation is to report container types with an explicit note
that interior fragments are expected to be identified as their embedded content,
and to treat a `pdf → jpg` call as *evidence of an embedded image*, not as a
misclassification.

### 5.5 Why the failures happen, and which are fixable

| failure | cause | fixable? |
|---|---|---|
| png ↔ zip (35% of png) | PNG IDAT chunks are zlib/DEFLATE — the same compressor ZIP uses | **No** at 512 B; the byte streams are near-identical. Longer fragments help (precision 0.037 → 0.140) but do not solve it |
| xml → html (56%) | both are angle-bracket markup; the distinction is near-arbitrary in one block | Partly — merge the classes, or accept and report `markup` |
| pdf/zip interiors | container formats hold other formats | Not an error (§5.4) |
| txt/csv/html boundaries | all printable ASCII; separated by delimiter statistics | Yes — more context helps most here |

The first row is the important one for a forensic reader: **some confusions are
information-theoretic, not engineering failures.** No architecture distinguishes
two DEFLATE streams from different container formats when both are headerless
and 512 bytes long. Recognising which errors are irreducible tells an examiner
where corroborating evidence (file-system metadata, adjacent block continuity,
carving context) is *required* rather than optional.

### 5.6 What this means for the project's hypothesis

The three mechanisms were designed to make the model learn format structure
rather than corpus quirks. Read through the forensic lens, they did something
subtly worse than "nothing": augmentation and entropy aux features produced
**statistically significant in-corpus gains that did not transfer**
(p = 0.004 and 0.007), which is the exact signature of fitting the training
corpus harder. The entropy adversary improved the headline gap metric (p = 0.014)
while significantly *reducing* both in-corpus accuracy and cross-corpus
macro-F1 — an apparent improvement that an uncritical reading of the gap alone
would have reported as success.

Fragment length, by contrast, moved every operational measure at once:
accuracy 0.536 → 0.671, png precision 0.037 → 0.140, zip precision 0.189 →
0.377, and calibration error 0.242 → 0.185. **The intervention that helped was
the one that gave the model more evidence, not the one that regularised how it
learned.**

---

## 6. Where the remaining errors are

Error analysis of the cross-corpus confusion matrix (48,506 errors):

| category | share |
|---|---|
| container interior (pdf/zip fragments called jpg/png/gif/zip/txt) | **34.9%** |
| html ↔ xml | 6.1% |
| everything else | 59.0% |

**About a third of "errors" are labelling artefacts.** A PDF's interior is
FlateDecode (DEFLATE) streams and embedded DCTDecode (JPEG) images, so a chunk
from the middle of a PDF holding a JPEG genuinely *is* jpg bytes.

| label scheme | cross-corpus accuracy |
|---|---|
| strict (9 classes) | 0.5741 |
| html+xml merged | 0.6001 |
| excluding pdf/zip container fragments | 0.6459 |
| coarse families (text/markup/image/archive/doc) | 0.6624 |

Worst classes, and where they go:

| class | recall | mostly predicted as |
|---|---|---|
| xml | 0.096 | html 56% |
| png | 0.167 | zip 35%, gif 23% |
| zip | 0.321 | pdf 27%, png 16% |
| pdf | 0.327 | zip 25%, jpg 18% |

`png → zip` is close to information-theoretically forced: PNG's IDAT chunks are
zlib/DEFLATE, the same compressor ZIP uses. Headerless at 512 bytes, the two are
near-identical. No architecture fixes that — but **more bytes do**, which is
consistent with §4.

---

## 7. Bugs found and fixed

1. **`histogram_bayes_ceiling` was not a bound.** Documented as an "analytic
   upper bound"/"entropy ceiling", but the logreg baseline (0.7578) and the CNN
   (~0.80) both exceed its 0.6889. Renamed `byte_marginal_reference`; old name
   kept as an alias. The reframing is itself a result: everything above that
   line is accuracy earned from byte **order**, since the marginal cannot
   produce it.
2. **Missing `model.eval()` at inference.** Dropout stayed active and BatchNorm
   kept updating, so identical bytes gave different answers and per-class recall
   read 0.08 where evaluation reported 0.797. `load_model` now returns eval-mode
   models.
3. **Header leakage at 4096 B.** `fragment_file` skipped the header block only
   when `len(data) > 2*block_size`, so files between 4 KB and 8 KB contributed
   exactly one fragment — their header, magic bytes intact. Harmless at 512 B
   (<0.1%) but **5.53% of PNG fragments** at 4096, which would have inflated the
   headline block-size result in §4.
4. **Checkpoint namespace collision.** Checkpoints were named
   `{config}_s{seed}.pt` with no experiment namespace, so the 4096-byte runs
   were overwriting the 512-byte models of the same name. Now namespaced under
   `checkpoints/{experiment}/`.
5. **`torch.load` under torch ≥2.6.** `weights_only` defaults to True and the
   checkpoint stored numpy strings; every eval would have failed *after*
   training completed.
6. **Inflated container statistic** (my own analysis script): the
   `cm[keep, keep]` submatrix dropped errors where a non-container fragment was
   *predicted* as a container, inflating "excluding containers" from 0.6459 to
   0.7045.
7. **Confounded block-size figure**: the 512 arm was initially drawn from the
   8,000/class matrix while the 4096 arm was 4,000/class. `make_figures.py` now
   refuses to plot arms whose training budgets differ.

---

## 8. Limitations

- **Block-size arms are n=2.** The pooled result is solid (p=0.00009) but
  per-config tests are weak. 4096-byte runs cost ~40 min each on this card
  (batch 256 puts VRAM at ~96% of 6 GB).
- **Subsampled**: 8,000 of 50,000 available fragments/class at 512 B;
  4,000/class in the block-size comparison; 6 epochs throughout.
- **One cross-corpus point**: GovDocs1 only. DFRWS would make it a trend.
- **Thin classes**: GovDocs `png` has 1,811 fragments @512 B and 683 @4096 B,
  from only 88 source files. The fix is extracting more GovDocs zips (021+),
  not tuning the model.
- **`encrypted` untested**: absent from both corpora.

---

## 9. What to do next

1. **Run the full ablation at 4096 B with more seeds.** §4 used n=2; the
   interesting question is whether `C_aux`'s positive trend survives at the
   block size that actually works.
2. **Push further on length** — 8192/16384 bytes. Block size is the one factor
   with a large, reproducible effect; find where it saturates.
3. **Report container-aware metrics** alongside strict accuracy (§5).
4. **DFRWS as a second away-corpus** (`fraggen/data/dfrws.py` is stubbed).
5. **Reconsider the project's framing.** The mechanisms were designed to close
   the cross-corpus gap and measurably do not. The honest paper is
   *"fragment length, not training tricks, governs cross-corpus generalisation"*
   — a stronger and better-supported claim than the original hypothesis.

---

## Reproducing

```bash
bash scripts/run_overnight.sh          # full pipeline, resumable, no network
python scripts/significance.py --results reports/ablations_gpu
python scripts/blocksize_compare.py
python scripts/make_figures.py --results reports/ablations_gpu \
    --results-512 reports/blocksize_512 --results-4k reports/blocksize_4096
```

Figures in `reports/figures/`; per-run JSON in `reports/*/`.
