# FragGen — Project Review Document

> **Path note (repository cleanup, 2026-09-14).** This review predates Dataset V4
> and the reorganisation. Paths and commands (`fraggen.*`, `scripts/classify.py`,
> `scripts/demo.py`, `reports/ablations_gpu/`, …) refer to the old layout; the
> results are unchanged. For the path map see
> [`docs/REMOVED_FILES.md`](../REMOVED_FILES.md) §5, and for the current system see
> [`RESULTS_AND_DISCUSSION.md`](RESULTS_AND_DISCUSSION.md).

**Cross-corpus file-fragment type identification for digital forensics**

CSE4074 Forensics Analytics · file-fragment classification under corpus shift

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Why it matters forensically](#2-why-it-matters-forensically)
3. [Objectives](#3-objectives)
4. [Background](#4-background)
5. [Datasets](#5-datasets)
6. [Methodology](#6-methodology)
7. [Experimental design](#7-experimental-design)
8. [Results](#8-results)
9. [Results and discussion — the analyst's reading](#9-results-and-discussion--the-analysts-reading)
10. [Confidence calibration](#10-confidence-calibration)
11. [Limitations and threats to validity](#11-limitations-and-threats-to-validity)
12. [Conclusions](#12-conclusions)
13. [Future work](#13-future-work)
14. [How to run everything](#14-how-to-run-everything)
15. [Engineering log — defects found and fixed](#15-engineering-log--defects-found-and-fixed)

---

## 1. Problem statement

Given a **raw block of bytes** recovered from unallocated disk space — with no
filename, no extension, no filesystem metadata, and taken from the *middle* of a
file so no magic signature is present — determine which **file type** it came
from.

Formally: learn `f : {0..255}^L → C` where `L ∈ {512, 4096}` bytes and `C` is a
set of 9 file types (txt, csv, html, xml, pdf, jpg, png, gif, zip).

The specific problem this project targets is **corpus shift**. Prior fragment
classifiers report high accuracy on their own benchmark and drop sharply on data
from a different source, because they memorise the encoders, tool-chains and
vocabulary of the corpus they were trained on. The quantity of interest is
therefore not accuracy but the **gap**:

```
GAP = accuracy(in-corpus test) − accuracy(cross-corpus test)
```

Train on one corpus (FFT-75), test on a genuinely independent one (GovDocs1),
and measure how much survives.

---

## 2. Why it matters forensically

File carving recovers deleted files from unallocated space when the file system
no longer records where they were. Classical carvers (Scalpel, Foremost) work by
matching header/footer signatures — which fails precisely when it matters most:

- the header sector has been overwritten,
- the file is **fragmented** and only interior blocks survive,
- the volume is damaged or partially wiped.

For interior fragments there is no signature to match, so type identification
must come from the byte content itself. That capability feeds three real tasks:

| task | what fragment typing provides |
|---|---|
| **Triage** | estimate the composition of unallocated space before committing examiner hours |
| **Targeted carving** | find candidate JPEG/PDF regions to attempt reassembly on |
| **Prioritisation** | rank which disk regions an examiner should look at first |

The forensic requirement is stricter than the ML one. An examiner does not just
need a label; they need to know **how much weight the label can bear** — which
is why this report spends as much space on precision, calibration and operating
points as on accuracy.

---

## 3. Objectives

1. Build a fragment classifier that operates on headerless byte blocks.
2. Measure the cross-corpus gap honestly on two independent real corpora.
3. Test whether three proposed mechanisms shrink that gap:
   augmentation, entropy-aware normalisation (aux features + adversary),
   contrastive pre-training.
4. Interpret the resulting scores **as a forensic analyst would** — precision,
   confidence reliability, and the operating points at which the tool can and
   cannot support a finding.

---

## 4. Background

**FiFTy (Mittal et al., 2019)** established the modern baseline: a byte-embedding
neural network over fragments, released with the **FFT-75** dataset (75 file
types, 512 B and 4096 B variants). Its reported accuracies are in-corpus.

**GovDocs1 (Garfinkel et al., DFRWS 2009)** is a corpus of ~1M real documents
harvested from .gov domains, with true extensions. Because the files are real
and independently sourced, fragmenting them yields a genuine "away game" for a
model trained on FFT-75.

The gap between the two is the phenomenon under study. The three mechanisms
tested here are standard responses to domain shift — input perturbation
(augmentation), nuisance-factor removal (gradient-reversal adversary), and
representation learning (contrastive pre-training).

---

## 5. Datasets

| corpus | role | fragments | source |
|---|---|---|---|
| **FFT-75** (FiFTy scenario 1) | train + in-corpus test | 6.14M @ 512 B; 6.14M @ 4096 B | IEEE DataPort |
| **GovDocs1** (zips 000–020) | cross-corpus test | 113,884 @ 512 B; 53,929 @ 4096 B | digitalcorpora.org |

### 5.1 Label space

The configured 10-class label set was reduced to **9**. `encrypted` appears in
neither corpus — FFT-75 scenario 1 has no encrypted type and GovDocs1 has no
`.aes`/`.gpg` files — so retaining it would create a classifier head with zero
training and zero test examples. See `configs/real.yaml`.

Source types are mapped by **name** onto the target set, so the label space is
identical across corpora and the gap is meaningful. Two classes absorb several
source types: `zip ← {7z, bz2, gz, rar, zip}` and `txt ← {txt, log}`.

### 5.2 Fragmentation protocol

- Non-overlapping blocks of `L` bytes.
- **Block 0 of every file is discarded**, so no magic signature is ever visible.
  A file shorter than `2L` yields nothing rather than falling back to its header.
- Verified: **<0.1%** of fragments begin with their own format's magic bytes,
  at both block sizes (`scripts/verify_processed.py`).

### 5.3 Class balance

FFT-75 is balanced by construction (50k/class available). GovDocs1 is not — it
reflects what real .gov servers hold:

| class | @512 B | @4096 B |
|---|---|---|
| txt, html, pdf, jpg | 20,000 | 10,000 |
| gif | 15,420 | 5,503 |
| csv | 5,780 | 2,198 |
| zip | 5,565 | 4,206 |
| xml | 5,308 | 1,339 |
| **png** | **1,811** | **683** |

`png` is drawn from only 88 source files, so per-class png figures carry wide
error bars. This is a property of the corpus, not of the model.

---

## 6. Methodology

### 6.1 Architecture

`ByteCNN` (599,377 parameters):

```
bytes (B, L) ──► Embedding(256 → 32)          learn a vector per byte VALUE
              ──► Conv1d(32→64,  k=9) + BN + ReLU + MaxPool + Dropout
              ──► Conv1d(64→128, k=9) + BN + ReLU + MaxPool + Dropout
              ──► Conv1d(128→256,k=9) + BN + ReLU + MaxPool + Dropout
              ──► Global average pool                 (B, 256)
              ──► [concat 8 entropy aux features]     (B, 264)
              ──► Linear(→256) + ReLU + Dropout + Linear(→9)
```

Global average pooling makes the network **length-agnostic**, so the identical
architecture runs at 512 and 4096 bytes — which is what allows the block-size
comparison to be clean.

Byte *embedding* rather than raw byte values matters: it lets the model learn
that 0x00 and 0xFF are categorically different rather than numerically distant,
and the convolutions then read n-gram structure (JPEG markers, PDF keywords,
DEFLATE noise) rather than mere byte frequencies.

### 6.2 The three mechanisms under test

| mechanism | implementation | intent |
|---|---|---|
| **Augmentation** | cyclic shift, byte-replacement noise, per-bit flips, SpecAugment-style masking | force invariance to encoder quirks |
| **Entropy aux features** | 8 statistics (entropy, mean, std, printable ratio, zero ratio, 0xFF ratio, newline ratio, mean abs delta) concatenated into the head | let the classifier factor out overall randomness rather than entangle it |
| **Entropy adversary** | gradient-reversal head predicting an entropy bucket from the backbone | stop the backbone from encoding raw randomness level |
| **Contrastive pre-training** | NT-Xent (SimCLR) on two augmented views before supervised fine-tuning | learn structure from unlabelled fragments |

### 6.3 Evaluation metrics

| metric | question it answers | audience |
|---|---|---|
| accuracy | how often is the model right | ML |
| macro-F1 | ...averaged fairly over imbalanced classes | ML |
| GAP | how much is lost off-corpus | the project's thesis |
| **precision** | if the tool says X, how often is it X | **analyst** |
| **ECE** | can the printed confidence be quoted | **analyst** |
| **coverage/selective accuracy** | what reliability is buyable by abstaining | **analyst** |

---

## 7. Experimental design

**Ablation ladder** — each configuration adds one mechanism:

| run | augment | aux | adversary | contrastive |
|---|---|---|---|---|
| A_plain | – | – | – | – |
| B_aug | ✓ | – | – | – |
| C_aux | ✓ | ✓ | – | – |
| D_adv | ✓ | ✓ | ✓ | – |
| E_full | ✓ | ✓ | ✓ | ✓ |

**Runs**: 5 configs × **12 random seeds** = 60 training runs at 512 B, plus
5 × 2 × 2 = 20 runs for the block-size comparison. Every run trains, evaluates
on both corpora, and writes a JSON report.

**Controls applied:**

- *Seeds*: 12 per configuration. A 3-seed pilot produced a conclusion that
  **reversed** at 12 seeds (§11), so seed count is treated as a first-class
  design parameter.
- *Matched budgets*: the block-size arms use identical fragments/class and
  epochs, so the only variable is fragment length.
- *Device consistency*: CPU and GPU results are never mixed within one table.
- *Significance*: Welch's t-test against the `A_plain` control, plus a stated
  minimum detectable difference so null results can be read correctly.

---

## 8. Results

### 8.1 Ablation, 512 bytes, 12 seeds

| run | in-corpus acc | cross acc | cross macro-F1 | GAP |
|---|---|---|---|---|
| baseline (logreg on byte histogram) | 0.7578 | 0.4877 | – | 0.2701 |
| A_plain | 0.8007 ±0.0119 | 0.5243 ±0.0280 | 0.4437 ±0.0255 | 0.2764 ±0.0269 |
| B_aug | 0.8145 ±0.0080 | 0.5293 ±0.0190 | 0.4500 ±0.0158 | 0.2851 ±0.0178 |
| C_aux | 0.8129 ±0.0070 | 0.5402 ±0.0207 | 0.4556 ±0.0181 | 0.2728 ±0.0201 |
| D_adv | 0.7643 ±0.0215 | 0.5152 ±0.0171 | 0.4201 ±0.0222 | **0.2492** ±0.0228 |
| E_full | 0.7927 ±0.0122 | 0.5239 ±0.0258 | 0.4380 ±0.0222 | 0.2687 ±0.0255 |

**Significance vs control (cross-corpus accuracy) — nothing is significant:**

| run | difference | t | p |
|---|---|---|---|
| B_aug | +0.0051 | +0.52 | 0.610 |
| C_aux | +0.0159 | +1.58 | 0.130 |
| D_adv | −0.0091 | −0.96 | 0.348 |
| **E_full** | **−0.0003** | −0.03 | **0.975** |

The complete FragGen stack is **statistically indistinguishable from a plain
byte-CNN** on the metric the project exists to improve.

**What is significant:**

| effect | difference | p | interpretation |
|---|---|---|---|
| B_aug, in-corpus | +0.0138 | 0.004 | augmentation improves fit to FFT-75 |
| C_aux, in-corpus | +0.0122 | 0.007 | so do aux features |
| both, cross-corpus | ~0 | n.s. | neither transfers |
| D_adv, GAP | −0.0273 | 0.014 | adversary shrinks the gap |
| D_adv, in-corpus | −0.0364 | 0.000 | by getting worse at home |
| D_adv, cross macro-F1 | −0.0236 | 0.025 | and worse away too |

Augmentation and aux features buy **in-corpus fit that does not transfer** —
the exact signature of corpus overfitting the mechanisms were meant to prevent.

> **On the GAP metric.** GAP rewards a model that degrades uniformly. `D_adv`
> demonstrates this: the best GAP in the matrix, significantly *worse* on both
> accuracies that matter. **GAP must never be reported alone.**

### 8.2 Block size, matched budget

| config | cross acc @512 | cross acc @4096 | change | p |
|---|---|---|---|---|
| **A_plain** | 0.5007 ±0.0151 | **0.6812** ±0.0141 | **+0.1805** | 0.007 |
| B_aug | 0.5510 ±0.0029 | 0.6705 ±0.0640 | +0.1196 | 0.230 |
| C_aux | 0.4988 ±0.0292 | 0.6444 ±0.0103 | +0.1456 | 0.064 |
| D_adv | 0.4588 ±0.0122 | 0.5447 ±0.0545 | +0.0859 | 0.256 |
| E_full | 0.4755 ±0.0321 | 0.5856 ±0.0612 | +0.1101 | 0.193 |
| **pooled** | **0.4970** | **0.6253** | **+0.1283** | **0.00009** |

**+0.128 from fragment length against ~0.000 from three mechanisms** — an order
of magnitude apart. At 4096 B the plain CNN is again the best configuration.

---

## 9. Results and discussion — the analyst's reading

Accuracy answers *"how often is the model right"*. An examiner needs four
different questions answered, and the answers are less flattering.

### 9.1 If the tool reports type X, how often is it X? (precision)

| reported | precision | recall | n called | operational verdict |
|---|---|---|---|---|
| jpg | 0.888 | 0.836 | 9,414 | **reliable** — can support a finding |
| xml | 0.790 | 0.048 | 81 | trustworthy when claimed, rarely claimed |
| html | 0.768 | 0.734 | 9,549 | indicative — corroborate |
| txt | 0.744 | 0.587 | 7,888 | indicative — corroborate |
| gif | 0.701 | 0.906 | 7,111 | indicative — corroborate |
| pdf | 0.678 | 0.463 | 6,833 | indicative — corroborate |
| csv | 0.422 | 0.858 | 4,464 | **weak** — over-reports; lead only |
| zip | 0.377 | 0.706 | 7,876 | **unreliable** — do not report |
| png | 0.140 | 0.146 | 713 | **unreliable** — do not report |

*(4096 B. At 512 B png precision is **0.037** — wrong 96% of the time it says PNG.)*

Three consequences that "accuracy = 0.67" conceals:

1. **Competence is per-type.** One headline number averages a defensible JPEG
   identifier with a PNG detector worse than chance. Findings must be reported
   per type with their own reliability attached.
2. **`xml` shows the classic forensic asymmetry.** Precision 0.79, recall 0.048
   — claimed 81 times out of 1,339 actual XML fragments. **Absence of evidence
   is not evidence of absence**: "no XML on this disk" is unsupportable.
3. **`csv` and `zip` over-report.** High recall, low precision — they act as
   catch-alls. Their positives are triage leads, not findings.

### 9.2 Can the confidence value be quoted?

| stated confidence | actual accuracy | overstatement |
|---|---|---|
| 0.990 | 0.823 | +0.167 |
| 0.927 | 0.681 | +0.247 |
| 0.845 | 0.575 | **+0.271** |
| 0.755 | 0.529 | +0.226 |
| 0.652 | 0.509 | +0.143 |

**ECE = 0.185 @4096 B, 0.242 @512 B.** Systematically overconfident, and worst
in the middle of the range where an examiner is most likely making a judgement
call. Addressed in §10.

### 9.3 What reliability can be bought by abstaining?

| threshold | coverage | accuracy | operational use |
|---|---|---|---|
| 0.00 | 100.0% | 0.671 | sweep everything |
| 0.70 | 80.9% | 0.721 | broad triage |
| 0.90 | 54.5% | 0.801 | prioritisation |
| 0.95 | 46.0% | 0.823 | prioritisation |
| 0.99 | 30.8% | 0.833 | best available |

**90% reliability is not achievable at any threshold** at 4096 B; at 512 B only
on the top 0.5% of blocks.

> **Central operational conclusion.** FragGen is a **triage instrument, not an
> identification instrument.** It ranks where an examiner should look and gives
> a coarse composition estimate of unallocated space. It does not establish what
> any individual fragment is. An 0.83-accuracy filter over 30% of a terabyte
> image saves real examiner time — but the claim must match the evidence.

### 9.4 When is an "error" actually a finding?

14,206 test fragments are labelled `pdf` or `zip`; strict accuracy on them is
0.535 against 0.720 elsewhere. But a PDF's interior is FlateDecode streams and
embedded DCTDecode images. The label says "pdf" because the *file* was a PDF;
the model says "jpg" because *those bytes are JPEG*.

For a carver reconstructing content, **the model's answer is the more useful
one** — it locates image data at a specific offset inside a container. Scored as
classification error it counts against the tool; read as an investigative
result it is what the examiner wanted. Report `pdf → jpg` as *evidence of an
embedded image*, not as a misclassification.

Error decomposition (48,506 cross-corpus errors):

| category | share |
|---|---|
| container interior | **34.9%** |
| html ↔ xml | 6.1% |
| genuine | 59.0% |

### 9.5 Which failures are irreducible?

| failure | cause | fixable? |
|---|---|---|
| png ↔ zip | PNG IDAT is zlib/DEFLATE — the same compressor ZIP uses | **No** at 512 B; longer fragments help (0.037 → 0.140) but do not solve it |
| xml → html | both angle-bracket markup | Partly — merge, or report `markup` |
| pdf/zip interiors | containers hold other formats | Not an error (§9.4) |
| txt/csv/html | all printable ASCII; separated by delimiter statistics | Yes — most helped by context |

**Some confusions are information-theoretic, not engineering failures.** No
architecture separates two DEFLATE streams from different containers when both
are headerless and 512 bytes long. Knowing which errors are irreducible tells an
examiner where corroborating evidence — filesystem metadata, adjacent-block
continuity, carving context — is *required* rather than optional.

### 9.6 What this means for the hypothesis

The mechanisms were designed to make the model learn format structure rather
than corpus quirks. They did something subtly worse than nothing: augmentation
and aux features produced **significant in-corpus gains that did not transfer**,
and the adversary improved the headline GAP while significantly *reducing* both
in-corpus accuracy and cross-corpus macro-F1.

Fragment length moved every operational measure at once:

| measure | @512 B | @4096 B |
|---|---|---|
| cross-corpus accuracy | 0.536 | **0.671** |
| png precision | 0.037 | **0.140** |
| zip precision | 0.189 | **0.377** |
| calibration error (ECE) | 0.242 | **0.185** |

**The intervention that helped gave the model more evidence; the ones that
failed only regularised how it learned.**

---

## 10. Confidence calibration

§9.2 showed the raw softmax cannot be quoted. **Temperature scaling**
(Guo et al., 2017) rescales the logits by a single learned scalar `T`:

```
p = softmax(logits / T)
```

`T` is fitted by minimising NLL on a held-out **in-corpus** split — all an
analyst realistically has labels for. Because the transform is monotonic it
**cannot change any prediction**: accuracy, precision and recall are bit-identical
before and after, which is exactly the property evidence handling requires. The
implementation asserts this.

### Results (T = 1.2903)

| test set | ECE before | ECE after | change | verdict |
|---|---|---|---|---|
| **in-corpus** (FFT-75) | 0.0575 | **0.0279** | −51% | **quotable as a probability** |
| **cross-corpus** (GovDocs1) | 0.1844 | 0.1393 | −24% | **still unusable** |

Reliability after calibration, in-corpus:

| stated | actual |
|---|---|
| 0.856 | 0.878 |
| 0.927 | 0.932 |
| 0.992 | 0.995 |

### Interpretation — the important part

**Calibration does not transfer across corpora.** A temperature fitted on FFT-75
makes FFT-75 confidences honest (ECE 0.028, quotable) but leaves GovDocs1
confidences overstated (ECE 0.139).

Forensically this is the decisive point: **real evidence is by definition
cross-corpus.** An examiner never has labelled fragments from the suspect disk
to calibrate on. So the calibrated confidence is trustworthy exactly where it is
not needed, and untrustworthy exactly where it is.

**Practical guidance implemented in the tool:**

- `classify.py` loads the fitted temperature automatically and applies it.
- It prints an explicit warning that the calibration was fitted in-corpus and
  reports the cross-corpus ECE.
- On out-of-corpus evidence, confidence should be treated as a **ranking**
  (which blocks to examine first), never as a probability.
- To quote a number, use the **empirical** accuracy at a chosen threshold from
  the operating-point table (§9.3), which is measured, not modelled.

---

## 11. Limitations and threats to validity

| limitation | impact | mitigation / status |
|---|---|---|
| **Statistical power** | at 12 seeds the minimum detectable difference is ~0.019 | stated explicitly; C_aux's +0.0159 sits just under it and is unresolved |
| **3-seed pilot reversed** | a 3-seed run showed augmentation *hurting* (−0.0228); at 12 seeds it was +0.0051 (p=0.610) | the reason 12 seeds were run; documented as the main methodological lesson |
| **Block-size arms are n=2** | per-config tests weak | pooled test decisive (p=0.00009); direction consistent across all 5 |
| **Subsampled training** | 8,000 of 50,000 fragments/class | GPU-bounded; full-data run is future work |
| **One away-corpus** | GovDocs1 only | DFRWS prep is stubbed (`fraggen/data/dfrws.py`) |
| **Thin classes** | png = 683 fragments @4096 B from 88 files | fix is extracting more corpus zips, not tuning |
| **`encrypted` untested** | absent from both corpora | label space reduced to 9, documented |
| **Calibration does not transfer** | confidence unquotable on real evidence | measured and reported (§10) |

---

## 12. Conclusions

1. **The cross-corpus gap is real on real corpora**: 0.80 in-corpus → 0.52
   cross-corpus for a plain byte-CNN at 512 bytes.
2. **The byte-CNN is worth having**: +0.037 cross-corpus over a flat-feature
   logistic-regression baseline. Sequence modelling helps.
3. **None of the three mechanisms helps.** At 12 seeds, no configuration differs
   significantly from the plain CNN on cross-corpus accuracy; the full stack is
   indistinguishable (p=0.975). Two mechanisms significantly improve *in-corpus*
   accuracy without transferring — corpus overfitting.
4. **Fragment length dominates**: +0.128 cross-corpus accuracy (p=0.00009),
   improving accuracy, per-class precision and calibration simultaneously.
5. **The tool is a triage instrument.** Court-grade single-fragment
   identification is unreachable at any confidence threshold.
6. **Confidence is quotable in-corpus after calibration, and not cross-corpus** —
   which is where evidence actually lives.

**The defensible thesis is therefore:** *fragment length, not training-time
regularisation, governs cross-corpus generalisation in file-fragment
classification.* This is better supported than the original hypothesis and more
actionable for practitioners: **carve larger blocks before training harder.**

---

## 13. Future work

1. **Full ablation at 4096 B with 12 seeds** — does C_aux's positive trend
   survive at the block size that works?
2. **Push further on length**: 8192, 16384 bytes — find where it saturates.
3. **Cross-corpus calibration**: unsupervised/transductive calibration, or
   conformal prediction, which gives distribution-free coverage guarantees and
   would directly address §10's failure.
4. **DFRWS 2006/2007** as a second away-corpus.
5. **Container-aware evaluation** as the default reporting mode.
6. **Full 450k training set**, now a ~6-minute GPU run.

---

## 14. How to run everything

### Setup

```bash
pip install -r requirements.txt
# GPU (optional but ~16x faster at 512 B, ~7x at 4096 B):
pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

### Data preparation

```bash
# GovDocs1: extract only the types in the label set
mkdir -p data/govdocs1_files
for z in data/Govdoc1/*.zip; do
  unzip -q -o -C "$z" '*/*.html' '*/*.pdf' '*/*.txt' '*/*.jpg' '*/*.gif' \
    '*/*.xml' '*/*.csv' '*/*.png' '*/*.gz' -d data/govdocs1_files
done

# FFT-75 -> FragmentSets (streams the npz; never loads the 3.1 GB array)
python scripts/prep_fft75.py --src data/FFT-75/512 --out data/processed \
    --block-size 512 --per-class 50000 --config configs/real.yaml

# GovDocs1 -> cross-corpus test set
python -m fraggen.data.govdocs --root data/govdocs1_files \
    --out data/processed/govdocs_512_test.npz --block-size 512 \
    --per-class 20000 --config configs/real.yaml

# sanity check: shapes, balance, and magic-byte leakage
python scripts/verify_processed.py "data/processed/*.npz"
```

### Experiments

```bash
bash scripts/run_overnight.sh      # everything, resumable, no network needed
```

or individually:

```bash
python scripts/run_ablations.py --limit 8000 --epochs 6 \
    --seeds 1337,1338,1339 --outdir reports/ablations_gpu
python scripts/significance.py --results reports/ablations_gpu
python scripts/blocksize_compare.py
python scripts/make_figures.py --results reports/ablations_gpu \
    --results-512 reports/blocksize_512 --results-4k reports/blocksize_4096
```

### Forensic analysis and calibration

```bash
python scripts/forensic_analysis.py \
    --model checkpoints/blocksize_4096/A_plain_s1337.pt \
    --data data/processed/govdocs_4096_test.npz

python scripts/calibrate.py \
    --model checkpoints/blocksize_4096/A_plain_s1337.pt \
    --val data/processed/fft75_4096_val.npz \
    --in-test data/processed/fft75_4096_test.npz \
    --cross data/processed/govdocs_4096_test.npz
```

### Using the classifier on evidence

```bash
# what type is this?
python scripts/classify.py --model checkpoints/blocksize_4096/A_plain_s1337.pt \
    --file suspect.bin

# per-offset type map across a raw image, abstaining when unsure
python scripts/classify.py --model checkpoints/blocksize_4096/A_plain_s1337.pt \
    --file disk.raw --map --min-conf 0.9
```

It reports calibrated confidence automatically when a `.calib.json` exists, and
warns that the calibration was fitted in-corpus.

### Live demo

```bash
python scripts/demo.py            # four acts: problem, model, gap, finding
python scripts/demo.py --quick    # faster
```

> **Note when presenting**: `demo.py` scores **files** (majority vote over their
> blocks); the tables above score **fragments**. File-level numbers are higher.

### Script reference

| script | purpose |
|---|---|
| `prep_fft75.py` | FFT-75 npz → FragmentSet, streamed |
| `verify_processed.py` | shapes, balance, magic-byte leak test |
| `run_ablations.py` | ablation ladder, `--seeds`, resumable |
| `significance.py` | Welch t-tests + minimum detectable difference |
| `blocksize_compare.py` | 512 vs 4096 with matched budgets |
| `forensic_analysis.py` | precision tiers, calibration, operating points |
| `calibrate.py` | fit temperature; report in- vs cross-corpus ECE |
| `error_analysis.py` | container/markup error decomposition |
| `make_figures.py` | report figures |
| `classify.py` | **use the model on real evidence** |
| `demo.py` | narrated live demo |
| `run_overnight.sh` | full pipeline |

---

## 15. Engineering log — defects found and fixed

Recording these because several would have silently corrupted results.

| # | defect | consequence if missed |
|---|---|---|
| 1 | `histogram_bayes_ceiling` documented as an upper bound | three classifiers exceeded it; renamed `byte_marginal_reference`. The reframing is a result: everything above the line is accuracy from byte **order** |
| 2 | Missing `model.eval()` at inference | dropout/BatchNorm live during prediction; identical bytes gave different answers, recall read 0.08 vs 0.797 |
| 3 | Header leakage at 4096 B | files of 4–8 KB contributed only their header block — **5.53% of PNG fragments** carried magic bytes, inflating the headline block-size result |
| 4 | Checkpoint namespace collision | 4096-byte runs overwrote 512-byte models of the same name |
| 5 | `torch.load` under torch ≥2.6 | `weights_only=True` default + numpy strings → every eval would fail *after* training |
| 6 | Inflated container statistic | a submatrix dropped errors where non-containers were predicted as containers; reported 0.7045 vs true 0.6459 |
| 7 | Confounded block-size figure | 512 arm from the 8k/class matrix vs 4096 arm at 4k/class; the plot script now refuses mismatched budgets |
| 8 | Broad `except Exception` hid a missing import | a calibrated model silently reported as uncalibrated |

---

## References

- Mittal, Thakur, Mylonas, Ramanathan — *FiFTy: Large-scale File Fragment Type
  Identification using Neural Networks*, 2019. FFT-75 dataset.
- Garfinkel, Farrell, Roussev, Dinolt — *Bringing Science to Digital Forensics
  with Standardized Forensic Corpora*, DFRWS 2009. GovDocs1.
- Guo, Pleiss, Sun, Weinberger — *On Calibration of Modern Neural Networks*,
  ICML 2017. Temperature scaling.
- Ganin & Lempitsky — *Unsupervised Domain Adaptation by Backpropagation*,
  ICML 2015. Gradient reversal.
- Chen, Kornblith, Norouzi, Hinton — *A Simple Framework for Contrastive
  Learning of Visual Representations*, ICML 2020. NT-Xent.
