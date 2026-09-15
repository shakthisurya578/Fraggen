# FragGen — Project Status

**Cross-corpus file-fragment type identification for digital forensics**
CSE4074 Forensics Analytics

*Status as of 2026-09-10. Every number below is read from the result files in
`reports/`, not from memory.*

---

## 1. What the project is

Given a **raw block of bytes** from unallocated disk space — no filename, no
extension, no filesystem metadata, and taken from the *middle* of a file so no
magic signature is present — identify the **file type**.

The specific target is **corpus shift**. Prior classifiers score well on their
own benchmark and collapse on data from another source, because they memorise
one corpus's encoders and tool-chains. So the headline quantity is the **gap**:

```
GAP = accuracy(in-corpus test) − accuracy(cross-corpus test)
```

Train on FFT-75; test on GovDocs1 and DFRWS 2007.

**Label space: 9 classes** — `txt, csv, html, xml, pdf, jpg, png, gif, zip`.
The configured `encrypted` class was dropped because *neither* corpus contains
an encrypted type; keeping it would create a classifier head with zero training
and zero test examples. See `configs/real.yaml`.

---

## 2. Headline result

**Two things move cross-corpus accuracy. Neither is a FragGen mechanism.**

| lever | cross-corpus gain | evidence |
|---|---|---|
| **Bigger fragments** (512 → 4096 B) | **+0.128** | pooled, p = 0.00009 |
| **More training data** (8k → 50k frag/class) | **+0.108** | n = 3, t ≈ 7.6 |
| Container-aware reporting | +0.049 | relabelling only |
| Contiguous-run aggregation | +0.027 | errors are correlated |
| **All three FragGen mechanisms** | **~0.000** | n = 12, p = 0.975 |

Both winning levers give the model **more evidence**. The three mechanisms —
designed to regularise *how* it learns — do nothing measurable.

**Defensible thesis:** *fragment length and training-set size, not training-time
regularisation, govern cross-corpus generalisation in file-fragment
classification.* Practical advice: **carve larger blocks before training harder.**

---

## 3. What is DONE

### 3.1 Data pipeline ✅

| corpus | role | 512 B | 4096 B |
|---|---|---|---|
| **FFT-75** (FiFTy scenario 1) | train + in-corpus test | ✅ train/val/test | ✅ train/val/test |
| **GovDocs1** (zips 000–020) | cross-corpus #1 | ✅ 113,884 frags | ✅ 53,929 frags |
| **DFRWS 2007** | cross-corpus #2 | ✅ prepped | ✅ 6,733 frags |

- FFT-75 label order taken from the FiFTy repo's `labels.json` — the `.npz`
  files carry only integer indices, so a wrong ordering would have silently
  mislabelled all 6.14M fragments.
- **Middle-of-file only**: block 0 of every file is discarded. Verified at
  **<0.1% magic-byte leakage** on both block sizes (`verify_processed.py`).
- DFRWS layout tables converted from *sector* ranges to *byte* offsets by
  `scripts/dfrws_layout_to_truth.py`, with header-block skipping so the results
  stay comparable to FFT-75 / GovDocs1.

### 3.2 Ablation matrix — 5 configs × 12 seeds ✅

`reports/ablations_gpu/` — 60 runs, 512 B, 8k fragments/class, 6 epochs.

| run | in-corpus acc | cross acc | cross macro-F1 | GAP |
|---|---|---|---|---|
| baseline (logreg) | 0.7578 | 0.4877 | – | 0.2701 |
| A_plain | 0.8007 ±0.0119 | 0.5243 ±0.0280 | 0.4437 ±0.0255 | 0.2764 ±0.0269 |
| B_aug | 0.8145 ±0.0080 | 0.5293 ±0.0190 | 0.4500 ±0.0158 | 0.2851 ±0.0178 |
| C_aux | 0.8129 ±0.0070 | 0.5402 ±0.0207 | 0.4556 ±0.0181 | 0.2728 ±0.0201 |
| D_adv | 0.7643 ±0.0215 | 0.5152 ±0.0171 | 0.4201 ±0.0222 | **0.2492** ±0.0228 |
| E_full | 0.7927 ±0.0122 | 0.5239 ±0.0258 | 0.4380 ±0.0222 | 0.2687 ±0.0255 |

**Significance (Welch t-test vs the A_plain control), cross-corpus accuracy —
nothing is significant:**

| run | difference | t | p |
|---|---|---|---|
| B_aug | +0.0051 | +0.52 | 0.610 |
| C_aux | +0.0159 | +1.58 | 0.130 |
| D_adv | −0.0091 | −0.96 | 0.348 |
| **E_full** | **−0.0003** | −0.03 | **0.975** |

The full stack is **statistically indistinguishable from a plain byte-CNN** on
the metric the project exists to improve.

**What *is* significant tells the story:**

| effect | difference | p | reading |
|---|---|---|---|
| B_aug, in-corpus | +0.0138 | 0.004 | augmentation fits FFT-75 better… |
| C_aux, in-corpus | +0.0122 | 0.007 | …so do aux features… |
| both, cross-corpus | ~0 | n.s. | …but neither transfers |
| D_adv, GAP | −0.0273 | 0.014 | adversary shrinks the gap… |
| D_adv, in-corpus | −0.0364 | 0.000 | …by getting worse at home |
| D_adv, cross macro-F1 | −0.0236 | 0.025 | …and worse away too |

> **The GAP metric is a trap.** It rewards a model that degrades uniformly.
> `D_adv` posts the best GAP in the matrix while being significantly *worse* on
> both accuracies that matter. Never report GAP alone.

### 3.3 Block-size comparison ✅

`reports/blocksize_512/` vs `reports/blocksize_4096/` — matched at 4k/class,
6 epochs, 2 seeds, so the **only** variable is fragment length.

| config | cross @512 | cross @4096 | change | p |
|---|---|---|---|---|
| **A_plain** | 0.5007 ±0.0151 | **0.6812** ±0.0141 | **+0.1805** | 0.007 |
| B_aug | 0.5510 ±0.0029 | 0.6705 ±0.0640 | +0.1196 | 0.230 |
| C_aux | 0.4988 ±0.0292 | 0.6444 ±0.0103 | +0.1456 | 0.064 |
| D_adv | 0.4588 ±0.0122 | 0.5447 ±0.0545 | +0.0859 | 0.256 |
| E_full | 0.4755 ±0.0321 | 0.5856 ±0.0612 | +0.1101 | 0.193 |
| **pooled** | **0.4970** | **0.6253** | **+0.1283** | **0.00009** |

At 4096 B the **plain CNN is again the best config**, and the adversary the
worst. The mechanisms do not become useful with more context.

### 3.4 Training-set size ✅

`reports/fulldata_512/` — A_plain, all 50k fragments/class, 3 seeds.

| training data | in-corpus | cross-corpus | cross macro-F1 | GAP |
|---|---|---|---|---|
| 8k/class (n=12) | 0.8007 | 0.5243 ±0.0280 | 0.4437 | 0.2764 |
| **50k/class (n=3)** | **0.8739** ±0.0041 | **0.6325** ±0.0205 | **0.5291** | **0.2415** |
| change | +0.073 | **+0.108** | +0.085 | −0.035 |

Unlike the entropy adversary, more data improves **every** measure at once.

### 3.5 Third corpus — DFRWS 2007 ✅ (with a major caveat)

`reports/eval_dfrws_4096.json`, model `blocksize_4096/A_plain_s1337`.

| test corpus | n | accuracy | macro-F1 | GAP |
|---|---|---|---|---|
| FFT-75 (in-corpus) | 36,000 | 0.8535 | 0.8446 | – |
| GovDocs1 | 53,929 | 0.6713 | 0.5444 | +0.1822 |
| **DFRWS 2007** | 6,733 | **0.7855** | **0.2156** | +0.0680 |

> ⚠️ **DFRWS covers only 3 of the 9 classes.** Its fragments are jpg (65.8%),
> pdf (26.6%) and zip (7.6%); `txt, csv, html, xml, png, gif` have **zero**
> examples. That is why macro-F1 reads 0.2156 — it averages over six absent
> classes — and why 0.7855 accuracy is **not comparable** to GovDocs1's 0.6713
> across nine classes. The figure is dominated by jpg, which the model handles
> well.
>
> Per-class on DFRWS: jpg recall 0.973 / precision 0.881 · pdf recall 0.261 /
> precision 0.936 · zip recall 1.000 / precision 0.437.
>
> **Report DFRWS as a 3-class check, not as a second full cross-corpus point.**

### 3.6 Forensic reliability analysis ✅

`scripts/forensic_analysis.py` → `reports/forensic_4096.json`, `forensic_512.json`.

**Precision — the question an analyst actually asks** (4096 B, GovDocs1):

| reported | precision | recall | operational verdict |
|---|---|---|---|
| jpg | 0.888 | 0.836 | **reliable** — can support a finding |
| xml | 0.790 | 0.048 | trustworthy when claimed, rarely claimed |
| html | 0.768 | 0.734 | indicative — corroborate |
| txt | 0.744 | 0.587 | indicative — corroborate |
| gif | 0.701 | 0.906 | indicative — corroborate |
| pdf | 0.678 | 0.463 | indicative — corroborate |
| csv | 0.422 | 0.858 | **weak** — over-reports; lead only |
| zip | 0.377 | 0.706 | **unreliable** — do not report |
| png | 0.140 | 0.146 | **unreliable** — do not report |

*(At 512 B `png` precision is **0.037** — wrong 96% of the time it says PNG.)*

`xml` shows the classic forensic asymmetry: precision 0.790, recall 0.048.
**Absence of evidence is not evidence of absence** — "no XML on this disk" is
unsupportable from this tool.

**Operating points** — abstaining buys reliability:

| threshold | coverage | accuracy |
|---|---|---|
| 0.00 | 100.0% | 0.671 |
| 0.90 | 54.5% | 0.801 |
| 0.99 | 30.8% | 0.833 |

**90% reliability is not achievable at any threshold.** FragGen is a **triage
instrument, not an identification instrument**: it ranks where an examiner
should look; it does not establish what a fragment is.

**Error decomposition** (48,506 cross-corpus errors): container interior
**34.9%** · html↔xml 6.1% · genuine 59.0%. A block inside a PDF called `jpg`
may be *correctly* identifying an embedded object — forensically a finding, not
an error.

### 3.7 Confidence calibration ✅

`fraggen/calibration.py`, `scripts/calibrate.py`. Temperature scaling,
**T = 1.2903**, fitted on a held-out in-corpus split.

| test set | ECE before | ECE after | verdict |
|---|---|---|---|
| in-corpus (FFT-75) | 0.0575 | **0.0279** | **quotable as a probability** |
| cross-corpus (GovDocs1) | 0.1844 | 0.1393 | **still unusable** |

Temperature scaling is monotonic, so **no prediction changes** — accuracy,
precision and recall are bit-identical before and after (asserted in code).

> **The important finding is that calibration does not transfer.** Real evidence
> is by definition cross-corpus, and an examiner never has labelled fragments
> from the suspect disk to calibrate on. So calibrated confidence is trustworthy
> exactly where it is *not* needed. `classify.py` applies T automatically and
> prints this caveat.

Calibrations exist for all 10 `blocksize_4096` models; 2 of 60 in `ablations_gpu`.

### 3.8 Tooling ✅

| script | purpose |
|---|---|
| `prep_fft75.py` | FFT-75 npz → FragmentSet, streamed (never loads the 3.1 GB array) |
| `dfrws_layout_to_truth.py` | DFRWS sector layout → byte-offset truth CSV |
| `verify_processed.py` | shapes, class balance, **magic-byte leak test** |
| `run_ablations.py` | ablation ladder, `--seeds`, resumable, namespaced checkpoints |
| `significance.py` | Welch t-tests + minimum detectable difference |
| `blocksize_compare.py` | 512 vs 4096 with matched budgets |
| `forensic_analysis.py` | precision tiers, calibration, operating points |
| `calibrate.py` | fit temperature; report in- vs cross-corpus ECE |
| `run_aggregation.py` | accuracy vs contiguous-run length |
| `error_analysis.py` | container / markup error decomposition |
| `make_figures.py` | 5 report figures |
| `classify.py` | **use the model on real evidence** |
| `demo.py` | narrated 4-act live demo |
| `run_overnight.sh` | full unattended pipeline, resumable |

### 3.9 Deliverables ✅

- `reports/FINDINGS.md` — experimental write-up
- `reports/REVIEW1.md` — 15-section review document
- `reports/figures/` — 5 figures (gap, per-class, confusion, reject curve, block size)
- `reports/*/summary.md` — per-experiment tables
- `significance.txt`, `blocksize_comparison.txt`, `forensic_*.json`

### 3.10 Defects found and fixed ✅

| # | defect | consequence if missed |
|---|---|---|
| 1 | `histogram_bayes_ceiling` documented as an upper bound | three classifiers exceeded it; renamed `byte_marginal_reference` |
| 2 | Missing `model.eval()` at inference | dropout/BatchNorm live during prediction; recall read 0.08 vs 0.797 |
| 3 | Header leakage at 4096 B | **5.53% of PNG fragments** carried magic bytes — would have inflated the block-size headline |
| 4 | Checkpoint namespace collision | 4096-byte runs overwrote 512-byte models of the same name |
| 5 | `torch.load` under torch ≥2.6 | `weights_only=True` + numpy strings → every eval fails *after* training |
| 6 | Inflated container statistic | dropped errors where non-containers were predicted as containers (0.7045 vs true 0.6459) |
| 7 | Confounded block-size figure | mismatched training budgets; the plot script now refuses them |
| 8 | Broad `except Exception` | hid a missing import; a calibrated model reported as uncalibrated |

---

## 4. What is TO BE DONE

### Priority 1 — cheap, high value

**1.1 Combine the two winning levers: 4096 B + full training data** 🔴
The most obvious gap in the whole project. 4096 has only ever been trained at
4k/class; full data has only ever been run at 512 B. Each gives ~+0.11–0.13
independently. If they stack, the best configuration is materially better than
anything measured so far.
*Blocker:* prep a larger 4096 training set — memory-limited to ~20k/class at
2 GB free RAM. ~15 min prep + ~30 min training.

**1.2 Re-scope the mechanisms' null result** 🔴
Every ablation conclusion was drawn at **8k/class** — a CPU-era compromise never
revisited once the GPU arrived. The honest claim today is *"at 8k/class, no
mechanism helps"*, not a universal one. Re-running the ladder at full data
(~90 min) would either generalise the null result or overturn it. `C_aux`'s
+0.0159 sits just under the 0.019 detectability threshold and is the candidate
most likely to change.

**1.3 Update the stale reports** 🟠
`FINDINGS.md` (Aug 22 07:21) and `REVIEW1.md` (Aug 22 07:36) **predate** the
full-data result (07:55) and all DFRWS work (Aug 24). Both currently describe
GovDocs1 as the only cross-corpus point and omit the +0.108 data effect.

**1.4 Block-size arms are n=2** 🟠
Pooled significance is decisive (p=0.00009) but per-config tests are weak. Three
more seeds per arm (~2 h at 4096) would let each row stand on its own.

### Priority 2 — strengthens the science

**2.1 Fix DFRWS class coverage** 🟠
Only 3 of 9 classes are populated, making it a jpg-dominated 3-class check.
`data/DFRWS/dfrws-2006-challenge.zip` is downloaded but **unused** — the 2006
image adds HTML, Text and Office fragments and would broaden coverage. Until
then, DFRWS macro-F1 (0.2156) must not be quoted as comparable.

**2.2 Multi-source training** 🟡
Train on FFT-75 **+** DFRWS, test on GovDocs1. Multi-source domain
generalisation is the standard fix for exactly this failure mode and is
untested here.

**2.3 Cross-corpus calibration** 🟡
Temperature scaling fails off-corpus (§3.7). Conformal prediction gives
distribution-free coverage guarantees and would directly address the gap that
currently stops confidence being reportable on real evidence.

**2.4 Push fragment length further** 🟡
8192 / 16384 bytes — find where the effect saturates. *Blocker:* FFT-75 ships
only 512 and 4096, so this needs a training corpus built from another source.

### Priority 3 — completeness

**3.1 Thin classes** 🟡 — GovDocs `png` has 1,811 fragments @512 B and 683
@4096 B from just 88 source files. The fix is extracting more GovDocs zips
(021+), not tuning the model.

**3.2 `encrypted` class untested** 🟡 — absent from all three corpora. DFRWS
"Random" filler sectors are high-entropy and may serve as a proxy; verify the
entropy before claiming so.

**3.3 Calibrate remaining checkpoints** 🟢 — 2 of 60 in `ablations_gpu`, none in
`blocksize_512` / `fulldata_512`.

**3.4 Reassembly evaluation** 🟢 — `dfrws.py` preserves `file_ids` for
reassembly scoring, but no reassembly metric is implemented. DFRWS is a
*carving* challenge; using it only for per-fragment classification leaves its
main value on the table.

**3.5 Container-aware metrics as default** 🟢 — implemented in `metrics.py` and
reported, but strict 9-class accuracy is still the headline everywhere.

---

## 5. Known limitations (carry these into any write-up)

| limitation | status |
|---|---|
| Mechanism null result scoped to 8k/class | **not yet generalised** (§4.1.2) |
| Statistical power: MDD ≈ 0.019 at 12 seeds | stated; `C_aux` unresolved |
| A 3-seed pilot *reversed* at 12 seeds | augmentation looked harmful (−0.0228), was +0.0051 (p=0.610) |
| Block-size arms n=2 | pooled result solid, per-config weak |
| DFRWS covers 3/9 classes | **major** — see §3.5 |
| Calibration does not transfer cross-corpus | measured, unresolved |
| png thin (88 source files) | corpus property, not model |
| `encrypted` untested | absent from all corpora |
| 90% reliability unreachable | triage tool, not identification |

---

## 6. How to run everything

```bash
# full pipeline — resumable, no network needed
bash scripts/run_overnight.sh
```

```bash
# individual stages
python scripts/run_ablations.py --limit 8000 --epochs 6 \
    --seeds 1337,1338,1339 --outdir reports/ablations_gpu
python scripts/significance.py --results reports/ablations_gpu
python scripts/blocksize_compare.py
python scripts/make_figures.py --results reports/ablations_gpu \
    --results-512 reports/blocksize_512 --results-4k reports/blocksize_4096
```

```bash
# forensic interpretation + calibration
python scripts/forensic_analysis.py \
    --model checkpoints/blocksize_4096/A_plain_s1337.pt \
    --data data/processed/govdocs_4096_test.npz
python scripts/calibrate.py --model checkpoints/blocksize_4096/A_plain_s1337.pt \
    --val data/processed/fft75_4096_val.npz \
    --in-test data/processed/fft75_4096_test.npz \
    --cross data/processed/govdocs_4096_test.npz
```

```bash
# use it on evidence
python scripts/classify.py --model checkpoints/blocksize_4096/A_plain_s1337.pt \
    --file suspect.bin --map --min-conf 0.9
```

```bash
# live demo (4 acts: problem, model, gap, finding)
python scripts/demo.py
```

**Best model to date:** `checkpoints/blocksize_4096/A_plain_s1337.pt` —
cross-corpus 0.6812, calibrated.

> **When presenting:** `demo.py` scores **files** (majority vote over their
> blocks); every table above scores **fragments**. File-level numbers run higher,
> and you will be asked about it.

---

## 7. Suggested next session

1. Prep 4096 train at ~20k/class → run `A_plain` → test whether the two levers
   stack. *(~45 min)*
2. Re-run the 5-config ladder at full data, 3 seeds, to re-scope the null
   result. *(~90 min)*
3. Regenerate `FINDINGS.md` / `REVIEW1.md` with the full-data and DFRWS
   results. *(~15 min)*
