# V4 Training Results — 2026-09-14

> **Path note (repository cleanup, 2026-09-14).** Written before the reorganisation:
> `reports/v4_results/` is now this folder, `checkpoints/v4_results/` is now
> `checkpoints/v4/`, and `FINDINGS.md` is now `docs/thesis/FINDINGS.md`. The numbers
> are unchanged. The full five-config ablation, including B–E, is in
> [`ABLATION_RESULTS.md`](ABLATION_RESULTS.md). For the path map see
> [`docs/REMOVED_FILES.md`](../../../docs/REMOVED_FILES.md).

Trained the plain byte-CNN on the newly-built, balanced `data/processed_v4/`
dataset (see `reports/DATASET_V4_AND_CLEANUP_2026-09-14.md` for how it was
built and audited). 3 seeds, 6 epochs, full training set (61,940 fragments,
no subsampling — v4 was already balanced at build time).

```bash
python scripts/run_ablations.py --only A_plain --seeds 1337,1338,1339 \
    --epochs 6 --limit 0 --block-size 4096 \
    --train-npz data/processed_v4/train.npz \
    --in-npz data/processed_v4/val.npz \
    --cross-npz data/processed_v4/test.npz \
    --outdir reports/v4_results
```

Total wall-clock: 30.4 minutes on the RTX 4050 (GPU util 99–100% throughout).

---

## ⚠️ Read this before the numbers

**These are NOT cross-corpus accuracy figures**, and should not be compared
directly to the 0.52–0.68 numbers in `FINDINGS.md`. `val.npz` and `test.npz`
are held-out **files**, not a held-out **corpus** — both splits are drawn
from the same source pool (govdocs1 + dfrws + the newly-ingested rico/tgif/
uci_csv/github_zip/wiki_pages) as `train.npz`, just different documents. This
was flagged in the data audit and applies to every V4 result below.

What this run answers: *given a balanced, multi-source, real-document
training set, how well does the model classify unseen documents drawn from
the same sources?* That's a legitimate and useful question — just a
different, easier one than cross-corpus generalization.

---

## Headline numbers

| split | accuracy | macro-F1 |
|---|---|---|
| val ("in-corpus") | **0.9082 ± 0.0049** | 0.9087 |
| test ("cross-corpus" label, same caveat applies) | **0.8938 ± 0.0036** | **0.8976 ± 0.0041** |
| gap | 0.0144 ± 0.0024 | — |

(mean ± std across 3 seeds: 1337, 1338, 1339 — full per-seed detail in
`reports/v4_results/summary.md`)

For context, the previous best cross-corpus result in this project
(FFT-75 → GovDocs1, 4096 B, full training data) was **0.6812**. V4's 0.89 is
not a better model — it's answering an easier question on a different,
better-balanced dataset. The small, tight val↔test gap (1.4 points, std
0.0024) confirms this: a genuine cross-corpus gap would not be this small or
this stable across random seeds.

## Per-class breakdown (test split, mean of 3 seeds)

| class | recall | precision | note |
|---|---|---|---|
| **png** | **1.000** | **0.999** | essentially solved — RICO screenshots are visually/structurally uniform |
| xml | 0.991 | 0.981 | strong — full-length Wikipedia articles are distinctive |
| csv | 0.958 | 0.953 | strong, **despite only reaching 82% of its fragment cap** at build time |
| gif | 0.891 | 0.973 | strong |
| html | 0.920 | 0.962 | strong |
| txt | 0.891 | 0.891 | solid |
| jpg | 0.948 | 0.818 | high recall, but see below — absorbs some pdf-embedded images |
| zip | 0.865 | 0.726 | absorbs pdf-embedded compressed streams (see below) |
| **pdf** | **0.620** | 0.877 | **the weak class** — but see the confusion breakdown |

**csv is a genuinely good sign.** The data audit worried that its 82%-of-cap
shortfall (8,204 vs. the 10,000 target) might mean unreliable training. It
didn't — csv actually recalls better than txt or html here. The class was
never *badly* thin (unlike old V3's 278-fragment png), just short of the
cap; that turned out to not matter.

## Why pdf is weak — and it isn't really a model failure

Confusion breakdown for true-pdf fragments (seed 1337, test split):

| predicted as | share |
|---|---|
| pdf (correct) | 58.8% |
| **zip** | **24.9%** |
| **jpg** | **15.4%** |
| txt / gif | 0.9% |

This is the same container-format phenomenon documented in `FINDINGS.md`
§9.4/§9.5: a PDF's interior is FlateDecode-compressed streams (byte-identical
to ZIP's DEFLATE) and often contains embedded JPEG images (literally JPEG
byte streams). A 4096-byte fragment from inside such a region genuinely *is*
zip- or jpg-shaped at the byte level — calling it "pdf" would be wrong at the
byte level even though the source *file* was a PDF. ~40% of pdf's "errors"
fall into these two categories, meaning the *effective* pdf error rate on
content that is actually pdf-structured (not an embedded object) is much
smaller than the raw 38% miss rate suggests.

This also explains the two other classes with lower precision:
- **zip precision 0.726**: 22.5% of "zip" predictions are actually pdf
  fragments hitting the exact reason above.
- **jpg precision 0.818**: 11.2% of "jpg" predictions are actually pdf
  fragments containing embedded JPEG data.

This is consistent across all 3 seeds and matches the earlier
FFT-75/GovDocs1 findings exactly — it's a property of the PDF format itself,
not an artifact of this particular dataset or training run.

## What this means

1. **The balancing worked as intended.** Every class (including the
   previously-starved png/xml/csv/gif/zip) now trains to a usable accuracy —
   no class is a near-zero-precision dead weight the way png (0.037–0.14
   precision) and pdf-dominance were in V2/V3.
2. **The pdf/zip/jpg container confusion is real and expected**, not a bug —
   it reproduces the identical finding already documented for the
   FFT-75/GovDocs1 cross-corpus work, on a completely different data source
   (real GovDocs1 PDFs either way, but different jpg/zip sources this time:
   RICO wasn't involved here since jpg comes from govdocs1+dfrws only in
   V4). That consistency across two independent builds is itself evidence
   the explanation is correct.
3. **Do not report the 0.89 accuracy as a cross-corpus result.** If asked
   "how good is FragGen at cross-corpus generalization," the answer is still
   the FFT-75→GovDocs1 numbers in `FINDINGS.md` (0.52 @512B / 0.68 @4096B).
   If asked "how good is FragGen at classifying unseen real documents from
   its own training sources," the answer is this: 0.89, and it's now a
   defensible number because every class actually has enough data.

## Suggested next step, if useful

This run used only the plain-CNN config (A_plain). Given the FFT-75/GovDocs1
ablation work found the three FragGen mechanisms (augmentation, entropy
adversary, contrastive) made no measurable difference to cross-corpus
accuracy, it would be worth checking whether that null result also holds
in-corpus on V4 — i.e. run D_adv / E_full on the same data and compare. Not
done here since the ask was "train and check accuracy," not a full ablation
re-run; say the word and I'll launch it (~1–1.5 hours for the remaining 4
configs × 3 seeds at this GPU's demonstrated ~2.5–4 min/epoch on v4-sized
data).
