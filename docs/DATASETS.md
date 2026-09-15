# FragGen datasets — download & prep

> Raw corpora live under **`data/raw/`** and are not in git (44.8 GiB, third-party
> licences). The folder-by-folder inventory is in
> [`data/raw/README.md`](../data/raw/README.md). This file covers the three corpora
> behind the cross-corpus study (FFT-75, GovDocs1, DFRWS) and how Dataset V4 is built.

Target label space (`configs/real.yaml → classes`):
`txt, csv, html, xml, pdf, jpg, png, gif, zip`

Each corpus's native file types are mapped onto this set **by name** (see
`src/preprocessing/fft75.py → ALIAS`); types outside the set are dropped. This keeps
the label space identical across corpora, so the cross-corpus **gap** is meaningful.

---

## 1. FFT-75 (FiFTy) — cross-corpus study, training corpus

Labelled 512B and 4096B fragments for 75 file types, ~100k+ fragments per type,
already split 80/10/10 (train/test/val) across 6 scenarios.

- Code and docs: https://github.com/mittalgovind/fifty
- Dataset mirror (needs a free IEEE account): https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset
- Papers With Code entry: https://paperswithcode.com/dataset/fft-75

Download a scenario archive (`512_1.tar.gz` for 512-byte blocks, `4k_1.tar.gz` for
4096-byte), extract it into `data/raw/FFT-75/`, then convert it to a FragGen
`FragmentSet` with `scripts/prep_fft75.py` (below).

---

## 2. GovDocs1 — cross-corpus test corpus, and the main V4 source

~1M real .gov files with known extensions. You fragment them yourself, so each
chunk inherits its file's type; this is a genuinely different source from FFT-75.

- Landing page: https://digitalcorpora.org/corpora/file-corpora/files/
- Full corpus (1000 zips of 1000 files): https://downloads.digitalcorpora.org/corpora/files/govdocs1/
- By-type subsets (smaller, faster): https://downloads.digitalcorpora.org/corpora/files/govdocs1/by_type/

This project uses thread zips `000.zip`–`020.zip` (`data/raw/Govdoc1/`), extracted to
`data/raw/govdocs1_files/`.

The fragmenter skips each file's first block, so the model is only ever tested on
middle-of-file fragments with no magic-header giveaway.

Citation: Garfinkel, Farrell, Roussev, Dinolt — *Bringing Science to Digital
Forensics with Standardized Forensic Corpora*, DFRWS 2009.

---

## 3. DFRWS 2006 & 2007 — carving images

Raw images (no filesystem) containing files and fragments, with published
ground-truth answers.

- Challenges hub: https://dfrws.org/forensic-challenges/
- 2007 repo (image + details): https://github.com/dfrws/dfrws2007-challenge

The ground truth for the 2007 image, `data/raw/DFRWS/dfrws2007_gt_4096.csv`, was produced by
`scripts/dfrws_layout_to_truth.py --block-size 4096`. In Dataset V4, DFRWS contributes
pdf/jpg/zip fragments to the **test split only**.

---

## 4. Rebuilding the cross-corpus FragmentSets

The prepared sets from the August study are archived in
`archive/data/processed_cross_corpus_fft75_govdocs1/`. To rebuild them:

```bash
# 1. GovDocs1: extract only the files whose types map onto the target label set.
#    NOTE: this unzip needs '*/*.pdf' patterns -- a bare '*.pdf' matches nothing
#    because the members are named '000/000009.pdf'.
mkdir -p data/raw/govdocs1_files
for z in data/raw/Govdoc1/*.zip; do
  unzip -q -o -C "$z" '*/*.html' '*/*.htm' '*/*.pdf' '*/*.txt' '*/*.text' \
    '*/*.log' '*/*.jpg' '*/*.jpeg' '*/*.gif' '*/*.xml' '*/*.csv' '*/*.tsv' \
    '*/*.png' '*/*.gz' '*/*.zip' '*/*.bz2' '*/*.rar' '*/*.7z' -d data/raw/govdocs1_files
done

OUT=archive/data/processed_cross_corpus_fft75_govdocs1

# 2. FFT-75 (scenario 1, 512B) -> train/val/test FragmentSets
python scripts/prep_fft75.py --src data/raw/FFT-75/512 --out $OUT \
    --block-size 512 --per-class 50000 --per-class-eval 6000 --config configs/real.yaml

# 3. GovDocs1 -> cross-corpus test FragmentSet
python -m src.preprocessing.govdocs --root data/raw/govdocs1_files \
    --out $OUT/govdocs_512_test.npz --block-size 512 --per-class 20000 --config configs/real.yaml

# 4. DFRWS -> FragmentSet
python -m src.preprocessing.dfrws --image data/raw/DFRWS/dfrws-2007-challenge.img \
    --truth data/raw/DFRWS/dfrws2007_gt_4096.csv --out $OUT/dfrws_4096_test.npz --block-size 4096

# 5. Verify shapes, balance and magic-byte leakage
python scripts/verify_processed.py "$OUT/*.npz"
```

| file | n | notes |
|---|---|---|
| `fft75_512_train.npz` | 450,000 | 50k × 9 classes, balanced |
| `fft75_512_val.npz` | 54,000 | 6k × 9 |
| `fft75_512_test.npz` | 54,000 | 6k × 9 (in-corpus "home turf") |
| `govdocs_512_test.npz` | 113,884 | cross-corpus "away game", imbalanced |

### Two things that differ from the original plan

**The label space is 9 classes, not 10.** `encrypted` has *no source* in either
corpus: FFT-75 scenario 1 has no encrypted type, and GovDocs1 has no `.aes`/`.gpg`
files. Keeping it would give the classifier a head with zero training and zero test
examples, so `configs/real.yaml` is `default.yaml` minus that class.

**The FFT-75 scenario-1 label order** is taken from the FiFTy repo's
`fifty/utilities/labels.json` and hard-coded in `scripts/prep_fft75.py`
(`FFT75_SCENARIO1`). The downloaded `.npz` files carry only integer indices, so if you
switch to a different scenario archive that list must change, or every label is
silently wrong.

`ALIAS` collapses source types: `zip` ← {7z, bz2, gz, rar, zip} and `txt` ← {txt, log}.
GovDocs per-class yields are capped by the corpus itself: only 88 PNG and 127 .gz files
exist in zips 000–020. Report per-class metrics, not just overall accuracy.

---

## 5. Dataset V4

V4 combines GovDocs1 and DFRWS with sources ingested for the classes that were starved
in V2/V3: RICO screenshots (png), TGIF (gif), CSV files and ZIP archives from public
GitHub repositories, and full Simple-Wikipedia articles (xml).

```bash
# xml source: split the Simple-Wikipedia dump into one file per article
python scripts/extract_wiki_pages.py \
    --dump data/raw/downloads/simplewiki-latest-pages-articles.xml.bz2 \
    --out data/raw/wiki_pages --min-size 8192

# build: file-level 70/15/15 split, per-class cap, header exclusion, magic-leak filter
python scripts/build_processed_v4.py
```

Outputs: `data/processed_v4/{train,val,test}.npz` and
`data/metadata/v4_{split_manifest,fragment_metadata}.csv` + `v4_build_report.md`.
The design and audit are in `reports/datasets/DATASET_V4_AND_CLEANUP_2026-09-14.md`.

**Licences.** GovDocs1 is distributed for research by Digital Corpora; DFRWS, FFT-75,
RICO, TGIF and Wikipedia (CC BY-SA) each carry their own terms, and files collected
from GitHub repositories keep their original repository licences. None of the data is
covered by this repository's MIT licence, and none of it should be pushed to a public
repository.

---

## Sources
- [FiFTy repo (mittalgovind/fifty)](https://github.com/mittalgovind/fifty)
- [FFT-75 on IEEE DataPort](https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset)
- [GovDocs1 — Digital Corpora](https://digitalcorpora.org/corpora/file-corpora/files/)
- [GovDocs1 downloads](https://downloads.digitalcorpora.org/corpora/files/govdocs1/)
- [DFRWS forensic challenges](https://dfrws.org/forensic-challenges/)
- [DFRWS 2007 challenge repo](https://github.com/dfrws/dfrws2007-challenge)
