# FragGen V4 — Viva Script (about 14 minutes)

A timed, spoken walkthrough. **Say** is a suggested wording, **Show** is what should
be on screen, and **Run** is the command to type. The live demo is in
[`DEMO_GUIDE.md`](DEMO_GUIDE.md); prepared answers are in [`EXAMINER_QA.md`](EXAMINER_QA.md).

## Before the examiners arrive (checklist)

```bash
python -m unittest discover -s tests -t .      # all tests pass, checkpoints load
python demo/cli/viva_demo.py --no-pause        # warm-up run: caches, GPU init
streamlit run demo/streamlit/app.py            # leave running in a second terminal
```

Have these figures open in an image viewer, in this order: `fig8`, `fig4`, `fig1`,
`fig2`, `fig7`, `fig6` (all in `reports/figures/`).

---

## 1. Opening — 0:00 to 0:30

**Say:** "FragGen classifies the type of a file from a single 4096-byte block taken
from the middle of the file, with no name, no header and no filesystem. That is the
situation in unallocated disk space. I'll show how the data was built, the model, a
live demo, the results, and what the model gets wrong and why."

## 2. The forensic problem — 0:30 to 2:00

**Say:**
- "When a file is deleted, the directory entry goes, but the data blocks often stay.
  Carvers recover blocks, and block types decide how to reassemble them."
- "Signature tools look at the header. We deliberately never let the model see it,
  because a recovered mid-file block has no header."

**Show:** `fig8_pipeline_architecture.png`, top row.

## 3. Data — 2:00 to 4:00

**Show:** `fig4_dataset_class_distribution.png`

**Say:**
- "Dataset V4 has 88,805 real fragments over nine types. Earlier versions were
  unusable: V3 was 49% PDF with PNG at 0.2%."
- "V4 adds real sources for the starved classes (RICO, TGIF, GitHub CSV and ZIP files,
  Wikipedia articles) and caps each class at 10,000 fragments. Panel (b) shows where
  each class comes from."
- "Leakage controls: the split is by file, not by fragment; fragments start at offset
  4096; anything still beginning with a magic signature is dropped. The audit found no
  identical file shared between train and test."

## 4. Model — 4:00 to 5:30

**Show:** `fig8_pipeline_architecture.png`, middle panel.

**Say:**
- "Each byte value is embedded into 32 dimensions, followed by three convolution blocks
  with kernel 9, which halve the length each time: 4096 to 512 positions with 256
  channels. Then global average pooling and a two-layer classifier. That's 465 thousand
  parameters."
- "A file gets a majority vote over its fragments, but every fragment's prediction is
  kept and exported."
- "The dashed boxes are the four mechanisms the project was designed to test. The
  deployed model uses none of them, and the ablation explains why."

## 5. Live demo — 5:30 to 9:30

**Run:** `python demo/cli/viva_demo.py`

**Say at each step (press Enter to advance):**
1. PDF: "Twelve fragments after the header. Nine PDF, three ZIP. The verdict is PDF, and
   the note explains the ZIP calls, which we'll come back to."
2. PNG, CSV, ZIP: go through these quickly. "Unanimous, high confidence."
3. Tricky PDF: "This PDF's vote flips to ZIP. It isn't a random failure. The script now
   parses the PDF's own stream objects…"
4. Stream table: "Every fragment called ZIP lies entirely inside a FlateDecode stream.
   FlateDecode *is* DEFLATE, the compression inside ZIP. The fragments that contain
   PDF object syntax are called PDF with high confidence."

**Optional, if time allows:** in the Streamlit app, upload a file the examiner provides,
then point to the timeline and the download buttons.

## 6. Results — 9:30 to 11:00

**Show:** `fig1_confusion_matrix.png`, then `fig2_precision_recall.png`

**Say:**
- "Test accuracy is 0.894 ± 0.004 over three seeds; macro-F1 is 0.898. PNG, XML and CSV
  are essentially solved."
- "Most of the error is one row: PDF, with 24% predicted ZIP and 13% JPG."
- **The key caveat, said plainly:** "This 89% is on unseen *files* from the same sources.
  It is not a cross-corpus result. When we trained on FFT-75 and tested on GovDocs1, a
  genuinely different corpus, accuracy was 0.68 at this block size. Both numbers are
  reported, and they answer different questions."

## 7. Why PDF looks like ZIP — 11:00 to 12:15

**Show:** `fig7_pdf_container_analysis.png`, panel (b).

**Say:** "To check the explanation rather than assert it, I took all 1,500 GovDocs1 PDF
fragments in the test split, re-read each from its source file, byte-verified, and
mapped it to the stream it sits in. PDF-syntax fragments: 100% called PDF. JPEG streams:
two thirds called JPG. DEFLATE streams: a third called ZIP. The model is faithful to
the bytes; the file-extension label is what's ambiguous."

## 8. Ablation — 12:15 to 13:00

**Show:** `fig6_ablation_comparison.png`

<!-- V4_ABLATION_SCRIPT -->
**Say:** "Each rung adds one mechanism: augmentation, entropy features, an entropy
adversary, contrastive pre-training. Same data, epochs and seeds, compared with paired
tests against the plain CNN. None of them beats it here either — the entropy adversary
drops accuracy by three points, and its McNemar test is significant in all three seeds,
meaning it isn't noise, it's a consistent, negative effect. Contrastive pre-training
costs seven times the training time for a result that's also worse. Same conclusion as
the cross-corpus study: the plain CNN is the one to deploy."
<!-- /V4_ABLATION_SCRIPT -->

## 9. Limitations and future work — 13:00 to 13:45

**Say:**
- "V4 isn't cross-corpus; extension labels penalise container formats; nine types; three
  seeds; fragments classified independently."
- "Next: container-aware labels, models over neighbouring blocks, holding out a whole
  source to get a balanced cross-corpus test, and calibration that survives a corpus shift."

## 10. Close — 13:45 to 14:00

**Say:** "In short: a reproducible pipeline from raw bytes to an explained verdict, an
honest separation between the same-source and cross-corpus numbers, and a measured,
not assumed, account of the model's main error. Thank you."

---

## If something goes wrong

| problem | fallback |
|---|---|
| GPU error or slow start | `set FRAGGEN_DEVICE=cpu` (Windows) or `export FRAGGEN_DEVICE=cpu`, then re-run; CPU takes about 70 ms for a 12-fragment file |
| Streamlit will not start | use the CLI: `python demo/cli/demo.py tricky_pdf_container_example.pdf --map` |
| Laptop fails entirely | present from `docs/thesis/RESULTS_AND_DISCUSSION.pdf`; its figures match the live output |
| Asked for a number not in this script | `reports/ablations/v4/ABLATION_RESULTS.md`, `reports/tables/`, `docs/viva/EXAMINER_QA.md` |
