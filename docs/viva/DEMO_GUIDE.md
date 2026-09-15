# FragGen V4 — Demo Guide

How to run, present and troubleshoot the live demonstration. For the spoken
walkthrough see [`VIVA_SCRIPT.md`](VIVA_SCRIPT.md); for likely questions see
[`EXAMINER_QA.md`](EXAMINER_QA.md).

> **Scope, in one sentence.** The demo runs the Dataset V4 plain byte-CNN
> (`checkpoints/v4/A_plain_s1337.pt`). V4 test accuracy is **0.8938 ± 0.0036**
> (3 seeds) on *unseen files from the training source pool*. The project's
> cross-corpus result, **0.6812 ± 0.0141** (train FFT-75 → test GovDocs1, 4096 B,
> 2 seeds), answers a different and harder question and is not replaced by
> anything here. Sources: `reports/ablations/v4/` and
> `reports/ablations/cross_corpus_fft75_govdocs1/blocksize_4096/`.

---

## 1. Components

| component | command (from the repository root) | shows |
|---|---|---|
| CLI | `python demo/cli/demo.py <file>` | per-fragment table, vote breakdown, final prediction, notes, JSON + CSV report |
| Viva script | `python demo/cli/viva_demo.py` | narrated six-step run on PDF, PNG, CSV and ZIP, then the tricky PDF with its stream-level explanation |
| Streamlit app | `streamlit run demo/streamlit/app.py` | upload or sample picker, metadata, vote chart, confidence chart, fragment timeline (plus a PDF stream map for PDFs), table, JSON/CSV download |
| Figures | `python scripts/make_publication_figures.py` | the 8 publication figures in `reports/figures/` |

All four import the same inference code, `src/inference/classifier.py`. No
component re-implements fragmenting, calibration or voting, so they cannot
disagree about a file.

## 2. Setup

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -t .        # 26 tests; confirms the checkpoints load
```

The demo needs only `checkpoints/v4/`, `checkpoints/calibration/` and
`demo/samples/`; the raw corpora are not needed. A GPU is optional. To force CPU,
for example while a GPU is busy training, set `FRAGGEN_DEVICE=cpu` (or pass
`--device cpu` to the CLI or viva script).

## 3. Running each component

### CLI

```bash
python demo/cli/demo.py sample.pdf                       # bare sample names resolve to demo/samples/
python demo/cli/demo.py demo/samples/sample.zip --map    # adds a byte-offset map
python demo/cli/demo.py tricky_pdf_container_example.pdf # PDF whose vote flips to ZIP
python demo/cli/demo.py disk.img --offset 1048576 --length 65536 --keep-header --map
```

Reports are written to `demo/outputs/cli/<file>.json` and `<file>.csv`
(`--no-save` turns this off).

### Viva script

```bash
python demo/cli/viva_demo.py                 # PDF, PNG, CSV, ZIP, then the tricky PDF (press Enter to advance)
python demo/cli/viva_demo.py --no-pause      # unattended, e.g. a rehearsal recording
python demo/cli/viva_demo.py --all           # all nine classes first
python demo/cli/viva_demo.py --with-xml-case # also XML read as HTML
```

The opening banner reads its accuracy figures from the report files at run time.

### Streamlit

```bash
streamlit run demo/streamlit/app.py
```

Run it from the repository root so `.streamlit/config.toml` (the dark theme) is
picked up, then open the printed URL (default `http://localhost:8501`). Uploads
are classified in memory and nothing is written to disk.

## 4. Suggested presentation order (about 10 minutes)

1. **Frame the problem (1 min).** Show `reports/figures/fig8_pipeline_architecture.png`:
   4096 raw bytes from the middle of a file, no header, 9 candidate types.
2. **Viva script, correct cases (3 min).** Run `python demo/cli/viva_demo.py` and walk
   through the six steps for PDF, then move quickly through PNG, CSV and ZIP.
3. **Viva script, tricky PDF (2 min).** Let the script reach the "WHY DID A PDF VOTE
   ZIP?" section. The stream table shows each ZIP call sitting inside FlateDecode
   (DEFLATE) data.
4. **Streamlit, live upload (2 min).** Ask the examiner for a file, or pick
   `tricky_pdf_container_example.pdf`, and point at the timeline's PDF-region strip.
5. **Results (2 min).** `fig1_confusion_matrix.png`, `fig6_ablation_comparison.png` and
   `fig7_pdf_container_analysis.png` (panel b shows the same effect across every test PDF).

Close with the scope sentence at the top of this guide.

## 5. Expected outputs

### CLI on `sample.pdf` (actual output, abridged)

```
Fragment   Prediction   Confidence Inside PDF region
----------------------------------------------------
1          ZIP          42.3%      FlateDecode (100%)
2          PDF          99.9%      FlateDecode (66%)
...
12         ZIP          54.1%      FlateDecode (100%)

Vote breakdown:
  PDF      9 / 12   ######################
  ZIP      3 / 12   ########

>>> Final prediction: PDF  (75.0% of fragments agree, mean confidence 92.7%)
```

The fourth column appears only for PDFs. It is computed from the file's own
stream objects (`src/explainability/container_analysis.py`), not by the model.

### Bundled samples

| file | outcome |
|---|---|
| `sample.{pdf,jpg,png,gif,txt,html,xml,csv,zip}` | each classified as its true type (checked by `tests/test_checkpoints.py`) |
| `tricky_pdf_container_example.pdf` | vote goes to **ZIP**: a known and explained limitation |
| `tricky_xml_html_boundary_example.xml` | read as **HTML**: a known and explained limitation |

These samples were chosen by hand to illustrate behaviour. They are **not** an
accuracy estimate; accuracy comes from the 13,754-fragment test split.

### JSON report (`src/explainability/report.py`)

```json
{
  "schema": "fraggen.file_report/v1",
  "file": {"name": "sample.pdf", "path": "...", "size_bytes": 55964},
  "model": {"checkpoint": "checkpoints/v4/A_plain_s1337.pt", "classes": ["txt", "..."],
            "block_size": 4096, "calibrated_confidence": true, "temperature": 1.0214},
  "fragmentation": {"header_skipped": true, "first_fragment_offset": 4096, "n_fragments": 12},
  "verdict": {"final_prediction": "pdf", "vote_share": 0.75,
              "mean_confidence_of_winning_fragments": 0.927, "vote_counts": {"pdf": 9, "zip": 3}},
  "notes": ["3 of 12 fragments were classified ZIP or JPG next to 9 PDF fragment(s). ..."],
  "fragments": [
    {"fragment_index": 1, "byte_offset": 4096, "predicted_class": "zip",
     "confidence": 0.423, "probability_vector": {"txt": 0.0, "...": 0.0, "zip": 0.423}}
  ]
}
```

(`...` marks values elided here; `vote_counts` in a real report lists all nine classes.)

### CSV report

One row per fragment:
`fragment_index, byte_offset, predicted_class, confidence, probability_vector` (a JSON
object), followed by `p_txt … p_zip` columns for spreadsheets.

## 6. Troubleshooting

| symptom | fix |
|---|---|
| `CUDA out of memory` or a slow start while training runs | `FRAGGEN_DEVICE=cpu` |
| "no complete 4096-byte fragment after the header block" | the input is under 8,192 bytes; the model never scores a header block, so there is nothing to classify |
| Streamlit shows a light theme | launch from the repository root (where `.streamlit/` lives) |
| `ModuleNotFoundError: src` | run the commands from the repository root |
| Port 8501 already in use | `streamlit run demo/streamlit/app.py --server.port 8502` |
