"""Explainable output for a classified file: JSON, CSV and plain-language notes.

Every consumer (CLI, Streamlit, viva script) exports through these functions,
so a report downloaded from the web app is byte-for-byte the same structure as
one written by ``demo/cli/demo.py``.

Per-fragment fields (identical in JSON and CSV):
    fragment_index      1-based position among the fragments that were scored
    byte_offset         offset of the fragment inside the original file
    predicted_class     argmax of the probability vector
    confidence          probability of the predicted class (calibrated if a
                        temperature was available -- see ``calibrated``)
    probability_vector  the full distribution over the model's classes
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Dict, List

REPORT_SCHEMA = "fraggen.file_report/v1"
FRAGMENT_FIELDS = ["fragment_index", "byte_offset", "predicted_class", "confidence",
                   "probability_vector"]

# pdf and zip are containers whose interiors are DEFLATE streams and embedded
# JPEG data; html/xml share markup syntax. Used by explain_verdict().
_CONTAINER_LOOKALIKES = {"pdf": ("zip", "jpg")}
_MARKUP_PAIR = {"xml": "html", "html": "xml"}


def fragment_records(verdict) -> List[Dict]:
    """One dict per fragment with exactly the documented FRAGMENT_FIELDS."""
    return [{
        "fragment_index": f.index,
        "byte_offset": f.byte_offset,
        "predicted_class": f.predicted_class,
        "confidence": f.confidence,
        "probability_vector": dict(zip(verdict.classes, f.probabilities)),
    } for f in verdict.fragments]


def verdict_to_dict(verdict) -> Dict:
    """Full JSON-serialisable report: file metadata, model/calibration
    provenance, the vote, plain-language notes, and every fragment."""
    return {
        "schema": REPORT_SCHEMA,
        "file": {
            "name": verdict.name,
            "path": verdict.file_path,
            "size_bytes": verdict.file_size_bytes,
        },
        "model": {
            "checkpoint": verdict.checkpoint,
            "classes": verdict.classes,
            "block_size": verdict.block_size,
            "calibrated_confidence": verdict.calibrated,
            "temperature": verdict.temperature,
        },
        "fragmentation": {
            "header_skipped": verdict.header_skipped,
            "first_fragment_offset": verdict.start_offset,
            "n_fragments": verdict.n_fragments,
        },
        "verdict": {
            "final_prediction": verdict.final_prediction,
            "vote_share": verdict.final_confidence,
            "mean_confidence_of_winning_fragments": verdict.mean_winner_confidence,
            "vote_counts": verdict.vote_counts,
        },
        "notes": explain_verdict(verdict),
        "fragments": fragment_records(verdict),
    }


def verdict_to_json(verdict) -> str:
    """The JSON report as a string (used for in-browser downloads)."""
    return json.dumps(verdict_to_dict(verdict), indent=2)


def save_json(verdict, out_path) -> Path:
    """Write the JSON report, creating parent folders as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(verdict_to_json(verdict), encoding="utf-8")
    return out_path


def verdict_to_csv(verdict) -> str:
    """One row per fragment. ``probability_vector`` is kept as a JSON object so
    the documented field survives intact, and is also spread into one
    ``p_<class>`` column per class for spreadsheet use."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(FRAGMENT_FIELDS + [f"p_{c}" for c in verdict.classes])
    for rec, frag in zip(fragment_records(verdict), verdict.fragments):
        w.writerow([rec["fragment_index"], rec["byte_offset"], rec["predicted_class"],
                    f"{rec['confidence']:.6f}",
                    json.dumps({k: round(v, 6) for k, v in rec["probability_vector"].items()})]
                   + [f"{p:.6f}" for p in frag.probabilities])
    return buf.getvalue()


def save_csv(verdict, out_path) -> Path:
    """Write the CSV report, creating parent folders as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(verdict_to_csv(verdict), encoding="utf-8")
    return out_path


def explain_verdict(verdict) -> List[str]:
    """Plain-language notes an examiner should read alongside the verdict.

    Rules are deliberately simple and only describe what the vote shows; they
    never change the prediction.
    """
    notes: List[str] = []
    n = verdict.n_fragments
    if n == 0:
        return [f"File has no complete {verdict.block_size}-byte fragment after the "
                f"header block, so no prediction is made."]
    votes = verdict.vote_counts
    winner = verdict.final_prediction

    if verdict.final_confidence < 0.5:
        notes.append(f"Low agreement: only {verdict.final_confidence:.0%} of fragments voted "
                     f"{winner.upper()}. Treat the file-level verdict as uncertain.")

    pdf_like = votes.get("pdf", 0)
    lookalike = sum(votes.get(c, 0) for c in _CONTAINER_LOOKALIKES["pdf"])
    if pdf_like > 0 and lookalike / n > 0.15:
        notes.append(
            f"{lookalike} of {n} fragments were classified ZIP or JPG next to {pdf_like} PDF "
            f"fragment(s). PDF compresses its streams with FlateDecode (the same DEFLATE "
            f"algorithm ZIP uses) and embeds JPEG images verbatim, so a fragment inside such "
            f"a stream is genuinely zip- or jpg-shaped at the byte level. This is an "
            f"understood property of the format, not a random error.")

    if winner in _MARKUP_PAIR and votes.get(_MARKUP_PAIR[winner], 0) > 0:
        other = _MARKUP_PAIR[winner]
        notes.append(f"{votes[other]} fragment(s) read as {other.upper()}: HTML and XML share "
                     f"angle-bracket syntax, and the declaration that separates them sits in "
                     f"the excluded header block.")

    if not verdict.calibrated:
        notes.append("Confidence is raw softmax (no calibration file found); read it as a "
                     "ranking, not a probability.")
    return notes
