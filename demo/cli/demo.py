#!/usr/bin/env python
"""FragGen V4 command-line demo: classify any file, fragment by fragment.

    python demo/cli/demo.py demo/samples/sample.pdf
    python demo/cli/demo.py sample.pdf                  # bare names resolve to demo/samples/
    python demo/cli/demo.py suspect.bin --map           # adds a byte-offset map
    python demo/cli/demo.py disk.raw --offset 1048576 --length 65536 --keep-header --map

Prints a per-fragment table (Fragment / Prediction / Confidence), the vote
breakdown, the final prediction and plain-language notes, then writes a JSON
and a CSV report to demo/outputs/cli/.

This file only formats results. Every model step (fragmenting, CNN, calibration,
voting) happens in src.inference.FragGenClassifier, the same code the
Streamlit app and the viva script call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explainability import (dominant_region, explain_verdict,  # noqa: E402
                                fragment_region_shares, is_pdf, pdf_stream_regions,
                                save_csv, save_json)
from src.inference import DEFAULT_CHECKPOINT, FragGenClassifier  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "demo" / "samples"
DEFAULT_OUT = REPO_ROOT / "demo" / "outputs" / "cli"


def resolve_input(arg: str) -> Optional[Path]:
    """Accept a real path, or a bare bundled-sample name such as 'sample.pdf'."""
    p = Path(arg)
    if p.exists():
        return p
    candidate = SAMPLES_DIR / p.name
    return candidate if candidate.exists() else None


def pct(p: float) -> str:
    """0.991 -> '99.1%'."""
    return f"{p * 100:.1f}%"


def pdf_region_shares(path: Path, verdict, offset: int):
    """For a whole PDF file, the share of each fragment inside FlateDecode /
    DCTDecode / other streams. None for anything else (explanation only)."""
    if offset != 0:
        return None
    data = path.read_bytes()
    if not is_pdf(data):
        return None
    return fragment_region_shares(pdf_stream_regions(data), verdict.start_offset,
                                  verdict.block_size, verdict.n_fragments)


def print_fragment_table(verdict, shares=None) -> None:
    """The requested Fragment / Prediction / Confidence table. For PDFs a
    fourth column names the PDF stream type the fragment mostly lies in."""
    head = f"{'Fragment':<10} {'Prediction':<12} {'Confidence':<11}"
    if shares is not None:
        head += "Inside PDF region"
    print("\n" + head)
    print("-" * max(34, len(head)))
    for i, frag in enumerate(verdict.fragments):
        line = f"{frag.index:<10} {frag.predicted_class.upper():<12} {pct(frag.confidence):<11}"
        if shares is not None:
            region = dominant_region(shares[i])
            line += f"{region} ({shares[i][region]:.0%})"
        print(line)
    print("-" * max(34, len(head)))


def print_offset_map(verdict) -> None:
    """--map: where in the input each call was made, with a confidence bar."""
    print(f"\n{'Offset':<14}{'Fragment':<10}{'Prediction':<12}{'Confidence':<11}")
    for frag in verdict.fragments:
        bar = "#" * int(round(frag.confidence * 20))
        print(f"0x{frag.byte_offset:010x}  {frag.index:<10}{frag.predicted_class.upper():<12}"
              f"{pct(frag.confidence):<11}{bar}")


def print_summary(verdict) -> None:
    """File facts, vote breakdown, final prediction and notes."""
    print(f"\nFile               : {verdict.file_path or verdict.name}")
    print(f"Bytes examined     : {verdict.file_size_bytes:,}")
    print(f"Block size         : {verdict.block_size} bytes "
          f"(first block {'skipped' if verdict.header_skipped else 'kept'})")
    print(f"Fragments scored   : {verdict.n_fragments}")
    print(f"Model              : {verdict.checkpoint}")
    print("Confidence         : " + (f"calibrated (temperature T={verdict.temperature:.4f})"
                                     if verdict.calibrated else "raw softmax (uncalibrated)"))

    print("\nVote breakdown:")
    total = max(verdict.n_fragments, 1)
    for cls, n in sorted(verdict.vote_counts.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"  {cls.upper():<5} {n:>4} / {total:<4} {'#' * int(round(30 * n / total))}")

    print(f"\n>>> Final prediction: {verdict.final_prediction.upper()}  "
          f"({pct(verdict.final_confidence)} of fragments agree, mean confidence "
          f"{pct(verdict.mean_winner_confidence)})")
    notes = explain_verdict(verdict)
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="FragGen V4: classify a file's 4096-byte fragments and vote on its type.")
    ap.add_argument("file", help="any file (or a bundled sample name such as sample.pdf)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                    help="model checkpoint (default: checkpoints/v4/A_plain_s1337.pt)")
    ap.add_argument("--calibration", default="auto",
                    help="calibration JSON; 'auto' = checkpoints/calibration/<stem>.calib.json")
    ap.add_argument("--no-calibration", action="store_true", help="report raw softmax")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="where JSON/CSV reports go")
    ap.add_argument("--keep-header", action="store_true",
                    help="also score the first block (off by default, as in evaluation)")
    ap.add_argument("--offset", type=int, default=0, help="start byte (disk images)")
    ap.add_argument("--length", type=int, default=0, help="bytes to read (0 = to the end)")
    ap.add_argument("--max-fragments", type=int, default=0, help="cap fragments scored (0 = all)")
    ap.add_argument("--map", action="store_true", help="also print a byte-offset map")
    ap.add_argument("--no-save", action="store_true", help="print only, write no reports")
    args = ap.parse_args()

    path = resolve_input(args.file)
    if path is None:
        print(f"error: file not found: {args.file}")
        return 1

    clf = FragGenClassifier(checkpoint_path=args.checkpoint,
                            calibration_path=None if args.no_calibration else args.calibration,
                            device=args.device)
    print(f"Model ready on {clf.device}  |  classes: {', '.join(clf.classes)}")

    verdict = clf.classify_file(path, skip_header=not args.keep_header,
                                max_fragments=args.max_fragments or None,
                                offset=args.offset, length=args.length)
    if verdict.too_small:
        print(f"\n{path.name}: {verdict.file_size_bytes:,} bytes leaves no complete "
              f"{verdict.block_size}-byte fragment after the header block; nothing to classify.")
        return 1

    print_fragment_table(verdict, pdf_region_shares(path, verdict, args.offset))
    if args.map:
        print_offset_map(verdict)
    print_summary(verdict)

    if not args.no_save:
        out_dir = Path(args.out_dir)
        json_path = save_json(verdict, out_dir / f"{path.name}.json")
        csv_path = save_csv(verdict, out_dir / f"{path.name}.csv")
        print(f"\nSaved: {json_path}\nSaved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
