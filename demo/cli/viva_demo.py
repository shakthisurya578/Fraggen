#!/usr/bin/env python
"""FragGen V4 viva demonstration: a narrated, step-by-step live run.

    python demo/cli/viva_demo.py                  # PDF, PNG, CSV, ZIP, then the tricky PDF
    python demo/cli/viva_demo.py --no-pause       # no Enter prompts (recording / CI)
    python demo/cli/viva_demo.py --all            # all nine classes first
    python demo/cli/viva_demo.py --with-xml-case  # also the XML->HTML boundary example
    python demo/cli/viva_demo.py --skip-tricky    # correct cases only

Every prediction comes from src.inference.FragGenClassifier.classify_bytes, the
same call the CLI and the Streamlit app make. Numbers quoted in the opening
banner are read from the result files by src.utils.results. The PDF->ZIP
explanation is computed from the PDF's own stream objects by
src.explainability.container_analysis.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explainability import (dominant_region, explain_verdict,  # noqa: E402
                                fragment_region_shares, pdf_stream_regions, save_json)
from src.inference import FragGenClassifier  # noqa: E402
from src.utils import results  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "demo" / "samples"
OUT_DIR = REPO_ROOT / "demo" / "outputs" / "viva"
SAMPLES = {c: f"sample.{c}" for c in ["pdf", "png", "csv", "zip", "jpg", "gif", "html", "xml", "txt"]}
SPEC_ORDER = ["pdf", "png", "csv", "zip"]
TRICKY_PDF = "tricky_pdf_container_example.pdf"
TRICKY_XML = "tricky_xml_html_boundary_example.xml"
RULE = "=" * 76


def pause(enabled: bool) -> None:
    """Wait for Enter so the presenter controls the pace (no-op with --no-pause)."""
    if not enabled:
        return
    try:
        input("\n    [Enter] ")
    except EOFError:
        pass


def banner() -> None:
    """Opening statement, with the two headline numbers read from report files."""
    v4 = results.v4_metric("A_plain", "test", "accuracy")
    cc = results.cross_corpus_4096("A_plain")
    print(RULE)
    print("  FragGen V4 -- file-fragment type classification -- viva demonstration")
    print(RULE)
    print("""
  Question: given 4096 raw bytes from the MIDDLE of an unknown file (no name,
  no header, no filesystem), which of 9 types did they come from?

  Pipeline: bytes -> skip header block -> 4096-byte fragments -> ByteCNN
            -> calibrated probabilities per fragment -> majority vote
""")
    if v4:
        print(f"  Dataset V4 test accuracy (unseen FILES, same source pool): "
              f"{v4.fmt()}  over {v4.n} seeds")
    if cc:
        print(f"  Cross-corpus accuracy (train FFT-75 -> test GovDocs1, 4096 B): "
              f"{cc.fmt()}  over {cc.n} seeds")
    print("  These answer different questions and are never compared directly.")


def present(clf: FragGenClassifier, label: str, filename: str, expected: str,
            interactive: bool, save: bool):
    """Narrate the six pipeline steps for one file. Returns (verdict, bytes)."""
    path = SAMPLES_DIR / filename
    print(f"\n{RULE}\n  {label}\n{RULE}")
    if not path.exists():
        print(f"  sample missing: {path}")
        return None, b""

    data = path.read_bytes()
    print(f"\n[1/6] Read {path.name}: {len(data):,} bytes")

    fragments, start = clf.fragment_bytes(data)
    print(f"[2/6] Skip the first {clf.block_size} bytes (header / magic signature), split the rest"
          f"\n      -> {len(fragments)} fragments, first one at offset {start}")

    t0 = time.perf_counter()
    verdict = clf.classify_bytes(data, name=path.name, file_path=path)
    ms = (time.perf_counter() - t0) * 1000
    print(f"[3/6] ByteCNN scored all {verdict.n_fragments} fragments in {ms:.0f} ms on {clf.device}")
    print(f"[4/6] One probability vector per fragment "
          f"({'calibrated, T=' + format(clf.temperature, '.4f') if clf.calibrated else 'raw softmax'}):")

    rows = verdict.fragments
    shown = rows if len(rows) <= 16 else rows[:12]
    print(f"\n      {'Fragment':<10}{'Offset':<12}{'Prediction':<12}{'Confidence'}")
    for f in shown:
        print(f"      {f.index:<10}{f.byte_offset:<12}{f.predicted_class.upper():<12}{f.confidence:.1%}")
    if len(shown) < len(rows):
        print(f"      ... {len(rows) - len(shown)} more (full list in the JSON report)")

    print("\n[5/6] Majority vote:")
    for cls, n in sorted(verdict.vote_counts.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"      {cls.upper():<5} {n:>3}  {'#' * n}")

    ok = verdict.final_prediction == expected
    print(f"\n[6/6] >>> FINAL PREDICTION: {verdict.final_prediction.upper()}   "
          f"({verdict.final_confidence:.0%} of fragments agree)   "
          f"true type: {expected.upper()}   {'CORRECT' if ok else 'MISMATCH'}")
    for note in explain_verdict(verdict):
        print(f"      note: {note}")

    if save:
        out = save_json(verdict, OUT_DIR / f"{expected}_{path.name}.json")
        print(f"      report: {out.relative_to(REPO_ROOT).as_posix()}")
    pause(interactive)
    return verdict, data


def explain_pdf_container(verdict, data: bytes) -> None:
    """Show, fragment by fragment, which PDF stream type each prediction sits in.
    Everything printed is computed from this file's bytes."""
    regions = pdf_stream_regions(data)
    by_type = Counter(r.label for r in regions)
    shares = fragment_region_shares(regions, verdict.start_offset, verdict.block_size,
                                    verdict.n_fragments)
    print(f"\n{RULE}\n  WHY DID A PDF VOTE ZIP?  (parsed from the PDF's own stream objects)\n{RULE}")
    print(f"""
  A PDF is a container. Its bytes mostly sit inside 'stream ... endstream'
  objects, each with a /Filter:
    /FlateDecode = zlib/DEFLATE, the same compression inside a ZIP archive
    /DCTDecode   = a JPEG image stored verbatim

  This file has {len(regions)} streams: {by_type.get('FlateDecode', 0)} FlateDecode, {by_type.get('DCTDecode', 0)} DCTDecode, {by_type.get('other_stream', 0)} other.
""")
    print(f"  {'Frag':<6}{'Offset':<10}{'Prediction':<12}{'Conf':<8}{'Flate':>7}{'JPEG':>7}{'outside':>9}")
    for f, s in zip(verdict.fragments, shares):
        print(f"  {f.index:<6}{f.byte_offset:<10}{f.predicted_class.upper():<12}{f.confidence:<8.0%}"
              f"{s['FlateDecode']:>7.0%}{s['DCTDecode']:>7.0%}{s['outside_streams']:>9.0%}")

    table = Counter((dominant_region(s), f.predicted_class) for f, s in zip(verdict.fragments, shares))
    print("\n  Where each prediction lands (fragment counts):")
    for region in ["FlateDecode", "DCTDecode", "other_stream", "outside_streams"]:
        row = {c: n for (r, c), n in table.items() if r == region}
        if row:
            parts = ", ".join(f"{c.upper()} {n}" for c, n in sorted(row.items(), key=lambda kv: -kv[1]))
            print(f"    mostly {region:<16} -> {parts}")
    zip_frags = [s for f, s in zip(verdict.fragments, shares) if f.predicted_class == "zip"]
    if zip_frags:
        in_flate = sum(1 for s in zip_frags if dominant_region(s) == "FlateDecode")
        print(f"\n  {in_flate} of the {len(zip_frags)} fragments called ZIP lie mostly inside "
              f"FlateDecode streams: at the byte level they ARE deflate data.")
    print("""
  Takeaway for the examiner: the file-level vote can be fooled by a PDF that is
  mostly compressed payload, but each fragment-level call is faithful to the
  bytes it saw. The per-fragment trace plus this stream map makes that visible.""")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="all nine classes before the tricky case")
    ap.add_argument("--classes", nargs="*", help="explicit class list, e.g. pdf jpg")
    ap.add_argument("--skip-tricky", action="store_true", help="do not show the tricky PDF")
    ap.add_argument("--with-xml-case", action="store_true", help="also show XML read as HTML")
    ap.add_argument("--no-pause", action="store_true", help="do not wait for Enter")
    ap.add_argument("--no-save", action="store_true", help="do not write JSON reports")
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    args = ap.parse_args()
    interactive, save = not args.no_pause, not args.no_save

    wanted = list(SAMPLES) if args.all else [c.lower() for c in (args.classes or SPEC_ORDER)]
    banner()
    pause(interactive)

    print("\nLoading checkpoints/v4/A_plain_s1337.pt ...")
    clf = FragGenClassifier(device=args.device)
    print(f"Loaded on {clf.device}; classes = {', '.join(clf.classes)}")

    scoreboard = []
    for cls in wanted:
        if cls not in SAMPLES:
            print(f"(unknown class '{cls}' skipped)")
            continue
        v, _ = present(clf, f"DEMONSTRATION: {cls.upper()}", SAMPLES[cls], cls, interactive, save)
        if v is not None:
            scoreboard.append((cls, v.final_prediction))

    if not args.skip_tricky:
        v, data = present(clf, "KNOWN LIMITATION: a PDF dominated by compressed streams",
                          TRICKY_PDF, "pdf", interactive, save)
        if v is not None:
            explain_pdf_container(v, data)
            pause(interactive)
    if args.with_xml_case:
        present(clf, "KNOWN LIMITATION: XML near the HTML boundary", TRICKY_XML, "xml",
                interactive, save)

    print(f"\n{RULE}\n  SUMMARY (bundled, pre-selected samples: an illustration, not an accuracy estimate)\n{RULE}")
    for cls, pred in scoreboard:
        print(f"  {cls.upper():<5} -> {pred.upper():<5} {'CORRECT' if cls == pred else 'MISMATCH'}")
    n_ok = sum(c == p for c, p in scoreboard)
    print(f"\n  {n_ok}/{len(scoreboard)} correct. Measured accuracy lives in reports/ablations/v4/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
