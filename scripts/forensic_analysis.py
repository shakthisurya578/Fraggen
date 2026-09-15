"""Forensic-analyst reading of the classifier's scores.

Accuracy answers "how often is the model right". An investigator needs different
questions answered:

  * If the tool reports JPG, how often is that true?          -> precision
  * Can I trust the confidence value it prints?               -> calibration
  * What threshold do I set to work at 95% reliability, and
    how much of the disk do I still get to identify?          -> operating points
  * Which file types is this tool competent to testify about,
    and which should I refuse to draw conclusions on?         -> reliability tiers

Outputs a table an analyst could actually work from.

Usage:
  python scripts/forensic_analysis.py \
      --model checkpoints/v4/A_plain_s1337.pt \
      --data data/processed_v4/test.npz
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.models.checkpoint import load_model
from src.preprocessing.fragments import FragmentSet

CONTAINERS = {"pdf", "zip"}


@torch.no_grad()
def probs(model, X, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.int64)).to(device)
        logits, _, _ = model(xb)
        out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/v4/A_plain_s1337.pt")
    ap.add_argument("--data", default="data/processed_v4/test.npz")
    ap.add_argument("--out", default=None, help="write JSON summary here")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(str(ROOT / args.model), device)
    fs = FragmentSet.load(ROOT / args.data)
    classes = [str(c) for c in fs.classes]
    P = probs(model, fs.X, device)
    pred, conf, y = P.argmax(1), P.max(1), fs.y

    print("=" * 78)
    print(f"  FORENSIC RELIABILITY REPORT")
    print(f"  model {Path(args.model).name}   evidence {Path(args.data).name}")
    print(f"  {len(fs):,} fragments of {fs.block_size} bytes, corpus '{fs.corpus}'")
    print("=" * 78)

    overall = float((pred == y).mean())
    print(f"\noverall accuracy: {overall:.4f}   "
          f"(random guessing would be {1/len(classes):.3f})")

    # ---------- 1. precision: the question an analyst actually asks ----------
    print("\n" + "-" * 78)
    print("1. IF THE TOOL SAYS X, HOW OFTEN IS IT X?  (precision)")
    print("-" * 78)
    print(f"  {'reported':>8s} {'precision':>10s} {'recall':>8s} {'n called':>9s}   interpretation")
    rows = []
    for c, name in enumerate(classes):
        called = pred == c
        actual = y == c
        prec = float((y[called] == c).mean()) if called.any() else float("nan")
        rec = float((pred[actual] == c).mean()) if actual.any() else float("nan")
        rows.append((name, prec, rec, int(called.sum()), int(actual.sum())))
    for name, prec, rec, ncall, nact in sorted(rows, key=lambda r: -(r[1] if r[1] == r[1] else -1)):
        if prec != prec:
            verdict = "never reported"
        elif prec >= 0.85:
            verdict = "RELIABLE - can support a finding"
        elif prec >= 0.65:
            verdict = "indicative - corroborate before relying"
        elif prec >= 0.40:
            verdict = "WEAK - treat as a lead only"
        else:
            verdict = "UNRELIABLE - do not report this class"
        print(f"  {name:>8s} {prec:10.3f} {rec:8.3f} {ncall:9,}   {verdict}")

    # ---------- 2. is the confidence value trustworthy? ----------
    print("\n" + "-" * 78)
    print("2. IS THE CONFIDENCE VALUE TRUSTWORTHY?  (calibration)")
    print("-" * 78)
    print("  When the tool prints a confidence, an analyst needs the empirical")
    print("  accuracy at that confidence to match it. Otherwise the number is")
    print("  decoration and cannot be quoted.\n")
    print(f"  {'stated conf':>13s} {'actual acc':>11s} {'n':>9s}   {'gap':>7s}")
    edges = [0.0, .3, .4, .5, .6, .7, .8, .9, .95, 1.001]
    ece, tot = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 30:
            continue
        stated, actual = float(conf[m].mean()), float((pred[m] == y[m]).mean())
        ece += m.sum() / tot * abs(stated - actual)
        flag = "  <- overconfident" if stated - actual > 0.10 else ""
        print(f"  {stated:13.3f} {actual:11.3f} {int(m.sum()):9,}   {stated-actual:+7.3f}{flag}")
    print(f"\n  expected calibration error (ECE): {ece:.4f}")
    print("  " + ("well calibrated - confidence can be quoted" if ece < 0.05 else
                  "MISCALIBRATED - the printed confidence overstates reliability;" if ece < 0.15 else
                  "SEVERELY MISCALIBRATED - do not quote raw confidence"))
    if ece >= 0.05:
        print("  Report a threshold-based reliability instead (section 3).")

    # ---------- 3. operating points ----------
    print("\n" + "-" * 78)
    print("3. OPERATING POINTS  (abstain to buy reliability)")
    print("-" * 78)
    print("  A carver need not answer on every block. Raising the confidence")
    print("  threshold discards ambiguous blocks and raises accuracy on the rest.\n")
    print(f"  {'threshold':>10s} {'coverage':>9s} {'accuracy':>9s}   what it means")
    for thr in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        keep = conf >= thr
        if keep.sum() < 50:
            continue
        cov, acc = float(keep.mean()), float((pred[keep] == y[keep]).mean())
        note = ""
        if acc >= 0.95:
            note = "court-grade, but see coverage"
        elif acc >= 0.90:
            note = "high confidence triage"
        elif acc >= 0.80:
            note = "usable for prioritisation"
        print(f"  {thr:10.2f} {cov:9.1%} {acc:9.3f}   {note}")

    # target-reliability lookup: the number an analyst wants
    print()
    for target in [0.90, 0.95, 0.99]:
        order = np.argsort(-conf)
        correct = (pred == y)[order]
        run = np.cumsum(correct) / np.arange(1, len(correct) + 1)
        ok = np.flatnonzero(run >= target)
        if len(ok) == 0:
            print(f"  {target:.0%} reliability: NOT ACHIEVABLE at any threshold")
            continue
        k = ok[-1]
        print(f"  {target:.0%} reliability -> keep the top {k+1:,} blocks "
              f"({(k+1)/len(conf):.1%} of the disk), threshold {conf[order][k]:.3f}")

    # ---------- 4. container caveat ----------
    print("\n" + "-" * 78)
    print("4. WHAT COUNTS AS AN ERROR  (container formats)")
    print("-" * 78)
    idx = {c: i for i, c in enumerate(classes)}
    cont = [idx[c] for c in CONTAINERS if c in idx]
    if cont:
        m = np.isin(y, cont)
        strict = float((pred[m] == y[m]).mean())
        print(f"  pdf/zip fragments: {int(m.sum()):,}  strict accuracy {strict:.3f}")
        print("  A PDF's interior is DEFLATE streams and embedded JPEG images, so a")
        print("  block from inside a PDF that is called 'jpg' is not necessarily a")
        print("  mistake - it may be correctly identifying an embedded object.")
        print("  Forensically this is a FINDING, not an error: it says the container")
        print("  holds image data at that offset, which is what a carver wants to know.")
        non = ~m
        print(f"\n  accuracy on non-container fragments: "
              f"{float((pred[non] == y[non]).mean()):.3f}  ({int(non.sum()):,} fragments)")

    # ---------- 5. bottom line ----------
    print("\n" + "=" * 78)
    print("  ANALYST'S BOTTOM LINE")
    print("=" * 78)
    reliable = [n for n, p, *_ in rows if p == p and p >= 0.85]
    weak = [n for n, p, *_ in rows if p == p and p < 0.40]
    print(f"  Report findings on : {', '.join(reliable) if reliable else '(none)'}")
    print(f"  Refuse to report on: {', '.join(weak) if weak else '(none)'}")
    print(f"  Never quote raw confidence as a probability unless ECE < 0.05"
          f"  (here {ece:.3f}).")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "model": args.model, "data": args.data, "overall_accuracy": overall,
            "ece": ece,
            "per_class": [{"class": n, "precision": p, "recall": r,
                           "n_called": nc, "n_actual": na} for n, p, r, nc, na in rows],
        }, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
