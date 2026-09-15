"""Fit a confidence calibration for a trained model, and report whether it holds.

Fits a temperature on a held-out IN-CORPUS split (all an analyst realistically
has), then measures calibration error on both the in-corpus test set and the
cross-corpus set. The second number is the one that matters: a temperature that
only works on the training corpus is no use on evidence from somewhere else.

Writes checkpoints/calibration/<checkpoint stem>.calib.json (override with
--out); src.inference.FragGenClassifier picks it up automatically.

Usage (the command that produced checkpoints/calibration/A_plain_s1337.calib.json):
  python scripts/calibrate.py --model checkpoints/v4/A_plain_s1337.pt \
      --val data/processed_v4/val.npz \
      --in-test data/processed_v4/val.npz \
      --cross data/processed_v4/test.npz
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.checkpoint import load_model
from src.preprocessing.fragments import FragmentSet
from src.utils.calibration import (fit_temperature, apply_temperature,
                                   expected_calibration_error, reliability_table)


@torch.no_grad()
def get_logits(model, X, device, batch=512, limit=0):
    model.eval()
    if limit and len(X) > limit:
        idx = np.random.default_rng(0).choice(len(X), limit, replace=False)
        X = X[np.sort(idx)]
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.int64)).to(device)
        lg, _, _ = model(xb)
        out.append(lg.cpu().numpy())
    return np.concatenate(out), X


def report(name, P, y):
    pred, conf = P.argmax(1), P.max(1)
    correct = (pred == y)
    ece = expected_calibration_error(conf, correct)
    return ece, float(correct.mean()), conf, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val", required=True, help="held-out split to FIT on")
    ap.add_argument("--in-test", required=True)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--out", default=None,
                    help="default: checkpoints/calibration/<model stem>.calib.json")
    ap.add_argument("--limit", type=int, default=40000,
                    help="cap fragments per split (speed)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpath = ROOT / args.model
    model, cfg = load_model(str(mpath), device)

    # ---- fit T on the held-out in-corpus split ----
    val = FragmentSet.load(ROOT / args.val)
    lg_v, Xv = get_logits(model, val.X, device, limit=args.limit)
    yv = val.y[:len(Xv)] if len(Xv) == len(val.X) else None
    if yv is None:  # subsampled -> recompute labels on the same indices
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(val.X), args.limit, replace=False))
        lg_v, _ = get_logits(model, val.X[idx], device)
        yv = val.y[idx]

    T = fit_temperature(lg_v, yv)
    print("=" * 74)
    print(f"  CONFIDENCE CALIBRATION — {mpath.name}")
    print("=" * 74)
    print(f"\nfitted temperature T = {T:.4f}   "
          f"({'softening over-confident scores' if T > 1 else 'sharpening'})")
    print("T is monotonic in the logits, so no prediction changes:")
    print("accuracy, precision and recall are identical before and after.\n")

    results = {}
    for tag, path in [("in-corpus", args.in_test), ("cross-corpus", args.cross)]:
        fs = FragmentSet.load(ROOT / path)
        rng = np.random.default_rng(1)
        idx = (np.sort(rng.choice(len(fs.X), args.limit, replace=False))
               if args.limit and len(fs.X) > args.limit else np.arange(len(fs.X)))
        lg, _ = get_logits(model, fs.X[idx], device)
        y = fs.y[idx]

        P_raw = apply_temperature(lg, 1.0)
        P_cal = apply_temperature(lg, T)
        ece_raw, acc_raw, conf_raw, corr = report(tag, P_raw, y)
        ece_cal, acc_cal, conf_cal, _ = report(tag, P_cal, y)

        print("-" * 74)
        print(f"{tag.upper()}  ({len(y):,} fragments)   accuracy {acc_raw:.4f}")
        print("-" * 74)
        print(f"  ECE before : {ece_raw:.4f}")
        print(f"  ECE after  : {ece_cal:.4f}   "
              f"({(ece_raw-ece_cal)/max(ece_raw,1e-9):+.0%} change)")
        assert abs(acc_raw - acc_cal) < 1e-9, "temperature changed a prediction!"

        print(f"\n  {'stated':>8s} {'actual':>8s} {'n':>8s}   {'stated':>8s} "
              f"{'actual':>8s} {'n':>8s}")
        print(f"  {'--- before ---':^26s}   {'--- after ---':^26s}")
        rb = reliability_table(conf_raw, corr)
        ra = reliability_table(conf_cal, corr)
        for i in range(max(len(rb), len(ra))):
            l = (f"  {rb[i][0]:8.3f} {rb[i][1]:8.3f} {rb[i][2]:8,}" if i < len(rb)
                 else " " * 26)
            r = (f"   {ra[i][0]:8.3f} {ra[i][1]:8.3f} {ra[i][2]:8,}" if i < len(ra)
                 else "")
            print(l + r)

        verdict = ("quotable — confidence may be reported as a probability"
                   if ece_cal < 0.05 else
                   "improved but still overstated — quote empirical accuracy at a threshold"
                   if ece_cal < 0.12 else
                   "still unusable — do not report raw confidence")
        print(f"\n  verdict: {verdict}")
        results[tag] = {"ece_before": ece_raw, "ece_after": ece_cal,
                        "accuracy": acc_raw, "n": int(len(y)), "verdict": verdict}

    out = (ROOT / args.out) if args.out else (ROOT / "checkpoints" / "calibration"
                                              / f"{mpath.stem}.calib.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "temperature": T, "fitted_on": args.val,
        "block_size": int(getattr(cfg, "block_size", 0)), "results": results,
    }, indent=2))
    print("\n" + "=" * 74)
    print(f"  wrote {out}  —  src.inference picks it up automatically")
    print("=" * 74)


if __name__ == "__main__":
    raise SystemExit(main())
