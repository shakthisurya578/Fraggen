"""Classify the fragments of an arbitrary file or raw disk image.

This is FragGen actually being *used*: no labels, no npz, just bytes in and
predicted types out. Splits the input into block_size chunks, runs the trained
model over them, and reports what each region looks like -- the carving view.

  # what type is this file, judged from its middle bytes?
  python scripts/classify.py --model checkpoints/E_full.pt --file suspect.bin

  # per-block map across a raw image, abstaining when unsure
  python scripts/classify.py --model checkpoints/E_full.pt --file disk.raw \
      --map --min-conf 0.6

  # only look at one region
  python scripts/classify.py --model checkpoints/E_full.pt --file disk.raw \
      --offset 1048576 --length 65536 --map
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fraggen.eval import load_model


def fragments_from_bytes(data: bytes, block: int, skip_header: bool):
    """Non-overlapping blocks. Optionally drop block 0, which carries the magic
    signature -- the model is trained on middle-of-file fragments."""
    start = block if (skip_header and len(data) > 2 * block) else 0
    n = (len(data) - start) // block
    if n <= 0:
        return np.empty((0, block), dtype=np.uint8), start
    buf = np.frombuffer(data, dtype=np.uint8, count=n * block, offset=start)
    return buf.reshape(n, block), start


def load_calibration(model_path):
    """Temperature written by scripts/calibrate.py, if it exists."""
    p = Path(model_path).with_suffix(".calib.json")
    if not p.exists():
        return None
    # narrow: a malformed file should be reported, not silently downgraded to
    # "uncalibrated" -- a broad except here once hid a missing json import and
    # made a calibrated model look uncalibrated.
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"warning: could not read {p.name}: {e}")
        return None


@torch.no_grad()
def predict_probs(model, X, device, batch=1024, temperature=1.0):
    # eval() is essential, not cosmetic: in train mode dropout stays active and
    # BatchNorm keeps updating running stats, so the same bytes give different
    # answers each run and accuracy collapses (0.80 recall -> 0.11 observed).
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.int64)).to(device)
        logits, _, _ = model(xb)
        # temperature scaling is monotonic, so dividing here cannot change any
        # prediction -- only the reported confidence
        out.append(torch.softmax(logits / temperature, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", required=True, help="any file, or a raw disk image")
    ap.add_argument("--block-size", type=int, default=None,
                    help="default: the model's trained block size")
    ap.add_argument("--offset", type=int, default=0, help="start byte")
    ap.add_argument("--length", type=int, default=0, help="bytes to read (0 = all)")
    ap.add_argument("--map", action="store_true",
                    help="print a per-block type map instead of just the summary")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="abstain below this confidence (shown as '?')")
    ap.add_argument("--keep-header", action="store_true",
                    help="include the first block (normally skipped)")
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="stop after N blocks (0 = no limit)")
    ap.add_argument("--raw-conf", action="store_true",
                    help="report uncalibrated softmax instead of the fitted temperature")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model, device)
    classes = [str(c) for c in cfg.classes]
    block = args.block_size or int(getattr(cfg, "block_size", 512))

    p = Path(args.file)
    with open(p, "rb") as f:
        f.seek(args.offset)
        data = f.read(args.length) if args.length else f.read()

    X, start = fragments_from_bytes(data, block, not args.keep_header)
    if args.max_blocks:
        X = X[:args.max_blocks]
    if len(X) == 0:
        print(f"{p.name}: {len(data)} bytes is too small for a {block}-byte block")
        return 1

    cal = None if args.raw_conf else load_calibration(args.model)
    T = float(cal["temperature"]) if cal else 1.0
    P = predict_probs(model, X, device, temperature=T)
    pred = P.argmax(1)
    conf = P.max(1)
    keep = conf >= args.min_conf

    print(f"file      : {p}")
    print(f"size      : {len(data):,} bytes   block={block}   blocks={len(X):,}"
          f"   (first block {'kept' if args.keep_header else 'skipped'})")
    if cal:
        cross = cal.get("results", {}).get("cross-corpus", {})
        ece = cross.get("ece_after")
        print(f"confidence: calibrated, T={T:.3f} (fitted on {Path(cal['fitted_on']).name})")
        if ece is not None and ece >= 0.05:
            print(f"  ! calibration was fitted IN-CORPUS and does not transfer: "
                  f"cross-corpus ECE {ece:.3f}.")
            print(f"  ! On evidence from another source, treat confidence as a "
                  f"RANKING, not a probability.")
    else:
        print(f"confidence: UNCALIBRATED raw softmax - overstates reliability by")
        print(f"  up to 0.27. Run scripts/calibrate.py, and do not quote these as "
              f"probabilities.")

    if args.map:
        print(f"\noffset          type       conf")
        for i in range(len(X)):
            off = args.offset + start + i * block
            name = classes[pred[i]] if keep[i] else "?"
            bar = "#" * int(round(conf[i] * 20))
            print(f"  0x{off:010x}  {name:9s}  {conf[i]:.3f} {bar}")

    counts = Counter(classes[c] for c in pred[keep])
    n_kept = int(keep.sum())
    print(f"\ncomposition ({n_kept:,}/{len(X):,} blocks above conf {args.min_conf}):")
    for name, n in counts.most_common():
        share = n / max(n_kept, 1)
        print(f"  {name:9s} {n:>7,}  {share:6.1%}  {'#' * int(round(share * 40))}")
    if n_kept < len(X):
        print(f"  {'?':9s} {len(X)-n_kept:>7,}  {(len(X)-n_kept)/len(X):6.1%}  (abstained)")

    if counts:
        top, n_top = counts.most_common(1)[0]
        print(f"\nverdict   : {top}  ({n_top/max(n_kept,1):.1%} of scored blocks, "
              f"mean conf {conf[keep][pred[keep] == classes.index(top)].mean():.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
