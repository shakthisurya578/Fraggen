"""How much does aggregating CONTIGUOUS blocks buy?

A carver rarely sees an isolated fragment. Unallocated space contains runs of
consecutive sectors from the same file, and a run can vote. This measures
accuracy as a function of run length -- the single most realistic way to make
the classifier more usable without retraining anything.

Run length 1 is the per-fragment number reported everywhere else, so this table
is directly comparable to the headline results.

Usage:
  python scripts/run_aggregation.py --model checkpoints/v4/A_plain_s1337.pt --block 4096
"""
from __future__ import annotations
import argparse, os, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.checkpoint import load_model
from src.preprocessing.fft75 import ALIAS

CORPUS = ROOT / "data" / "raw" / "govdocs1_files"
WANT = ["txt", "csv", "html", "xml", "pdf", "jpg", "png", "gif", "zip"]


def collect(per_type, min_size, seed=0):
    rng = random.Random(seed)
    found = {w: [] for w in WANT}
    subdirs = sorted([d for d in CORPUS.iterdir() if d.is_dir()])
    rng.shuffle(subdirs)
    for d in subdirs:
        if all(len(v) >= per_type for v in found.values()):
            break
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        rng.shuffle(entries)
        for e in entries:
            ext = e.name.rsplit(".", 1)[-1].lower() if "." in e.name else ""
            c = ALIAS.get(ext)
            if c in found and len(found[c]) < per_type:
                try:
                    if e.stat().st_size >= min_size:
                        found[c].append(Path(e.path))
                except OSError:
                    pass
    return found


def frags(data, block):
    start = block                      # always drop the header block
    n = max(0, (len(data) - start) // block)
    if n == 0:
        return np.empty((0, block), np.uint8)
    return np.frombuffer(data, np.uint8, count=n * block, offset=start).reshape(n, block)


@torch.no_grad()
def probs(model, X, device, batch=256):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.int64)).to(device)
        lg, _, _ = model(xb)
        out.append(torch.softmax(lg, 1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/v4/A_plain_s1337.pt")
    ap.add_argument("--block", type=int, default=4096)
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--runs", default="1,2,4,8,16,32")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(str(ROOT / args.model), device)
    classes = [str(c) for c in cfg.classes]
    lengths = [int(x) for x in args.runs.split(",")]
    maxrun = max(lengths)

    files = collect(args.per_type, min_size=args.block * (maxrun + 2))
    n_files = sum(len(v) for v in files.values())
    print("=" * 72)
    print(f"  CONTIGUOUS-RUN AGGREGATION   block={args.block}  model={Path(args.model).name}")
    print(f"  {n_files} GovDocs1 files (cross-corpus), runs of {lengths}")
    print("=" * 72)

    # collect per-file probability sequences once
    seqs = []
    for t, paths in files.items():
        ci = classes.index(t)
        for p in paths:
            X = frags(p.read_bytes(), args.block)
            if len(X) < 1:
                continue
            seqs.append((ci, probs(model, X, device)))

    print(f"\n  {'run len':>8s} {'bytes':>9s} {'accuracy':>9s} {'n runs':>9s}   delta")
    base = None
    rows = []
    for L in lengths:
        ok = tot = 0
        for ci, P in seqs:
            n = len(P) // L
            for k in range(n):
                seg = P[k * L:(k + 1) * L]
                # sum of probabilities = soft vote; more stable than hard majority
                if int(seg.sum(0).argmax()) == ci:
                    ok += 1
                tot += 1
        acc = ok / max(tot, 1)
        if base is None:
            base = acc
        d = acc - base
        rows.append((L, acc, tot))
        print(f"  {L:>8d} {L*args.block:>9,} {acc:>9.4f} {tot:>9,}   "
              f"{d:+.4f}" if L > 1 else
              f"  {L:>8d} {L*args.block:>9,} {acc:>9.4f} {tot:>9,}   (baseline)")

    print()
    best = max(rows, key=lambda r: r[1])
    print(f"  best: runs of {best[0]} ({best[0]*args.block:,} bytes) -> {best[1]:.4f}")
    print(f"  gain over single fragments: {best[1]-base:+.4f}")
    print("""
  A run is contiguous sectors from one file. Carvers already work this way --
  this costs no retraining, only a decision to aggregate before reporting.""")


if __name__ == "__main__":
    raise SystemExit(main())
