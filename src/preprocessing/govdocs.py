"""GovDocs1 fragmenter: turn real files (known types) into labelled fragments.

This is FragGen's primary *cross-corpus* test source. GovDocs1 files come with
known extensions, so each fragment inherits its file's type. Run this on the
user's machine where the corpus lives. Optional source-level augmentation
(re-encoding images at varied JPEG quality) further stresses generalisation.

Usage:
  python -m src.preprocessing.govdocs --root /path/to/govdocs1 --out data/test_govdocs.npz \
      --block-size 512 --per-class 20000
"""
from __future__ import annotations
import argparse, os, random
from pathlib import Path
import numpy as np
from .fragments import FragmentSet
from .fft75 import ALIAS


def _iter_files(root):
    for dirpath, _, names in os.walk(root):
        for n in names:
            ext = n.rsplit(".", 1)[-1].lower() if "." in n else ""
            canon = ALIAS.get(ext)
            if canon:
                yield canon, os.path.join(dirpath, n)


def fragment_file(path, block_size, max_frags=None, skip_header=True):
    """Non-overlapping block_size chunks. Skips the very first block so the
    classifier sees middle-of-file fragments (no magic header giveaway).

    A file shorter than 2*block_size yields nothing rather than falling back to
    its header block. The old fallback (`len(data) > 2*block_size else 0`) let
    every file between 1x and 2x block_size contribute exactly one fragment --
    its header, magic bytes included. Harmless at 512 B (<0.1% of fragments) but
    at 4096 B it leaked the signature into 5.5% of PNG fragments, which would
    flatter the larger block size in a 512-vs-4096 comparison."""
    with open(path, "rb") as f:
        data = f.read()
    start = block_size if skip_header else 0
    frags = []
    for off in range(start, len(data) - block_size + 1, block_size):
        frags.append(np.frombuffer(data[off:off + block_size], dtype=np.uint8))
        if max_frags and len(frags) >= max_frags:
            break
    return frags


def build(root, target_classes, block_size=512, per_class=20000,
          per_file_cap=50, seed=0) -> FragmentSet:
    rng = random.Random(seed)
    tgt = {c: i for i, c in enumerate(target_classes)}
    buckets = {c: [] for c in target_classes}
    files = list(_iter_files(root))
    rng.shuffle(files)
    for canon, path in files:
        if canon not in tgt or len(buckets[canon]) >= per_class:
            continue
        try:
            frags = fragment_file(path, block_size, max_frags=per_file_cap)
        except Exception:
            continue
        buckets[canon].extend(frags)
    X, y = [], []
    for c, i in tgt.items():
        fr = buckets[c][:per_class]
        X.extend(fr); y.extend([i] * len(fr))
    if not X:
        raise RuntimeError("no fragments produced -- check --root and file types")
    X = np.stack(X).astype(np.uint8)
    y = np.array(y, dtype=np.int64)
    perm = np.random.default_rng(seed).permutation(len(y))
    return FragmentSet(X[perm], y[perm], list(target_classes), corpus="govdocs1")


def main():
    from ..config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--per-class", type=int, default=20000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    fs = build(args.root, cfg.classes, args.block_size, args.per_class)
    fs.save(args.out)
    print("saved", args.out, "n=", len(fs))


if __name__ == "__main__":
    main()
