"""DFRWS 2006/2007 carving-challenge prep.

The DFRWS challenges ship a raw disk image plus a ground-truth mapping of which
byte ranges belong to which file/type -- so they test the FULL carving job
end-to-end, not just single-fragment classification. This loader reads the raw
image and a ground-truth CSV you provide (columns: start,end,type[,file_id]) and
emits labelled block_size fragments; file_id is preserved for reassembly eval.

Usage:
  python -m src.preprocessing.dfrws --image dfrws2006.raw --truth dfrws2006_gt.csv \
      --out data/test_dfrws.npz --block-size 512
"""
from __future__ import annotations
import argparse, csv
import numpy as np
from .fragments import FragmentSet
from .fft75 import ALIAS


def load_truth(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build(image_path, truth_path, target_classes, block_size=512):
    tgt = {c: i for i, c in enumerate(target_classes)}
    data = np.fromfile(image_path, dtype=np.uint8)
    X, y, fids = [], [], []
    for r in load_truth(truth_path):
        canon = ALIAS.get(str(r["type"]).lower())
        if canon not in tgt:
            continue
        s, e = int(r["start"]), int(r["end"])
        fid = r.get("file_id", "")
        for off in range(s, min(e, len(data)) - block_size + 1, block_size):
            X.append(data[off:off + block_size]); y.append(tgt[canon]); fids.append(fid)
    if not X:
        raise RuntimeError("no labelled fragments -- check truth CSV columns")
    fs = FragmentSet(np.stack(X).astype(np.uint8), np.array(y, np.int64),
                     list(target_classes), corpus="dfrws")
    fs.file_ids = np.array(fids)  # attached for reassembly evaluation
    return fs


def main():
    from ..config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--block-size", type=int, default=512)
    args = ap.parse_args()
    cfg = load_config(args.config)
    fs = build(args.image, args.truth, cfg.classes, args.block_size)
    fs.save(args.out)
    print("saved", args.out, "n=", len(fs))


if __name__ == "__main__":
    main()
