"""Evaluation: in-corpus vs cross-corpus, the GAP, reject option, entropy ceiling.

CLI:
  python -m src.training.eval --model checkpoints/model.pt \
      --in-corpus data/test_in.npz \
      --cross data/test_govdocs.npz data/test_dfrws.npz \
      --report reports/eval.json
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

from ..preprocessing.fragments import FragmentSet
# load_model now lives in src/models/checkpoint.py (moved verbatim) so that
# inference code does not depend on this module; re-exported here.
from ..models.checkpoint import load_model  # noqa: F401
from ..utils import metrics as M
from ..preprocessing.entropy import histogram_bayes_ceiling


@torch.no_grad()
def predict(model, fs: FragmentSet, device, batch=1024):
    model.eval()
    # cast per batch: fs.X.astype(int64) up front is 8x the array (466 MB for
    # the 113k GovDocs set) and is only ever consumed one batch at a time
    probs = []
    t0 = time.time()
    for i in range(0, len(fs.X), batch):
        xb = torch.from_numpy(fs.X[i:i + batch].astype(np.int64)).to(device)
        logits, _, _ = model(xb)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    dt = time.time() - t0
    P = np.concatenate(probs)
    bytes_processed = fs.X.size
    sec_per_gb = dt / max(bytes_processed / 1e9, 1e-12)
    return P, sec_per_gb


def evaluate_corpus(model, fs, cfg, device):
    P, sec_per_gb = predict(model, fs, device)
    pred = P.argmax(1)
    n_classes = len(fs.classes)
    ceil_acc, _ = histogram_bayes_ceiling(fs.X, fs.y, n_classes)
    return {
        "corpus": fs.corpus,
        "n": int(len(fs)),
        "accuracy": M.accuracy(fs.y, pred),
        "macro_f1": M.macro_f1(fs.y, pred, n_classes),
        "reject": M.reject_at_coverage(P, fs.y, cfg.reject.target_coverage),
        "reject_curve": M.reject_curve(P, fs.y),
        "taxonomy": M.taxonomy_views(fs.y, pred, fs.classes),
        "high_entropy": M.high_entropy_confusion(fs.y, pred, fs.classes),
        "confusion": M.full_confusion(fs.y, pred, fs.classes),
        "entropy_ceiling": ceil_acc,
        "sec_per_gb": sec_per_gb,
    }


def report(model, in_fs, cross_list, cfg, device):
    res = {"in_corpus": evaluate_corpus(model, in_fs, cfg, device), "cross_corpus": []}
    for fs in cross_list:
        res["cross_corpus"].append(evaluate_corpus(model, fs, cfg, device))
    in_acc = res["in_corpus"]["accuracy"]
    gaps = []
    for c in res["cross_corpus"]:
        g = M.corpus_gap(in_acc, c["accuracy"])
        c["gap_vs_in_corpus"] = g
        gaps.append(g)
    res["headline"] = {
        "in_corpus_accuracy": in_acc,
        "mean_cross_corpus_accuracy": float(np.mean([c["accuracy"] for c in res["cross_corpus"]])) if cross_list else None,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "max_gap": float(np.max(gaps)) if gaps else None,
    }
    return res


def print_summary(res):
    h = res["headline"]
    print("\n==================== FragGen evaluation ====================")
    print(f"in-corpus accuracy      : {h['in_corpus_accuracy']:.4f}"
          f"  (macro-F1 {res['in_corpus']['macro_f1']:.4f}, "
          f"entropy ceiling {res['in_corpus']['entropy_ceiling']:.4f})")
    for c in res["cross_corpus"]:
        print(f"cross [{c['corpus']:>16s}] acc: {c['accuracy']:.4f}"
              f"  macro-F1 {c['macro_f1']:.4f}  GAP {c['gap_vs_in_corpus']:+.4f}")
    if h["mean_gap"] is not None:
        print(f"--> HEADLINE mean cross-corpus GAP : {h['mean_gap']:+.4f}  "
              f"(max {h['max_gap']:+.4f})   [lower is better]")
    for c in res["cross_corpus"]:
        t = c.get("taxonomy")
        if t:
            print(f"  [{c['corpus']}] strict {t['strict']:.4f} | "
                  f"html+xml merged {t['markup_merged']:.4f} | "
                  f"coarse {t['coarse_families']:.4f} | "
                  f"excl. pdf/zip containers {t.get('excl_containers', float('nan')):.4f}")
    r = res["in_corpus"]["reject"]
    print(f"reject option (cov {r['coverage']:.2f}) selective acc: {r['selective_accuracy']:.4f}")
    print(f"throughput: {res['in_corpus']['sec_per_gb']:.2f} sec/GB")
    print("============================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--in-corpus", required=True)
    ap.add_argument("--cross", nargs="*", default=[])
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.model, device)
    in_fs = FragmentSet.load(args.in_corpus)
    cross = [FragmentSet.load(p) for p in args.cross]
    res = report(model, in_fs, cross, cfg, device)
    print_summary(res)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote report", args.report)


if __name__ == "__main__":
    main()
