"""Reference cross-corpus GAP on the REAL corpora, using the torch-free baseline.

This answers the question the whole project rests on: does the cross-corpus
collapse that FragGen sets out to fix actually happen between FFT-75 and
GovDocs1? The synthetic demo shows gap ~0.37, but that gap was *designed in*.
This measures it on real data, on CPU, in minutes -- before committing hours of
CNN training.

Train on FFT-75 -> test on FFT-75 test (home) and GovDocs1 (away).

Usage:
  python scripts/real_baseline.py --train-limit 10000 --kind logreg
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.fragments import FragmentSet
from src.models.baseline import FeatureBaseline
from src.utils import metrics as M
from src.preprocessing.entropy import histogram_bayes_ceiling

# the FFT-75/GovDocs1 FragmentSets were archived with the cross-corpus study
PROC = Path("archive/data/processed_cross_corpus_fft75_govdocs1")


def subsample(fs: FragmentSet, per_class: int, seed=1337) -> FragmentSet:
    if not per_class:
        return fs
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(len(fs.classes)):
        idx = np.flatnonzero(fs.y == c)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)
    return fs.subset(np.sort(np.concatenate(keep)))


def per_class_recall(y, pred, n_classes):
    out = []
    for c in range(n_classes):
        m = y == c
        out.append(float((pred[m] == c).mean()) if m.any() else float("nan"))
    return out


def evaluate(clf, fs, cov):
    P = clf.predict_proba(fs)
    pred = P.argmax(1)
    nC = len(fs.classes)
    ceil, _ = histogram_bayes_ceiling(fs.X, fs.y, nC)
    return {
        "corpus": fs.corpus,
        "n": int(len(fs)),
        "accuracy": M.accuracy(fs.y, pred),
        "macro_f1": M.macro_f1(fs.y, pred, nC),
        "per_class_recall": per_class_recall(fs.y, pred, nC),
        "reject": M.reject_at_coverage(P, fs.y, cov),
        "high_entropy": M.high_entropy_confusion(fs.y, pred, list(fs.classes)),
        "entropy_ceiling": ceil,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=10000,
                    help="fragments per class for training (0 = all)")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="fragments per class for evaluation (0 = all)")
    ap.add_argument("--kind", default="logreg", choices=["logreg", "mlp"])
    # default writes a *_rerun file so a re-run can never overwrite the validated
    # reports/ablations/cross_corpus_fft75_govdocs1/real_baseline.json
    ap.add_argument("--report", default="reports/ablations/cross_corpus_fft75_govdocs1/real_baseline_rerun.json")
    args = ap.parse_args()

    t_start = time.time()
    train = subsample(FragmentSet.load(PROC / "fft75_512_train.npz"), args.train_limit)
    test_in = subsample(FragmentSet.load(PROC / "fft75_512_test.npz"), args.eval_limit)
    cross = subsample(FragmentSet.load(PROC / "govdocs_512_test.npz"), args.eval_limit)
    test_in.corpus = "fft75(in)"
    classes = [str(c) for c in train.classes]

    print(f"train {len(train):,} | in-corpus {len(test_in):,} | cross {len(cross):,}")
    print(f"fitting {args.kind} baseline ...")
    t0 = time.time()
    clf = FeatureBaseline(args.kind).fit(train)
    print(f"  fit took {time.time()-t0:.1f}s")

    r_in = evaluate(clf, test_in, 0.9)
    r_cross = evaluate(clf, cross, 0.9)
    gap = M.corpus_gap(r_in["accuracy"], r_cross["accuracy"])

    print("\n================ FragGen: REAL cross-corpus baseline ================")
    print(f"in-corpus  [FFT-75 ]  acc={r_in['accuracy']:.4f}  macroF1={r_in['macro_f1']:.4f}"
          f"  ceiling={r_in['entropy_ceiling']:.4f}")
    print(f"cross      [GovDocs]  acc={r_cross['accuracy']:.4f}  macroF1={r_cross['macro_f1']:.4f}"
          f"  ceiling={r_cross['entropy_ceiling']:.4f}")
    print(f"\n--> HEADLINE cross-corpus GAP = {gap:+.4f}   [this is what FragGen must shrink]")

    print(f"\nper-class recall           in      cross     gap")
    for i, c in enumerate(classes):
        a, b = r_in["per_class_recall"][i], r_cross["per_class_recall"][i]
        print(f"  {c:10s}            {a:6.3f}   {b:6.3f}   {a-b:+6.3f}")

    he_i, he_c = r_in["high_entropy"], r_cross["high_entropy"]
    print(f"\nhigh-entropy types only: in={he_i['accuracy']:.4f}  cross={he_c['accuracy']:.4f}"
          f"  gap={he_i['accuracy']-he_c['accuracy']:+.4f}")
    ri, rc = r_in["reject"], r_cross["reject"]
    print(f"reject@cov0.90 selective acc: in={ri['selective_accuracy']:.4f}"
          f"  cross={rc['selective_accuracy']:.4f}")
    print("=====================================================================\n")

    out = {
        "setup": {"kind": args.kind, "train_per_class": args.train_limit,
                  "n_train": len(train), "classes": classes},
        "in_corpus": r_in, "cross_corpus": r_cross,
        "headline": {"in_corpus_accuracy": r_in["accuracy"],
                     "cross_corpus_accuracy": r_cross["accuracy"], "gap": gap},
        "runtime_sec": time.time() - t_start,
    }
    p = Path(args.report)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
