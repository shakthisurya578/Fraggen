"""End-to-end smoke test WITHOUT torch.

Exercises the real data + feature + metrics pipeline: generate corpus A (home)
and corpus B (away), train the feature baseline on A, then evaluate in-corpus vs
cross-corpus and report the headline GAP, macro-F1, reject option, entropy
ceiling, and high-entropy confusion. Proves the scientific core computes end to
end; the torch CNN/contrastive path is a drop-in upgrade of the classifier.
"""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.synthetic import generate_corpus, corpus_A, corpus_B
from src.models.baseline import FeatureBaseline
from src.utils import metrics as M
from src.preprocessing.entropy import histogram_bayes_ceiling


def evaluate(clf, fs, target_cov=0.9):
    P = clf.predict_proba(fs); pred = P.argmax(1); nC = len(fs.classes)
    ceil, _ = histogram_bayes_ceiling(fs.X, fs.y, nC)
    return {
        "corpus": fs.corpus, "acc": M.accuracy(fs.y, pred),
        "macro_f1": M.macro_f1(fs.y, pred, nC),
        "reject": M.reject_at_coverage(P, fs.y, target_cov),
        "he": M.high_entropy_confusion(fs.y, pred, fs.classes),
        "ceiling": ceil,
    }


def main():
    A = generate_corpus(corpus_A(0), n_per_class=1200, block_size=512)
    n = len(A); perm = np.random.default_rng(0).permutation(n); cut = int(0.8 * n)
    train, test_in = A.subset(perm[:cut]), A.subset(perm[cut:])
    test_in.corpus = "synthetic-A(in)"
    away = generate_corpus(corpus_B(1), n_per_class=400, block_size=512)

    clf = FeatureBaseline("logreg").fit(train)
    r_in = evaluate(clf, test_in)
    r_away = evaluate(clf, away)
    gap = M.corpus_gap(r_in["acc"], r_away["acc"])

    print("\n===== FragGen smoke test (feature baseline) =====")
    print(f"in-corpus  [{r_in['corpus']:>16s}] acc={r_in['acc']:.3f} "
          f"macroF1={r_in['macro_f1']:.3f} ceiling={r_in['ceiling']:.3f}")
    print(f"cross      [{r_away['corpus']:>16s}] acc={r_away['acc']:.3f} "
          f"macroF1={r_away['macro_f1']:.3f}")
    print(f"--> cross-corpus GAP = {gap:+.3f}   (this is the number FragGen shrinks)")
    print(f"reject@cov{r_in['reject']['coverage']:.2f}: "
          f"selective_acc={r_in['reject']['selective_accuracy']:.3f} "
          f"(vs full {r_in['acc']:.3f})")
    print(f"high-entropy-only acc  in={r_in['he']['accuracy']:.3f} "
          f"away={r_away['he']['accuracy']:.3f}  "
          f"types={r_in['he']['labels']}")
    print("=================================================\n")
    assert 0.0 <= r_in["acc"] <= 1.0 and gap == gap, "metric sanity failed"
    print("SMOKE TEST PASSED: data + features + metrics + gap + reject all run.")


if __name__ == "__main__":
    main()
