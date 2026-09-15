"""Generate synthetic corpora A (home) and B (away) to demo the cross-corpus GAP.

Writes:
  <out>/train.npz     (corpus A, for pretrain + train)
  <out>/test_in.npz   (corpus A, held-out -> in-corpus test)
  <out>/test_away.npz (corpus B -> cross-corpus test)
"""
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.synthetic import generate_corpus, corpus_A, corpus_B
from src.preprocessing.fragments import FragmentSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="output folder (the archived copy is archive/data/synthetic)")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--per-class", type=int, default=1500)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    A = generate_corpus(corpus_A(0), n_per_class=args.per_class, block_size=args.block_size)
    # split A into train / in-corpus test
    n = len(A); rng = np.random.default_rng(0); perm = rng.permutation(n)
    cut = int(n * 0.8)
    train = A.subset(perm[:cut]); train.corpus = "synthetic-A-train"
    test_in = A.subset(perm[cut:]); test_in.corpus = "synthetic-A-test"
    away = generate_corpus(corpus_B(1), n_per_class=max(300, args.per_class // 4),
                           block_size=args.block_size)

    FragmentSet.save(train, out / "train.npz")
    FragmentSet.save(test_in, out / "test_in.npz")
    FragmentSet.save(away, out / "test_away.npz")
    print(f"train {len(train)}  test_in {len(test_in)}  test_away {len(away)}  -> {out}")


if __name__ == "__main__":
    main()
