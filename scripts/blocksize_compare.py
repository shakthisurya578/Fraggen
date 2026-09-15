"""512 vs 4096 byte fragments, same configs, same training budget.

Both arms are trained at the same fragments/class and epochs, so the only
variable is fragment length. Reports the per-config change and a Welch t-test.

Usage:
  python scripts/blocksize_compare.py
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["A_plain", "B_aug", "C_aux", "D_adv", "E_full"]


def load(p):
    rows = json.loads((ROOT / p / "summary.json").read_text())["rows"]
    g = {}
    for r in rows:
        g.setdefault(r["run"], []).append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="reports/ablations/cross_corpus_fft75_govdocs1/blocksize_512")
    ap.add_argument("--b", default="reports/ablations/cross_corpus_fft75_govdocs1/blocksize_4096")
    args = ap.parse_args()
    ga, gb = load(args.a), load(args.b)

    print("=== 512 vs 4096 byte fragments (matched training budget) ===\n")
    for metric, better in [("cross_acc", "higher"), ("gap", "lower"),
                           ("cross_f1", "higher"), ("in_acc", "higher")]:
        print(f"--- {metric} ({better} is better) ---")
        print(f"  {'config':10s} {'512':>16s} {'4096':>16s} {'change':>10s}  {'p':>7s}")
        for name in ORDER:
            if name not in ga or name not in gb:
                continue
            a = np.array([r[metric] for r in ga[name]], float)
            b = np.array([r[metric] for r in gb[name]], float)
            d = b.mean() - a.mean()
            try:
                _, p = stats.ttest_ind(b, a, equal_var=False)
            except Exception:
                p = float("nan")
            star = " *" if p < 0.05 else ""
            print(f"  {name:10s} {a.mean():7.4f}+/-{a.std(ddof=1):.4f} "
                  f"{b.mean():7.4f}+/-{b.std(ddof=1):.4f} {d:+10.4f}  {p:7.3f}{star}")
        print()

    # pooled across configs: does 4096 beat 512 overall?
    a_all = np.array([r["cross_acc"] for v in ga.values() for r in v])
    b_all = np.array([r["cross_acc"] for v in gb.values() for r in v])
    t, p = stats.ttest_ind(b_all, a_all, equal_var=False)
    print(f"pooled over all configs, cross-corpus accuracy:")
    print(f"  512 : {a_all.mean():.4f} +/-{a_all.std(ddof=1):.4f}  (n={len(a_all)})")
    print(f"  4096: {b_all.mean():.4f} +/-{b_all.std(ddof=1):.4f}  (n={len(b_all)})")
    print(f"  change {b_all.mean()-a_all.mean():+.4f}   t={t:+.2f}  p={p:.5f}")


if __name__ == "__main__":
    raise SystemExit(main())
