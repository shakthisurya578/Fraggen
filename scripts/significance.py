"""Are the ablation differences real, or seed noise?

Welch's t-test of every config against the A_plain control, on the two columns
that matter (cross-corpus accuracy, and the GAP). With only 3 seeds per config
the test is weak -- it takes |t| ~ 2.8 to clear p<0.05 at ~4 d.o.f. -- so this
mostly protects against reading structure into noise, which is exactly the risk
when every difference in the table is a couple of points wide.

Usage:
  python scripts/significance.py --results reports/ablations/cross_corpus_fft75_govdocs1/ablations_512_12seed
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["A_plain", "B_aug", "C_aux", "D_adv", "E_full"]
CONTROL = "A_plain"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/ablations/cross_corpus_fft75_govdocs1/ablations_512_12seed")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    d = ROOT / args.results
    rows = json.loads((d / "summary.json").read_text())["rows"]
    g = {}
    for r in rows:
        g.setdefault(r["run"], []).append(r)

    if CONTROL not in g:
        print(f"no {CONTROL} runs in {d}")
        return 1

    print(f"=== significance vs {CONTROL} ({d.name}) ===")
    print(f"seeds per config: " +
          ", ".join(f"{k}={len(v)}" for k, v in g.items()))

    for metric, better in [("cross_acc", "higher"), ("gap", "lower"),
                           ("in_acc", "higher"), ("cross_f1", "higher")]:
        base = np.array([r[metric] for r in g[CONTROL]], float)
        print(f"\n--- {metric}  ({better} is better) ---")
        print(f"  {CONTROL:9s} {base.mean():.4f} +/-{base.std(ddof=1):.4f}  (control)")
        for name in ORDER:
            if name == CONTROL or name not in g:
                continue
            v = np.array([r[metric] for r in g[name]], float)
            diff = v.mean() - base.mean()
            t, p = stats.ttest_ind(v, base, equal_var=False)
            mark = "SIGNIFICANT" if p < args.alpha else "n.s."
            arrow = "+" if diff > 0 else ""
            # ASCII only: piping this through tee on Windows encodes as cp1252,
            # which has no U+0394, and the whole stage dies on the delta sign.
            print(f"  {name:9s} {v.mean():.4f} +/-{v.std(ddof=1):.4f}   "
                  f"diff={diff:+.4f}  t={t:+.2f}  p={p:.3f}  {mark}")

    # how big an effect could 3 seeds even detect?
    pooled = np.mean([np.std([r["cross_acc"] for r in v], ddof=1)
                      for v in g.values() if len(v) > 1])
    n = int(np.median([len(v) for v in g.values()]))
    # use the actual seed count, not a hardcoded one: the whole point of this
    # line is to say what THIS run could resolve
    tcrit = stats.t.ppf(0.975, df=max(2 * n - 2, 1))
    mde = tcrit * pooled * np.sqrt(2 / max(n, 1))
    print(f"\ntypical per-config s.d. (cross_acc): {pooled:.4f}")
    print(f"minimum detectable difference at n={n}, p<0.05: ~{mde:.4f}")
    print(f"differences smaller than that cannot be resolved with {n} seeds,")
    print("regardless of how they look in the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
