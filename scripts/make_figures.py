"""Generate the report figures from completed ablation runs.

Reads the per-run JSON reports written by src.training.eval plus the driver's
summary.json, and writes PNGs to reports/figures/cross_corpus/.

  fig1_gap        in-corpus vs cross-corpus accuracy per config, with seed error
                  bars, and the GAP beneath. The headline chart.
  fig2_perclass   per-class recall in-corpus vs cross-corpus -- shows the
                  collapse is concentrated, not uniform.
  fig3_confusion  cross-corpus confusion heatmap, row-normalised.
  fig4_reject     risk-coverage curve: accuracy if the model may abstain.
  fig5_blocksize  512 vs 4096 comparison (only if 4K results exist).

Usage:
  python scripts/make_figures.py --results reports/ablations/cross_corpus_fft75_govdocs1/ablations_512_12seed
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "cross_corpus"
ORDER = ["A_plain", "B_aug", "C_aux", "D_adv", "E_full"]
NICE = {"A_plain": "A: plain CNN", "B_aug": "B: +augment", "C_aux": "C: +aux",
        "D_adv": "D: +adversary", "E_full": "E: full FragGen"}
C_IN, C_CROSS, C_GAP = "#4C72B0", "#DD8452", "#937860"


def load(results: Path):
    s = json.loads((results / "summary.json").read_text())
    groups = {}
    for r in s["rows"]:
        groups.setdefault(r["run"], []).append(r)
    return s.get("setup", {}), groups


def _ms(vals):
    vals = [v for v in vals if v == v]
    if not vals:
        return float("nan"), 0.0
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, sd


def fig_gap(setup, groups, baseline, out):
    names = [n for n in ORDER if n in groups]
    if not names:
        return
    ia = [_ms([r["in_acc"] for r in groups[n]]) for n in names]
    ca = [_ms([r["cross_acc"] for r in groups[n]]) for n in names]
    gp = [_ms([r["gap"] for r in groups[n]]) for n in names]
    x = np.arange(len(names))
    w = 0.38

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 7), height_ratios=[2, 1],
                                  sharex=True)
    ax.bar(x - w/2, [m for m, _ in ia], w, yerr=[s for _, s in ia], capsize=4,
           label="in-corpus (FFT-75)", color=C_IN)
    ax.bar(x + w/2, [m for m, _ in ca], w, yerr=[s for _, s in ca], capsize=4,
           label="cross-corpus (GovDocs1)", color=C_CROSS)
    if baseline:
        ax.axhline(baseline[1], ls="--", lw=1.2, c="grey")
        # anchor left and sit the label just under the line: bars are tall on the
        # right, and at ha="right" the text ran into the last cross-corpus bar
        ax.text(-0.45, baseline[1] - 0.045,
                f"logreg baseline cross = {baseline[1]:.3f}", fontsize=8,
                ha="left", color="grey")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=9)
    n_seeds = len(groups[names[0]])
    ax.set_title(f"FragGen ablation: accuracy and cross-corpus gap\n"
                 f"{setup.get('limit','?')} fragments/class, "
                 f"{setup.get('epochs','?')} epochs, {n_seeds} seeds "
                 f"(error bars = 1 s.d. across seeds)", fontsize=11)

    ax2.bar(x, [m for m, _ in gp], 0.5, yerr=[s for _, s in gp], capsize=4,
            color=C_GAP)
    ax2.set_ylabel("GAP (in − cross)")
    ax2.set_xticks(x, [NICE.get(n, n) for n in names], rotation=15)
    ax2.axhline(0, c="k", lw=0.8)
    for i, (m, s) in enumerate(gp):
        ax2.text(i, m + s + 0.006, f"{m:.3f}", ha="center", fontsize=9)
    ax2.set_ylim(0, max(m + s for m, s in gp) * 1.25)
    fig.tight_layout()
    fig.savefig(out / "fig1_gap.png", dpi=150)
    plt.close(fig)
    print("  wrote fig1_gap.png")


def fig_perclass(results, run, out):
    p = results / f"{run}.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    cls = [str(c) for c in r["in_corpus"]["confusion"]["labels"]]

    def recall(sec):
        cm = np.array(sec["confusion"]["matrix"], float)
        tot = cm.sum(1)
        return np.divide(np.diag(cm), tot, out=np.zeros_like(tot), where=tot > 0)

    ri, rc = recall(r["in_corpus"]), recall(r["cross_corpus"][0])
    order = np.argsort(-(ri - rc))
    x = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, ri[order], 0.4, label="in-corpus", color=C_IN)
    ax.bar(x + 0.2, rc[order], 0.4, label="cross-corpus", color=C_CROSS)
    for i, j in enumerate(order):
        d = ri[j] - rc[j]
        if d > 0.25:
            ax.text(i, max(ri[j], rc[j]) + 0.03, f"−{d:.2f}", ha="center",
                    fontsize=8, color="crimson")
    ax.set_xticks(x, [cls[j] for j in order])
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.set_title(f"Per-class recall, {run}: the collapse is concentrated, "
                 f"not uniform", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_perclass.png", dpi=150)
    plt.close(fig)
    print("  wrote fig2_perclass.png")


def fig_confusion(results, run, out):
    p = results / f"{run}.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    sec = r["cross_corpus"][0]
    cm = np.array(sec["confusion"]["matrix"], float)
    cls = [str(c) for c in sec["confusion"]["labels"]]
    tot = cm.sum(1, keepdims=True)
    norm = np.divide(cm, tot, out=np.zeros_like(cm), where=tot > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(cls)), cls, rotation=45, ha="right")
    ax.set_yticks(range(len(cls)), cls)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(cls)):
        for j in range(len(cls)):
            v = norm[i, j]
            if v >= 0.06:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.55 else "black")
    ax.set_title(f"Cross-corpus confusion ({run}, row-normalised)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out / "fig3_confusion.png", dpi=150)
    plt.close(fig)
    print("  wrote fig3_confusion.png")


def fig_reject(results, run, out):
    p = results / f"{run}.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sec, lab, c in [(r["in_corpus"], "in-corpus (FFT-75)", C_IN),
                        (r["cross_corpus"][0], "cross-corpus (GovDocs1)", C_CROSS)]:
        curve = sec.get("reject_curve")
        if not curve:
            continue
        cov = [p_[0] for p_ in curve]
        acc = [p_[1] for p_ in curve]
        ax.plot(cov, acc, marker="o", ms=3, label=lab, color=c)
    ax.set_xlabel("coverage (fraction of fragments the model answers on)")
    ax.set_ylabel("selective accuracy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f"Reject option ({run}): abstaining buys accuracy", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig4_reject.png", dpi=150)
    plt.close(fig)
    print("  wrote fig4_reject.png")


def fig_blocksize(dir512, dir4k, out):
    """512 vs 4096, only if both matrices exist.

    Both arms MUST come from runs with the same fragments/class and epochs --
    pointing this at the main 12-seed matrix (8000/class) while the 4096 arm is
    4000/class would compare block size and training budget at once and credit
    the difference entirely to block size.
    """
    try:
        s512, g512 = load(dir512)
        s4k, g4k = load(dir4k)
    except Exception:
        return
    if (s512.get("limit"), s512.get("epochs")) != (s4k.get("limit"), s4k.get("epochs")):
        print(f"  SKIP fig5: training budgets differ "
              f"({s512.get('limit')}/class,{s512.get('epochs')}ep vs "
              f"{s4k.get('limit')}/class,{s4k.get('epochs')}ep) -- "
              f"pass --results-512 pointing at the matched arm")
        return
    names = [n for n in ORDER if n in g512 and n in g4k]
    if not names:
        return
    x = np.arange(len(names))
    w = 0.38
    a512 = [_ms([r["cross_acc"] for r in g512[n]]) for n in names]
    a4k = [_ms([r["cross_acc"] for r in g4k[n]]) for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w/2, [m for m, _ in a512], w, yerr=[s for _, s in a512], capsize=4,
           label="512-byte fragments", color="#55A868")
    ax.bar(x + w/2, [m for m, _ in a4k], w, yerr=[s for _, s in a4k], capsize=4,
           label="4096-byte fragments", color="#C44E52")
    ax.set_xticks(x, [NICE.get(n, n) for n in names], rotation=15)
    ax.set_ylabel("cross-corpus accuracy")
    ax.legend(fontsize=9)
    ax.set_title("Does more context help? Cross-corpus accuracy by block size",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig5_blocksize.png", dpi=150)
    plt.close(fig)
    print("  wrote fig5_blocksize.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/ablations/cross_corpus_fft75_govdocs1/ablations_512_12seed")
    ap.add_argument("--results-512", default="reports/ablations/cross_corpus_fft75_govdocs1/blocksize_512",
                    help="512B arm of the block-size figure; must be budget-matched "
                         "to --results-4k (not the main matrix)")
    ap.add_argument("--results-4k", default="reports/ablations/cross_corpus_fft75_govdocs1/blocksize_4096")
    ap.add_argument("--focus", default="D_adv_s1337",
                    help="run used for per-class / confusion / reject figures")
    args = ap.parse_args()

    results = ROOT / args.results
    OUT.mkdir(parents=True, exist_ok=True)
    setup, groups = load(results)

    baseline = None
    bp = ROOT / "reports" / "ablations" / "cross_corpus_fft75_govdocs1" / "real_baseline.json"
    if bp.exists():
        h = json.loads(bp.read_text())["headline"]
        baseline = (h["in_corpus_accuracy"], h["cross_corpus_accuracy"], h["gap"])

    print(f"figures from {results} ->  {OUT}")
    fig_gap(setup, groups, baseline, OUT)
    focus = args.focus
    if not (results / f"{focus}.json").exists():
        cands = sorted(results.glob("*_s*.json"))
        focus = cands[0].stem if cands else focus
    fig_perclass(results, focus, OUT)
    fig_confusion(results, focus, OUT)
    fig_reject(results, focus, OUT)
    fig_blocksize(ROOT / args.results_512, ROOT / args.results_4k, OUT)


if __name__ == "__main__":
    main()
