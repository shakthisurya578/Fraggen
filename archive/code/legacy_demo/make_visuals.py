#!/usr/bin/env python
"""
make_visuals.py — Deliverable 3: publication-quality figures for the FragGen V4 demo.

Generates, using matplotlib only:
  1. fig1_confusion_matrix.png     -- row-normalised confusion matrix (test split)
  2. fig2_precision_recall.png     -- per-class precision/recall bars
  3. fig3_fragment_timeline.png    -- confidence-per-fragment for one sample file
  4. fig4_class_distribution.png   -- pie chart of V4's class balance
  5. fig5_training_curves.png      -- validation accuracy across training epochs

All figures pull from data that already exists (the v4 evaluation JSON, the
v4 npz splits, the training logs) or from a live run of the already-trained
model on one sample file (fig 3). Nothing here retrains anything.

Usage:
    python make_visuals.py
    python make_visuals.py --sample samples/tricky_pdf_container_example.pdf
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from fragcore import FragGenClassifier, DEFAULT_CHECKPOINT  # noqa: E402

EVAL_JSON = REPO_ROOT / "reports" / "v4_results" / "A_plain_s1337.json"
TRAIN_LOG_GLOB = REPO_ROOT / "reports" / "v4_results" / "logs"
PROCESSED_DIR = REPO_ROOT / "data" / "processed_v4"

# A restrained, colourblind-safe-ish palette used consistently across figures.
CLASS_COLOR = "#3B6EA5"
ACCENT_COLOR = "#D1495B"
GRID_COLOR = "#DDDDDD"


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1: confusion matrix
# ---------------------------------------------------------------------------
def fig_confusion_matrix(eval_data: dict, out_path: Path):
    sec = eval_data["cross_corpus"][0]
    cm = np.array(sec["confusion"]["matrix"], dtype=float)
    classes = [str(c) for c in sec["confusion"]["labels"]]
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), [c.upper() for c in classes], rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), [c.upper() for c in classes])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("FragGen V4 — Confusion Matrix (test split, row-normalised)\n"
                 f"Overall accuracy: {sec['accuracy']:.1%}  |  macro-F1: {sec['macro_f1']:.3f}",
                 fontsize=11)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = norm[i, j]
            if v >= 0.03:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 2: per-class precision / recall
# ---------------------------------------------------------------------------
def fig_precision_recall(eval_data: dict, out_path: Path):
    sec = eval_data["cross_corpus"][0]
    cm = np.array(sec["confusion"]["matrix"], dtype=float)
    classes = [str(c) for c in sec["confusion"]["labels"]]
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(len(classes)), where=cm.sum(1) > 0)
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(len(classes)), where=cm.sum(0) > 0)

    order = np.argsort(-recall)
    classes_o = [classes[i] for i in order]
    recall_o, precision_o = recall[order], precision[order]

    x = np.arange(len(classes_o))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w / 2, recall_o, w, label="Recall", color=CLASS_COLOR, zorder=3)
    ax.bar(x + w / 2, precision_o, w, label="Precision", color=ACCENT_COLOR, zorder=3)
    ax.set_xticks(x, [c.upper() for c in classes_o])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.legend(frameon=False)
    ax.set_title("FragGen V4 — Per-class Precision & Recall (test split)", fontsize=11)
    for i, (r, p) in enumerate(zip(recall_o, precision_o)):
        ax.text(i - w / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=7.5)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 3: confidence timeline for one live-classified sample file
# ---------------------------------------------------------------------------
def fig_fragment_timeline(sample_path: Path, out_path: Path):
    clf = FragGenClassifier()
    verdict = clf.classify_file(sample_path)
    if verdict.n_fragments == 0:
        print(f"  SKIP fig3: {sample_path} too small to fragment")
        return

    idx = [f.index for f in verdict.fragments]
    conf = [f.confidence for f in verdict.fragments]
    labels = [f.predicted_class for f in verdict.fragments]
    colors = [CLASS_COLOR if c == verdict.final_prediction else ACCENT_COLOR for c in labels]

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(idx, conf, color="#999999", linewidth=1, zorder=2)
    ax.scatter(idx, conf, c=colors, s=60, zorder=3, edgecolor="white", linewidth=0.6)
    for i, c, lab in zip(idx, conf, labels):
        ax.annotate(lab.upper(), (i, c), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, rotation=0)
    ax.set_xlabel("Fragment index (post header-skip)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(idx)
    ax.set_title(f"FragGen V4 — Fragment Confidence Timeline\n"
                 f"{sample_path.name}  ->  final vote: {verdict.final_prediction.upper()} "
                 f"({verdict.final_confidence:.0%} agreement)", fontsize=11)
    _style_axes(ax)
    # legend explaining colour meaning
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=CLASS_COLOR, label=f"agrees with final vote ({verdict.final_prediction.upper()})"),
        Patch(color=ACCENT_COLOR, label="disagrees with final vote"),
    ], loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}  (sample: {sample_path.name})")


# ---------------------------------------------------------------------------
# Figure 4: class distribution pie chart (from the actual dataset, not hardcoded)
# ---------------------------------------------------------------------------
def fig_class_distribution(out_path: Path):
    import zipfile

    def load_y_and_classes(npz_path):
        with zipfile.ZipFile(npz_path) as zf:
            with zf.open("y.npy") as f:
                y = np.load(f, allow_pickle=True)
            with zf.open("classes.npy") as f:
                classes = [str(c) for c in np.load(f, allow_pickle=True)]
        return np.asarray(y).ravel(), classes

    y_all, classes = [], None
    for split in ["train", "val", "test"]:
        p = PROCESSED_DIR / f"{split}.npz"
        if not p.exists():
            continue
        y, classes = load_y_and_classes(p)
        y_all.append(y)
    y_all = np.concatenate(y_all)
    counts = np.bincount(y_all, minlength=len(classes))

    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(classes))]
    wedges, _, autotexts = ax.pie(
        counts, labels=[c.upper() for c in classes], autopct=lambda p: f"{p:.1f}%",
        colors=colors, startangle=90, pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title(f"FragGen V4 — Class Distribution\n(train+val+test combined, n={counts.sum():,} fragments)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ---------------------------------------------------------------------------
# Figure 5: accuracy-across-epochs, parsed from the actual training logs
# ---------------------------------------------------------------------------
def fig_training_curves(out_path: Path):
    log_dir = TRAIN_LOG_GLOB
    logs = sorted(log_dir.glob("A_plain_s*.train.log"))
    if not logs:
        print("  SKIP fig5: no training logs found")
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    epoch_re = re.compile(r"\[train\] epoch (\d+)/(\d+) val_acc=([\d.]+)")
    any_plotted = False
    for log_path in logs:
        seed_m = re.search(r"s(\d+)\.train\.log", log_path.name)
        seed = seed_m.group(1) if seed_m else "?"
        epochs, accs = [], []
        for line in log_path.read_text().splitlines():
            m = epoch_re.search(line)
            if m:
                epochs.append(int(m.group(1)))
                accs.append(float(m.group(3)))
        if not epochs:
            continue
        ax.plot(epochs, accs, marker="o", markersize=4, label=f"seed {seed}")
        best_ep = int(np.argmax(accs))
        ax.scatter([epochs[best_ep]], [accs[best_ep]], s=90, facecolors="none",
                  edgecolors="black", linewidth=1.3, zorder=5)
        any_plotted = True

    if not any_plotted:
        print("  SKIP fig5: logs found but no epoch lines matched")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("FragGen V4 — Validation Accuracy Across Training Epochs\n"
                 "(circled point = best epoch; that checkpoint is what gets saved)", fontsize=11)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")
    print("  NOTE: only validation accuracy is logged during training "
          "(fraggen/train.py does not compute a per-epoch training-set "
          "accuracy), so this shows validation accuracy only -- labelled "
          "as such rather than implying a training-accuracy curve exists.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=str(HERE / "samples" / "sample.pdf"),
                    help="File to use for the fragment-confidence timeline (fig 3)")
    ap.add_argument("--out-dir", default=str(HERE / "figures"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating FragGen V4 figures...")
    if EVAL_JSON.exists():
        eval_data = json.loads(EVAL_JSON.read_text())
        fig_confusion_matrix(eval_data, out_dir / "fig1_confusion_matrix.png")
        fig_precision_recall(eval_data, out_dir / "fig2_precision_recall.png")
    else:
        print(f"  SKIP fig1/fig2: {EVAL_JSON} not found")

    fig_fragment_timeline(Path(args.sample), out_dir / "fig3_fragment_timeline.png")
    fig_class_distribution(out_dir / "fig4_class_distribution.png")
    fig_training_curves(out_dir / "fig5_training_curves.png")

    print(f"\nAll figures written to {out_dir}")


if __name__ == "__main__":
    main()
