#!/usr/bin/env python
"""Publication figures for FragGen V4 (matplotlib only) -> reports/figures/.

  fig1_confusion_matrix             A_plain, V4 test split, seeds pooled, row-normalised
  fig2_precision_recall             per-class precision vs recall (mean ± std over seeds), iso-F1
  fig3_fragment_confidence_timeline live inference on three demo samples
  fig4_dataset_class_distribution   V4 fragments per class, by split and by source
  fig5_training_curves              per-epoch selection-split accuracy per config; E_full pre-training loss
  fig6_ablation_comparison          val/test accuracy and macro-F1 per config, seeds shown
  fig7_pdf_container_analysis       (a) stream map vs predictions for the tricky PDF
                                    (b) all GovDocs1 PDF fragments in V4 test: prediction by stream type
  fig8_pipeline_architecture        inference pipeline + ByteCNN (SVG and PNG)

Every number drawn comes from a file: evaluation reports, training logs,
data/metadata, the checkpoints themselves, or a live run of src.inference.
Fig 6 reads reports/ablations/v4/ablation_summary.json for its p-values, so run
scripts/analyze_ablations.py first. A manifest with captions and data sources
is written to reports/figures/FIGURES.md.

Usage:
  python scripts/make_publication_figures.py                 # all figures
  python scripts/make_publication_figures.py --only 3,7 --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explainability import (REGION_LABELS, dominant_region,  # noqa: E402
                                fragment_region_shares, pdf_stream_regions)
from src.utils import results  # noqa: E402
from src.utils.plotting import (CONFIG_LABELS, CONFIG_ORDER, REGION_COLORS,  # noqa: E402
                                class_color, use_publication_style)

OUT = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
V4 = results.V4_DIR
DATA = ROOT / "data" / "processed_v4"
META = ROOT / "data" / "metadata" / "v4_fragment_metadata.csv"
SAMPLES = ROOT / "demo" / "samples"
CALIB = ROOT / "checkpoints" / "calibration" / "A_plain_s1337.calib.json"
CONFIG_COLORS = {"A_plain": "#2B2B2B", "B_aug": "#E69F00", "C_aux": "#009E73",
                 "D_adv": "#0072B2", "E_full": "#CC79A7"}
SPLIT_COLORS = {"train": "#34495E", "val": "#7F9CC4", "test": "#C5D3E8"}
SOURCE_COLORS = {"govdocs1": "#6F6F6F", "dfrws": "#A6761D", "rico": "#56B4E9", "tgif": "#CC79A7",
                 "uci_csv": "#8C564B", "github_zip": "#0072B2", "wiki_pages": "#B8860B"}
EPOCH_RE = re.compile(r"\[train\] epoch (\d+)/(\d+) val_acc=([\d.]+)")
PRETRAIN_RE = re.compile(r"\[pretrain\] epoch (\d+)/(\d+) loss=([\d.]+)")
MANIFEST = OUT / "figures_manifest.json"


def save(fig, name, caption, sources, svg=False):
    """Write PNG (300 dpi via rcParams) and optionally SVG; record caption + sources."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    if svg:
        fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest[name] = {"caption": caption, "sources": sources,
                      "files": [f"{name}.png"] + ([f"{name}.svg"] if svg else [])}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {name}.png" + (" + .svg" if svg else ""))


def per_class_prf(cm):
    """Per-class precision, recall, F1 from a confusion matrix (rows = true)."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    p = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    r = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return p, r, f


def class_legend(ax, classes, **kw):
    """Legend with one colour patch per class name."""
    return ax.legend(handles=[Patch(color=class_color(c), label=c.upper()) for c in classes], **kw)


# --------------------------------------------------------------------------- 1
def fig1_confusion_matrix():
    reps = results.v4_reports("A_plain")
    blocks = [results.split_block(r, "test") for r in reps.values()]
    labels = [str(c) for c in blocks[0]["confusion"]["labels"]]
    cm = np.sum([np.array(b["confusion"]["matrix"]) for b in blocks], axis=0).astype(float)
    norm = cm / cm.sum(1, keepdims=True)
    acc = results.v4_metric("A_plain", "test", "accuracy")
    f1 = results.v4_metric("A_plain", "test", "macro_f1")

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ticks = [c.upper() for c in labels]
    ax.set_xticks(range(len(labels)), ticks)
    ax.set_yticks(range(len(labels)), ticks)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if norm[i, j] >= 0.005:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.6 else "#222222")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(f"A_plain on the V4 test split ({len(blocks)} seeds pooled)\n"
                 f"accuracy {acc.fmt()}   ·   macro-F1 {f1.fmt()}")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("share of the true class")
    cb.outline.set_visible(False)
    save(fig, "fig1_confusion_matrix",
         f"Row-normalised confusion matrix of the plain byte-CNN on the Dataset V4 test split, "
         f"summed over {len(blocks)} training seeds ({int(cm.sum()):,} fragment evaluations). "
         f"Most off-diagonal mass is PDF fragments predicted as ZIP or JPG.",
         [f"reports/ablations/v4/A_plain_s{s}.json" for s in reps])


# --------------------------------------------------------------------------- 2
def fig2_precision_recall():
    reps = results.v4_reports("A_plain")
    per = defaultdict(lambda: defaultdict(list))
    labels = None
    for r in reps.values():
        b = results.split_block(r, "test")
        labels = [str(c) for c in b["confusion"]["labels"]]
        p, rc, f = per_class_prf(b["confusion"]["matrix"])
        for i, c in enumerate(labels):
            per[c]["p"].append(p[i]); per[c]["r"].append(rc[i]); per[c]["f"].append(f[i])
    m = {c: {k: (np.mean(v), np.std(v, ddof=1) if len(v) > 1 else 0.0) for k, v in per[c].items()}
         for c in labels}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.8, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    lo = min(min(m[c]["p"][0], m[c]["r"][0]) for c in labels) - 0.06
    rr = np.linspace(0.001, 1.0, 600)
    for f in (0.6, 0.7, 0.8, 0.9, 0.95):
        pp = f * rr / (2 * rr - f)
        ok = (rr > f / 2) & (pp <= 1.01) & (pp >= lo)
        ax.plot(rr[ok], pp[ok], color="#C8C8C8", lw=0.8, ls=":", zorder=0)
        if f / (2 - f) >= lo + 0.01:          # label only contours that reach the right edge
            ax.text(1.005, f / (2 - f), f"F1={f}", fontsize=7.5, color="#999999", va="center")
    for c in labels:
        (pm, ps), (rm, rs) = m[c]["p"], m[c]["r"]
        ax.errorbar(rm, pm, xerr=rs, yerr=ps, fmt="o", ms=7, color=class_color(c), capsize=2.5,
                    elinewidth=1, zorder=3)
        # hand-placed offsets for the crowded high-precision/high-recall corner
        dx, dy, ha = {"png": (-10, 4, "right"), "xml": (0, -20, "center"),
                      "csv": (0, -20, "center"), "html": (0, 9, "center"),
                      "gif": (-8, 5, "right")}.get(c, (6, 5, "left"))
        ax.annotate(c.upper(), (rm, pm), textcoords="offset points", xytext=(dx, dy), ha=ha,
                    fontsize=9, weight="bold", color=class_color(c))
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True)
    ax.set_title("(a) Precision vs recall per class", loc="left")

    order = sorted(labels, key=lambda c: -m[c]["f"][0])
    y = np.arange(len(order))
    bx.barh(y, [m[c]["f"][0] for c in order], xerr=[m[c]["f"][1] for c in order],
            color=[class_color(c) for c in order], capsize=2.5, height=0.65)
    for i, c in enumerate(order):
        bx.text(m[c]["f"][0] + m[c]["f"][1] + 0.006, i, f"{m[c]['f'][0]:.3f}", va="center", fontsize=8.5)
    bx.set_yticks(y, [c.upper() for c in order])
    bx.invert_yaxis()
    bx.set_xlim(lo, 1.06)
    bx.set_xlabel("F1")
    bx.grid(True, axis="x")
    bx.set_title("(b) F1 per class, sorted", loc="left")
    fig.suptitle(f"A_plain on the V4 test split: mean ± std over {len(reps)} seeds", y=1.0)
    fig.tight_layout()
    save(fig, "fig2_precision_recall",
         "Per-class precision against recall (error bars: seed std) with iso-F1 contours, and "
         "per-class F1. PDF has high precision but the lowest recall; ZIP and JPG lose precision "
         "to the PDF fragments they absorb.",
         [f"reports/ablations/v4/A_plain_s{s}.json" for s in reps])


# --------------------------------------------------------------------------- 3
def fig3_fragment_timeline(device):
    from src.inference import FragGenClassifier
    clf = FragGenClassifier(device=device)
    bs, kib = clf.block_size, 1024
    files = [("sample.pdf", "pdf"), ("tricky_pdf_container_example.pdf", "pdf"),
             ("tricky_xml_html_boundary_example.xml", "xml")]
    fig, axes = plt.subplots(len(files), 1, figsize=(12.5, 8.8))
    for ax, (fn, true) in zip(axes, files):
        v = clf.classify_file(SAMPLES / fn)
        centers = [(f.byte_offset + bs / 2) / kib for f in v.fragments]
        ax.axvspan(0, bs / kib, color="#EFEFEF", zorder=0)
        ax.text(bs / (2 * kib), 0.5, "header\nblock\n(not\nscored)", ha="center", va="center",
                fontsize=7, color="#888888")
        for f in v.fragments:
            ax.axvspan(f.byte_offset / kib, (f.byte_offset + bs) / kib, ymin=0, ymax=0.07,
                       color=class_color(f.predicted_class), lw=0)
        ax.plot(centers, [f.confidence for f in v.fragments], color="#9A9A9A", lw=1, zorder=1)
        ax.scatter(centers, [f.confidence for f in v.fragments], s=40, zorder=2, edgecolor="white",
                   linewidth=0.6, c=[class_color(f.predicted_class) for f in v.fragments])
        ax.axhline(0.5, ls="--", lw=0.8, color="#BBBBBB")
        ax.set_ylim(0, 1.06)
        ax.set_xlim(0, (v.start_offset + v.n_fragments * bs) / kib)
        ax.set_ylabel("confidence")
        votes = ", ".join(f"{c.upper()} {n}" for c, n in
                          sorted(v.vote_counts.items(), key=lambda kv: -kv[1]) if n)
        ax.set_title(f"{fn}   true type {true.upper()}   ·   votes: {votes}   ·   "
                     f"final: {v.final_prediction.upper()} ({v.final_confidence:.0%})",
                     loc="left", fontsize=10.5)
        class_legend(ax, [c for c in v.classes if v.vote_counts.get(c)], loc="lower right",
                     bbox_to_anchor=(1.0, 0.1), ncol=4, fontsize=8)
    axes[-1].set_xlabel("byte offset in file (KiB).  Coloured strip: predicted class of each "
                        "4096-byte fragment")
    fig.suptitle(f"Fragment confidence timelines: A_plain seed 1337, calibrated confidence "
                 f"(T = {clf.temperature:.4f})", y=1.0)
    fig.tight_layout()
    save(fig, "fig3_fragment_confidence_timeline",
         "Per-fragment calibrated confidence across three files. A mostly-correct PDF with three "
         "low-confidence ZIP fragments; a PDF whose majority vote flips to ZIP; and an XML file "
         "read as HTML.",
         ["checkpoints/v4/A_plain_s1337.pt", "checkpoints/calibration/A_plain_s1337.calib.json"]
         + [f"demo/samples/{f}" for f, _ in files])


# --------------------------------------------------------------------------- 4
def fig4_class_distribution():
    by_split, by_source = defaultdict(Counter), defaultdict(Counter)
    with open(META, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_split[r["class"]][r["split"]] += 1
            by_source[r["class"]][r["source"]] += 1
    # cross-check the metadata against the label arrays stored in the npz files
    split_totals, classes = {}, None
    for split in ("train", "val", "test"):
        with zipfile.ZipFile(DATA / f"{split}.npz") as zf:
            with zf.open("y.npy") as fh:
                y = np.load(fh)
            with zf.open("classes.npy") as fh:
                classes = [str(c) for c in np.load(fh, allow_pickle=True)]
        counts = np.bincount(y, minlength=len(classes))
        for i, c in enumerate(classes):
            if counts[i] != by_split[c][split]:
                raise SystemExit(f"metadata/npz mismatch for {split}/{c}: {by_split[c][split]} vs {counts[i]}")
        split_totals[split] = int(len(y))
    total = sum(split_totals.values())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.2, 5.0))
    x = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for split in ("train", "val", "test"):
        vals = np.array([by_split[c][split] for c in classes])
        ax.bar(x, vals, bottom=bottom, color=SPLIT_COLORS[split], width=0.7,
               label=f"{split} ({split_totals[split]:,})")
        bottom += vals
    for i, t in enumerate(bottom):
        ax.text(i, t + total * 0.002, f"{int(t):,}", ha="center", va="bottom", fontsize=8)
    ax.axhline(total / len(classes), ls="--", lw=0.9, color="#999999")
    ax.text(-0.45, total / len(classes) * 0.975, "equal share", va="top", ha="left",
            fontsize=8, color="#777777")
    ax.set_xticks(x, [c.upper() for c in classes])
    ax.set_ylabel("fragments")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax.set_title("(a) Fragments per class, by split", loc="left")

    sources = sorted({s for c in classes for s in by_source[c]},
                     key=lambda s: -sum(by_source[c][s] for c in classes))
    bottom = np.zeros(len(classes))
    for s in sources:
        vals = np.array([by_source[c][s] for c in classes])
        bx.bar(x, vals, bottom=bottom, color=SOURCE_COLORS.get(s, "#BBBBBB"), width=0.7, label=s)
        bottom += vals
    bx.set_xticks(x, [c.upper() for c in classes])
    bx.set_ylabel("fragments")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    bx.set_title("(b) Fragments per class, by source corpus", loc="left")
    fig.suptitle(f"Dataset V4: {total:,} fragments of 4096 B (train {split_totals['train']:,} · "
                 f"val {split_totals['val']:,} · test {split_totals['test']:,})", y=1.0)
    fig.tight_layout()
    save(fig, "fig4_dataset_class_distribution",
         "Dataset V4 class balance. The per-class cap keeps every class near an equal share; CSV "
         "falls short because its sources ran out of usable files. Panel (b) shows which real "
         "corpus supplied each class.",
         ["data/metadata/v4_fragment_metadata.csv",
          "data/processed_v4/{train,val,test}.npz (label arrays, cross-checked)"])


# --------------------------------------------------------------------------- 5
def _parse_logs():
    acc, loss = defaultdict(dict), defaultdict(dict)
    for p in sorted((V4 / "logs").glob("*_s*.train.log")):
        cfg, seed = p.stem.replace(".train", "").rsplit("_s", 1)
        a, l_ = {}, {}
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if (m := EPOCH_RE.search(line)):
                a[int(m.group(1))] = float(m.group(3))
            if (m := PRETRAIN_RE.search(line)):
                l_[int(m.group(1))] = float(m.group(3))
        if a:
            acc[cfg][int(seed)] = a
        if l_:
            loss[cfg][int(seed)] = l_
    return acc, loss


def fig5_training_curves():
    acc, loss = _parse_logs()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.0, 4.8), gridspec_kw={"width_ratios": [1.7, 1]})
    for cfg in [c for c in CONFIG_ORDER if c in acc]:
        seeds = sorted(acc[cfg])
        epochs = sorted(set.intersection(*[set(acc[cfg][s]) for s in seeds]))
        M = np.array([[acc[cfg][s][e] for e in epochs] for s in seeds])
        mu = M.mean(0)
        sd = M.std(0, ddof=1) if len(seeds) > 1 else np.zeros_like(mu)
        ax.plot(epochs, mu, marker="o", ms=4, lw=1.6, color=CONFIG_COLORS[cfg],
                label=f"{CONFIG_LABELS[cfg]}  (n={len(seeds)})")
        ax.fill_between(epochs, mu - sd, mu + sd, color=CONFIG_COLORS[cfg], alpha=0.12, lw=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy on the 10% selection split")
    ax.grid(True, axis="y")
    ax.legend(loc="lower right")
    ax.set_title("(a) Per-epoch accuracy used for best-epoch selection (mean ± std)", loc="left")
    if loss:
        for cfg in loss:
            seeds = sorted(loss[cfg])
            epochs = sorted(set.intersection(*[set(loss[cfg][s]) for s in seeds]))
            M = np.array([[loss[cfg][s][e] for e in epochs] for s in seeds])
            bx.errorbar(epochs, M.mean(0), yerr=M.std(0, ddof=1) if len(seeds) > 1 else None,
                        marker="o", color=CONFIG_COLORS.get(cfg, "#444"), capsize=3,
                        label=f"{cfg} (n={len(seeds)})")
        bx.set_xlabel("pre-training epoch")
        bx.set_ylabel("NT-Xent loss")
        bx.legend()
        bx.grid(True, axis="y")
    else:
        bx.text(0.5, 0.5, "no contrastive pre-training logs yet", ha="center", va="center",
                transform=bx.transAxes, color="#888888")
        bx.set_axis_off()
    bx.set_title("(b) Contrastive pre-training loss", loc="left")
    fig.suptitle("Training dynamics on Dataset V4. train.py logs no per-epoch training-set "
                 "accuracy, so none is plotted", y=1.0, fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig5_training_curves",
         "Accuracy after each epoch on the 10% split carved from train.npz, which train.py uses "
         "to keep the best epoch (band: seed std); and the NT-Xent loss during E_full's "
         "contrastive pre-training. train.py records no per-epoch training-set accuracy, so no "
         "train-vs-validation curve is shown.",
         ["reports/ablations/v4/logs/*.train.log"])


# --------------------------------------------------------------------------- 6
def fig6_ablation_comparison():
    configs = [c for c in CONFIG_ORDER if results.v4_reports(c)]
    summary_path = V4 / "ablation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    panels = [("val", "accuracy", "(a) val accuracy"), ("test", "accuracy", "(b) test accuracy"),
              ("test", "macro_f1", "(c) test macro-F1")]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.9))
    for ax, (split, metric, title) in zip(axes, panels):
        vals = {c: [results.split_block(r, split)[metric] for r in results.v4_reports(c).values()]
                for c in configs}
        x = np.arange(len(configs))
        means = [np.mean(vals[c]) for c in configs]
        stds = [np.std(vals[c], ddof=1) if len(vals[c]) > 1 else 0 for c in configs]
        ax.bar(x, means, yerr=stds, capsize=3, width=0.62, alpha=0.85,
               color=[CONFIG_COLORS[c] for c in configs])
        rng = np.random.default_rng(0)
        for i, c in enumerate(configs):
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals[c])), vals[c], s=16, color="white",
                       edgecolor="#222222", linewidth=0.8, zorder=3)
        ax.axhline(means[0], ls="--", lw=0.9, color="#555555")
        allv = [v for c in configs for v in vals[c]]
        lo, hi = min(allv) - 0.012, max(allv) + 0.016
        ax.set_ylim(lo, hi)
        if summary:
            tests = summary["paired_t_vs_control"]["results"].get(f"{split}_{metric}", {})
            for i, c in enumerate(configs[1:], start=1):
                t = tests.get(f"{c}_vs_A_plain")
                if t:
                    ax.text(i, means[i] + stds[i] + (hi - lo) * 0.02,
                            f"Δ {t['mean_diff']:+.4f}\np={t.get('p_holm', t['p']):.2f}",
                            ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x, [c.replace("_", "\n") for c in configs])
        ax.grid(True, axis="y")
        ax.set_title(title, loc="left")
    counts = sorted({len(results.v4_reports(c)) for c in configs})
    seeds_txt = (f"{counts[0]} seed{'s' if counts[0] != 1 else ''}" if len(counts) == 1
                 else f"{counts[0]}–{counts[-1]} seeds per config")
    fig.suptitle(f"Dataset V4 ablation ladder: mean ± std over {seeds_txt} (dots: seeds; dashed: "
                 f"A_plain; Δ vs A_plain with Holm-adjusted seed-paired t-test p)", y=1.0, fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig6_ablation_comparison",
         "Validation and test accuracy and test macro-F1 for the cumulative ablation ladder on "
         "Dataset V4. Annotations give the seed-paired difference from A_plain and its "
         "Holm-adjusted p-value.",
         [f"reports/ablations/v4/{c}_s*.json" for c in configs]
         + ["reports/ablations/v4/ablation_summary.json"])


# --------------------------------------------------------------------------- 7
def test_pdf_fragments_by_region(clf):
    """For every GovDocs1 PDF fragment in V4 test: re-read the source PDF, verify the
    stored bytes match the file at the recorded offset, map the fragment to the
    PDF stream it lies in, and record the model's prediction."""
    d = np.load(DATA / "test.npz", allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    y, sources, fids, offs = d["y"], d["sources"], d["file_ids"], d["offsets"]
    sel = np.flatnonzero((y == classes.index("pdf")) & (sources == "govdocs1"))
    X = d["X"][sel]
    pred = clf.predict_fragments(X).argmax(1)
    by_file = defaultdict(list)
    for k, i in enumerate(sel):
        by_file[str(fids[i])].append(k)
    rows, verified, mismatched, missing = [], 0, 0, 0
    for fid, ks in by_file.items():
        path = ROOT / "data" / "raw" / "govdocs1_files" / fid.split("/", 1)[1]
        if not path.exists():
            missing += len(ks)
            continue
        raw = path.read_bytes()
        regions = pdf_stream_regions(raw)
        for k in ks:
            off = int(offs[sel[k]])
            if raw[off:off + clf.block_size] != X[k].tobytes():
                mismatched += 1
                continue
            verified += 1
            sh = fragment_region_shares(regions, off, clf.block_size, 1)[0]
            rows.append((dominant_region(sh), classes[pred[k]]))
    table = defaultdict(Counter)
    for region, p in rows:
        table[region][p] += 1
    info = {"model": "checkpoints/v4/A_plain_s1337.pt", "split": "data/processed_v4/test.npz",
            "pdf_fragments_from_govdocs1": int(len(sel)), "bytes_verified_against_source": verified,
            "byte_mismatches": mismatched, "source_missing": missing, "files": len(by_file),
            "by_dominant_region": {r: dict(table[r]) for r in REGION_LABELS if table[r]}}
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "v4_test_pdf_fragments_by_stream_region.json").write_text(json.dumps(info, indent=2),
                                                                         encoding="utf-8")
    with open(TABLES / "v4_test_pdf_fragments_by_stream_region.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dominant_region", "n_fragments"] + [f"predicted_{c}" for c in classes]
                   + [f"share_{c}" for c in classes])
        for r in REGION_LABELS:
            n = sum(table[r].values())
            if n:
                w.writerow([r, n] + [table[r][c] for c in classes]
                           + [f"{table[r][c] / n:.4f}" for c in classes])
    return info, table, classes


def fig7_pdf_container(device):
    from src.inference import FragGenClassifier
    clf = FragGenClassifier(device=device)
    bs, kib = clf.block_size, 1024
    path = SAMPLES / "tricky_pdf_container_example.pdf"
    data = path.read_bytes()
    v = clf.classify_bytes(data, name=path.name, file_path=path)
    regions = pdf_stream_regions(data)
    shares = fragment_region_shares(regions, v.start_offset, bs, v.n_fragments)
    info, table, classes = test_pdf_fragments_by_region(clf)

    fig = plt.figure(figsize=(12.8, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 1.35], hspace=0.62)
    ax = fig.add_subplot(gs[0])
    cx = fig.add_subplot(gs[1], sharex=ax)
    bx = fig.add_subplot(gs[2])
    end = (v.start_offset + v.n_fragments * bs) / kib

    ax.broken_barh([(0, len(data) / kib)], (2.1, 0.8), facecolors=REGION_COLORS["outside_streams"], alpha=0.35)
    for lab in ("FlateDecode", "DCTDecode", "other_stream"):
        spans = [(r.start / kib, (r.end - r.start) / kib) for r in regions if r.label == lab]
        if spans:
            ax.broken_barh(spans, (2.1, 0.8), facecolors=REGION_COLORS[lab])
    frag_spans = [(f.byte_offset / kib, bs / kib) for f in v.fragments]
    ax.broken_barh(frag_spans, (1.1, 0.8), facecolors=[class_color(f.predicted_class) for f in v.fragments],
                   edgecolor="white", linewidth=0.8)
    ax.broken_barh(frag_spans, (0.1, 0.8), facecolors=[REGION_COLORS[dominant_region(s)] for s in shares],
                   edgecolor="white", linewidth=0.8)
    for row in (1.1, 0.1):
        ax.broken_barh([(0, bs / kib)], (row, 0.8), facecolors="#F2F2F2", edgecolor="#BBBBBB", hatch="///")
    ax.set_yticks([2.5, 1.5, 0.5], ["PDF streams\n(parsed from bytes)", "model prediction\nper fragment",
                                    "stream type holding\nmost of the fragment"])
    ax.set_ylim(0, 3)
    ax.set_xlim(0, end)
    ax.tick_params(axis="x", labelbottom=False)
    votes = ", ".join(f"{c.upper()} {n}" for c, n in sorted(v.vote_counts.items(), key=lambda kv: -kv[1]) if n)
    n_by = Counter(r.label for r in regions)
    ax.set_title(f"(a) {path.name}: {len(regions)} streams ({n_by.get('FlateDecode', 0)} FlateDecode, "
                 f"{n_by.get('DCTDecode', 0)} DCTDecode); fragment votes {votes} → "
                 f"{v.final_prediction.upper()}", loc="left", fontsize=10.5)
    handles = [Patch(color=REGION_COLORS[r], label=r) for r in REGION_LABELS[:3] if any(
        dominant_region(s) == r for s in shares) or n_by.get(r)]
    handles.append(Patch(color=REGION_COLORS["outside_streams"], alpha=0.35,
                         label="outside streams (PDF object syntax)"))
    handles += [Patch(color=class_color(c), label=f"pred {c.upper()}") for c in v.classes if v.vote_counts.get(c)]
    handles += [Patch(facecolor="#F2F2F2", edgecolor="#BBBBBB", hatch="///", label="header (not scored)")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.13), ncol=4, fontsize=8)

    centers = [(f.byte_offset + bs / 2) / kib for f in v.fragments]
    cx.bar(centers, [s["FlateDecode"] for s in shares], width=bs / kib * 0.92,
           color=REGION_COLORS["FlateDecode"], alpha=0.22, label="FlateDecode share of fragment")
    cx.bar(centers, [s["DCTDecode"] for s in shares], width=bs / kib * 0.92,
           bottom=[s["FlateDecode"] for s in shares], color=REGION_COLORS["DCTDecode"], alpha=0.3,
           label="DCTDecode share of fragment")
    cx.plot(centers, [f.confidence for f in v.fragments], color="#444444", lw=1)
    cx.scatter(centers, [f.confidence for f in v.fragments], s=34, zorder=3, edgecolor="white",
               c=[class_color(f.predicted_class) for f in v.fragments], label="prediction confidence")
    cx.set_ylim(0, 1.05)
    cx.set_ylabel("share / confidence")
    cx.set_xlabel("byte offset (KiB)")
    cx.legend(loc="lower left", ncol=3, fontsize=8)

    regs = [r for r in REGION_LABELS if table[r]]
    show = ["pdf", "zip", "jpg"]
    left = np.zeros(len(regs))
    for c in show + ["other"]:
        vals = np.array([(sum(n for k, n in table[r].items() if k not in show) if c == "other"
                          else table[r][c]) / sum(table[r].values()) for r in regs])
        bx.barh(range(len(regs)), vals, left=left, color=class_color(c) if c != "other" else "#CFCFCF",
                label=f"predicted {c.upper()}")
        for i, (l_, w) in enumerate(zip(left, vals)):
            if w >= 0.06:
                bx.text(l_ + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if c != "other" else "#333333")
        left += vals
    bx.set_yticks(range(len(regs)), [f"{r}\n(n={sum(table[r].values()):,})" for r in regs])
    bx.invert_yaxis()
    bx.set_xlim(0, 1)
    bx.set_xlabel("share of fragments")
    bx.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, fontsize=8.5)
    bx.set_title(f"(b) All {info['bytes_verified_against_source']:,} GovDocs1 PDF fragments in the V4 "
                 f"test split, grouped by the stream type holding most of their bytes (A_plain s1337)",
                 loc="left", fontsize=10.5, pad=24)
    fig.suptitle("Why PDF fragments are called ZIP or JPG: prediction depends on the stream "
                 "type the fragment lies in", y=0.995)
    save(fig, "fig7_pdf_container_analysis",
         "PDF container analysis. (a) For one PDF, stream objects parsed from its bytes against "
         "the per-fragment predictions. (b) Every GovDocs1 PDF fragment in the V4 test split, "
         "each byte-verified against its source file and grouped by the stream type holding most "
         "of its bytes: ZIP predictions concentrate in FlateDecode data and JPG predictions in "
         "DCTDecode data.",
         ["demo/samples/tricky_pdf_container_example.pdf", "data/processed_v4/test.npz",
          "data/raw/govdocs1_files/", "checkpoints/v4/A_plain_s1337.pt",
          "reports/tables/v4_test_pdf_fragments_by_stream_region.{csv,json}"])


# --------------------------------------------------------------------------- 8
def _box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#333333", ls="-", fs=9.0, weight="normal", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.4",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight,
            color="#1F1F1F", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, ls="-", color="#444444"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, ls=ls, shrinkA=1, shrinkB=1))


def fig8_architecture():
    import torch
    from src.models.checkpoint import load_model
    from src.models.cnn import AUX_DIM, N_ENTROPY_BUCKETS
    from src.utils.config import load_config

    model, cfg = load_model(str(ROOT / "checkpoints" / "v4" / "A_plain_s1337.pt"), torch.device("cpu"))
    total = sum(p.numel() for p in model.parameters())
    infer = sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("proj.", "adv.")))
    m, bs, ncls = cfg.model, int(cfg.block_size), len(cfg.classes)
    ch, k, emb, drop = list(m.channels), m.kernel_size, m.emb_dim, m.dropout
    T = json.loads(CALIB.read_text(encoding="utf-8"))["temperature"]
    ladder = load_config(ROOT / "configs" / "real.yaml")   # mechanism settings used by B-E

    fig, ax = plt.subplots(figsize=(15.5, 9.2))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 92)
    ax.axis("off")

    top = [("Input\nfile, upload or\ndisk-image region", "#F7F7F7"),
           (f"Discard block 0\n(first {bs} B:\nheader / magic)", "#FBE9E2"),
           (f"Split into N\n{bs}-byte\nfragments", "#F7F7F7"),
           ("ByteCNN\nscores every\nfragment", "#E3EEF8"),
           (f"softmax(logits / T)\nT = {T:.4f}\n(calibration)", "#F7F7F7"),
           ("Majority vote\nover the N\nfragment labels", "#F7F7F7"),
           ("Verdict +\nJSON / CSV\nreport", "#E8F4EC")]
    w, h, y0, step = 17.6, 14.0, 72.0, 21.3
    for i, (t, fc) in enumerate(top):
        _box(ax, 1.0 + i * step, y0, w, h, t, fc=fc, weight="bold" if i == 3 else "normal", fs=8.8)
        if i:
            _arrow(ax, 1.0 + (i - 1) * step + w + 0.5, y0 + h / 2, 1.0 + i * step - 0.5, y0 + h / 2)
    ax.text(75, 89.5, "Inference pipeline: src/inference/classifier.py (one implementation shared by "
                      "the CLI, Streamlit app, viva script and figures)", ha="center", fontsize=10.5,
            weight="bold")

    ax.add_patch(FancyBboxPatch((3, 31), 144, 29.5, boxstyle="round,pad=0.4,rounding_size=2",
                                fc="#F4F8FC", ec="#6C8EBF", lw=1.3))
    _arrow(ax, 1.0 + 3 * step + w / 2, y0 - 0.6, 1.0 + 3 * step + w / 2, 60.9, ls="--", color="#6C8EBF")
    ax.text(75, 57.2, f"ByteCNN, deployed as A_plain: {infer:,} parameters on the inference path "
                      f"({total:,} including the projection head, which inference does not use)",
            ha="center", fontsize=10, weight="bold", color="#2F4F7F")
    lengths = [bs // (2 ** i) for i in range(len(ch) + 1)]
    layers = [(f"Embedding\n256 byte values → {emb}", f"(N, {bs}) → (N, {emb}, {bs})")]
    cin = emb
    for i, c in enumerate(ch):
        layers.append((f"Conv1d {cin}→{c}, k={k}\nBatchNorm · ReLU\nMaxPool ÷2 · Dropout {drop}",
                       f"→ (N, {c}, {lengths[i + 1]})"))
        cin = c
    layers.append(("Global average\npooling over\nfragment length", f"→ (N, {ch[-1]})"))
    layers.append((f"Linear {ch[-1]}→256 · ReLU\nDropout {drop}\nLinear 256→{ncls}", f"→ (N, {ncls}) logits"))
    lw_, lstep = 19.5, 23.6
    for i, (t, shape) in enumerate(layers):
        x = 6 + i * lstep
        _box(ax, x, 38.5, lw_, 13.5, t, fc="#FFFFFF", ec="#6C8EBF", fs=8.8)
        ax.text(x + lw_ / 2, 34.3, shape, ha="center", fontsize=8.3, color="#2F4F7F", family="DejaVu Sans Mono")
        if i:
            _arrow(ax, 6 + (i - 1) * lstep + lw_ + 0.4, 45.25, x - 0.4, 45.25, color="#6C8EBF")

    ax.text(75, 26.8, "Ablation mechanisms (cumulative ladder B → E, evaluated in reports/ablations/v4). "
                      "The deployed A_plain model uses none of them.",
            ha="center", fontsize=10, weight="bold", color="#7A4E00")
    aug = ladder.augment
    mech = [
        (f"B_aug · fragment augmentation\n(training only)\nbit flips p={aug.bit_flip_prob}, "
         f"byte noise p={aug.byte_noise_prob},\nmasking p={aug.mask_prob}, cyclic shift", 6 + 0.5 * lstep),
        (f"C_aux · {AUX_DIM} entropy / byte-statistics\nfeatures concatenated to the\npooled vector "
         f"(head input {ch[-1]} → {ch[-1] + AUX_DIM})", 6 + 5 * lstep),
        (f"D_adv · entropy adversary\n(training only)\ngradient reversal λ={ladder.train.adversary_lambda}"
         f" →\n{N_ENTROPY_BUCKETS} entropy buckets", 6 + 4 * lstep),
        (f"E_full · contrastive pre-training\n(training only)\nprojection {ch[-1]}→256→{m.proj_dim}, NT-Xent,\n"
         f"{ladder.contrastive.epochs} epochs, τ={ladder.contrastive.temperature}", 6 + 4.5 * lstep),
    ]
    bw, bh = 33.0, 17.5
    for i, (t, _target_x) in enumerate(mech):
        _box(ax, 4.0 + i * 36.0, 4.0, bw, bh, t, fc="#FFF8EC", ec="#C98A1B", ls="--", fs=8.6)
    save(fig, "fig8_pipeline_architecture",
         "FragGen inference pipeline and ByteCNN architecture. Layer sizes, parameter counts, "
         "the calibration temperature and the ablation hyper-parameters are read from the "
         "checkpoint, the calibration file and configs/real.yaml, not typed by hand.",
         ["checkpoints/v4/A_plain_s1337.pt", "checkpoints/calibration/A_plain_s1337.calib.json",
          "configs/real.yaml", "src/models/cnn.py"], svg=True)


def write_manifest_md():
    """reports/figures/FIGURES.md: caption and data sources for every figure."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    L = ["# Figures", "", "Generated by `scripts/make_publication_figures.py` (matplotlib only, "
         "300 dpi PNG; the architecture diagram is also SVG). Cross-corpus figures from the "
         "August study are in `cross_corpus/`.", ""]
    for name in sorted(manifest):
        e = manifest[name]
        L += [f"## {name}", "", f"![{name}]({e['files'][0]})", "", e["caption"], "",
              "Data: " + ", ".join(f"`{s}`" for s in e["sources"]), ""]
    (OUT / "FIGURES.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="", help="comma-separated figure numbers, e.g. 1,3,7")
    ap.add_argument("--device", default=None, help="cpu / cuda for the live-inference figures")
    args = ap.parse_args()
    use_publication_style()
    wanted = {int(x) for x in args.only.split(",") if x.strip()} or set(range(1, 9))
    jobs = {1: fig1_confusion_matrix, 2: fig2_precision_recall,
            3: lambda: fig3_fragment_timeline(args.device), 4: fig4_class_distribution,
            5: fig5_training_curves, 6: fig6_ablation_comparison,
            7: lambda: fig7_pdf_container(args.device), 8: fig8_architecture}
    for n in sorted(wanted):
        print(f"figure {n}")
        jobs[n]()
    write_manifest_md()


if __name__ == "__main__":
    main()
