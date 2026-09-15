#!/usr/bin/env python
"""Dataset V4 ablation analysis: A_plain against B_aug, C_aux, D_adv and E_full.

Reads experiment outputs only:
  reports/ablations/v4/<config>_s<seed>.json          evaluation reports (src.training.eval)
  reports/ablations/v4/logs/<config>_s<seed>.train.log per-epoch training logs
  reports/ablations/v4/driver_summary_*.json           training wall-clock (driver output)
  checkpoints/v4/<config>_s<seed>.pt + data/processed_v4/test.npz
                                                       re-scored for fragment-level McNemar tests

Writes:
  reports/ablations/v4/ablation_summary.json    every number below, machine-readable
  reports/ablations/v4/ABLATION_RESULTS.md      the human-readable report
  reports/ablations/v4/test_predictions.npz     per-fragment test predictions per run
  reports/tables/v4_*.csv / .md                 main, per-class, per-seed, significance, curves
  reports/confusion_matrices/v4_*.csv           pooled counts, row-normalised, long format

Statistics
  * mean ± sample std (ddof=1) over seeds 1337/1338/1339
  * seed-paired t-test of each config against A_plain (same seed = same weight
    init, split and data order), with a 95% CI of the mean difference and
    Holm correction across the four comparisons; plus the same test between
    consecutive rungs (the marginal contribution of each added mechanism)
  * exact McNemar test on the 13,754 test fragments for each seed pair: does
    this trained model make different mistakes from its A_plain twin?

Usage:
  python scripts/analyze_ablations.py
  python scripts/analyze_ablations.py --no-mcnemar      # JSON reports only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.plotting import CONFIG_LABELS, CONFIG_ORDER  # noqa: E402
from src.utils.results import split_block  # noqa: E402

V4 = ROOT / "reports" / "ablations" / "v4"
TABLES = ROOT / "reports" / "tables"
CMS = ROOT / "reports" / "confusion_matrices"
CONTROL = "A_plain"
SPLIT_NAMES = {"val": "val.npz", "test": "test.npz"}
MECHANISM = {"A_plain": "(none: plain byte-CNN)", "B_aug": "fragment augmentation",
             "C_aux": "entropy auxiliary features", "D_adv": "entropy adversary (gradient reversal)",
             "E_full": "contrastive pre-training"}
FLAGS = {"A_plain": "--no-aug --no-aux --no-adv --no-contrastive",
         "B_aug": "--no-aux --no-adv --no-contrastive", "C_aux": "--no-adv --no-contrastive",
         "D_adv": "--no-contrastive", "E_full": "(all mechanisms on)"}
HEADLINE = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
EPOCH_RE = re.compile(r"\[train\] epoch (\d+)/(\d+) val_acc=([\d.]+)")
PRETRAIN_RE = re.compile(r"\[pretrain\] epoch (\d+)/(\d+) loss=([\d.]+)")


# ------------------------------------------------------------------ helpers
def mean_std(values):
    """{"mean", "std", "values"} with sample std (0 for a single value)."""
    values = [float(v) for v in values]
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "values": values}


def fmt(ms, d=4):
    return f"{ms['mean']:.{d}f} ± {ms['std']:.{d}f}"


def per_class_from_confusion(cm):
    """Precision, recall and F1 per class from a confusion matrix (rows = true).
    Zero-division yields 0, matching sklearn's zero_division=0 used in evaluation."""
    cm = np.asarray(cm, dtype=float)
    tp, pred_tot, true_tot = np.diag(cm), cm.sum(0), cm.sum(1)
    prec = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    rec = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m, adjusted, running = len(pvalues), [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired_test(control, treatment):
    """Seed-paired t-test of treatment - control, with a 95% CI of the mean difference."""
    c, t = np.asarray(control, float), np.asarray(treatment, float)
    d = t - c
    n = len(d)
    out = {"n": n, "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if n > 1 else 0.0}
    if n < 2 or out["sd_diff"] == 0.0:
        out.update(t=float("nan"), p=float("nan"), ci95=[float("nan"), float("nan")])
        return out
    tstat, p = stats.ttest_rel(t, c)
    half = stats.t.ppf(0.975, n - 1) * out["sd_diff"] / math.sqrt(n)
    out.update(t=float(tstat), p=float(p), ci95=[out["mean_diff"] - half, out["mean_diff"] + half])
    return out


def mcnemar_exact(correct_a, correct_b):
    """Exact (binomial) McNemar test. b = A right & B wrong, c = A wrong & B right."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), n, 0.5)))
    return {"a_only_correct": b, "b_only_correct": c, "p": p}


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def train_fragment_count():
    """Number of fragments in data/processed_v4/train.npz, read from its label array."""
    import zipfile
    with zipfile.ZipFile(ROOT / "data" / "processed_v4" / "train.npz") as zf:
        with zf.open("y.npy") as f:
            return int(len(np.load(f)))


# ------------------------------------------------------------------ loading
def load_runs():
    """(config, seed) -> evaluation report, for every config in the ladder."""
    runs = {}
    for cfg in CONFIG_ORDER:
        for p in sorted(V4.glob(f"{cfg}_s*.json")):
            runs[(cfg, int(p.stem.rsplit("_s", 1)[1]))] = json.loads(p.read_text(encoding="utf-8"))
    return runs


def run_metrics(report, checks):
    """Headline and per-class metrics for both splits of one run. Values that
    can be recomputed from the confusion matrix are cross-checked."""
    out = {}
    for split in SPLIT_NAMES:
        b = split_block(report, split)
        labels = [str(c) for c in b["confusion"]["labels"]]
        cm = np.array(b["confusion"]["matrix"])
        prec, rec, f1 = per_class_from_confusion(cm)
        checks["max_abs_acc_vs_confusion"] = max(checks.get("max_abs_acc_vs_confusion", 0.0),
                                                 abs(np.trace(cm) / cm.sum() - b["accuracy"]))
        checks["max_abs_macro_f1_vs_confusion"] = max(checks.get("max_abs_macro_f1_vs_confusion", 0.0),
                                                      abs(float(f1.mean()) - b["macro_f1"]))
        tax = b.get("taxonomy", {})
        out[split] = {
            "n": int(b["n"]), "accuracy": float(b["accuracy"]), "macro_f1": float(b["macro_f1"]),
            "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()),
            "accuracy_markup_merged": tax.get("markup_merged"),
            "accuracy_coarse_families": tax.get("coarse_families"),
            "accuracy_excl_containers": tax.get("excl_containers"),
            "labels": labels, "confusion": cm.astype(int).tolist(),
            "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]),
                              "f1": float(f1[i]), "support": int(cm[i].sum())}
                          for i, c in enumerate(labels)},
        }
    return out


def parse_training_logs():
    """Per-epoch accuracy on the 10% selection split carved from train.npz (the
    number train.py uses to keep the best epoch), plus E_full pre-training loss."""
    rows = []
    for p in sorted((V4 / "logs").glob("*_s*.train.log")):
        cfg, seed = p.stem.replace(".train", "").rsplit("_s", 1)
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = EPOCH_RE.search(line)
            if m:
                rows.append([cfg, int(seed), "train", int(m.group(1)), "selection_split_accuracy", float(m.group(3))])
            m = PRETRAIN_RE.search(line)
            if m:
                rows.append([cfg, int(seed), "pretrain", int(m.group(1)), "nt_xent_loss", float(m.group(3))])
    return rows


def training_minutes():
    """(config, seed) -> training minutes recorded by the driver, where available."""
    out = {}
    for p in sorted(V4.glob("driver_summary_*.json")):
        for r in json.loads(p.read_text(encoding="utf-8")).get("rows", []):
            tm = r.get("train_min")
            if tm is not None and not (isinstance(tm, float) and math.isnan(tm)):
                out[(r["run"], int(r["seed"]))] = float(tm)
    return out


def score_test_fragments(runs, device_arg):
    """Re-score test.npz with every checkpoint (same batch size and softmax as
    src.training.eval) so McNemar can pair decisions fragment by fragment."""
    import torch
    from src.inference import resolve_device
    from src.models.checkpoint import load_model
    from src.preprocessing.fragments import FragmentSet

    device = resolve_device(device_arg)
    test = FragmentSet.load(ROOT / "data" / "processed_v4" / "test.npz")
    preds = {}
    for (cfg, seed) in sorted(runs):
        ck = ROOT / "checkpoints" / "v4" / f"{cfg}_s{seed}.pt"
        if not ck.exists():
            continue
        model, _ = load_model(str(ck), device)
        out = []
        with torch.no_grad():
            for i in range(0, len(test.X), 1024):
                xb = torch.from_numpy(test.X[i:i + 1024].astype(np.int64)).to(device)
                logits, _, _ = model(xb)
                out.append(torch.softmax(logits, dim=1).argmax(1).cpu().numpy())
        preds[f"{cfg}_s{seed}"] = np.concatenate(out).astype(np.int16)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return test.y.astype(np.int16), preds, str(device)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-mcnemar", action="store_true", help="skip re-scoring checkpoints")
    ap.add_argument("--device", default=None, help="cpu / cuda for re-scoring (default: auto)")
    args = ap.parse_args()

    runs = load_runs()
    configs = [c for c in CONFIG_ORDER if any(k[0] == c for k in runs)]
    seeds = sorted({s for _, s in runs})
    if CONTROL not in configs:
        sys.exit("no A_plain reports found in reports/ablations/v4")
    checks = {}
    per_run = {k: run_metrics(v, checks) for k, v in runs.items()}
    minutes = training_minutes()

    # ---- aggregate over seeds
    agg = {}
    for cfg in configs:
        cseeds = sorted(s for c, s in runs if c == cfg)
        entry = {"mechanism_added": MECHANISM[cfg], "train_flags": FLAGS[cfg], "seeds": cseeds,
                 "train_minutes": mean_std([minutes[(cfg, s)] for s in cseeds if (cfg, s) in minutes])
                 if any((cfg, s) in minutes for s in cseeds) else None, "splits": {}}
        for split in SPLIT_NAMES:
            rs = [per_run[(cfg, s)][split] for s in cseeds]
            e = {m: mean_std([r[m] for r in rs]) for m in HEADLINE}
            for m in ["accuracy_markup_merged", "accuracy_coarse_families", "accuracy_excl_containers"]:
                if all(r[m] is not None for r in rs):
                    e[m] = mean_std([r[m] for r in rs])
            labels = rs[0]["labels"]
            e["per_class"] = {c: {k: mean_std([r["per_class"][c][k] for r in rs])
                                  for k in ["precision", "recall", "f1"]} for c in labels}
            e["confusion_pooled"] = np.sum([np.array(r["confusion"]) for r in rs], axis=0).tolist()
            e["labels"] = labels
            e["n"] = rs[0]["n"]
            entry["splits"][split] = e
        agg[cfg] = entry

    # ---- paired statistics
    def paired_block(pairs, name):
        block = {"comparison": name, "results": {}}
        for split in SPLIT_NAMES:
            for metric in ["accuracy", "macro_f1"]:
                key = f"{split}_{metric}"
                tests = {}
                for a, b in pairs:
                    common = sorted(set(agg[a]["seeds"]) & set(agg[b]["seeds"]))
                    tests[f"{b}_vs_{a}"] = paired_test(
                        [per_run[(a, s)][split][metric] for s in common],
                        [per_run[(b, s)][split][metric] for s in common])
                ps = [t["p"] for t in tests.values()]
                valid = [i for i, p in enumerate(ps) if not math.isnan(p)]
                adj = holm([ps[i] for i in valid]) if valid else []
                for j, i in enumerate(valid):
                    list(tests.values())[i]["p_holm"] = adj[j]
                block["results"][key] = tests
        return block

    vs_control = paired_block([(CONTROL, c) for c in configs if c != CONTROL], "each config vs A_plain")
    consecutive = paired_block(list(zip(configs[:-1], configs[1:])), "each rung vs the previous rung")

    mcnemar, prediction_checks, scored_on = {}, {}, None
    if not args.no_mcnemar:
        y, preds, scored_on = score_test_fragments(runs, args.device)
        np.savez_compressed(V4 / "test_predictions.npz", y=y, **preds)
        for key, p in preds.items():
            cfg, seed = key.rsplit("_s", 1)
            recomputed = float((p == y).mean())
            reported = per_run[(cfg, int(seed))]["test"]["accuracy"]
            prediction_checks[key] = {"recomputed_accuracy": recomputed, "reported_accuracy": reported,
                                      "abs_diff": abs(recomputed - reported)}
        for cfg in configs:
            if cfg == CONTROL:
                continue
            per_seed = {}
            for s in seeds:
                ka, kb = f"{CONTROL}_s{s}", f"{cfg}_s{s}"
                if ka in preds and kb in preds:
                    per_seed[str(s)] = mcnemar_exact(preds[ka] == y, preds[kb] == y)
            if per_seed:
                mcnemar[cfg] = per_seed
        # Holm across the four configs within each seed
        for s in seeds:
            cfgs = [c for c in mcnemar if str(s) in mcnemar[c]]
            adj = holm([mcnemar[c][str(s)]["p"] for c in cfgs])
            for c, a in zip(cfgs, adj):
                mcnemar[c][str(s)]["p_holm"] = a

    summary = {
        "setup": {"dataset": "data/processed_v4", "block_size": 4096, "epochs": 6,
                  "train_fragments": train_fragment_count(),
                  "val_fragments": agg[CONTROL]["splits"]["val"]["n"],
                  "test_fragments": agg[CONTROL]["splits"]["test"]["n"],
                  "seeds": seeds, "config_file": "configs/real.yaml",
                  "note": "val.npz and test.npz are held-out FILES from the training source pool; "
                          "not a cross-corpus evaluation"},
        "configs": agg,
        "paired_t_vs_control": vs_control,
        "paired_t_consecutive": consecutive,
        "mcnemar_test_fragments": mcnemar,
        "consistency_checks": {**checks, "prediction_rescoring": prediction_checks,
                               "rescoring_device": scored_on},
    }
    (V4 / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- tables
    main_rows = []
    for cfg in configs:
        a = agg[cfg]["splits"]
        t = vs_control["results"]["test_accuracy"].get(f"{cfg}_vs_{CONTROL}")
        mc = mcnemar.get(cfg, {})
        main_rows.append([
            cfg, MECHANISM[cfg], len(agg[cfg]["seeds"]),
            fmt(a["val"]["accuracy"]), fmt(a["val"]["macro_f1"]),
            fmt(a["test"]["accuracy"]), fmt(a["test"]["macro_f1"]),
            fmt(a["test"]["macro_precision"]), fmt(a["test"]["macro_recall"]),
            "" if t is None else f"{t['mean_diff']:+.4f}",
            "" if t is None else f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}]",
            "" if t is None else f"{t['p']:.3f}",
            "" if t is None else f"{t.get('p_holm', float('nan')):.3f}",
            "" if not mc else f"{sum(v['p_holm'] < 0.05 for v in mc.values())}/{len(mc)}",
            "" if agg[cfg]["train_minutes"] is None else f"{agg[cfg]['train_minutes']['mean']:.1f}",
        ])
    main_header = ["config", "mechanism added", "seeds", "val accuracy", "val macro-F1",
                   "test accuracy", "test macro-F1", "test macro-precision", "test macro-recall",
                   "Δ test acc vs A_plain", "95% CI of Δ", "paired t p", "Holm p",
                   "McNemar seeds p_holm<0.05", "train min (mean)"]
    write_csv(TABLES / "v4_ablation_main.csv", main_header, main_rows)
    md = ["| " + " | ".join(main_header) + " |", "|" + "---|" * len(main_header)]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in main_rows]
    (TABLES / "v4_ablation_main.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    pc_rows = []
    for cfg in configs:
        for split in SPLIT_NAMES:
            for c, m in agg[cfg]["splits"][split]["per_class"].items():
                pc_rows.append([cfg, split, c] + [f"{m[k]['mean']:.4f}" for k in ["precision", "recall", "f1"]]
                               + [f"{m[k]['std']:.4f}" for k in ["precision", "recall", "f1"]])
    write_csv(TABLES / "v4_per_class_metrics.csv",
              ["config", "split", "class", "precision_mean", "recall_mean", "f1_mean",
               "precision_std", "recall_std", "f1_std"], pc_rows)

    seed_rows = []
    for (cfg, s) in sorted(per_run, key=lambda k: (CONFIG_ORDER.index(k[0]), k[1])):
        r = per_run[(cfg, s)]
        seed_rows.append([cfg, s] + [f"{r[sp][m]:.4f}" for sp in SPLIT_NAMES for m in HEADLINE]
                         + [f"{minutes[(cfg, s)]:.1f}" if (cfg, s) in minutes else ""])
    write_csv(TABLES / "v4_per_seed_runs.csv",
              ["config", "seed"] + [f"{sp}_{m}" for sp in SPLIT_NAMES for m in HEADLINE] + ["train_minutes"],
              seed_rows)

    sig_rows = []
    for block in (vs_control, consecutive):
        for key, tests in block["results"].items():
            for name, t in tests.items():
                sig_rows.append([block["comparison"], key, name, t["n"], f"{t['mean_diff']:+.5f}",
                                 f"{t['ci95'][0]:+.5f}", f"{t['ci95'][1]:+.5f}", f"{t['t']:.3f}",
                                 f"{t['p']:.4f}", f"{t.get('p_holm', float('nan')):.4f}"])
    for cfg, per_seed in mcnemar.items():
        for s, m in per_seed.items():
            sig_rows.append(["McNemar exact, test fragments", f"seed {s}", f"{cfg}_vs_{CONTROL}",
                             m["a_only_correct"] + m["b_only_correct"],
                             f"A-only {m['a_only_correct']} / {cfg}-only {m['b_only_correct']}",
                             "", "", "", f"{m['p']:.4g}", f"{m['p_holm']:.4g}"])
    write_csv(TABLES / "v4_significance.csv",
              ["test", "metric / seed", "comparison", "n", "mean diff (or discordant counts)",
               "ci95 low", "ci95 high", "t", "p", "p_holm"], sig_rows)

    write_csv(TABLES / "v4_training_curves.csv",
              ["config", "seed", "phase", "epoch", "metric", "value"], parse_training_logs())

    # ---- confusion matrices
    long_rows = []
    for cfg in configs:
        for split in SPLIT_NAMES:
            e = agg[cfg]["splits"][split]
            labels, cm = e["labels"], np.array(e["confusion_pooled"], dtype=float)
            write_csv(CMS / f"v4_{cfg}_{split}_pooled_counts.csv", ["true \\ predicted"] + labels,
                      [[labels[i]] + [int(v) for v in cm[i]] for i in range(len(labels))])
            rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            write_csv(CMS / f"v4_{cfg}_{split}_pooled_rownorm.csv", ["true \\ predicted"] + labels,
                      [[labels[i]] + [f"{v:.4f}" for v in rn[i]] for i in range(len(labels))])
            for s in agg[cfg]["seeds"]:
                m = per_run[(cfg, s)][split]["confusion"]
                for i, t in enumerate(labels):
                    for j, p in enumerate(labels):
                        long_rows.append([cfg, s, split, t, p, m[i][j]])
    write_csv(CMS / "v4_all_runs_long.csv", ["config", "seed", "split", "true", "predicted", "count"], long_rows)

    write_markdown(summary, configs, seeds, mcnemar, prediction_checks)
    print(f"analysed {len(runs)} runs over configs {configs}; wrote {V4 / 'ABLATION_RESULTS.md'}")


def write_markdown(summary, configs, seeds, mcnemar, prediction_checks):
    """ABLATION_RESULTS.md. Every sentence of interpretation is conditional on the
    computed statistics, so the text cannot claim more than the numbers show."""
    agg, vs, cons = summary["configs"], summary["paired_t_vs_control"], summary["paired_t_consecutive"]
    L = ["# Dataset V4 ablation study", "",
         "_Generated by `scripts/analyze_ablations.py` from the evaluation reports in this folder. "
         "Do not edit by hand; re-run the script._", "",
         "> **Scope.** `val.npz` and `test.npz` are *held-out files* drawn from the same source pool "
         "as `train.npz`. These are **not** cross-corpus numbers. The project's cross-corpus result "
         "(train FFT-75 → test GovDocs1) is in `../cross_corpus_fft75_govdocs1/` and "
         "`docs/thesis/FINDINGS.md`.", "",
         "## Setup", "",
         f"- Data: `data/processed_v4`: {summary['setup']['train_fragments']:,} train, "
         f"{summary['setup']['val_fragments']:,} val and {summary['setup']['test_fragments']:,} test "
         "fragments, 4096 B, 9 classes",
         f"- Training: `configs/real.yaml`, 6 epochs, every training fragment (`--limit 0`), "
         f"seeds {', '.join(map(str, seeds))}; best epoch kept by accuracy on a 10% split of `train.npz`",
         "- Driver: `python scripts/run_ablations.py` (the defaults reproduce this protocol)", "",
         "### The cumulative ladder", "",
         "| config | adds | `src.training.train` flags |", "|---|---|---|"]
    L += [f"| {c} | {MECHANISM[c]} | `{FLAGS[c]}` |" for c in configs]
    L += ["", "The ladder is cumulative: each rung keeps everything before it. Contrastive "
          "pre-training therefore enters at **E_full**, and its marginal effect is "
          "E_full − D_adv (see the consecutive-rung table). In this code base the third rung is "
          "`C_aux` (entropy auxiliary features). There is no separate `C_contrastive` "
          "configuration, and the ladder was kept unchanged so every V4 number stays comparable "
          "with the earlier ablation matrices.", "",
          "## Results (mean ± std over seeds)", ""]
    L += (TABLES / "v4_ablation_main.md").read_text(encoding="utf-8").splitlines()
    L += ["", "Δ is the seed-paired mean difference in test accuracy; the CI and p-values come "
          "from a paired t-test (df = 2). Holm correction is applied across the four "
          "comparisons. McNemar counts the seeds whose exact test on the 13,754 test "
          "fragments is significant after Holm correction.", ""]

    # interpretation, generated from the numbers
    L += ["## What the statistics support", ""]
    tests = vs["results"]["test_accuracy"]
    sig_t = [c for c in configs if c != CONTROL and tests.get(f"{c}_vs_{CONTROL}", {}).get("p_holm", 1) < 0.05]
    for c in configs:
        if c == CONTROL:
            continue
        t = tests[f"{c}_vs_{CONTROL}"]
        f1 = vs["results"]["test_macro_f1"][f"{c}_vs_{CONTROL}"]
        s = (f"- **{c}** ({MECHANISM[c]}): test accuracy {fmt(agg[c]['splits']['test']['accuracy'])} "
             f"vs {fmt(agg[CONTROL]['splits']['test']['accuracy'])}; Δ = {t['mean_diff']:+.4f} "
             f"(95% CI {t['ci95'][0]:+.4f} to {t['ci95'][1]:+.4f}, paired p = {t['p']:.3f}, "
             f"Holm {t.get('p_holm', float('nan')):.3f}); Δ macro-F1 = {f1['mean_diff']:+.4f} "
             f"(paired p = {f1['p']:.3f}).")
        if c in mcnemar:
            per = mcnemar[c]
            n_sig = sum(v["p_holm"] < 0.05 for v in per.values())
            better = sum(v["b_only_correct"] > v["a_only_correct"] for v in per.values())
            s += (f" Fragment-level McNemar: significant in {n_sig}/{len(per)} seeds; {c} fixes more "
                  f"fragments than it breaks in {better}/{len(per)} seeds.")
        L.append(s)
    L.append("")
    if not sig_t:
        L.append("No mechanism changes V4 test accuracy by an amount distinguishable from "
                 "seed-to-seed variation at n = 3 seeds (seed-paired t-test, Holm-corrected).")
    else:
        L.append("After Holm correction, the seed-paired t-test finds a test-accuracy difference for: "
                 + ", ".join(sig_t) + ".")
    L += ["With 3 seeds the paired t-test has 2 degrees of freedom, so only large, consistent "
          "differences can reach significance. A non-significant result means *no detectable "
          "effect at this sample size*, not *no effect*. McNemar asks a different, narrower "
          "question: whether two specific trained models make different mistakes on this fixed "
          "test set. It says nothing about the expected effect of the training recipe.", "",
          "## Marginal contribution of each rung (paired t, test accuracy)", "",
          "| step | Δ test accuracy | 95% CI | p | Holm p |", "|---|---|---|---|---|"]
    for name, t in cons["results"]["test_accuracy"].items():
        b, a = name.split("_vs_")
        L.append(f"| {a} → {b} | {t['mean_diff']:+.4f} | [{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] | "
                 f"{t['p']:.3f} | {t.get('p_holm', float('nan')):.3f} |")

    L += ["", "## Per-class F1 on test (mean over seeds)", "",
          "| class | " + " | ".join(configs) + " |", "|---|" + "---|" * len(configs)]
    for c in agg[CONTROL]["splits"]["test"]["labels"]:
        L.append(f"| {c} | " + " | ".join(
            f"{agg[k]['splits']['test']['per_class'][c]['f1']['mean']:.3f}" for k in configs) + " |")

    tv = [m for m in ["accuracy_markup_merged", "accuracy_excl_containers"] if m in agg[CONTROL]["splits"]["test"]]
    if tv:
        L += ["", "## Container-aware views of test accuracy (from the evaluation reports)", "",
              "| config | strict | html+xml merged | excluding pdf/zip fragments |", "|---|---|---|---|"]
        for k in configs:
            e = agg[k]["splits"]["test"]
            L.append(f"| {k} | {fmt(e['accuracy'])} | {fmt(e['accuracy_markup_merged'])} | "
                     f"{fmt(e['accuracy_excl_containers'])} |")

    L += ["", "## Consistency checks", "",
          f"- Accuracy recomputed from each confusion matrix vs the reported value: max |Δ| = "
          f"{summary['consistency_checks']['max_abs_acc_vs_confusion']:.2e}",
          f"- Macro-F1 recomputed from each confusion matrix vs the reported value: max |Δ| = "
          f"{summary['consistency_checks']['max_abs_macro_f1_vs_confusion']:.2e}"]
    if prediction_checks:
        worst = max(v["abs_diff"] for v in prediction_checks.values())
        exact = sum(v["abs_diff"] == 0 for v in prediction_checks.values())
        L.append(f"- Test accuracy re-scored from checkpoints (device: "
                 f"{summary['consistency_checks']['rescoring_device']}) vs reported: {exact}/"
                 f"{len(prediction_checks)} identical, max |Δ| = {worst:.2e}")
    L += ["", "## Files", "",
          "- `ablation_summary.json`: every number above, machine-readable",
          "- `test_predictions.npz`: per-fragment test predictions of every run (the McNemar input)",
          "- `../../tables/v4_*.csv`: main, per-class, per-seed, significance and training-curve tables",
          "- `../../confusion_matrices/v4_*`: pooled confusion matrices (counts, row-normalised) and "
          "long-format per-run counts",
          "- `driver_summary_A_plain.*`, `driver_summary_B_to_E_part{1,2}.*`: raw driver outputs "
          "(part 1/2: the driver was restarted mid-run, see `RUN_PROVENANCE.md`). The A_plain "
          "`.md` header says \"FFT-75/GovDocs1\" because the label was hard-coded at the time; "
          "its numbers are the V4 ones.",
          "- `RUN_PROVENANCE.md`: how and when each B_aug…E_full run was produced, including the "
          "mid-run GPU-memory issue and its fix.", ""]
    (V4 / "ABLATION_RESULTS.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
