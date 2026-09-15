"""Ablation matrix: what does each FragGen mechanism contribute?

Trains a cumulative ladder of models -- each run switches on one more mechanism --
and evaluates every one on the same two held-out sets ("in" and "cross").
Defaults reproduce the Dataset V4 protocol: data/processed_v4, 4096 B, every
training fragment, 6 epochs, seeds 1337-1339 (val.npz = "in", test.npz = "cross";
both are held-out FILES from the training source pool, not a different corpus).
The August cross-corpus study (train FFT-75, test GovDocs1) used this same
driver with the archived FragmentSets -- see the example below.

  A_plain   vanilla byte-CNN, no FragGen mechanisms
  B_aug     + fragment augmentation
  C_aux     + entropy aux features in the head
  D_adv     + entropy adversary (gradient reversal)
  E_full    + contrastive pretraining      == full FragGen

With --seeds, each config is trained several times from different random seeds
and reported as mean +/- std. That matters here: the test sets are big enough
that sampling noise is ~0.0015, but training-seed variance is not, and the
differences between configs are only a few points.

Compare against the torch-free reference in reports/real_baseline.json.

Usage:
  python scripts/run_ablations.py                                  # full V4 ladder
  python scripts/run_ablations.py --only B_aug,C_aux,D_adv,E_full  # reuses A_plain
  # cross-corpus re-run on the archived FFT-75/GovDocs1 FragmentSets:
  python scripts/run_ablations.py --block-size 512 --limit 8000 --seeds 1337 \
      --train-npz archive/data/processed_cross_corpus_fft75_govdocs1/fft75_512_train.npz \
      --in-npz archive/data/processed_cross_corpus_fft75_govdocs1/fft75_512_test.npz \
      --cross-npz archive/data/processed_cross_corpus_fft75_govdocs1/govdocs_512_test.npz \
      --train-label FFT-75 --in-label FFT-75 --cross-label GovDocs1 \
      --baseline reports/ablations/cross_corpus_fft75_govdocs1/real_baseline.json --outdir reports/ablations/cross_corpus_rerun
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed_v4"
PY = sys.executable

# cumulative: each entry disables everything NOT yet switched on
LADDER = [
    ("A_plain", ["--no-aug", "--no-aux", "--no-adv", "--no-contrastive"]),
    ("B_aug",   ["--no-aux", "--no-adv", "--no-contrastive"]),
    ("C_aux",   ["--no-adv", "--no-contrastive"]),
    ("D_adv",   ["--no-contrastive"]),
    ("E_full",  []),
]


def run(cmd, log_path):
    """Stream a subprocess to a log file; raise with a tail on failure."""
    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if p.returncode != 0:
        tail = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise RuntimeError(f"FAILED: {' '.join(cmd)}\n" + "\n".join(tail))


def row_from_report(path, name, seed, train_min=float("nan")):
    r = json.loads(Path(path).read_text())
    c = r["cross_corpus"][0]
    return {"run": name, "seed": seed,
            "in_acc": r["in_corpus"]["accuracy"], "in_f1": r["in_corpus"]["macro_f1"],
            "cross_acc": c["accuracy"], "cross_f1": c["macro_f1"],
            "gap": c["gap_vs_in_corpus"], "train_min": train_min}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="train fragments per class (0 = all)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--config", default="configs/real.yaml")
    ap.add_argument("--only", default=None, help="comma-separated run names to do")
    ap.add_argument("--seeds", default="1337,1338,1339", help="comma-separated training seeds")
    ap.add_argument("--outdir", default="reports/ablations/v4")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if a run's report already exists")
    ap.add_argument("--block-size", type=int, default=4096,
                    help="fragment length; must match the npz files below")
    ap.add_argument("--train-npz", default=str(PROC / "train.npz"))
    ap.add_argument("--in-npz", default=str(PROC / "val.npz"))
    ap.add_argument("--cross-npz", default=str(PROC / "test.npz"))
    # labels only affect the summary.md header (they were hard-coded to
    # FFT-75/GovDocs1 before, which mislabelled the V4 runs)
    ap.add_argument("--train-label", default="Dataset V4 train")
    ap.add_argument("--in-label", default="V4 val (held-out files)")
    ap.add_argument("--cross-label", default="V4 test (held-out files)")
    ap.add_argument("--baseline", default=None,
                    help="optional baseline JSON shown as an extra row "
                         "(e.g. reports/ablations/cross_corpus_fft75_govdocs1/real_baseline.json for FFT-75 runs)")
    args = ap.parse_args()

    for p in (args.train_npz, args.in_npz, args.cross_npz):
        if not Path(p).exists():
            ap.error(f"missing dataset: {p}")

    outdir = ROOT / args.outdir
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    # Namespace checkpoints by results dir. Flat {config}_s{seed}.pt names collide
    # across experiments: a 4096-byte run would silently overwrite the 512-byte
    # model of the same name, leaving checkpoints that no longer match the report
    # sitting beside them.
    ckpt_dir = ROOT / "checkpoints" / outdir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    todo = LADDER
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        todo = [x for x in LADDER if x[0] in want]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows, t_all = [], time.time()
    for name, flags in todo:
        for seed in seeds:
            tag = f"{name}_s{seed}"
            print(f"\n{'='*64}\n  {tag}   flags: {' '.join(flags) or '(all mechanisms on)'}"
                  f"\n{'='*64}")
            ckpt = ckpt_dir / f"{tag}.pt"
            rep = outdir / f"{tag}.json"

            if rep.exists() and not args.force:
                rows.append(row_from_report(rep, name, seed))
                print(f"  [reused existing {rep.name}]  GAP={rows[-1]['gap']:+.4f}")
                emit(rows, outdir, args, seeds)
                continue

            t0 = time.time()
            # -u so the child's per-epoch prints reach the log file as they
            # happen; block-buffered they only appear when the run ends, which
            # makes a long job indistinguishable from a hung one.
            train_cmd = [PY, "-u", "-m", "src.training.train", "all",
                         "--config", args.config,
                         "--train", args.train_npz,
                         "--out", str(ckpt),
                         "--limit", str(args.limit),
                         "--epochs", str(args.epochs),
                         "--seed", str(seed)]
            if args.block_size:
                train_cmd += ["--block-size", str(args.block_size)]
            run(train_cmd + flags, outdir / "logs" / f"{tag}.train.log")
            t_train = (time.time() - t0) / 60
            print(f"  trained in {t_train:.1f} min")

            run([PY, "-u", "-m", "src.training.eval",
                 "--model", str(ckpt),
                 "--in-corpus", args.in_npz,
                 "--cross", args.cross_npz,
                 "--report", str(rep)],
                outdir / "logs" / f"{tag}.eval.log")

            rows.append(row_from_report(rep, name, seed, t_train))
            r = rows[-1]
            print(f"  in={r['in_acc']:.4f}  cross={r['cross_acc']:.4f}  GAP={r['gap']:+.4f}")
            emit(rows, outdir, args, seeds)

    print(f"\nall runs done in {(time.time()-t_all)/60:.1f} min")


def _ms(vals):
    """mean and (sample) std as a display string."""
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s, (f"{m:.4f}" if len(vals) == 1 else f"{m:.4f} ±{s:.4f}")


def emit(rows, outdir, args, seeds):
    (outdir / "summary.json").write_text(json.dumps(
        {"setup": {"limit": args.limit, "epochs": args.epochs, "seeds": seeds},
         "rows": rows}, indent=2))

    base = None
    bp = ROOT / args.baseline if args.baseline else None
    if bp is not None and bp.exists():
        b = json.loads(bp.read_text())["headline"]
        base = (b["in_corpus_accuracy"], b["cross_corpus_accuracy"], b["gap"])

    order = [n for n, _ in LADDER]
    groups = {}
    for r in rows:
        groups.setdefault(r["run"], []).append(r)

    # --limit 0 means "use everything"; printing a literal 0 reads as "trained
    # on nothing" in a report table
    lim = "all available" if not args.limit else f"{args.limit:,}"
    L = ["# FragGen ablation matrix", "",
         f"Train: {args.train_label}, {lim} fragments/class, {args.epochs} epochs, "
         f"seeds {seeds}. In-corpus test: {args.in_label}. "
         f"Cross-corpus test: {args.cross_label}.", ""]

    multi = any(len(v) > 1 for v in groups.values())
    L += [("| run | n | in-corpus acc | cross acc | cross macro-F1 | **GAP** |"
           if multi else
           "| run | in-corpus acc | macro-F1 | cross acc | macro-F1 | **GAP** |"),
          "|---|---|---|---|---|---|"]
    if base:
        L.append(f"| _baseline(logreg)_ | {'1 | ' if multi else ''}{base[0]:.4f} | "
                 f"{base[1]:.4f} | {'- | ' if multi else '- | '}**{base[2]:+.4f}** |"
                 if multi else
                 f"| _baseline(logreg)_ | {base[0]:.4f} | - | {base[1]:.4f} | - | "
                 f"**{base[2]:+.4f}** |")
    for name in order:
        g = groups.get(name)
        if not g:
            continue
        if multi:
            _, _, ia = _ms([r["in_acc"] for r in g])
            _, _, ca = _ms([r["cross_acc"] for r in g])
            _, _, cf = _ms([r["cross_f1"] for r in g])
            gm, gs, gg = _ms([r["gap"] for r in g])
            L.append(f"| {name} | {len(g)} | {ia} | {ca} | {cf} | **{gg}** |")
        else:
            r = g[0]
            L.append(f"| {name} | {r['in_acc']:.4f} | {r['in_f1']:.4f} | "
                     f"{r['cross_acc']:.4f} | {r['cross_f1']:.4f} | **{r['gap']:+.4f}** |")
    L += ["", "Lower GAP is better; in-corpus accuracy should not collapse to get it."]

    if multi:
        L += ["", "## Per-seed detail", "",
              "| run | seed | in acc | cross acc | cross F1 | GAP |", "|---|---|---|---|---|---|"]
        for name in order:
            for r in groups.get(name, []):
                L.append(f"| {name} | {r['seed']} | {r['in_acc']:.4f} | "
                         f"{r['cross_acc']:.4f} | {r['cross_f1']:.4f} | {r['gap']:+.4f} |")
    (outdir / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
