#!/usr/bin/env bash
# Unattended pipeline, ordered most-valuable-first so that whatever finishes is
# the most useful subset. Every stage is resumable: the driver reuses any run
# whose JSON report already exists, so re-running this picks up where it stopped.
#
# Needs no network: torch+CUDA installed, both corpora already prepped on disk.
#
# Sizing note: at 4096 bytes a run takes ~40 min on this card (batch 256 puts
# VRAM at ~96% of 6 GB), not the ~17 min an isolated benchmark suggested. So the
# block-size comparison runs at 4,000 fragments/class rather than 8,000, and is
# repeated at 512 with the SAME setting -- otherwise "4096 is better" would be
# confounded with "trained on more data".
set -u

cd "F:/sem 7/CSE4074 - Forensics Analytics/Project/fraggen" || exit 1
PY='C:/Users/shakthisurya/AppData/Local/Programs/Python/Python311/python.exe'

stamp() { date "+%H:%M:%S"; }
banner() { echo; echo "=================================================="; \
           echo "  $1   [$(stamp)]"; echo "=================================================="; }

# ---------------------------------------------------------------- stage 1
# The primary question: do the mechanisms do anything? At n=3 the minimum
# detectable difference was 0.0395 and every observed effect was smaller, so
# n=3 could not answer it. 12 seeds resolves ~0.02.
banner "STAGE 1/5  512B, 5 configs x 12 seeds  (statistical power)"
"$PY" -u scripts/run_ablations.py \
    --limit 8000 --epochs 6 \
    --seeds 1337,1338,1339,1340,1341,1342,1343,1344,1345,1346,1347,1348 \
    --outdir reports/ablations_gpu
echo "stage 1 exit=$?"

banner "STAGE 2/5  significance tests"
"$PY" -u scripts/significance.py --results reports/ablations_gpu \
    | tee reports/ablations_gpu/significance.txt
echo "stage 2 exit=$?"

# ---------------------------------------------------------------- stage 3
# Block-size comparison, matched settings on both sides.
banner "STAGE 3/5  512B @ 4000/class, 2 seeds  (block-size control arm)"
"$PY" -u scripts/run_ablations.py \
    --limit 4000 --epochs 6 --seeds 1337,1338 \
    --outdir reports/blocksize_512
echo "stage 3 exit=$?"

banner "STAGE 4/5  4096B @ 4000/class, 2 seeds  (~40 min/run, writes as it goes)"
"$PY" -u scripts/run_ablations.py \
    --limit 4000 --epochs 6 --seeds 1337,1338 \
    --block-size 4096 \
    --train-npz data/processed/fft75_4096_train.npz \
    --in-npz    data/processed/fft75_4096_test.npz \
    --cross-npz data/processed/govdocs_4096_test.npz \
    --outdir reports/blocksize_4096
echo "stage 4 exit=$?"

# ---------------------------------------------------------------- stage 5
banner "STAGE 5/5  figures + error analysis"
"$PY" -u scripts/make_figures.py --results reports/ablations_gpu \
    --results-4k reports/blocksize_4096 --focus D_adv_s1337
for r in reports/ablations_gpu/D_adv_s1337.json reports/blocksize_4096/A_plain_s1337.json; do
  [ -f "$r" ] && { echo "--- $r ---"; "$PY" -u scripts/error_analysis.py "$r"; }
done
echo "stage 5 exit=$?"

banner "PIPELINE COMPLETE"
for f in reports/ablations_gpu/summary.md reports/blocksize_512/summary.md \
         reports/blocksize_4096/summary.md; do
  [ -f "$f" ] && { echo; echo "### $f"; cat "$f"; }
done
