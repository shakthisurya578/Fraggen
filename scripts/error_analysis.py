"""Where does cross-corpus accuracy actually go?

Splits the confusion matrix into errors that are *semantically defensible* and
errors that are genuine failure, then reports what accuracy would be under
coarser label taxonomies. The point: a 512-byte chunk from the middle of a PDF
that holds an embedded JPEG stream IS jpg bytes, and calling it "pdf" is a
labelling artefact of container formats, not a model mistake. Same for zip.
And html vs xml is a near-arbitrary split at 512 bytes.

Usage:
  python scripts/error_analysis.py reports/ablations/A_plain_s1337.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

# containers whose interior legitimately contains other formats' bytes
CONTAINERS = {"pdf", "zip"}
# what a container's fragments plausibly decode to
CONTAINER_INTERIOR = {"jpg", "png", "gif", "zip", "pdf", "txt"}
# groupings for coarse-taxonomy scoring
COARSE = {"txt": "text", "csv": "text", "html": "markup", "xml": "markup",
          "pdf": "doc", "jpg": "image", "png": "image", "gif": "image",
          "zip": "archive"}
MERGE_MARKUP = {"html": "markup", "xml": "markup"}


def regroup(cm, classes, mapping):
    """Collapse a confusion matrix under a class -> group mapping."""
    groups = sorted(set(mapping.get(c, c) for c in classes))
    gi = {g: i for i, g in enumerate(groups)}
    out = np.zeros((len(groups), len(groups)), dtype=np.int64)
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            out[gi[mapping.get(ci, ci)], gi[mapping.get(cj, cj)]] += cm[i, j]
    return out, groups


def acc(cm):
    return float(np.trace(cm) / max(cm.sum(), 1))


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1
                else "reports/ablations/v4/A_plain_s1337.json")
    r = json.loads(path.read_text())
    sec = r["cross_corpus"][0]
    cm = np.array(sec["confusion"]["matrix"], dtype=np.int64)
    classes = [str(c) for c in sec["confusion"]["labels"]]
    idx = {c: i for i, c in enumerate(classes)}
    total = cm.sum()

    print(f"=== {path.name}  corpus={sec['corpus']} ===")
    print(f"as-labelled accuracy : {acc(cm):.4f}   (n={total:,})\n")

    # ---- biggest confusions ----
    print("top 10 confusions (true -> predicted):")
    errs = [(cm[i, j], classes[i], classes[j])
            for i in range(len(classes)) for j in range(len(classes)) if i != j]
    for n, t, p in sorted(errs, reverse=True)[:10]:
        print(f"  {t:6s} -> {p:6s} {n:>7,}  ({n/total:5.2%} of all fragments)")

    # ---- how much error is container-interior? ----
    cont_err = sum(cm[idx[c], idx[p]]
                   for c in CONTAINERS if c in idx
                   for p in CONTAINER_INTERIOR if p in idx and p != c)
    markup_err = 0
    if "xml" in idx and "html" in idx:
        markup_err = cm[idx["xml"], idx["html"]] + cm[idx["html"], idx["xml"]]
    tot_err = total - np.trace(cm)
    print(f"\ntotal errors            : {tot_err:,}")
    print(f"  container interior    : {cont_err:,}  ({cont_err/tot_err:5.1%} of errors)")
    print(f"    (pdf/zip fragments called jpg/png/gif/zip/pdf/txt)")
    print(f"  html<->xml            : {markup_err:,}  ({markup_err/tot_err:5.1%} of errors)")
    print(f"  everything else       : {tot_err-cont_err-markup_err:,} "
          f"({(tot_err-cont_err-markup_err)/tot_err:5.1%} of errors)")

    # ---- accuracy under coarser taxonomies ----
    print("\naccuracy under alternative label schemes:")
    print(f"  as-is (9 classes)          : {acc(cm):.4f}")
    m, g = regroup(cm, classes, MERGE_MARKUP)
    print(f"  html+xml merged (8)        : {acc(m):.4f}   groups={g}")
    m2, g2 = regroup(cm, classes, COARSE)
    print(f"  coarse families ({len(g2)})         : {acc(m2):.4f}   groups={g2}")

    # Excluding fragments whose TRUE label is a container. Filter on rows only:
    # taking the cm[keep, keep] submatrix would also drop non-container fragments
    # that were *predicted* as pdf/zip (gif->pdf, gif->zip, ...), removing real
    # errors from the denominator and inflating the score.
    cont_rows = [idx[c] for c in CONTAINERS if c in idx]
    keep = [i for i in range(len(classes)) if i not in cont_rows]
    sub = cm[keep, :]                      # all predictions kept, container rows dropped
    correct = sum(cm[i, i] for i in keep)
    print(f"  excl. container fragments  : {correct/max(sub.sum(),1):.4f}   "
          f"(drops {cm[cont_rows].sum():,} pdf/zip fragments)")

    # ---- per-class recall, worst first ----
    print("\nper-class recall (cross-corpus), worst first:")
    rec = [(cm[i].sum() and cm[i, i] / cm[i].sum(), classes[i], cm[i].sum())
           for i in range(len(classes))]
    for v, c, n in sorted(rec):
        # where does this class actually go?
        i = idx[c]
        top = np.argsort(-cm[i])
        goes = ", ".join(f"{classes[j]} {cm[i,j]/max(cm[i].sum(),1):.0%}"
                         for j in top[:3] if j != i and cm[i, j] > 0)
        print(f"  {c:6s} {v:.3f}  (n={n:,})   mostly -> {goes}")


if __name__ == "__main__":
    raise SystemExit(main())
