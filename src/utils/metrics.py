"""Evaluation metrics for FragGen -- built around the cross-corpus GAP."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

# classes that are information-theoretically hard at small block sizes
HIGH_ENTROPY = ["jpg", "png", "gif", "zip", "encrypted", "pdf"]


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def macro_f1(y_true, y_pred, n_classes):
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)),
                          average="macro", zero_division=0))


def corpus_gap(in_acc: float, cross_acc: float) -> float:
    """The headline metric: how much accuracy is lost off the training corpus."""
    return float(in_acc - cross_acc)


def reject_curve(probs, y_true, steps=21):
    """Risk-coverage curve for the abstain option.
    Returns list of (coverage, selective_accuracy) sorted by increasing coverage."""
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    order = np.argsort(-conf)              # most confident first
    correct = (pred == y_true)[order]
    n = len(y_true)
    out = []
    for cov in np.linspace(1.0 / steps, 1.0, steps):
        k = max(1, int(round(cov * n)))
        out.append((k / n, float(correct[:k].mean())))
    return out


def reject_at_coverage(probs, y_true, target_coverage=0.9):
    """Pick a confidence threshold that keeps ~target_coverage of samples;
    report the accuracy on the kept (non-abstained) subset."""
    probs = np.asarray(probs)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    thr = np.quantile(conf, 1.0 - target_coverage)
    keep = conf >= thr
    cov = float(keep.mean())
    sel_acc = float((pred[keep] == np.asarray(y_true)[keep]).mean()) if keep.any() else float("nan")
    return {"threshold": float(thr), "coverage": cov, "selective_accuracy": sel_acc}


def high_entropy_confusion(y_true, y_pred, classes):
    """Confusion matrix restricted to the look-random types."""
    idx = [classes.index(c) for c in HIGH_ENTROPY if c in classes]
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    m = np.isin(yt, idx)
    cm = confusion_matrix(yt[m], yp[m], labels=idx)
    return {"labels": [classes[i] for i in idx], "matrix": cm.tolist(),
            "accuracy": accuracy(yt[m], yp[m]) if m.any() else float("nan")}


def full_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return {"labels": classes, "matrix": cm.tolist()}


# ---- container-aware scoring -------------------------------------------------
# pdf and zip are *containers*: their interiors are FlateDecode (DEFLATE) streams
# and embedded DCTDecode (JPEG) images, so a mid-file fragment of a PDF holding a
# JPEG genuinely IS jpg bytes. Scoring it wrong measures the labelling, not the
# model. Likewise png's IDAT is zlib -- the same compressor zip uses -- and
# html vs xml is close to arbitrary in a single block. Report the strict number
# alongside these views rather than instead of it.
CONTAINERS = ["pdf", "zip"]
COARSE_MAP = {"txt": "text", "csv": "text", "html": "markup", "xml": "markup",
              "pdf": "doc", "jpg": "image", "png": "image", "gif": "image",
              "zip": "archive", "encrypted": "archive"}
MARKUP_MAP = {"html": "markup", "xml": "markup"}


def _grouped_accuracy(y_true, y_pred, classes, mapping):
    lut = np.array([hash(mapping.get(c, c)) for c in classes])
    return float((lut[np.asarray(y_true)] == lut[np.asarray(y_pred)]).mean())


def taxonomy_views(y_true, y_pred, classes):
    """Accuracy under progressively coarser (and more defensible) label schemes."""
    classes = [str(c) for c in classes]
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    out = {"strict": accuracy(yt, yp),
           "markup_merged": _grouped_accuracy(yt, yp, classes, MARKUP_MAP),
           "coarse_families": _grouped_accuracy(yt, yp, classes, COARSE_MAP)}
    cont_idx = [i for i, c in enumerate(classes) if c in CONTAINERS]
    if cont_idx:
        keep = ~np.isin(yt, cont_idx)
        out["excl_containers"] = (float((yt[keep] == yp[keep]).mean())
                                  if keep.any() else float("nan"))
        out["n_container_fragments"] = int((~keep).sum())
    return out
