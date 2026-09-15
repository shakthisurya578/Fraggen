"""Entropy + statistical features for fragments, and the entropy-ceiling estimator.

Design note (the "entropy-aware" idea): high-entropy types (JPEG, PNG, ZIP,
encrypted) all look like near-random noise, so a naive model over-weights raw
randomness. We (1) expose entropy/statistical aux features so the classifier can
*factor out* the overall randomness level rather than entangle it, and
(2) provide an entropy-ceiling estimate that quantifies how much accuracy is even
theoretically possible from the byte-value marginal alone.
"""
from __future__ import annotations
import numpy as np

_LOG2 = np.log(2.0)


def byte_histograms(X: np.ndarray) -> np.ndarray:
    """Per-fragment normalised byte histogram. X: (N, L) uint8 -> (N, 256) float32."""
    X = np.asarray(X, dtype=np.uint8)
    N, L = X.shape
    counts = np.zeros((N, 256), dtype=np.float64)
    # vectorised bincount per row
    for i in range(N):
        counts[i] = np.bincount(X[i], minlength=256)
    return (counts / max(L, 1)).astype(np.float32)


def shannon_entropy(X: np.ndarray) -> np.ndarray:
    """Byte-level Shannon entropy in bits (0..8) per fragment. X: (N, L) -> (N,)."""
    hist = byte_histograms(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(hist > 0, np.log(hist) / _LOG2, 0.0)
    return (-(hist * logs).sum(axis=1)).astype(np.float32)


def aux_features(X: np.ndarray) -> np.ndarray:
    """Compact statistical descriptor per fragment: (N, 8).

    [entropy/8, mean/255, std/128, printable_ratio, zero_ratio, ff_ratio,
     newline_ratio, mean_abs_delta/255]  -- all roughly in [0,1].
    """
    X = np.asarray(X, dtype=np.uint8).astype(np.int32)
    N, L = X.shape
    ent = shannon_entropy(X) / 8.0
    mean = X.mean(axis=1) / 255.0
    std = X.std(axis=1) / 128.0
    printable = ((X >= 32) & (X < 127)).mean(axis=1)
    zeros = (X == 0).mean(axis=1)
    ffs = (X == 255).mean(axis=1)
    newlines = (X == 10).mean(axis=1)
    delta = np.abs(np.diff(X, axis=1)).mean(axis=1) / 255.0 if L > 1 else np.zeros(N)
    feats = np.stack([ent, mean, std, printable, zeros, ffs, newlines, delta], axis=1)
    return feats.astype(np.float32)


AUX_DIM = 8


def entropy_bucket(X: np.ndarray, n_buckets: int = 8) -> np.ndarray:
    """Discretise entropy into buckets -- the target of the entropy adversary head."""
    ent = shannon_entropy(X)  # 0..8 bits
    edges = np.linspace(0, 8, n_buckets + 1)[1:-1]
    return np.digitize(ent, edges).astype(np.int64)


def byte_marginal_reference(X: np.ndarray, y: np.ndarray, n_classes: int):
    """Accuracy of a plug-in naive-Bayes classifier that sees ONLY the byte
    histogram -- i.e. how far you get from byte *frequencies* alone, ignoring
    the order the bytes appear in.

    NOT an upper bound, despite what this function used to claim. It fits one
    average histogram per class and scores each fragment's histogram against
    them, which is the plug-in decision under a specific (and wrong) generative
    model, not the Bayes-optimal decision over all classifiers. Measured
    FFT-75 numbers make that concrete: this reference scores 0.6889 while a
    logistic regression on the same histogram features reaches 0.7578 and the
    byte-CNN reaches ~0.80. A bound that three classifiers exceed is not a bound.

    What it *is* good for: everything above this line is accuracy the model
    earned from byte ORDER (n-gram structure, markers, periodicity), because
    the marginal alone cannot produce it. That gap is the interesting quantity
    -- for high-entropy types (jpg/png/zip) whose marginals are near-uniform and
    nearly identical, the marginal carries almost no signal, so essentially all
    of their accuracy comes from sequence structure.

    Returns (reference_accuracy, per_class_recall).
    """
    hist = byte_histograms(X)                 # (N, 256)
    logp = np.log(hist + 1e-9)                 # (N, 256)
    # class-conditional mean log-likelihood proxy via per-class average histogram
    class_hist = np.zeros((n_classes, 256))
    for c in range(n_classes):
        m = y == c
        if m.any():
            class_hist[c] = hist[m].mean(axis=0)
    class_logp = np.log(class_hist + 1e-9)     # (C, 256)
    # log-likelihood of each fragment histogram under each class (multinomial-ish)
    scores = hist @ class_logp.T               # (N, C)
    pred = scores.argmax(axis=1)
    acc = float((pred == y).mean())
    per_class = np.array([
        float((pred[y == c] == c).mean()) if (y == c).any() else float("nan")
        for c in range(n_classes)
    ])
    return acc, per_class


# Back-compat alias. The old name asserted an upper bound this quantity does not
# provide; kept so existing reports/scripts keep running, but prefer the new name.
histogram_bayes_ceiling = byte_marginal_reference
