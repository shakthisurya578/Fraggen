"""Fragment augmentation.

Two purposes:
  1. Regularise the classifier so it can't latch onto one corpus's encoder quirks
     (the cross-corpus generalisation goal).
  2. Produce two stochastic "views" of a fragment for contrastive pretraining.

These operate on raw uint8 byte arrays. For *source-level* augmentation
(re-encoding images at different JPEG qualities etc.) see data/govdocs.py, which
runs at file-prep time where the original bytes still exist.
"""
from __future__ import annotations
import numpy as np


def augment_fragment(x: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    """Return an augmented copy of a single fragment x: (L,) uint8."""
    L = x.shape[0]
    out = x.astype(np.uint8).copy()

    # cyclic shift: container/stream fragments have no fixed phase
    if getattr(cfg, "cyclic_shift", True):
        s = int(rng.integers(0, L))
        out = np.roll(out, s)

    # byte replacement noise (simulates alternate encoder output / bit rot)
    p_byte = getattr(cfg, "byte_noise_prob", 0.0)
    if p_byte > 0:
        m = rng.random(L) < p_byte
        if m.any():
            out[m] = rng.integers(0, 256, size=int(m.sum()), dtype=np.uint8)

    # per-bit flips
    p_bit = getattr(cfg, "bit_flip_prob", 0.0)
    if p_bit > 0:
        flip = (rng.random((L, 8)) < p_bit)
        if flip.any():
            bits = np.unpackbits(out[:, None], axis=1)
            bits = np.bitwise_xor(bits, flip.astype(np.uint8))
            out = np.packbits(bits, axis=1)[:, 0]

    return out


def contrastive_views(x: np.ndarray, cfg, rng: np.random.Generator):
    """Two augmented views + contiguous masking for SimCLR-style pretraining."""
    v1 = augment_fragment(x, cfg, rng)
    v2 = augment_fragment(x, cfg, rng)
    mp = getattr(cfg, "mask_prob", 0.0)
    if mp > 0:
        v1 = _mask(v1, mp, rng)
        v2 = _mask(v2, mp, rng)
    return v1, v2


def _mask(x: np.ndarray, mask_prob: float, rng: np.random.Generator) -> np.ndarray:
    L = x.shape[0]
    span = max(1, int(L * mask_prob))
    start = int(rng.integers(0, max(1, L - span)))
    out = x.copy()
    out[start:start + span] = 0
    return out


def augment_batch(X: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    """Augment (N, L) in one vectorised pass.

    Same operations and the same distribution as calling augment_fragment on
    each row, but without the per-row Python loop -- contrastive pretraining
    augments every sample twice per epoch, so that loop dominated pretraining.
    Draw order differs from the per-row path, so results are not bit-comparable
    to it for a given seed; each row is still an independent draw.
    """
    X = np.asarray(X, dtype=np.uint8)
    N, L = X.shape
    out = X.copy()

    if getattr(cfg, "cyclic_shift", True):
        # row i rolled right by s_i  ->  out[i, j] = X[i, (j - s_i) % L]
        s = rng.integers(0, L, size=N)
        idx = (np.arange(L)[None, :] - s[:, None]) % L
        out = out[np.arange(N)[:, None], idx]

    p_byte = getattr(cfg, "byte_noise_prob", 0.0)
    if p_byte > 0:
        m = rng.random((N, L)) < p_byte
        n_hit = int(m.sum())
        if n_hit:
            out[m] = rng.integers(0, 256, size=n_hit, dtype=np.uint8)

    p_bit = getattr(cfg, "bit_flip_prob", 0.0)
    if p_bit > 0:
        flip = rng.random((N, L, 8)) < p_bit
        if flip.any():
            bits = np.unpackbits(out[:, :, None], axis=2)      # (N, L, 8)
            bits = np.bitwise_xor(bits, flip.astype(np.uint8))
            out = np.packbits(bits, axis=2)[:, :, 0]
    return out


def _mask_batch(X: np.ndarray, mask_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Contiguous zero-span per row, vectorised."""
    N, L = X.shape
    span = max(1, int(L * mask_prob))
    start = rng.integers(0, max(1, L - span), size=N)
    cols = np.arange(L)[None, :]
    m = (cols >= start[:, None]) & (cols < (start + span)[:, None])
    out = X.copy()
    out[m] = 0
    return out


def contrastive_views_batch(X: np.ndarray, cfg, rng: np.random.Generator):
    """Batched counterpart of contrastive_views for (N, L) inputs."""
    v1 = augment_batch(X, cfg, rng)
    v2 = augment_batch(X, cfg, rng)
    mp = getattr(cfg, "mask_prob", 0.0)
    if mp > 0:
        v1 = _mask_batch(v1, mp, rng)
        v2 = _mask_batch(v2, mp, rng)
    return v1, v2
