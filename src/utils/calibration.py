"""Confidence calibration, so a reported score means what it says.

The raw softmax of a trained CNN is not a probability. Measured on GovDocs1 the
model states 0.845 and is right 0.575 of the time -- an examiner quoting that
number would overstate the evidence by 27 points.

Temperature scaling (Guo et al., 2017) fixes this with a single learned scalar T:

    p = softmax(logits / T)

T > 1 softens over-confident distributions. It is fitted by minimising negative
log-likelihood on a held-out split, and because it is monotonic in the logits it
*cannot change any prediction* -- accuracy, precision and recall are identical
before and after. Only the reported confidence changes, which is exactly the
property you want for evidence: the calibration step cannot be accused of
altering what the tool found.

Forensic caveat, measured in scripts/calibrate.py: a temperature fitted on the
training corpus does not necessarily transfer to a different corpus. Report the
cross-corpus ECE, not just the in-corpus one.
"""
from __future__ import annotations
import numpy as np


def expected_calibration_error(conf, correct, n_bins=15):
    """Weighted average |confidence - accuracy| over equal-width bins."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def fit_temperature(logits, labels, lr=0.02, max_iter=300):
    """Learn a single scalar T minimising NLL on a held-out split.

    Optimised in log-space so T stays positive without a constraint.
    """
    import torch
    lg = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)          # T = exp(0) = 1
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T):
    """Softmax of logits / T. Monotonic -> argmax, and every ranking, unchanged."""
    z = np.asarray(logits, dtype=np.float64) / float(T)
    z = z - z.max(axis=1, keepdims=True)                # stable softmax
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_table(conf, correct, edges=(0.0, .3, .4, .5, .6, .7, .8, .9, .95, 1.001)):
    """Rows of (stated confidence, empirical accuracy, n) for a report."""
    conf, correct = np.asarray(conf), np.asarray(correct)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 30:
            continue
        rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows
