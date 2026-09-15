"""Shared test fixtures: a checkpoint-free classifier with a predictable model."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference import FragGenClassifier  # noqa: E402

CLASSES = ["txt", "csv", "html", "xml", "pdf", "jpg", "png", "gif", "zip"]


class FirstByteModel(torch.nn.Module):
    """Predicts class (first byte of the fragment) % n_classes with a logit
    margin of `margin`, and returns the (logits, z, adv) triple ByteCNN returns."""

    def __init__(self, n_classes: int = len(CLASSES), margin: float = 4.0):
        super().__init__()
        self.n, self.margin = n_classes, margin

    def forward(self, x):
        logits = torch.zeros(x.shape[0], self.n)
        logits[torch.arange(x.shape[0]), x[:, 0] % self.n] = self.margin
        return logits, None, None


def stub_classifier(block_size: int = 16, temperature: float = 1.0) -> FragGenClassifier:
    """A FragGenClassifier wired to FirstByteModel; no checkpoint on disk needed."""
    clf = FragGenClassifier.__new__(FragGenClassifier)
    clf.checkpoint_path = Path("stub.pt")
    clf.device = torch.device("cpu")
    clf.model, clf.cfg = FirstByteModel(), None
    clf.classes = list(CLASSES)
    clf.block_size = block_size
    clf.temperature = temperature
    clf.calibrated = temperature != 1.0
    clf.calibration_file = None
    return clf


def blocks(first_bytes, block_size: int = 16, header: bool = True) -> bytes:
    """Build input bytes: an optional header block, then one block per entry
    whose first byte selects the stub model's predicted class."""
    out = bytearray(b"H" * block_size if header else b"")
    for b in first_bytes:
        out += bytes([b]) + b"\x00" * (block_size - 1)
    return bytes(out)
