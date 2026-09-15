"""Checkpoint loading, shared by evaluation, calibration and inference.

``load_model`` was moved here unchanged from the former ``fraggen/eval.py`` so
that inference code (``src.inference``) does not have to import the training/
evaluation module. ``src.training.eval`` re-exports it for older callers.

A checkpoint is a dict ``{"state_dict": ..., "config": {...}}`` written by
``src.training.train.save_model``. The stored config (not configs/*.yaml) is
what rebuilds the architecture, so a checkpoint always loads into exactly the
network it was trained as.
"""
from __future__ import annotations
import torch

from ..utils.config import load_config
from .cnn import ByteCNN


def load_model(path, device):
    """Rebuild the ByteCNN described by the checkpoint's own config, load its
    weights, and return ``(model, cfg)`` with the model already in eval mode."""
    # weights_only=False: the checkpoint carries its config dict alongside the
    # tensors, and we wrote this file ourselves
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = load_config()
    cfg._raw = ckpt["config"]
    for k, v in ckpt["config"].items():
        setattr(cfg, k, _ns(v))
    model = ByteCNN(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    # a loaded checkpoint is for inference: hand it back in eval mode so callers
    # can't accidentally predict with dropout/BatchNorm still in training mode
    model.eval()
    return model, cfg


def _ns(v):
    """Recursively turn the checkpoint's nested config dict into attribute access."""
    from types import SimpleNamespace
    if isinstance(v, dict):
        return SimpleNamespace(**{k: _ns(x) for k, x in v.items()})
    if isinstance(v, list):
        return [_ns(x) for x in v]
    return v
