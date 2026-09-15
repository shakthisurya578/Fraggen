"""Byte-embedding 1D-CNN classifier with:
  - contrastive projection head (SimCLR-style pretraining)
  - optional entropy-aware auxiliary features fused into the head
  - optional entropy-adversary head (gradient reversal) so the backbone does not
    encode the raw randomness level -- the "entropy-aware normalisation" idea.
Supports 512 and 4096 block sizes (global pooling makes length flexible).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

AUX_DIM = 8
N_ENTROPY_BUCKETS = 8


def torch_aux_features(x: torch.Tensor) -> torch.Tensor:
    """Mirror of features.entropy.aux_features, computed on a byte tensor.
    x: (B, L) long in [0,255] -> (B, 8) float."""
    xf = x.float()
    B, L = xf.shape
    # byte histogram -> entropy. Accumulated with scatter_add rather than a
    # one_hot: one_hot materialises a (B, L, 256) float tensor (134 MB at
    # B=256, L=512) on every forward pass. This is bit-identical and ~240x faster.
    hist = torch.zeros(B, 256, device=x.device, dtype=xf.dtype)
    hist.scatter_add_(1, x, torch.ones_like(xf))
    hist = hist / L                                    # (B, 256)
    ent = -(hist * torch.log2(hist + 1e-9)).sum(dim=1) / 8.0
    mean = xf.mean(dim=1) / 255.0
    std = xf.std(dim=1) / 128.0
    printable = ((x >= 32) & (x < 127)).float().mean(dim=1)
    zeros = (x == 0).float().mean(dim=1)
    ffs = (x == 255).float().mean(dim=1)
    newlines = (x == 10).float().mean(dim=1)
    delta = (x[:, 1:].float() - x[:, :-1].float()).abs().mean(dim=1) / 255.0
    return torch.stack([ent, mean, std, printable, zeros, ffs, newlines, delta], dim=1)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class ByteCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.use_aux = getattr(m, "use_entropy_aux", True)
        self.use_adv = getattr(m, "use_entropy_adversary", True)
        self.emb = nn.Embedding(256, m.emb_dim)
        chans = list(m.channels)
        layers = []
        in_c = m.emb_dim
        for c in chans:
            layers += [nn.Conv1d(in_c, c, m.kernel_size, padding=m.kernel_size // 2),
                       nn.BatchNorm1d(c), nn.ReLU(), nn.MaxPool1d(2),
                       nn.Dropout(m.dropout)]
            in_c = c
        self.conv = nn.Sequential(*layers)
        self.backbone_dim = chans[-1]

        feat_dim = self.backbone_dim + (AUX_DIM if self.use_aux else 0)
        n_classes = len(cfg.classes)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(m.dropout),
            nn.Linear(256, n_classes))
        self.proj = nn.Sequential(
            nn.Linear(self.backbone_dim, 256), nn.ReLU(),
            nn.Linear(256, m.proj_dim))
        if self.use_adv:
            self.adv = nn.Sequential(
                nn.Linear(self.backbone_dim, 128), nn.ReLU(),
                nn.Linear(128, N_ENTROPY_BUCKETS))

    def backbone(self, x):
        h = self.emb(x).transpose(1, 2)      # (B, emb, L)
        h = self.conv(h)                     # (B, C, L')
        return h.mean(dim=2)                 # (B, C) global average pool

    def forward(self, x, adv_lambda=0.0):
        z = self.backbone(x)                 # (B, C)
        if self.use_aux:
            feat = torch.cat([z, torch_aux_features(x)], dim=1)
        else:
            feat = z
        logits = self.classifier(feat)
        adv_logits = None
        if self.use_adv and adv_lambda > 0:
            adv_logits = self.adv(grad_reverse(z, adv_lambda))
        return logits, z, adv_logits

    def project(self, x):
        return self.proj(self.backbone(x))   # (B, proj_dim) for contrastive
