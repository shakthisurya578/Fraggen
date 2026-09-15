"""NT-Xent (SimCLR) contrastive loss for unlabelled-fragment pretraining."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    """z1, z2: (B, D) projections of two augmented views of the same fragments."""
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2B, D)
    sim = z @ z.t() / temperature                        # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)                    # positive is the other view
    return F.cross_entropy(sim, targets)
