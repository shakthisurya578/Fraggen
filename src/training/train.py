"""Training: contrastive pretraining + classifier fine-tuning with entropy adversary.

CLI examples:
  python -m src.training.train pretrain --train data/train.npz --out checkpoints/enc.pt
  python -m src.training.train train    --train data/train.npz --init checkpoints/enc.pt \
                                    --out checkpoints/model.pt
  python -m src.training.train all       --train data/train.npz --out checkpoints/model.pt
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.config import load_config, config_to_dict
from ..preprocessing.fragments import FragmentSet, train_val_split, torch_dataset
from ..preprocessing.augment import contrastive_views_batch
from ..models.cnn import ByteCNN, N_ENTROPY_BUCKETS
from ..models.contrastive import nt_xent
from ..preprocessing.entropy import entropy_bucket


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- contrastive pretraining ----------------
def pretrain_contrastive(model, fs: FragmentSet, cfg, device=None):
    if not cfg.contrastive.enabled:
        return model
    device = device or _device()
    model.to(device).train()
    opt = torch.optim.Adam(
        list(model.emb.parameters()) + list(model.conv.parameters()) +
        list(model.proj.parameters()), lr=cfg.contrastive.lr)
    rng = np.random.default_rng(cfg.seed)
    X = fs.X
    bs = cfg.contrastive.batch_size
    for ep in range(cfg.contrastive.epochs):
        perm = rng.permutation(len(X))
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            a, b = contrastive_views_batch(X[idx], cfg.augment, rng)
            v1 = torch.from_numpy(a.astype(np.int64)).to(device)
            v2 = torch.from_numpy(b.astype(np.int64)).to(device)
            z1, z2 = model.project(v1), model.project(v2)
            loss = nt_xent(z1, z2, cfg.contrastive.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        print(f"[pretrain] epoch {ep+1}/{cfg.contrastive.epochs} loss={tot/len(X):.4f}")
    return model


# ---------------- supervised fine-tuning ----------------
def train_classifier(model, fs: FragmentSet, cfg, device=None):
    device = device or _device()
    model.to(device)
    tr, val = train_val_split(fs, cfg.train.val_frac, cfg.seed)
    dl = DataLoader(torch_dataset(tr, cfg, augment=True, seed=cfg.seed),
                    batch_size=cfg.train.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    ce = torch.nn.CrossEntropyLoss()
    lam = cfg.train.adversary_lambda if model.use_adv else 0.0
    best = {"val_acc": -1.0}
    for ep in range(cfg.train.epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits, _, adv = model(xb, adv_lambda=lam)
            loss = ce(logits, yb)
            if adv is not None:
                eb = torch.from_numpy(entropy_bucket(xb.cpu().numpy(),
                                      N_ENTROPY_BUCKETS)).to(device)
                loss = loss + ce(adv, eb)
            opt.zero_grad(); loss.backward(); opt.step()
        va = _quick_acc(model, val, device)
        print(f"[train] epoch {ep+1}/{cfg.train.epochs} val_acc={va:.4f}")
        if va > best["val_acc"]:
            best = {"val_acc": va, "state": {k: v.cpu().clone()
                                             for k, v in model.state_dict().items()}}
    if "state" in best:
        model.load_state_dict(best["state"])
    return model


@torch.no_grad()
def _quick_acc(model, fs, device, batch=1024):
    model.eval()
    # cast per batch, not up front: .astype(int64) on the whole split is an 8x
    # blow-up (a 45k x 512 val split alone is 184 MB, the full set 1.8 GB)
    preds = []
    for i in range(0, len(fs.X), batch):
        xb = torch.from_numpy(fs.X[i:i + batch].astype(np.int64)).to(device)
        logits, _, _ = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    return float((pred == fs.y).mean())


def build_model(cfg):
    return ByteCNN(cfg)


def save_model(model, cfg, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config_to_dict(cfg)}, path)
    print("saved", path)


def subsample_per_class(fs: FragmentSet, per_class: int, seed=0) -> FragmentSet:
    """Balanced subsample -- lets the ablation matrix run in minutes, not hours."""
    if not per_class:
        return fs
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(len(fs.classes)):
        idx = np.flatnonzero(fs.y == c)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        keep.append(idx)
    return fs.subset(np.sort(np.concatenate(keep)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pretrain", "train", "all"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--train", required=True, help="training FragmentSet .npz")
    ap.add_argument("--init", default=None, help="encoder checkpoint to warm-start")
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="fragments per class to train on (0 = all)")
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg.seed (init, subsample, split, augmentation)")
    # ---- ablation switches: each disables one FragGen mechanism ----
    ap.add_argument("--no-aug", action="store_true",
                    help="disable fragment augmentation")
    ap.add_argument("--no-adv", action="store_true",
                    help="disable the entropy-adversary (gradient reversal) head")
    ap.add_argument("--no-contrastive", action="store_true",
                    help="skip contrastive pretraining")
    ap.add_argument("--no-aux", action="store_true",
                    help="disable entropy aux features in the classifier head")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.block_size:
        cfg.block_size = args.block_size
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.no_aug:
        cfg.augment.bit_flip_prob = 0.0
        cfg.augment.byte_noise_prob = 0.0
        cfg.augment.mask_prob = 0.0
        cfg.augment.cyclic_shift = False
    if args.no_adv:
        cfg.model.use_entropy_adversary = False
    if args.no_aux:
        cfg.model.use_entropy_aux = False
    if args.no_contrastive:
        cfg.contrastive.enabled = False

    set_seed(cfg.seed)
    fs = FragmentSet.load(args.train)
    if args.limit:
        fs = subsample_per_class(fs, args.limit, cfg.seed)
    # plain str, not numpy.str_ from the npz: keeps the checkpoint free of numpy
    # objects so torch.load can read it under weights_only=True (torch>=2.6)
    cfg.classes = [str(c) for c in fs.classes]
    print(f"[setup] n_train={len(fs):,}  classes={len(cfg.classes)}  "
          f"aug={not args.no_aug}  adv={not args.no_adv}  "
          f"aux={not args.no_aux}  contrastive={cfg.contrastive.enabled}")
    model = build_model(cfg)
    if args.init:
        model.load_state_dict(torch.load(args.init)["state_dict"], strict=False)

    if args.mode in ("pretrain", "all"):
        model = pretrain_contrastive(model, fs, cfg)
    if args.mode in ("train", "all"):
        model = train_classifier(model, fs, cfg)
    save_model(model, cfg, args.out)


if __name__ == "__main__":
    main()
