"""Common fragment container + torch Dataset + npz IO."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class FragmentSet:
    """A labelled set of fragments from one corpus."""
    X: np.ndarray            # (N, L) uint8
    y: np.ndarray            # (N,) int64
    classes: list            # label names, index = class id
    corpus: str = "unknown"
    file_ids: np.ndarray = None  # (N,) source-file id per fragment, for
                                 # reassembly eval (DFRWS). None when the
                                 # corpus has no per-file provenance.

    def __len__(self):
        return self.X.shape[0]

    @property
    def block_size(self):
        return self.X.shape[1]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.file_ids is None else {"file_ids": np.asarray(self.file_ids)}
        np.savez_compressed(path, X=self.X.astype(np.uint8), y=self.y.astype(np.int64),
                            classes=np.array(self.classes), corpus=self.corpus, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        corpus = d["corpus"].item() if "corpus" in d else "unknown"
        # absent in every npz written before file_ids existed -- keep loading those
        return cls(X=d["X"], y=d["y"].astype(np.int64),
                   classes=list(d["classes"]), corpus=str(corpus),
                   file_ids=d["file_ids"] if "file_ids" in d else None)

    def subset(self, idx):
        fids = None if self.file_ids is None else self.file_ids[idx]
        return FragmentSet(self.X[idx], self.y[idx], self.classes, self.corpus, fids)


def train_val_split(fs: FragmentSet, val_frac: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(fs)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return fs.subset(tr_idx), fs.subset(val_idx)


# ---- torch glue (imported lazily so non-torch tooling still works) ----
def torch_dataset(fs: FragmentSet, cfg=None, augment: bool = False, seed: int = 0):
    import torch
    from torch.utils.data import Dataset
    from .augment import augment_fragment

    class _DS(Dataset):
        def __init__(self):
            self.X = fs.X
            self.y = fs.y
            self.aug = augment
            self.acfg = getattr(cfg, "augment", None) if cfg else None
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return self.X.shape[0]

        def __getitem__(self, i):
            x = self.X[i]
            if self.aug and self.acfg is not None:
                x = augment_fragment(x, self.acfg, self.rng)
            return torch.from_numpy(x.astype(np.int64)), int(self.y[i])

    return _DS()
