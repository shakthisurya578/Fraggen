"""Dataset V2 FragmentSet container with full provenance metadata.

Preserves for every fragment:
  - X: (N, 4096) uint8
  - y: (N,) int64
  - classes: (9,) target classes
  - file_ids: (N,) source file string identifiers
  - offsets: (N,) byte offset of fragment within source file / disk
  - sources: (N,) source corpus name ('govdocs1', 'dfrws', 'fft75')
  - split: (N,) assigned split ('train', 'val', 'test')
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np


@dataclass
class FragmentSetV2:
    """A collection of 4096-byte fragments with file-level provenance."""
    X: np.ndarray             # (N, L) uint8
    y: np.ndarray             # (N,) int64
    classes: list[str]        # 9 target class names
    file_ids: np.ndarray      # (N,) string array of source file IDs
    offsets: np.ndarray       # (N,) int64 array of byte offsets
    sources: np.ndarray       # (N,) string array of source corpora
    split: np.ndarray = None  # (N,) string array of split names ('train', 'val', 'test')

    def __len__(self) -> int:
        return self.X.shape[0]

    @property
    def block_size(self) -> int:
        return self.X.shape[1] if self.X.ndim > 1 else 0

    def subset(self, idx: Sequence[int] | np.ndarray) -> FragmentSetV2:
        idx = np.asarray(idx)
        sp = None if self.split is None else self.split[idx]
        return FragmentSetV2(
            X=self.X[idx],
            y=self.y[idx],
            classes=list(self.classes),
            file_ids=self.file_ids[idx],
            offsets=self.offsets[idx],
            sources=self.sources[idx],
            split=sp,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.split is not None:
            extra["split"] = np.asarray(self.split, dtype=object)
        np.savez_compressed(
            path,
            X=self.X.astype(np.uint8),
            y=self.y.astype(np.int64),
            classes=np.array([str(c) for c in self.classes]),
            file_ids=np.asarray(self.file_ids, dtype=object),
            offsets=self.offsets.astype(np.int64),
            sources=np.asarray(self.sources, dtype=object),
            **extra,
        )

    @classmethod
    def load(cls, path: str | Path) -> FragmentSetV2:
        d = np.load(path, allow_pickle=True)
        classes = [str(c) for c in d["classes"]]
        sp = d["split"] if "split" in d else None
        return cls(
            X=d["X"],
            y=d["y"].astype(np.int64),
            classes=classes,
            file_ids=d["file_ids"],
            offsets=d["offsets"].astype(np.int64),
            sources=d["sources"],
            split=sp,
        )
