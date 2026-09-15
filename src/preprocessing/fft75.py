"""FFT-75 (FiFTy) loader -> FragmentSet on FragGen's target label space.

FFT-75 ships labelled 512B/4096B fragments for 75 file types as .npz arrays.
The label INDEX ordering depends on the FiFTy scenario you downloaded, so we map
by *name*: you provide the ordered list of class names for your download (see
DATASETS.md), and we keep only the types that exist in the target label set
(configs/default.yaml -> classes), remapping the rest away.
"""
from __future__ import annotations
import numpy as np
from .fragments import FragmentSet

# Extension/type aliases -> FragGen target classes. Extend as needed.
ALIAS = {
    "jpg": "jpg", "jpeg": "jpg",
    "png": "png", "gif": "gif",
    "zip": "zip", "z": "zip", "gz": "zip", "bz2": "zip", "rar": "zip", "7z": "zip",
    "pdf": "pdf",
    "txt": "txt", "text": "txt", "log": "txt",
    "csv": "csv", "tsv": "csv",
    "html": "html", "htm": "html",
    "xml": "xml",
    "encrypted": "encrypted", "aes": "encrypted", "gpg": "encrypted", "rand": "encrypted",
}


def _detect_keys(d):
    for kx, ky in [("X", "y"), ("x", "y"), ("data", "labels"), ("arr_0", "arr_1")]:
        if kx in d and ky in d:
            return kx, ky
    raise KeyError(f"could not find (X,y) in npz keys: {list(d.keys())}")


def load_fft75(npz_path, src_class_names, target_classes) -> FragmentSet:
    """npz_path: a FiFTy .npz with X (N,L) uint8 and y (N,) int.
    src_class_names: ordered names for the label indices in this file.
    target_classes: FragGen target label list."""
    d = np.load(npz_path, allow_pickle=True)
    kx, ky = _detect_keys(d)
    X = d[kx].astype(np.uint8)
    y_src = d[ky].astype(np.int64).ravel()
    return remap_by_name(X, y_src, src_class_names, target_classes, corpus="fft75")


def remap_by_name(X, y_src, src_class_names, target_classes, corpus="fft75"):
    tgt_index = {c: i for i, c in enumerate(target_classes)}
    # source id -> target id (or -1 to drop)
    lut = np.full(len(src_class_names), -1, dtype=np.int64)
    for si, name in enumerate(src_class_names):
        canon = ALIAS.get(str(name).lower(), str(name).lower())
        if canon in tgt_index:
            lut[si] = tgt_index[canon]
    mapped = lut[y_src]
    keep = mapped >= 0
    return FragmentSet(X[keep], mapped[keep], list(target_classes), corpus=corpus)
