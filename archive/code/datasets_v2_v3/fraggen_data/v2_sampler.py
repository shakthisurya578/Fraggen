"""Corpus-balanced training sampler and index generator for Dataset V2.

Provides balanced sampling across heterogeneous forensic corpora:
  - FFT-75: 55-60% target (default 57.5%)
  - GovDocs1: 40-45% target (default 42.5%)
  - DFRWS: 0% in training (reserved strictly for evaluation)
  - Hard cap: No corpus exceeds 60% of training fragments

Prevents model bias towards synthetic/in-corpus artifacts by rebalancing
the dominant FFT-75 corpus with realistic GovDocs1 document fragments.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence, Iterator, Optional
import numpy as np
import torch
from torch.utils.data import Sampler


def build_corpus_balanced_index(
    sources: np.ndarray | Sequence[str],
    y: Optional[np.ndarray | Sequence[int]] = None,
    target_fft_ratio: float = 0.575,
    target_govdocs_ratio: float = 0.425,
    max_corpus_ratio: float = 0.60,
    strategy: str = "stratified",
    seed: int = 42,
) -> np.ndarray:
    """Build a balanced training index array into train.npz.

    Parameters
    ----------
    sources : np.ndarray
        Array of corpus source strings ('govdocs1', 'fft75', etc.)
    y : np.ndarray, optional
        Array of integer class labels, used when strategy='stratified'
    target_fft_ratio : float
        Target fraction for FFT-75 (e.g. 0.575 = 57.5%)
    target_govdocs_ratio : float
        Target fraction for GovDocs1 (e.g. 0.425 = 42.5%)
    max_corpus_ratio : float
        Upper bound on any single corpus contribution (default 0.60)
    strategy : str
        'stratified' (class-stratified in FFT-75) or 'uniform'
    seed : int
        RNG seed for reproducibility

    Returns
    -------
    np.ndarray
        1D int64 array of sorted unique indices into train.npz
    """
    sources = np.asarray(sources)
    if y is not None:
        y = np.asarray(y)

    # 1. Identify indices by corpus
    gov_mask = (sources == "govdocs1")
    fft_mask = (sources == "fft75")
    dfrws_mask = (sources == "dfrws")

    n_dfrws = int(np.sum(dfrws_mask))
    if n_dfrws > 0:
        raise ValueError(
            f"DFRWS fragments detected in training split ({n_dfrws}). "
            "DFRWS must be evaluation only!"
        )

    gov_indices = np.where(gov_mask)[0]
    fft_indices = np.where(fft_mask)[0]

    n_gov = len(gov_indices)
    n_fft_avail = len(fft_indices)

    if n_gov == 0:
        raise ValueError("No GovDocs1 fragments found in sources!")

    # 2. Compute target sample size for FFT-75
    # Since GovDocs1 is the smaller corpus (125k vs 1.14M), we retain 100% of GovDocs1
    # and subsample FFT-75 to achieve the desired target ratio.
    # n_fft / (n_fft + n_gov) = target_fft_ratio
    # n_fft = n_gov * (target_fft_ratio / target_govdocs_ratio)
    target_fft = int(round(n_gov * (target_fft_ratio / target_govdocs_ratio)))

    if target_fft > n_fft_avail:
        raise ValueError(
            f"Requested {target_fft} FFT-75 fragments, but only {n_fft_avail} available."
        )

    total_sampled = n_gov + target_fft
    actual_fft_ratio = target_fft / total_sampled
    actual_gov_ratio = n_gov / total_sampled

    if actual_fft_ratio > max_corpus_ratio:
        raise ValueError(
            f"FFT-75 ratio {actual_fft_ratio:.4f} exceeds max cap {max_corpus_ratio:.4f}"
        )
    if actual_gov_ratio > max_corpus_ratio:
        raise ValueError(
            f"GovDocs1 ratio {actual_gov_ratio:.4f} exceeds max cap {max_corpus_ratio:.4f}"
        )

    rng = np.random.default_rng(seed)

    # 3. Sample FFT-75 indices
    if strategy == "stratified" and y is not None:
        # Class-stratified subsampling of FFT-75
        fft_by_cls = defaultdict(list)
        for idx in fft_indices:
            fft_by_cls[y[idx]].append(idx)

        sampled_fft = []
        for cls_label in sorted(fft_by_cls.keys()):
            cls_indices = fft_by_cls[cls_label]
            cls_quota = int(round(target_fft * len(cls_indices) / n_fft_avail))
            chosen = rng.choice(cls_indices, size=cls_quota, replace=False)
            sampled_fft.extend(chosen)

        # Handle any 1-2 fragment rounding discrepancies
        if len(sampled_fft) < target_fft:
            needed = target_fft - len(sampled_fft)
            pool = list(set(fft_indices) - set(sampled_fft))
            sampled_fft.extend(rng.choice(pool, size=needed, replace=False))
        elif len(sampled_fft) > target_fft:
            sampled_fft = sampled_fft[:target_fft]
        sampled_fft = np.array(sampled_fft, dtype=np.int64)
    elif strategy == "uniform" or y is None:
        sampled_fft = rng.choice(fft_indices, size=target_fft, replace=False).astype(np.int64)
    else:
        raise ValueError(f"Unknown strategy: '{strategy}'. Choose 'stratified' or 'uniform'.")

    # 4. Merge GovDocs1 and sampled FFT-75 indices
    balanced_indices = np.sort(np.concatenate([gov_indices, sampled_fft]).astype(np.int64))
    return balanced_indices


class CorpusBalancedSampler(Sampler[int]):
    """PyTorch Sampler that draws batches with corpus balancing.

    Can be initialized either from a precomputed index array (e.g. from
    `train_index.npy`) or dynamically from corpus source arrays.
    """

    def __init__(
        self,
        indices: Optional[np.ndarray | Sequence[int]] = None,
        sources: Optional[np.ndarray | Sequence[str]] = None,
        y: Optional[np.ndarray | Sequence[int]] = None,
        target_fft_ratio: float = 0.575,
        target_govdocs_ratio: float = 0.425,
        max_corpus_ratio: float = 0.60,
        strategy: str = "stratified",
        seed: int = 42,
        shuffle: bool = True,
    ):
        super().__init__()
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        if indices is not None:
            self.indices = np.asarray(indices, dtype=np.int64)
            self.dynamic = False
        elif sources is not None:
            self.sources = np.asarray(sources)
            self.y = np.asarray(y) if y is not None else None
            self.target_fft_ratio = target_fft_ratio
            self.target_govdocs_ratio = target_govdocs_ratio
            self.max_corpus_ratio = max_corpus_ratio
            self.strategy = strategy
            self.dynamic = True
            # Build initial index
            self.indices = build_corpus_balanced_index(
                sources=self.sources,
                y=self.y,
                target_fft_ratio=self.target_fft_ratio,
                target_govdocs_ratio=self.target_govdocs_ratio,
                max_corpus_ratio=self.max_corpus_ratio,
                strategy=self.strategy,
                seed=self.seed,
            )
        else:
            raise ValueError("Must provide either 'indices' or 'sources'.")

    def set_epoch(self, epoch: int) -> None:
        """Sets epoch for reproducible per-epoch shuffling or dynamic resampling."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        if self.dynamic:
            # Dynamically resample with per-epoch seed
            epoch_seed = self.seed + self._epoch
            indices = build_corpus_balanced_index(
                sources=self.sources,
                y=self.y,
                target_fft_ratio=self.target_fft_ratio,
                target_govdocs_ratio=self.target_govdocs_ratio,
                max_corpus_ratio=self.max_corpus_ratio,
                strategy=self.strategy,
                seed=epoch_seed,
            )
        else:
            indices = self.indices.copy()

        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.indices)


def verify_corpus_balance(
    train_index: np.ndarray,
    train_sources: np.ndarray,
    train_file_ids: np.ndarray,
    train_classes: np.ndarray,
    val_file_ids: set[str],
    test_file_ids: set[str],
    max_corpus_ratio: float = 0.60,
    target_fft_range: tuple[float, float] = (0.55, 0.60),
    target_gov_range: tuple[float, float] = (0.40, 0.45),
) -> dict:
    """Verify all invariants on the balanced training index.

    Returns dictionary of verification metrics. Raises AssertionError on failure.
    """
    n_total = len(train_index)
    assert n_total > 0, "train_index is empty!"
    assert len(np.unique(train_index)) == n_total, "train_index contains duplicate indices!"

    sampled_sources = train_sources[train_index]
    sampled_fids = train_file_ids[train_index]
    sampled_classes = train_classes[train_index]

    source_counts = Counter(sampled_sources)
    n_fft = source_counts.get("fft75", 0)
    n_gov = source_counts.get("govdocs1", 0)
    n_dfrws = source_counts.get("dfrws", 0)

    # 1. DFRWS must be 0
    assert n_dfrws == 0, f"DFRWS found in training set: {n_dfrws} fragments!"

    # 2. Corpus ratios
    fft_pct = n_fft / n_total
    gov_pct = n_gov / n_total

    assert fft_pct <= max_corpus_ratio, (
        f"FFT-75 ratio {fft_pct*100:.2f}% exceeds {max_corpus_ratio*100:.1f}% cap!"
    )
    assert gov_pct <= max_corpus_ratio, (
        f"GovDocs1 ratio {gov_pct*100:.2f}% exceeds {max_corpus_ratio*100:.1f}% cap!"
    )
    assert target_fft_range[0] <= fft_pct <= target_fft_range[1], (
        f"FFT-75 ratio {fft_pct*100:.2f}% outside target range {target_fft_range}"
    )
    assert target_gov_range[0] <= gov_pct <= target_gov_range[1], (
        f"GovDocs1 ratio {gov_pct*100:.2f}% outside target range {target_gov_range}"
    )

    # 3. Leakage check: zero overlap with val and test
    sampled_unique_files = set(sampled_fids)
    val_overlap = sampled_unique_files.intersection(val_file_ids)
    test_overlap = sampled_unique_files.intersection(test_file_ids)

    assert len(val_overlap) == 0, f"Leakage detected! {len(val_overlap)} files overlap with Validation!"
    assert len(test_overlap) == 0, f"Leakage detected! {len(test_overlap)} files overlap with Test!"

    return {
        "n_total": n_total,
        "n_fft": n_fft,
        "n_gov": n_gov,
        "fft_pct": fft_pct,
        "gov_pct": gov_pct,
        "unique_files_total": len(sampled_unique_files),
        "unique_files_gov": len(set(sampled_fids[sampled_sources == "govdocs1"])),
        "unique_files_fft": len(set(sampled_fids[sampled_sources == "fft75"])),
        "val_overlap": len(val_overlap),
        "test_overlap": len(test_overlap),
        "class_counts": Counter(sampled_classes),
    }
