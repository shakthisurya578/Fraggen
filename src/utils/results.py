"""Headline numbers read from the experiment output files, never typed by hand.

Anything that quotes a result (viva script, Streamlit "About" box, figure
titles) goes through here, so a displayed number always traces back to a
report file on disk.

Conventions: mean over seeds, sample standard deviation (ddof=1), the same as
scripts/run_ablations.py. For Dataset V4 reports, "val" is the in_corpus block
(val.npz) and "test" is cross_corpus[0] (test.npz). Both are held-out FILES
from the training source pool.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
V4_DIR = REPO_ROOT / "reports" / "ablations" / "v4"
CROSS_CORPUS_DIR = REPO_ROOT / "reports" / "ablations" / "cross_corpus_fft75_govdocs1"


@dataclass(frozen=True)
class MeanStd:
    """A mean with its seed standard deviation and the number of seeds."""
    mean: float
    std: float
    n: int

    def fmt(self, digits: int = 4) -> str:
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"

    def pct(self, digits: int = 1) -> str:
        return f"{100 * self.mean:.{digits}f}% ± {100 * self.std:.{digits}f}"


def mean_std(values: List[float]) -> MeanStd:
    """Mean and sample std (0.0 for a single value)."""
    return MeanStd(statistics.mean(values),
                   statistics.stdev(values) if len(values) > 1 else 0.0, len(values))


def v4_reports(config: str = "A_plain", results_dir: Path = V4_DIR) -> Dict[int, dict]:
    """seed -> parsed evaluation report for one ablation config."""
    out = {}
    for p in sorted(Path(results_dir).glob(f"{config}_s*.json")):
        seed = int(p.stem.rsplit("_s", 1)[1])
        out[seed] = json.loads(p.read_text(encoding="utf-8"))
    return out


def split_block(report: dict, split: str) -> dict:
    """The evaluation block for "val" (in_corpus) or "test" (cross_corpus[0])."""
    return report["in_corpus"] if split == "val" else report["cross_corpus"][0]


def v4_metric(config: str = "A_plain", split: str = "test",
              metric: str = "accuracy") -> Optional[MeanStd]:
    """Mean ± std of a top-level metric ("accuracy" or "macro_f1") across seeds."""
    reps = v4_reports(config)
    if not reps:
        return None
    return mean_std([split_block(r, split)[metric] for r in reps.values()])


def cross_corpus_4096(config: str = "A_plain") -> Optional[MeanStd]:
    """Cross-corpus accuracy (train FFT-75 -> test GovDocs1) at 4096 B, from the
    block-size study's driver summary."""
    p = CROSS_CORPUS_DIR / "blocksize_4096" / "summary.json"
    if not p.exists():
        return None
    rows = [r for r in json.loads(p.read_text(encoding="utf-8"))["rows"] if r["run"] == config]
    return mean_std([r["cross_acc"] for r in rows]) if rows else None
