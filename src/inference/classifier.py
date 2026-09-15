"""FragGen inference engine -- the ONE place where raw bytes become predictions.

Pipeline (identical for every caller):
  1. take raw bytes: a file, a region of a disk image, or an in-memory upload
  2. drop the first block (the header, where magic signatures such as %PDF- live)
  3. split the rest into non-overlapping 4096-byte fragments
  4. ByteCNN forward pass on every fragment -> logits
  5. temperature-scaled softmax -> one probability vector per fragment
  6. argmax per fragment, then a majority vote across the input

Callers: demo/cli/demo.py, demo/cli/viva_demo.py, demo/streamlit/app.py and
scripts/make_publication_figures.py. None of them re-implements any step above,
so the CLI, the web app and the viva script cannot disagree about a file.

Scope: the default checkpoint is the Dataset V4 A_plain model (seed 1337). Its
accuracy is a held-out-FILE result on the training source pool
(reports/ablations/v4/). The project's cross-corpus result (FFT-75 -> GovDocs1)
is a different experiment, documented in docs/thesis/FINDINGS.md.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from ..models.checkpoint import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SIZE = 4096
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "v4" / "A_plain_s1337.pt"
CALIBRATION_DIR = REPO_ROOT / "checkpoints" / "calibration"


def calibration_path_for(checkpoint: Union[str, Path]) -> Path:
    """Calibration files live in checkpoints/calibration/ and are named after the
    checkpoint they were fitted for: A_plain_s1337.pt -> A_plain_s1337.calib.json."""
    return CALIBRATION_DIR / f"{Path(checkpoint).stem}.calib.json"


DEFAULT_CALIBRATION = calibration_path_for(DEFAULT_CHECKPOINT)


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Pick the compute device: explicit argument, else $FRAGGEN_DEVICE, else
    CUDA when available, else CPU."""
    choice = device or os.environ.get("FRAGGEN_DEVICE")
    if choice:
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _display_path(p: Path) -> str:
    """Repository-relative path when possible (keeps reports portable)."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


@dataclass
class FragmentResult:
    """One scored fragment."""
    index: int                 # 1-based position among the scored fragments
    byte_offset: int           # absolute offset of the fragment's first byte
    predicted_class: str       # argmax of `probabilities`
    confidence: float          # probability of predicted_class (calibrated if available)
    probabilities: List[float] = field(default_factory=list)  # order = FileVerdict.classes


@dataclass
class FileVerdict:
    """Everything the pipeline knows about one classified input."""
    name: str                        # display name (file name or upload name)
    file_path: Optional[str]         # None for in-memory uploads
    file_size_bytes: int             # number of bytes examined
    block_size: int
    header_skipped: bool
    start_offset: int                # absolute offset of the first scored fragment
    n_fragments: int
    fragments: List[FragmentResult]
    final_prediction: str            # majority-vote winner ("unknown" if no fragments)
    final_confidence: float          # vote share of the winner (agreement, not a probability)
    mean_winner_confidence: float    # mean confidence of the fragments that voted for it
    vote_counts: dict                # class -> number of fragments
    classes: List[str]
    calibrated: bool
    temperature: float
    checkpoint: str

    @property
    def too_small(self) -> bool:
        """True when no complete fragment remained after the header block."""
        return self.n_fragments == 0


class FragGenClassifier:
    """Load a trained ByteCNN once, then classify any number of inputs.

    Loading moves weights to the device and is the slow part, so every caller
    constructs one instance and reuses it (Streamlit caches it per process).
    """

    def __init__(self, checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
                 calibration_path: Union[str, Path, None] = "auto",
                 device: Optional[str] = None):
        """calibration_path: "auto" looks for checkpoints/calibration/<stem>.calib.json;
        None disables calibration; any other value is an explicit file."""
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        self.model, self.cfg = load_model(str(self.checkpoint_path), self.device)
        self.classes = [str(c) for c in self.cfg.classes]
        self.block_size = int(getattr(self.cfg, "block_size", BLOCK_SIZE))

        # Temperature scaling only changes the reported confidence, never the
        # predicted class (argmax(z) == argmax(z / T) for T > 0). A missing file
        # means raw softmax, and every verdict says so via `calibrated`. A
        # malformed file raises instead of silently downgrading to raw softmax.
        self.temperature = 1.0
        self.calibrated = False
        self.calibration_file: Optional[Path] = None
        if calibration_path == "auto":
            calibration_path = calibration_path_for(self.checkpoint_path)
        if calibration_path:
            p = Path(calibration_path)
            if p.exists():
                cal = json.loads(p.read_text(encoding="utf-8"))
                self.temperature = float(cal["temperature"])
                self.calibrated = True
                self.calibration_file = p

    # ---------------------------------------------------------------- steps 2-3
    def fragment_bytes(self, data: bytes, skip_header: bool = True):
        """Split bytes into non-overlapping block_size fragments.

        With skip_header=True the first block is always discarded, exactly like
        the Dataset V4 builder (fragments start at offset 4096). Inputs with no
        complete block after it yield zero fragments rather than silently
        falling back to classifying the header.

        Returns (fragments uint8 array of shape (n, block_size), start_offset).
        """
        start = self.block_size if skip_header else 0
        n = max(0, (len(data) - start) // self.block_size)
        if n == 0:
            return np.empty((0, self.block_size), dtype=np.uint8), start
        buf = np.frombuffer(data, dtype=np.uint8, count=n * self.block_size, offset=start)
        return buf.reshape(n, self.block_size), start

    # ---------------------------------------------------------------- steps 4-5
    @torch.no_grad()
    def predict_fragments(self, fragments: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Forward every fragment through the model; return (n, n_classes)
        probabilities from softmax(logits / T)."""
        self.model.eval()
        out = []
        for i in range(0, len(fragments), batch_size):
            x = torch.from_numpy(fragments[i:i + batch_size].astype(np.int64)).to(self.device)
            logits, _, _ = self.model(x)
            out.append(torch.softmax(logits / self.temperature, dim=1).cpu().numpy())
        if not out:
            return np.empty((0, len(self.classes)), dtype=np.float32)
        return np.concatenate(out, axis=0)

    # ---------------------------------------------------------------- steps 1-6
    def classify_bytes(self, data: bytes, name: str = "<bytes>",
                       file_path: Union[str, Path, None] = None, skip_header: bool = True,
                       max_fragments: Optional[int] = None, base_offset: int = 0) -> FileVerdict:
        """Run the complete pipeline on bytes already in memory.

        base_offset is added to every reported offset (use it when `data` is a
        region read from inside a larger image). max_fragments caps how many
        fragments are scored, taken from the start.
        """
        fragments, start = self.fragment_bytes(data, skip_header=skip_header)
        if max_fragments is not None and len(fragments) > max_fragments:
            fragments = fragments[:max_fragments]

        common = dict(
            name=name, file_path=None if file_path is None else str(file_path),
            file_size_bytes=len(data), block_size=self.block_size,
            header_skipped=skip_header, start_offset=base_offset + start,
            classes=list(self.classes), calibrated=self.calibrated,
            temperature=self.temperature, checkpoint=_display_path(self.checkpoint_path))

        if len(fragments) == 0:
            return FileVerdict(n_fragments=0, fragments=[], final_prediction="unknown",
                               final_confidence=0.0, mean_winner_confidence=0.0,
                               vote_counts={c: 0 for c in self.classes}, **common)

        probs = self.predict_fragments(fragments)
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        # Step 6: majority vote. Ties go to the class listed first in the
        # model's class order (np.argmax returns the first maximum).
        counts = np.bincount(pred, minlength=len(self.classes))
        winner = int(np.argmax(counts))

        frag_results = [
            FragmentResult(index=i + 1,
                           byte_offset=base_offset + start + i * self.block_size,
                           predicted_class=self.classes[pred[i]],
                           confidence=float(conf[i]),
                           probabilities=[float(p) for p in probs[i]])
            for i in range(len(fragments))
        ]
        return FileVerdict(
            n_fragments=len(fragments), fragments=frag_results,
            final_prediction=self.classes[winner],
            final_confidence=float(counts[winner] / len(pred)),
            mean_winner_confidence=float(conf[pred == winner].mean()),
            vote_counts={c: int(counts[i]) for i, c in enumerate(self.classes)},
            **common)

    def classify_file(self, file_path: Union[str, Path], skip_header: bool = True,
                      max_fragments: Optional[int] = None, offset: int = 0,
                      length: int = 0) -> FileVerdict:
        """Read a file (or `length` bytes from `offset`, e.g. inside a disk image)
        and classify it with classify_bytes."""
        path = Path(file_path)
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(length) if length else f.read()
        return self.classify_bytes(data, name=path.name, file_path=path,
                                   skip_header=skip_header, max_fragments=max_fragments,
                                   base_offset=offset)
