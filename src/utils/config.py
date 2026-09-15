"""Config loading: YAML -> nested SimpleNamespace with dict fallback."""
from __future__ import annotations
import copy
from types import SimpleNamespace
from pathlib import Path
import yaml


def _to_ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _to_dict(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


# src/utils/config.py -> parents[2] is the repository root
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, **overrides):
    path = Path(path) if path else DEFAULT_PATH
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    for k, v in overrides.items():
        raw[k] = v
    cfg = _to_ns(raw)
    cfg._raw = copy.deepcopy(raw)
    return cfg


def config_to_dict(cfg) -> dict:
    d = _to_dict(cfg)
    d.pop("_raw", None)
    return d
