"""Shared matplotlib styling so every figure and the web app use the same look.

CLASS_COLORS is a colour-blind-aware palette (Okabe-Ito based). It is
deliberately fixed per class so a PDF fragment is the same colour in the
confusion matrix, the fragment timeline, the Streamlit app and the thesis.
"""
from __future__ import annotations

import matplotlib

CLASS_COLORS = {
    "pdf": "#D55E00",   # vermillion
    "zip": "#0072B2",   # blue
    "jpg": "#009E73",   # bluish green
    "png": "#56B4E9",   # sky blue
    "gif": "#CC79A7",   # reddish purple
    "html": "#E69F00",  # orange
    "xml": "#B8860B",   # dark goldenrod
    "txt": "#7F7F7F",   # grey
    "csv": "#8C564B",   # brown
}

# PDF stream-region colours reuse the class they are byte-identical to
REGION_COLORS = {
    "FlateDecode": CLASS_COLORS["zip"],
    "DCTDecode": CLASS_COLORS["jpg"],
    "other_stream": "#BBBBBB",
    "outside_streams": CLASS_COLORS["pdf"],
}

# the cumulative ablation ladder, in order, with a readable label each
CONFIG_ORDER = ["A_plain", "B_aug", "C_aux", "D_adv", "E_full"]
CONFIG_LABELS = {
    "A_plain": "A  plain CNN",
    "B_aug": "B  + augmentation",
    "C_aux": "C  + entropy aux features",
    "D_adv": "D  + entropy adversary",
    "E_full": "E  + contrastive pre-training",
}


def class_color(name: str) -> str:
    """Colour for a class name (neutral grey for anything unknown)."""
    return CLASS_COLORS.get(str(name).lower(), "#444444")


def use_publication_style() -> None:
    """Consistent typography for exported figures: one sans-serif family,
    restrained sizes, light grid, no top/right spines, 300-dpi PNG output."""
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#E3E3E3",
        "grid.linewidth": 0.8,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "svg.fonttype": "none",
    })
