"""FragGen V4 interactive demo (Streamlit).

    streamlit run demo/streamlit/app.py      # run from the repository root

Upload any file or pick a bundled sample. The app shows file metadata, a
fragment timeline, a confidence chart, the vote, plain-language notes (with a
PDF stream map for PDFs), the per-fragment table, and JSON/CSV downloads.

Presentation only: every prediction comes from
src.inference.FragGenClassifier.classify_bytes, the same call the CLI and the
viva script make. Uploads are classified in memory; nothing is written to disk.
The dark theme is configured in .streamlit/config.toml at the repository root.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.explainability import (REGION_LABELS, dominant_region, explain_verdict,  # noqa: E402
                                fragment_region_shares, is_pdf, pdf_stream_regions,
                                verdict_to_csv, verdict_to_json)
from src.inference import FragGenClassifier  # noqa: E402
from src.utils import results  # noqa: E402
from src.utils.plotting import REGION_COLORS, class_color  # noqa: E402

SAMPLES_DIR = REPO_ROOT / "demo" / "samples"
BG, FG, MUTED, GRID = "#0e1117", "#d8dbe2", "#8b90a0", "#2a2e38"

st.set_page_config(page_title="FragGen V4 — File Fragment Classifier", page_icon="🧩",
                   layout="wide")
st.markdown("""
<style>
  .verdict { padding: 1rem 1.3rem; border-radius: 10px; margin: .4rem 0 1rem 0;
             background: linear-gradient(135deg, #1f6f54 0%, #14523d 100%);
             border: 1px solid #2a8a68; }
  .verdict h2 { margin: 0; color: #eafff5; font-size: 1.6rem; }
  .verdict p  { margin: .25rem 0 0 0; color: #b8ecd8; }
  .note { background: #241a12; border-left: 3px solid #d98c2b; border-radius: 4px;
          padding: .6rem .9rem; margin: .35rem 0; font-size: .9rem; color: #f0d9b8; }
</style>
""", unsafe_allow_html=True)


def cli_device():
    """Optional device override: `streamlit run demo/streamlit/app.py -- --device cpu`
    (otherwise $FRAGGEN_DEVICE, else CUDA when available, else CPU)."""
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--device", default=None)
    return ap.parse_known_args(sys.argv[1:])[0].device


@st.cache_resource(show_spinner="Loading the FragGen V4 model…")
def get_classifier() -> FragGenClassifier:
    """One model per server process, not one per interaction."""
    return FragGenClassifier(device=cli_device())


@st.cache_data(show_spinner=False)
def headline() -> dict:
    """Headline numbers read from the result files (never typed by hand)."""
    v4 = results.v4_metric("A_plain", "test", "accuracy")
    cc = results.cross_corpus_4096("A_plain")
    return {"v4": v4.fmt() if v4 else "n/a", "v4_n": v4.n if v4 else 0,
            "cc": cc.fmt() if cc else "n/a", "cc_n": cc.n if cc else 0}


def _dark(fig, axes) -> None:
    """Match matplotlib figures to the Streamlit dark theme."""
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=9)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        for side, spine in ax.spines.items():
            spine.set_color("#444a57")
            spine.set_visible(side in ("left", "bottom"))


def plot_timeline(verdict, shares=None):
    """Fragment timeline: one cell per fragment coloured by predicted class,
    plus (for PDFs) the stream type each fragment lies in."""
    rows = 2 if shares else 1
    fig, axes = plt.subplots(rows, 1, figsize=(11, 0.9 + 0.75 * rows), sharex=True, squeeze=False)
    axes = [a[0] for a in axes]
    idx = [f.index for f in verdict.fragments]
    axes[0].bar(idx, [1] * len(idx), width=0.92,
                color=[class_color(f.predicted_class) for f in verdict.fragments])
    axes[0].set_yticks([0.5], ["prediction"])
    if shares:
        axes[1].bar(idx, [1] * len(idx), width=0.92,
                    color=[REGION_COLORS[dominant_region(s)] for s in shares])
        axes[1].set_yticks([0.5], ["PDF region"])
    for ax in axes:
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("fragment index (4096 B each, header block skipped)")
    handles = [Patch(color=class_color(c), label=c.upper())
               for c in verdict.classes if verdict.vote_counts.get(c)]
    if shares:
        handles += [Patch(color=REGION_COLORS[r], label=r) for r in REGION_LABELS
                    if any(dominant_region(s) == r for s in shares)]
    leg = axes[0].legend(handles=handles, ncol=min(8, len(handles)), loc="lower left",
                         bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(FG)
    _dark(fig, axes)
    fig.tight_layout()
    return fig


def plot_confidence(verdict):
    """Confidence of each fragment's prediction, coloured by class."""
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    idx = [f.index for f in verdict.fragments]
    conf = [f.confidence for f in verdict.fragments]
    ax.plot(idx, conf, color="#5a6070", linewidth=1, zorder=1)
    ax.scatter(idx, conf, s=46, zorder=2, edgecolor=BG, linewidth=0.6,
               c=[class_color(f.predicted_class) for f in verdict.fragments])
    ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("fragment index")
    ax.set_ylabel("confidence")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    _dark(fig, [ax])
    fig.tight_layout()
    return fig


def plot_votes(verdict):
    """How many fragments voted for each class; the winner in its class colour."""
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    classes = list(verdict.classes)
    counts = [verdict.vote_counts[c] for c in classes]
    colors = [class_color(c) if c == verdict.final_prediction else "#3b4252" for c in classes]
    ax.barh([c.upper() for c in classes], counts, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("fragments")
    _dark(fig, [ax])
    fig.tight_layout()
    return fig


def show(fig) -> None:
    """Render a figure and free it (Streamlit reruns the script on every click)."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main() -> None:
    st.title("🧩 FragGen V4 — File Fragment Type Classifier")
    st.caption("Any file → 4096-byte fragments (header block skipped) → ByteCNN per fragment "
               "→ calibrated probabilities → majority vote.")

    h = headline()
    with st.expander("About this model — read before drawing conclusions"):
        st.markdown(f"""
- **Model**: plain byte-embedding CNN (`A_plain`, seed 1337) trained on **Dataset V4**:
  balanced, real documents from GovDocs1, DFRWS 2007, RICO, TGIF, GitHub CSV/ZIP and Wikipedia.
- **V4 test accuracy: {h['v4']}** ({h['v4_n']} seeds). These are *unseen files from the
  same source pool* as training.
- **Cross-corpus accuracy: {h['cc']}** ({h['cc_n']} seeds, train FFT-75 → test GovDocs1,
  4096 B). This is the harder, genuinely different-corpus question
  (`docs/thesis/FINDINGS.md`). The two numbers are not comparable.
- The model **never sees a file's first 4096 bytes**, where magic signatures live.
""")

    clf = get_classifier()
    with st.sidebar:
        st.header("Options")
        skip_header = st.checkbox("Skip header block (as evaluated)", value=True)
        st.caption(f"Device: `{clf.device}` · calibration: "
                   f"{'T=' + format(clf.temperature, '.4f') if clf.calibrated else 'off'}")

    up_col, sample_col = st.columns([3, 2])
    with up_col:
        uploaded = st.file_uploader("Upload a file to classify", type=None)
    with sample_col:
        samples = sorted(p.name for p in SAMPLES_DIR.glob("*.*"))
        choice = st.selectbox("…or pick a bundled sample", ["(none)"] + samples)

    if uploaded is not None:
        data, name = uploaded.getvalue(), uploaded.name
    elif choice != "(none)":
        data, name = (SAMPLES_DIR / choice).read_bytes(), choice
    else:
        st.info("Upload a file or pick a bundled sample to begin.")
        return

    with st.spinner("Classifying…"):
        verdict = clf.classify_bytes(data, name=name, skip_header=skip_header)

    if verdict.too_small:
        st.error(f"'{name}' has {len(data):,} bytes: no complete {verdict.block_size}-byte "
                 f"fragment remains after the header block.")
        return

    st.subheader("File metadata")
    m = st.columns(5)
    m[0].metric("File", name if len(name) <= 24 else name[:21] + "…")
    m[1].metric("Size", f"{len(data):,} B")
    m[2].metric("Fragments scored", verdict.n_fragments)
    m[3].metric("First fragment offset", f"{verdict.start_offset:,}")
    m[4].metric("Mean confidence (winner)", f"{verdict.mean_winner_confidence:.1%}")

    st.markdown(f"""
<div class="verdict"><h2>Final prediction: {verdict.final_prediction.upper()}</h2>
<p>{verdict.final_confidence:.1%} of fragments agree ·
{'calibrated' if verdict.calibrated else 'uncalibrated'} confidence</p></div>
""", unsafe_allow_html=True)

    shares = None
    if is_pdf(data):
        shares = fragment_region_shares(pdf_stream_regions(data), verdict.start_offset,
                                        verdict.block_size, verdict.n_fragments)

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Vote breakdown**")
        show(plot_votes(verdict))
    with right:
        st.markdown("**Confidence per fragment**")
        show(plot_confidence(verdict))
    st.markdown("**Fragment timeline**" + (" with PDF stream map" if shares else ""))
    show(plot_timeline(verdict, shares))

    for note in explain_verdict(verdict):
        st.markdown(f'<div class="note">{html.escape(note)}</div>', unsafe_allow_html=True)

    st.subheader("Per-fragment predictions")
    table = []
    for i, f in enumerate(verdict.fragments):
        row = {"Fragment": f.index, "Byte offset": f"0x{f.byte_offset:08x}",
               "Prediction": f.predicted_class.upper(), "Confidence": f"{f.confidence:.1%}",
               "Agrees with vote": "✓" if f.predicted_class == verdict.final_prediction else "✗"}
        if shares:
            region = dominant_region(shares[i])
            row["PDF region"] = f"{region} ({shares[i][region]:.0%})"
        table.append(row)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Export")
    d1, d2 = st.columns(2)
    d1.download_button("Download JSON report", verdict_to_json(verdict),
                       file_name=f"{name}.fraggen.json", mime="application/json")
    d2.download_button("Download CSV report", verdict_to_csv(verdict),
                       file_name=f"{name}.fraggen.csv", mime="text/csv")


if __name__ == "__main__":
    main()
