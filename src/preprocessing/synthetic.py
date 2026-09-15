"""Synthetic fragment generator.

Purpose: make the full FragGen pipeline runnable and demonstrable WITHOUT the
multi-GB real corpora (which live on the user's machine). Each class is drawn
from a byte process that mimics the real format's *true* structure (JPEG marker
bytes, PK signatures, PDF keywords, printable text, ...) on top of the right
entropy regime.

Crucially we can emit two corpora, A and B, that share the true class signal but
differ in *nuisance* parameters (vocabulary, marker frequency, entropy source,
byte bias). A model that memorises A's nuisance quirks -- exactly the failure
FragGen targets -- will drop on B, so the in-corpus vs cross-corpus GAP is
reproducible on synthetic data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .fragments import FragmentSet

DEFAULT_CLASSES = ["txt", "csv", "html", "xml", "pdf",
                   "jpg", "png", "gif", "zip", "encrypted"]

_WORDS_A = [b"the", b"case", b"file", b"data", b"disk", b"report", b"system",
            b"user", b"time", b"block", b"image", b"memory", b"trace", b"event"]
_WORDS_B = [b"evidence", b"court", b"sample", b"volume", b"sector", b"record",
            b"agent", b"kernel", b"packet", b"header", b"stream", b"offset"]
_TAGS_A = [b"div", b"span", b"p", b"a", b"table", b"tr", b"td", b"body", b"head"]
_TAGS_B = [b"section", b"article", b"ul", b"li", b"nav", b"main", b"figure"]


@dataclass
class CorpusParams:
    """Nuisance knobs that differ between corpora but preserve class identity."""
    name: str = "A"
    seed: int = 0
    words: list = field(default_factory=lambda: _WORDS_A)
    tags: list = field(default_factory=lambda: _TAGS_A)
    marker_freq: float = 1.0     # scales structural marker frequency (encoder quirk)
    byte_bias: int = 0           # small additive bias on high-entropy bytes
    entropy_struct: float = 1.0  # scales residual structure inside "random" streams


def corpus_A(seed=0):
    return CorpusParams("A", seed, _WORDS_A, _TAGS_A, 1.0, 0, 1.0)


def corpus_B(seed=1):
    # same formats, different encoder/vocabulary fingerprints -> the "away game"
    return CorpusParams("B", seed, _WORDS_B, _TAGS_B, 0.55, 7, 0.4)


def _rand_bytes(rng, n, bias=0):
    b = rng.integers(0, 256, size=n).astype(np.int32)
    if bias:
        b = (b + bias) % 256
    return b.astype(np.uint8)


def _text_body(rng, L, words, sep=b" "):
    out = bytearray()
    while len(out) < L:
        out += words[rng.integers(0, len(words))]
        out += sep
        if rng.random() < 0.08:
            out += b"\n"
    return np.frombuffer(bytes(out[:L]), dtype=np.uint8).copy()


def _gen_one(cls, rng, L, p: CorpusParams):
    if cls == "txt":
        return _text_body(rng, L, p.words, b" ")
    if cls == "csv":
        cols = []
        while sum(len(c) for c in cols) < L:
            row = b",".join(str(rng.integers(0, 9999)).encode() for _ in range(4)) + b"\n"
            cols.append(row)
        return np.frombuffer(b"".join(cols)[:L].ljust(L, b" "), dtype=np.uint8).copy()
    if cls in ("html", "xml"):
        out = bytearray()
        closing = cls == "xml"
        if cls == "xml" and rng.random() < 0.3 * p.marker_freq:
            out += b"<?xml version=\"1.0\"?>"
        while len(out) < L:
            t = p.tags[rng.integers(0, len(p.tags))]
            out += b"<" + t + b">"
            out += p.words[rng.integers(0, len(p.words))]
            if closing or rng.random() < 0.7:
                out += b"</" + t + b">"
            if rng.random() < 0.1:
                out += b"\n"
        return np.frombuffer(bytes(out[:L]), dtype=np.uint8).copy()
    if cls == "pdf":
        kws = [b"obj", b"endobj", b"stream", b"endstream", b"/Type", b"/Length", b"xref"]
        out = bytearray()
        while len(out) < L:
            if rng.random() < 0.5 * p.marker_freq:
                out += kws[rng.integers(0, len(kws))] + b" "
            else:
                # embedded high-entropy stream chunk
                n = int(rng.integers(8, 40))
                out += _rand_bytes(rng, n, p.byte_bias).tobytes()
        return np.frombuffer(bytes(out[:L]), dtype=np.uint8).copy()
    if cls == "jpg":
        out = _rand_bytes(rng, L, p.byte_bias)
        # 0xFF marker structure + 0xFF00 byte-stuffing quirk
        n_mark = max(1, int(L / 64 * p.marker_freq))
        pos = rng.integers(0, L - 1, size=n_mark)
        out[pos] = 0xFF
        out[np.clip(pos + 1, 0, L - 1)] = 0x00
        if rng.random() < 0.15:
            out[:2] = [0xFF, 0xD8]  # SOI
        return out
    if cls == "png":
        out = _rand_bytes(rng, L, p.byte_bias)
        # per-scanline filter byte periodicity (0..4) every ~stride bytes
        stride = 64
        for s in range(0, L, stride):
            out[s] = int(rng.integers(0, 5))
        if rng.random() < 0.1:
            out[:4] = [0x49, 0x44, 0x41, 0x54]  # "IDAT"
        return out
    if cls == "gif":
        out = _rand_bytes(rng, L, p.byte_bias)
        # LZW clear/stop code bytes appear periodically; block-size lead bytes
        for s in range(0, L, 255):
            out[s] = 0xFF if rng.random() < 0.5 * p.marker_freq else int(rng.integers(1, 255))
        return out
    if cls == "zip":
        out = _rand_bytes(rng, L, p.byte_bias)
        n_sig = max(1, int(L / 200 * p.marker_freq))
        for _ in range(n_sig):
            s = int(rng.integers(0, max(1, L - 4)))
            out[s:s + 4] = [0x50, 0x4B, 0x03, 0x04]  # PK\x03\x04
        return out
    if cls == "encrypted":
        return _rand_bytes(rng, L, 0)  # ~uniform, maximal entropy, no structure
    raise ValueError(cls)


def generate_corpus(params: CorpusParams, classes=None, n_per_class=2000,
                    block_size=512) -> FragmentSet:
    classes = classes or DEFAULT_CLASSES
    rng = np.random.default_rng(params.seed)
    X, y = [], []
    for ci, cls in enumerate(classes):
        for _ in range(n_per_class):
            X.append(_gen_one(cls, rng, block_size, params))
            y.append(ci)
    X = np.stack(X).astype(np.uint8)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return FragmentSet(X[perm], y[perm], classes, corpus=f"synthetic-{params.name}")
