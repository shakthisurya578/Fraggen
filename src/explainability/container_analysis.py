"""Map a PDF's internal stream objects, so fragment predictions can be explained.

A PDF is a container: most of its bytes sit inside ``stream ... endstream``
objects whose ``/Filter`` says how they are encoded. The two that matter for
FragGen are

    /FlateDecode   zlib/DEFLATE, the same compression ZIP uses
    /DCTDecode     a baseline JPEG stored verbatim

This module parses those regions straight from the file bytes (no PDF library,
no rendering) and reports, for every 4096-byte fragment, what share of it lies
inside each kind of region. That turns "the model called this PDF fragment
ZIP" into a checkable statement: "fragment 7 is 100% FlateDecode data".

The parser is intentionally conservative: it only reports regions it can
delimit exactly, and bytes it cannot attribute are counted as ``other``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

_STREAM_KW = re.compile(rb"stream(\r\n|\n|\r)")
_ENDSTREAM = b"endstream"
_FILTER = re.compile(rb"/Filter\s*(\[[^\]]*\]|/[A-Za-z0-9]+)")
_NAME = re.compile(rb"/([A-Za-z0-9]+)")

# region labels, in the order figures and tables use them
REGION_LABELS = ("FlateDecode", "DCTDecode", "other_stream", "outside_streams")


@dataclass(frozen=True)
class StreamRegion:
    """Byte range [start, end) of one stream's payload and its filter chain."""
    start: int
    end: int
    filters: Tuple[str, ...]

    @property
    def label(self) -> str:
        """Collapse a filter chain to one of REGION_LABELS. A chain containing
        DCTDecode is JPEG data, even if it is also Flate-wrapped."""
        if "DCTDecode" in self.filters:
            return "DCTDecode"
        if "FlateDecode" in self.filters:
            return "FlateDecode"
        return "other_stream"


def pdf_stream_regions(data: bytes) -> List[StreamRegion]:
    """Find every ``stream``/``endstream`` payload and read the ``/Filter`` from
    the object dictionary that precedes it (searched back to the previous
    ``obj`` keyword, at most 4096 bytes)."""
    regions: List[StreamRegion] = []
    pos = 0
    while True:
        m = _STREAM_KW.search(data, pos)
        if not m:
            break
        kw_start = m.start()
        # skip the "stream" inside "endstream"
        if data[max(0, kw_start - 3):kw_start] == b"end":
            pos = m.end()
            continue
        payload_start = m.end()
        end = data.find(_ENDSTREAM, payload_start)
        if end < 0:
            break
        endstream_at = end
        # the end-of-line marker before "endstream" is not part of the stream data
        if data[end - 2:end] == b"\r\n" and end - 2 >= payload_start:
            end -= 2
        elif data[end - 1:end] in (b"\n", b"\r") and end - 1 >= payload_start:
            end -= 1
        lookback = data[max(0, kw_start - 4096):kw_start]
        obj_at = lookback.rfind(b"obj")
        dictionary = lookback[obj_at:] if obj_at >= 0 else lookback
        filters: Tuple[str, ...] = ()
        fm = _FILTER.search(dictionary)
        if fm:
            filters = tuple(n.decode("ascii", "replace") for n in _NAME.findall(fm.group(1)))
        regions.append(StreamRegion(payload_start, end, filters))
        pos = endstream_at + len(_ENDSTREAM)
    return regions


def fragment_region_shares(regions: List[StreamRegion], start_offset: int,
                           block_size: int, n_fragments: int) -> List[Dict[str, float]]:
    """For fragment i covering [start_offset + i*block, +block), return the
    fraction of its bytes inside each REGION_LABELS category (sums to 1)."""
    out = []
    for i in range(n_fragments):
        a = start_offset + i * block_size
        b = a + block_size
        shares = {k: 0.0 for k in REGION_LABELS}
        covered = 0
        for r in regions:
            if r.end <= a or r.start >= b:
                continue
            overlap = min(b, r.end) - max(a, r.start)
            shares[r.label] += overlap
            covered += overlap
        shares["outside_streams"] = block_size - covered
        out.append({k: v / block_size for k, v in shares.items()})
    return out


def dominant_region(shares: Dict[str, float]) -> str:
    """The region label holding the largest share of a fragment's bytes."""
    return max(REGION_LABELS, key=lambda k: shares[k])


def is_pdf(data: bytes) -> bool:
    """True if the header carries the PDF signature (header is only inspected
    here for explanation; the classifier itself never sees block 0)."""
    return data[:1024].lstrip().startswith(b"%PDF-")
