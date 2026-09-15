"""Convert a DFRWS challenge layout table into the truth CSV `src.preprocessing.dfrws`
expects.

The DFRWS 2006/2007 carving challenges publish their ground truth as a layout
table keyed by *sector* ranges:

    Name,Type,FileIdx,FileSectorOffset,ImageSectorOffset,Scenario
    2.jpg,jpg,0,000000-000300,057145-057445,15

`src.preprocessing.dfrws.build()` wants *byte* offsets with columns
`start,end,type[,file_id]`, so this bridges the two. Sector ranges are
inclusive on both ends, hence `end = (last_sector + 1) * sector_size`.

Rows whose ImageSectorOffset is "not used" are fragments the challenge authors
never wrote into the image -- they are skipped, not mapped to offset 0.

`--skip-header` drops the first block of each file's FileIdx-0 fragment, which
is where the magic signature lives. The rest of FragGen is trained and
evaluated on middle-of-file fragments only, so leaving headers in would make
DFRWS results incomparable to the FFT-75/GovDocs1 numbers (and flatter).

  python scripts/dfrws_layout_to_truth.py \
      --layout data/DFRWS/challenge-layout.txt \
      --out    data/DFRWS/dfrws2007_gt.csv \
      --skip-header --block-size 512
"""
from __future__ import annotations
import argparse, csv, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.fft75 import ALIAS

NOT_USED = "not used"


def parse_range(text):
    """'057145-057445' -> (57145, 57445). Returns None for 'not used'."""
    text = (text or "").strip()
    if not text or text.lower() == NOT_USED:
        return None
    if "-" not in text:
        return None
    a, b = text.split("-", 1)
    return int(a), int(b)


def convert(layout_path, sector_size=512, block_size=512, skip_header=False,
            only_mapped=False):
    rows, stats = [], Counter()
    with open(layout_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rng = parse_range(r.get("ImageSectorOffset"))
            if rng is None:
                stats["skipped: not written to image"] += 1
                continue
            first, last = rng
            start, end = first * sector_size, (last + 1) * sector_size

            raw_type = str(r.get("Type", "")).lower().strip()
            canon = ALIAS.get(raw_type)
            if canon is None:
                stats[f"unmapped type: {raw_type}"] += 1
                if only_mapped:
                    continue

            # FileIdx 0 starting at file offset 0 is the fragment holding the
            # magic bytes; advance past one block so no signature is visible.
            fr = parse_range(r.get("FileSectorOffset"))
            if skip_header and str(r.get("FileIdx", "")).strip() == "0" \
                    and fr is not None and fr[0] == 0:
                start += block_size
                stats["header block dropped"] += 1

            if end - start < block_size:
                stats["skipped: shorter than one block"] += 1
                continue

            rows.append({
                "start": start,
                "end": end,
                "type": canon or raw_type,
                "file_id": r.get("Name", ""),
            })
            stats[f"kept: {canon or raw_type}"] += 1
    return rows, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True, help="DFRWS challenge layout table")
    ap.add_argument("--out", required=True, help="truth CSV for src.preprocessing.dfrws")
    ap.add_argument("--sector-size", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--skip-header", action="store_true",
                    help="drop the magic-byte block of each file (recommended)")
    ap.add_argument("--only-mapped", action="store_true",
                    help="emit only types in FragGen's label space")
    ap.add_argument("--image", default=None,
                    help="optional raw image: warn if the layout overruns it")
    args = ap.parse_args()

    rows, stats = convert(args.layout, args.sector_size, args.block_size,
                          args.skip_header, args.only_mapped)
    if not rows:
        raise SystemExit("no usable rows -- check the layout's column names")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["start", "end", "type", "file_id"])
        w.writeheader()
        w.writerows(rows)

    span = max(r["end"] for r in rows)
    print(f"wrote {args.out}: {len(rows)} rows, max byte offset {span:,}")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    if args.image:
        size = Path(args.image).stat().st_size
        print(f"\nimage {args.image}: {size:,} bytes")
        if span > size:
            print(f"  ! LAYOUT OVERRUNS IMAGE by {span - size:,} bytes -- these are"
                  f" almost certainly from different DFRWS challenge years.")
        else:
            print("  layout fits inside the image.")


if __name__ == "__main__":
    main()
