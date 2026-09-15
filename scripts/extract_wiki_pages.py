"""Extract individual article pages from a MediaWiki XML dump (.bz2) into
separate real files under data/raw/wiki_pages/.

This exists to fix a specific gap found by the V2/V3 data audit
(reports/DATA_AUDIT_2026-09-14.md): the previously-staged data/wikipedia_xml/ (now archive/data/wikipedia_xml_staging/)
individual page exports have a median size of 822 bytes -- far too small to
be usable interior-fragment source documents (need >= 8192 B). This script
pulls full-length real articles out of a bigger wiki dump instead.

Each output file is one real, complete <page>...</page> element exactly as it
appears in the dump -- not synthetic content, just sliced at the document
boundary the dump format itself defines. Talk/template/category pages and
redirects are skipped so what's kept looks like the "real world .xml
document" class the project is trying to represent, not wiki-internal
bookkeeping pages.

Usage:
  python scripts/extract_wiki_pages.py \
      --dump data/raw/downloads/simplewiki-latest-pages-articles.xml.bz2 \
      --out data/raw/wiki_pages --min-size 8192
"""
from __future__ import annotations
import argparse, bz2, re, time
from pathlib import Path

PAGE_START = b"<page>"
PAGE_END = b"</page>"
NS_RE = re.compile(rb"<ns>(-?\d+)</ns>")
REDIRECT_RE = re.compile(rb"<redirect\b")
TITLE_RE = re.compile(rb"<title>(.*?)</title>", re.DOTALL)
ID_RE = re.compile(rb"<id>(\d+)</id>")


def safe_slug(title: bytes, pageid: bytes) -> str:
    t = title.decode("utf-8", "replace")
    t = re.sub(r"[^A-Za-z0-9_-]+", "_", t)[:60].strip("_") or "page"
    return f"{pageid.decode()}_{t}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-size", type=int, default=8192)
    ap.add_argument("--max-pages", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming decompress: {args.dump}")
    t0 = time.time()
    n_seen = 0
    n_kept = 0
    n_redirect = 0
    n_nonarticle = 0
    n_too_small = 0
    buf = b""
    CHUNK = 1 << 20  # 1 MB decompressed chunks

    with bz2.open(args.dump, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            buf += chunk
            while True:
                start = buf.find(PAGE_START)
                if start == -1:
                    # keep a small tail in case <page> is split across reads
                    if len(buf) > len(PAGE_START):
                        buf = buf[-len(PAGE_START):]
                    break
                end = buf.find(PAGE_END, start)
                if end == -1:
                    buf = buf[start:]  # wait for more data
                    break
                end += len(PAGE_END)
                page = buf[start:end]
                buf = buf[end:]
                n_seen += 1

                ns_m = NS_RE.search(page)
                if not ns_m or ns_m.group(1) != b"0":
                    n_nonarticle += 1
                    continue
                if REDIRECT_RE.search(page):
                    n_redirect += 1
                    continue
                if len(page) < args.min_size:
                    n_too_small += 1
                    continue

                title_m = TITLE_RE.search(page)
                id_m = ID_RE.search(page)
                title = title_m.group(1) if title_m else b"untitled"
                pageid = id_m.group(1) if id_m else str(n_seen).encode()
                fname = safe_slug(title, pageid) + ".xml"
                (out_dir / fname).write_bytes(page)
                n_kept += 1

                if n_seen % 20000 == 0:
                    print(f"  ...{n_seen:,} pages scanned, {n_kept:,} kept "
                          f"({time.time()-t0:.0f}s elapsed)")
                if args.max_pages and n_kept >= args.max_pages:
                    break
            if args.max_pages and n_kept >= args.max_pages:
                break

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s")
    print(f"  pages seen        : {n_seen:,}")
    print(f"  non-article (ns!=0): {n_nonarticle:,}")
    print(f"  redirects skipped : {n_redirect:,}")
    print(f"  too small (<{args.min_size}B): {n_too_small:,}")
    print(f"  KEPT (written)    : {n_kept:,}  ->  {out_dir}")


if __name__ == "__main__":
    main()
