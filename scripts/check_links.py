#!/usr/bin/env python
"""Check that every relative link and image in the project's Markdown resolves.

    python scripts/check_links.py              # exit code 1 if an active document has a broken link

Checked: README.md, docs/, reports/, data/*/README.md, demo/outputs/README.md, archive/README.md.
External links (http, https, mailto) and pure #anchors are skipped. Inline code
spans and fenced code blocks are ignored, so example paths are not checked.

Documents written before the 2026-09-14 reorganisation (they carry a "Path note"
banner) are reported separately as legacy: their old paths are intentional and
explained by docs/REMOVED_FILES.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
CODE_SPAN = re.compile(r"`[^`]*`")
GLOBS = ["README.md", "docs/**/*.md", "reports/**/*.md", "data/*/README.md",
         "demo/outputs/README.md", "archive/README.md"]


def documents():
    seen = set()
    for g in GLOBS:
        for p in sorted(ROOT.glob(g)):
            if p.is_file() and p not in seen:
                seen.add(p)
                yield p


def links(text: str):
    """(line number, target) for every Markdown link outside code."""
    in_fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK.finditer(CODE_SPAN.sub("", line)):
            yield n, m.group(1)


def main() -> int:
    broken, legacy, checked = [], [], 0
    for doc in documents():
        text = doc.read_text(encoding="utf-8", errors="replace")
        is_legacy = "**Path note" in text[:1500]
        for n, target in links(text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            checked += 1
            path = unquote(target.split("#", 1)[0])
            if not path:
                continue
            if not (doc.parent / path).exists():
                (legacy if is_legacy else broken).append(
                    f"{doc.relative_to(ROOT).as_posix()}:{n} -> {target}")
    print(f"checked {checked} relative links in {len(list(documents()))} documents")
    if legacy:
        print(f"\nlegacy documents, old paths intentional ({len(legacy)}):")
        print("\n".join("  " + x for x in legacy))
    if broken:
        print(f"\nBROKEN ({len(broken)}):")
        print("\n".join("  " + x for x in broken))
        return 1
    print("\nno broken links in active documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
