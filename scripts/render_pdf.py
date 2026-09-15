#!/usr/bin/env python
"""Render a Markdown report to a print-quality PDF (Markdown -> HTML -> Edge/Chrome headless).

No extra Python packages: a small built-in Markdown converter (headings,
paragraphs, emphasis, inline code, links, images, fenced code, tables, nested
lists, block quotes, rules, raw HTML lines) plus a Chromium browser that is
already installed on the machine. Images are resolved relative to the Markdown
file, so figures from reports/figures/ are embedded in the PDF.

    python scripts/render_pdf.py docs/thesis/RESULTS_AND_DISCUSSION.md
    python scripts/render_pdf.py docs/FINAL_PROJECT_SUMMARY.md --out docs/FINAL_PROJECT_SUMMARY.pdf
    python scripts/render_pdf.py some.md --keep-html      # also keep the intermediate .html
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt;
       color: #1d1d1f; line-height: 1.45; }
body { margin: 0; }
h1 { font-size: 21pt; margin: 0 0 6pt 0; color: #10233f; border-bottom: 2px solid #10233f; padding-bottom: 4pt; }
h2 { font-size: 14.5pt; margin: 18pt 0 6pt 0; color: #10233f; border-bottom: 1px solid #c9d3e0;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 13pt 0 4pt 0; color: #22406b; page-break-after: avoid; }
h4 { font-size: 10.8pt; margin: 10pt 0 3pt 0; color: #22406b; page-break-after: avoid; }
p { margin: 4pt 0 6pt 0; text-align: justify; }
a { color: #1f5fa8; text-decoration: none; }
code { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 8.8pt; background: #f1f3f6;
       padding: 0 2pt; border-radius: 2pt; }
pre { background: #f5f7fa; border: 1px solid #dde3ea; border-radius: 3pt; padding: 6pt 8pt;
      font-size: 8.3pt; line-height: 1.3; white-space: pre-wrap; word-break: break-word;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; margin: 6pt 0 9pt 0; font-size: 8.7pt; width: auto; max-width: 100%;
        page-break-inside: avoid; }
th, td { border: 1px solid #cfd7e2; padding: 2.5pt 5pt; vertical-align: top; }
th { background: #eaf0f7; text-align: left; }
tr:nth-child(even) td { background: #fafbfd; }
td.r, th.r { text-align: right; } td.c, th.c { text-align: center; }
blockquote { margin: 7pt 0; padding: 5pt 10pt; background: #f4f7fb; border-left: 3pt solid #4b77b0; }
blockquote p { margin: 2pt 0; }
img { max-width: 100%; display: block; margin: 8pt auto 3pt auto; page-break-inside: avoid; }
hr { border: none; border-top: 1px solid #c9d3e0; margin: 12pt 0; }
ul, ol { margin: 3pt 0 6pt 0; padding-left: 18pt; }
li { margin: 1.5pt 0; }
.figure-caption { font-size: 8.8pt; color: #444; text-align: center; margin: 0 8% 10pt 8%; }
details > summary { font-weight: 600; }
.page-break { page-break-before: always; }
"""


# --------------------------------------------------------------------------- inline
def inline(text: str, base: Path) -> str:
    """Convert inline Markdown in one line/paragraph to HTML."""
    codes = []

    def keep_code(m):
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", keep_code, text)
    text = html.escape(text, quote=False)

    def img(m):
        alt, src = m.group(1), m.group(2)
        if not re.match(r"^[a-z]+://", src):
            src = (base / src).resolve().as_uri()
        return f'<img alt="{html.escape(alt)}" src="{src}">'

    text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", img, text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"(?<![\w])_(?!\s)([^_\n]+?)_(?![\w])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)


def split_row(line: str):
    """Split a Markdown table row on pipes that are not inside backticks."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells, cur, in_code = [], "", False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            cur += "|"; i += 2; continue
        if ch == "`":
            in_code = not in_code
        if ch == "|" and not in_code:
            cells.append(cur.strip()); cur = ""
        else:
            cur += ch
        i += 1
    cells.append(cur.strip())
    return cells


# --------------------------------------------------------------------------- blocks
def markdown_to_html(md: str, base: Path) -> str:
    lines = md.replace("\r\n", "\n").split("\n")
    out, i, n = [], 0, len(lines)
    para: list[str] = []

    def flush_para():
        if para:
            out.append(f"<p>{inline(' '.join(p.strip() for p in para), base)}</p>")
            para.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):                                  # fenced code
            flush_para()
            lang = stripped[3:].strip()
            buf = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            if lang == "mermaid":
                out.append('<pre class="mermaid-src">' + html.escape("\n".join(buf)) + "</pre>")
            else:
                out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        if not stripped:
            flush_para(); i += 1; continue

        if stripped.startswith("<") and not stripped.startswith("<http"):   # raw HTML block
            flush_para()                                                    # (until blank line)
            while i < n and lines[i].strip():
                out.append(lines[i].strip()); i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)                      # heading
        if m:
            flush_para()
            lvl = len(m.group(1))
            slug = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f'<h{lvl} id="{slug}">{inline(m.group(2), base)}</h{lvl}>')
            i += 1; continue

        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):                 # rule
            flush_para(); out.append("<hr>"); i += 1; continue

        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|?\s*:?-{2,}", lines[i + 1].strip()):
            flush_para()                                                 # table
            header = split_row(stripped)
            aligns = []
            for spec in split_row(lines[i + 1]):
                spec = spec.strip()
                aligns.append("r" if spec.endswith(":") and not spec.startswith(":") else
                              "c" if spec.startswith(":") and spec.endswith(":") else "")
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i])); i += 1
            cls = lambda k: f' class="{aligns[k]}"' if k < len(aligns) and aligns[k] else ""
            t = ["<table><thead><tr>"] + [f"<th{cls(k)}>{inline(h, base)}</th>" for k, h in enumerate(header)]
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>" + "".join(f"<td{cls(k)}>{inline(c, base)}</td>" for k, c in enumerate(r)) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        if stripped.startswith(">"):                                       # block quote
            flush_para()
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].lstrip()); i += 1
            out.append("<blockquote>" + markdown_to_html("\n".join(buf), base) + "</blockquote>")
            continue

        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):                         # (nested) list
            flush_para()
            stack = []                                                      # (indent, tag)
            html_parts = []
            while i < n and (re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]) or
                             (lines[i].startswith("  ") and lines[i].strip() and stack)):
                cur = lines[i]
                lm = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", cur)
                if not lm:                                                  # continuation line
                    html_parts[-1] = html_parts[-1][:-5] + " " + inline(cur.strip(), base) + "</li>"
                    i += 1; continue
                indent, marker, content = len(lm.group(1)), lm.group(2), lm.group(3)
                tag = "ol" if marker[0].isdigit() else "ul"
                while stack and indent < stack[-1][0]:
                    html_parts.append(f"</{stack.pop()[1]}>")
                if not stack or indent > stack[-1][0]:
                    stack.append((indent, tag)); html_parts.append(f"<{tag}>")
                html_parts.append(f"<li>{inline(content, base)}</li>")
                i += 1
            while stack:
                html_parts.append(f"</{stack.pop()[1]}>")
            out.append("".join(html_parts))
            continue

        para.append(line)
        i += 1
    flush_para()
    return "\n".join(out)


def find_browser() -> str:
    for b in BROWSERS:
        if Path(b).exists():
            return b
    for name in ("msedge", "chrome", "chromium", "google-chrome"):
        p = shutil.which(name)
        if p:
            return p
    sys.exit("no Edge/Chrome/Chromium found for PDF rendering")


def render(md_path: Path, out_pdf: Path, keep_html: bool = False) -> Path:
    """Markdown file -> styled HTML -> PDF via a headless Chromium browser."""
    md_path = md_path.resolve()
    body = markdown_to_html(md_path.read_text(encoding="utf-8"), md_path.parent)
    title = re.search(r"^#\s+(.+)$", md_path.read_text(encoding="utf-8"), re.M)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>"
           f"{html.escape(title.group(1) if title else md_path.stem)}</title>"
           f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_pdf = out_pdf.resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="fraggen_pdf_"))
    html_path = tmp_dir / (md_path.stem + ".html")
    html_path.write_text(doc, encoding="utf-8")
    cmd = [find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--print-to-pdf-no-header", f"--user-data-dir={tmp_dir / 'profile'}",
           f"--print-to-pdf={out_pdf}", html_path.as_uri()]
    if out_pdf.exists():
        out_pdf.unlink()
    subprocess.run(cmd, check=False, capture_output=True, timeout=180)
    if not out_pdf.exists() or out_pdf.stat().st_size == 0:
        sys.exit(f"PDF rendering failed for {md_path}")
    if keep_html:
        shutil.copy(html_path, out_pdf.with_suffix(".html"))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("markdown", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: same name with .pdf")
    ap.add_argument("--keep-html", action="store_true")
    args = ap.parse_args()
    out = render(args.markdown, args.out or args.markdown.with_suffix(".pdf"), args.keep_html)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
