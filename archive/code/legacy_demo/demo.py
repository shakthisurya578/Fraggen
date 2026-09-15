"""FragGen live demo — the whole project in one run.

Four acts:
  1. THE PROBLEM   a raw 512-byte chunk with no name, no header, no filesystem.
                   Shown as a hex dump. What type is it?
  2. THE MODEL     what FragGen predicts, and how confident it is.
  3. THE GAP       accuracy over many real files the model has never seen.
  4. THE FINDING   the same files at 4096 bytes -- the result the project landed on.

Usage:
  python scripts/demo.py                      # full narrated demo
  python scripts/demo.py --file suspect.bin   # just classify one file
  python scripts/demo.py --quick              # fewer files, faster
"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fraggen.eval import load_model
from fraggen.data.fft75 import ALIAS

CORPUS = ROOT / "data" / "govdocs1_files"
M512 = ROOT / "checkpoints" / "ablations_gpu" / "A_plain_s1337.pt"
M4096 = ROOT / "checkpoints" / "blocksize_4096" / "A_plain_s1337.pt"

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
GRN, RED, CYN, YEL = "\033[32m", "\033[31m", "\033[36m", "\033[33m"


def rule(ch="─", n=74):
    print(DIM + ch * n + OFF)


def act(n, title):
    print()
    rule("━")
    print(f"{BOLD}  ACT {n}   {title}{OFF}")
    rule("━")


def hexdump(buf, offset=0, rows=8):
    """Classic 16-bytes-per-row hex + ASCII view."""
    for r in range(rows):
        s = r * 16
        chunk = buf[s:s + 16]
        if not len(chunk):
            break
        hx = " ".join(f"{b:02x}" for b in chunk)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {DIM}{offset+s:08x}{OFF}  {hx:<47}  {DIM}|{OFF}{asc}{DIM}|{OFF}")


def fragments(data, block, skip_header=True):
    start = block if skip_header else 0
    n = max(0, (len(data) - start) // block)
    if n == 0:
        return np.empty((0, block), np.uint8)
    return np.frombuffer(data, np.uint8, count=n * block, offset=start).reshape(n, block)


@torch.no_grad()
def predict(model, X, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch].astype(np.int64)).to(device)
        logits, _, _ = model(xb)
        out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 9))


def collect(limit_per_type, want, seed=0):
    """Gather sample files per type without globbing the whole corpus."""
    rng = random.Random(seed)
    found = {w: [] for w in want}
    if not CORPUS.exists():
        return found
    subdirs = sorted([d for d in CORPUS.iterdir() if d.is_dir()])
    rng.shuffle(subdirs)
    for d in subdirs:
        if all(len(v) >= limit_per_type for v in found.values()):
            break
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        rng.shuffle(entries)
        for e in entries:
            ext = e.name.rsplit(".", 1)[-1].lower() if "." in e.name else ""
            canon = ALIAS.get(ext)
            if canon in found and len(found[canon]) < limit_per_type:
                try:
                    if e.stat().st_size > 20000:      # big enough for both block sizes
                        found[canon].append(Path(e.path))
                except OSError:
                    pass
    return found


def verdict_for(model, path, block, classes, device):
    data = path.read_bytes()
    X = fragments(data, block)
    if len(X) == 0:
        return None, 0.0, 0
    P = predict(model, X, device)
    pred = P.argmax(1)
    counts = np.bincount(pred, minlength=len(classes))
    top = int(counts.argmax())
    return classes[top], float(counts[top] / len(pred)), len(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="classify just this file")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-type", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not M512.exists():
        print(f"missing model {M512} -- train first (see README)")
        return 1
    m512, cfg = load_model(str(M512), device)
    classes = [str(c) for c in cfg.classes]

    # ---------------- single-file mode ----------------
    if args.file:
        p = Path(args.file)
        data = p.read_bytes()
        X = fragments(data, 512)
        if len(X) == 0:
            print("file too small for a 512-byte middle fragment")
            return 1
        P = predict(m512, X, device)
        pred, conf = P.argmax(1), P.max(1)
        counts = np.bincount(pred, minlength=len(classes))
        top = int(counts.argmax())
        print(f"\n{BOLD}{p.name}{OFF}  {len(data):,} bytes  ->  {len(X)} fragments")
        print(f"verdict: {BOLD}{classes[top]}{OFF}  "
              f"({counts[top]/len(pred):.0%} of blocks, mean conf {conf.mean():.3f})")
        return 0

    per_type = args.per_type or (4 if args.quick else 12)
    want = ["jpg", "png", "gif", "pdf", "html", "xml", "csv", "txt", "zip"]

    # ================= ACT 1 =================
    act(1, "THE PROBLEM — a chunk of bytes with no context")
    files = collect(per_type, want)
    demo_file = None
    for t in ["jpg", "html", "pdf"]:
        if files.get(t):
            demo_file = (t, files[t][0])
            break
    if demo_file is None:
        print("no corpus files found under data/govdocs1_files")
        return 1
    true_t, path = demo_file
    data = path.read_bytes()
    print(f"\n  A carver pulled this 512-byte block off a disk image.")
    print(f"  No filename. No extension. Not the start of the file, so no")
    print(f"  magic signature to look up. {BOLD}What file type is it?{OFF}\n")
    hexdump(data[512:1024], offset=512, rows=8)
    print(f"\n  {DIM}(truncated — 512 bytes total){OFF}")

    # ================= ACT 2 =================
    act(2, "THE MODEL — a byte-embedding 1D CNN reads it")
    X = fragments(data, 512)
    P1 = predict(m512, X[:1], device)[0]
    order = np.argsort(-P1)
    print(f"\n  {BOLD}That one block, on its own:{OFF}\n")
    for i in order[:4]:
        bar = "█" * int(round(P1[i] * 34))
        mark = f"  {GRN}<- ground truth{OFF}" if classes[i] == true_t else ""
        print(f"  {classes[i]:>5s}  {P1[i]:5.1%}  {bar}{mark}")
    hit1 = classes[int(order[0])] == true_t
    print(f"\n  single-block call: {BOLD}{classes[int(order[0])]}{OFF}  "
          f"{GRN + 'correct' + OFF if hit1 else RED + 'wrong' + OFF}")

    # the honest framing: one fragment is a weak signal, a file is many fragments
    Pall = predict(m512, X, device)
    counts = np.bincount(Pall.argmax(1), minlength=len(classes))
    top = int(counts.argmax())
    print(f"\n  {BOLD}All {len(X)} blocks of the same file, majority vote:{OFF}\n")
    for i in np.argsort(-counts)[:4]:
        if counts[i] == 0:
            continue
        share = counts[i] / len(X)
        bar = "█" * int(round(share * 34))
        mark = f"  {GRN}<- ground truth{OFF}" if classes[i] == true_t else ""
        print(f"  {classes[i]:>5s}  {share:5.1%}  {bar}{mark}")
    print(f"\n  file verdict: {BOLD}{classes[top]}{OFF}  "
          f"{GRN + 'correct' + OFF if classes[top] == true_t else RED + 'wrong' + OFF}")
    print(f"\n  {DIM}One 512-byte block is a weak signal — this is a genuinely hard{OFF}")
    print(f"  {DIM}problem. A carver has many blocks, and they vote.{OFF}")
    print(f"  {DIM}The model never sees a header: block 0 of every file is{OFF}")
    print(f"  {DIM}discarded in training and testing, on purpose.{OFF}")

    # ================= ACT 3 =================
    act(3, "THE GAP — real files the model has never seen")
    n_files = sum(len(v) for v in files.values())
    print(f"\n  {n_files} GovDocs1 files. The model trained on FFT-75 — a")
    print(f"  completely different corpus. This is the 'away game'.")
    print(f"  {DIM}Scored per FILE (majority vote over its blocks), not per fragment,{OFF}")
    print(f"  {DIM}so these numbers sit above the per-fragment figures in the report.{OFF}\n")
    t0 = time.time()
    res512 = {}
    for t in want:
        if not files.get(t):
            continue
        ok = 0
        for f in files[t]:
            v, _, _ = verdict_for(m512, f, 512, classes, device)
            ok += (v == t)
        res512[t] = (ok, len(files[t]))
        bar = "█" * int(round(ok / len(files[t]) * 26))
        col = GRN if ok / len(files[t]) >= .7 else (YEL if ok / len(files[t]) >= .4 else RED)
        print(f"  {t:>5s}  {col}{ok}/{len(files[t]):<3d}{OFF}  {bar}")
    tot_ok = sum(a for a, _ in res512.values())
    tot_n = sum(b for _, b in res512.values())
    print(f"\n  {BOLD}512-byte fragments: {tot_ok}/{tot_n} files correct "
          f"({tot_ok/max(tot_n,1):.0%}){OFF}   {DIM}[{time.time()-t0:.1f}s]{OFF}")

    # ================= ACT 4 =================
    act(4, "THE FINDING — same files, 4096-byte fragments")
    if not M4096.exists():
        print(f"\n  (4096 model not found at {M4096})")
        return 0
    m4096, cfg4 = load_model(str(M4096), device)
    cls4 = [str(c) for c in cfg4.classes]
    print(f"\n  Same architecture. Same training budget. Same files.")
    print(f"  The only change: each fragment is 8x longer.\n")
    res4k = {}
    for t in want:
        if not files.get(t):
            continue
        ok = 0
        for f in files[t]:
            v, _, _ = verdict_for(m4096, f, 4096, cls4, device)
            ok += (v == t)
        res4k[t] = (ok, len(files[t]))
        a512 = res512[t][0] / max(res512[t][1], 1)
        a4k = ok / max(len(files[t]), 1)
        d = a4k - a512
        arrow = f"{GRN}+{d:.0%}{OFF}" if d > 0 else (f"{RED}{d:.0%}{OFF}" if d < 0 else f"{DIM}same{OFF}")
        bar = "█" * int(round(a4k * 26))
        print(f"  {t:>5s}  {ok}/{len(files[t]):<3d}  {bar}  {arrow}")
    k_ok = sum(a for a, _ in res4k.values())
    k_n = sum(b for _, b in res4k.values())

    print()
    rule("═")
    print(f"  {BOLD}512-byte fragments   {tot_ok/max(tot_n,1):6.0%}{OFF}  "
          f"{DIM}({tot_ok}/{tot_n} files){OFF}")
    print(f"  {BOLD}4096-byte fragments  {k_ok/max(k_n,1):6.0%}{OFF}  "
          f"{DIM}({k_ok}/{k_n} files){OFF}   "
          f"{GRN}{(k_ok/max(k_n,1)) - (tot_ok/max(tot_n,1)):+.0%}{OFF}")
    rule("═")
    if tot_n < 60:
        print(f"  {DIM}Small live sample — run without --quick for more files. The{OFF}")
        print(f"  {DIM}statistically tested figures come from 80 full training runs:{OFF}")
    print(f"""
  Across 80 training runs, none of FragGen's three mechanisms
  (augmentation, entropy adversary, contrastive pretraining) beat a
  plain CNN cross-corpus — the full stack scored {BOLD}-0.0003, p=0.975{OFF}.

  Fragment length did: {BOLD}+0.128 cross-corpus accuracy, p=0.00009{OFF}.

  {CYN}For fragment carving, read more bytes before you train harder.{OFF}
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
