"""Convert a downloaded FFT-75 (FiFTy) scenario-1 .npz into a FragGen FragmentSet.

The FiFTy archives ship `x` (N, L) uint8 and `y` (N,) with label indices in the
*scenario-1* ordering (75 classes). We map those onto FragGen's target label
space by name via `src.preprocessing.fft75.ALIAS`, dropping types outside the set.

Memory note: the 512B train split is 6.14M x 512 = 3.1 GB uncompressed, which
does not fit comfortably in RAM alongside a filtered copy. So we stream `x.npy`
straight out of the npz's zip container in row-chunks and keep only the rows we
selected, never materialising the full array.

Usage:
  python -m scripts.prep_fft75 --src data/FFT-75/512 --out data/processed \
      --block-size 512 --per-class 50000 --per-class-eval 6000
"""
from __future__ import annotations
import argparse, ast, struct, sys, zipfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.fragments import FragmentSet
from src.preprocessing.fft75 import ALIAS

# Authoritative scenario-1 ordering, from the FiFTy repo:
# https://raw.githubusercontent.com/mittalgovind/fifty/master/fifty/utilities/labels.json
FFT75_SCENARIO1 = [
    "jpg", "arw", "cr2", "dng", "gpr", "nef", "nrw", "orf", "pef", "raf", "rw2",
    "3fr", "tiff", "heic", "bmp", "gif", "png", "ai", "eps", "psd", "mov", "mp4",
    "3gp", "avi", "mkv", "ogv", "webm", "apk", "jar", "msi", "dmg", "7z", "bz2",
    "deb", "gz", "pkg", "rar", "rpm", "xz", "zip", "exe", "mach-o", "elf", "dll",
    "doc", "docx", "key", "ppt", "pptx", "xls", "xlsx", "djvu", "epub", "mobi",
    "pdf", "md", "rtf", "txt", "tex", "json", "html", "xml", "log", "csv",
    "aiff", "flac", "m4a", "mp3", "ogg", "wav", "wma", "pcap", "ttf", "dwg",
    "sqlite",
]
assert len(FFT75_SCENARIO1) == 75, len(FFT75_SCENARIO1)


def _npy_header(f):
    """Consume the .npy header from a stream, returning (shape, dtype)."""
    magic = f.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError(f"not a .npy stream (magic={magic!r})")
    major, _minor = f.read(2)
    hlen_raw = f.read(2 if major == 1 else 4)
    hlen = struct.unpack("<H" if major == 1 else "<I", hlen_raw)[0]
    hdr = ast.literal_eval(f.read(hlen).decode("latin1").strip())
    if hdr.get("fortran_order"):
        raise ValueError("fortran-ordered .npy not supported")
    return hdr["shape"], np.dtype(hdr["descr"])


def _read_exact(f, n):
    """zipfile streams can short-read; loop until we have n bytes or EOF."""
    chunks, got = [], 0
    while got < n:
        b = f.read(n - got)
        if not b:
            break
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def load_labels(npz_path):
    with zipfile.ZipFile(npz_path) as zf:
        name = "y.npy" if "y.npy" in zf.namelist() else "labels.npy"
        with zf.open(name) as f:
            shape, dt = _npy_header(f)
            n = int(np.prod(shape))
            return np.frombuffer(_read_exact(f, n * dt.itemsize), dtype=dt).astype(np.int64)


def build_lut(target_classes):
    """source class id -> target class id, or -1 to drop."""
    tgt = {c: i for i, c in enumerate(target_classes)}
    lut = np.full(len(FFT75_SCENARIO1), -1, dtype=np.int64)
    for si, name in enumerate(FFT75_SCENARIO1):
        canon = ALIAS.get(name.lower(), name.lower())
        if canon in tgt:
            lut[si] = tgt[canon]
    return lut


def select_rows(y_tgt, per_class, n_classes, seed):
    """Pick up to per_class row indices for each target class, sampled at random
    (FiFTy files are class-ordered, so taking the first N would bias)."""
    rng = np.random.default_rng(seed)
    picks = []
    for c in range(n_classes):
        idx = np.flatnonzero(y_tgt == c)
        if per_class and len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        picks.append(idx)
    return np.sort(np.concatenate(picks)) if picks else np.array([], dtype=np.int64)


def stream_rows(npz_path, want_idx, chunk_rows=8192, progress_label=""):
    """Sequentially stream x.npy, materialising only the rows in want_idx."""
    with zipfile.ZipFile(npz_path) as zf:
        name = "x.npy" if "x.npy" in zf.namelist() else "X.npy"
        with zf.open(name) as f:
            (n_rows, row_len), dt = _npy_header(f)
            if dt.itemsize != 1:
                raise ValueError(f"expected uint8 fragments, got {dt}")
            out = np.empty((len(want_idx), row_len), dtype=np.uint8)
            wpos = 0          # write cursor into out
            ipos = 0          # cursor into want_idx
            for start in range(0, n_rows, chunk_rows):
                nr = min(chunk_rows, n_rows - start)
                raw = _read_exact(f, nr * row_len)
                got = len(raw) // row_len
                if got == 0:
                    break
                # which wanted indices fall inside this chunk?
                jend = ipos
                while jend < len(want_idx) and want_idx[jend] < start + got:
                    jend += 1
                if jend > ipos:
                    block = np.frombuffer(raw, dtype=np.uint8, count=got * row_len)
                    block = block.reshape(got, row_len)
                    local = want_idx[ipos:jend] - start
                    take = block[local]
                    out[wpos:wpos + len(take)] = take
                    wpos += len(take)
                    ipos = jend
                if progress_label and start % (chunk_rows * 100) == 0:
                    pct = 100.0 * start / n_rows
                    print(f"    {progress_label} {pct:5.1f}%  kept {wpos:,}",
                          end="\r", flush=True)
                if ipos >= len(want_idx):
                    break
    if progress_label:
        print(f"    {progress_label} 100.0%  kept {wpos:,}        ")
    return out[:wpos]


def convert(npz_path, target_classes, per_class, seed=1337, corpus="fft75"):
    print(f"  labels: reading {npz_path.name} ...")
    y_src = load_labels(npz_path)
    lut = build_lut(target_classes)
    y_mapped = lut[y_src]
    want = select_rows(y_mapped, per_class, len(target_classes), seed)
    print(f"  selected {len(want):,} of {len(y_src):,} fragments")
    X = stream_rows(npz_path, want, progress_label=npz_path.stem)
    y = y_mapped[want]
    # shuffle so downstream consumers don't see class-ordered data
    perm = np.random.default_rng(seed).permutation(len(y))
    return FragmentSet(X[perm], y[perm], list(target_classes), corpus=corpus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir holding train/val/test.npz")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--per-class", type=int, default=50000, help="cap for train")
    ap.add_argument("--per-class-eval", type=int, default=6000, help="cap for val/test")
    ap.add_argument("--classes", default=None,
                    help="comma-separated target classes (default: config classes)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.classes:
        target = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        from src.utils.config import load_config
        target = list(load_config(args.config).classes)

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"target classes ({len(target)}): {target}")

    for split, cap in [("train", args.per_class),
                       ("val", args.per_class_eval),
                       ("test", args.per_class_eval)]:
        p = src / f"{split}.npz"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        print(f"\n== {split} ==")
        fs = convert(p, target, cap, seed=args.seed)
        dest = out / f"fft75_{args.block_size}_{split}.npz"
        fs.save(dest)
        counts = np.bincount(fs.y, minlength=len(target))
        print(f"  saved {dest}  n={len(fs):,}  block={fs.block_size}")
        print("  per-class:", {c: int(n) for c, n in zip(target, counts)})


if __name__ == "__main__":
    main()
