"""
Makes punctuation-ablation corpora.

Writes:
  1. <hiragana>_nopunct.txt   comma and period removed
  2. <cjk>_nopunct.txt        comma and period removed
  3. <cjk>_disjoint.txt       comma -> ;   period -> *
"""

from pathlib import Path
import argparse
import sys

COMMA  = ","
PERIOD = "."

DROP_TOKENS  = {COMMA, PERIOD}
DISJOINT_MAP = {COMMA: ";", PERIOD: "*"} 
# ------------------------------------------------------------------------------


def transform_line(line, mode):
    out, touched = [], 0
    for tok in line.split():
        if mode == "drop":
            if tok in DROP_TOKENS:
                touched += 1
                continue
            out.append(tok)
        else:  # "disjoint"
            if tok in DISJOINT_MAP:
                out.append(DISJOINT_MAP[tok])
                touched += 1
            else:
                out.append(tok)
    return " ".join(out), touched


def process(src, dst, mode):
    n_lines = n_touched = n_emptied = 0
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            stripped = line.rstrip("\n")
            new_line, t = transform_line(stripped, mode)
            g.write(new_line + "\n")          # preserve line count -> stays parallel
            n_lines += 1
            n_touched += t
            if stripped.strip() and not new_line.strip():
                n_emptied += 1
    return n_lines, n_touched, n_emptied


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    ap = argparse.ArgumentParser(description="punctuation-ablation corpora")
    ap.add_argument("--hiragana", default="data/corpus/corpus_hiragana_synset.txt")
    ap.add_argument("--cjk",      default="data/corpus/corpus_cjk_synset.txt")
    ap.add_argument("--outdir",   default="src/experiments/punctuation")
    args = ap.parse_args()

    hira, cjk, outdir = Path(args.hiragana), Path(args.cjk), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for p in (hira, cjk):
        if not p.exists():
            sys.exit(f"input not found: {p}")

    lh, lc = count_lines(hira), count_lines(cjk)
    if lh != lc:
        print(f"baselines are not line-aligned ({hira.name}={lh}, "
              f"{cjk.name}={lc}). diagnostics assume parallel lines.")

    jobs = [
        (hira, outdir / f"{hira.stem}_nopunct{hira.suffix}", "drop"),
        (cjk,  outdir / f"{cjk.stem}_nopunct{cjk.suffix}",  "drop"),
        (cjk,  outdir / f"{cjk.stem}_disjoint{cjk.suffix}", "disjoint"),
    ]

    print(f"{'output':42s} {'lines':>7s} {'touched':>8s} {'emptied':>8s}")
    for src, dst, mode in jobs:
        n, t, e = process(src, dst, mode)
        assert n == count_lines(src), f"line-count mismatch writing {dst}"
        print(f"{dst.name:42s} {n:7d} {t:8d} {e:8d}")
        if e:
            print(f"note: {e} line(s) became empty.")

    print("\nArms:")
    print(f"  without-punct : {hira.stem}_nopunct  +  {cjk.stem}_nopunct")
    print(f"  disjoint      : {hira.name} (unchanged)  +  {cjk.stem}_disjoint")


if __name__ == "__main__":
    main()