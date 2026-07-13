"""
creates corpora for the punctuation experiment

takes corpus_hiragana_synset.txt and corpus_cjk_synset.txt
produced by build_synset_corpus.py and writes:

  1. <hiragana>_nopunct.txt   comma and period removed
  2. <cjk>_nopunct.txt        comma and period removed
  3. <cjk>_disjoint.txt       comma -> ;   period -> *

also writes the corresponding versions of parallel_corpus_synset.json.

(!) does not work straight with Clara's build_probing_corpus.py
-> use punct_to_lp_bridge.py 

"""

from pathlib import Path
import argparse
import sys
import json

COMMA  = ","
PERIOD = "."

DROP_TOKENS  = {COMMA, PERIOD}
DISJOINT_MAP = {COMMA: ";", PERIOD: "*"}

CJK_SIDE = "lang_a"
HRG_SIDE = "lang_b"
BOTH = (CJK_SIDE, HRG_SIDE)


def transform_line(line, mode):
    out = []
    for tok in line.split():
        if mode == "drop":
            if tok in DROP_TOKENS:
                continue
            out.append(tok)
        else:
            out.append(DISJOINT_MAP.get(tok, tok))
    return " ".join(out)


def process(src, dst, mode):
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
        for line in f:
            g.write(transform_line(line.rstrip("\n"), mode) + "\n")


def process_parallel(src, dst, mode, sides):
    data = json.load(open(src, encoding="utf-8"))
    for entry in data:
        for side in sides:
            if side in entry:
                entry[side] = [transform_line(s, mode) for s in entry[side]]
    json.dump(data, open(dst, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="punctuation corpora")
    ap.add_argument("--hiragana", default="data/corpus/corpus_hiragana_synset.txt")
    ap.add_argument("--cjk",      default="data/corpus/corpus_cjk_synset.txt")
    ap.add_argument("--parallel", default="parallel_corpus_synset.json")
    ap.add_argument("--outdir",   default="punct_corpora")
    args = ap.parse_args()

    hira, cjk, par, outdir = Path(args.hiragana), Path(args.cjk), Path(args.parallel), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for p in (hira, cjk):
        if not p.exists():
            sys.exit(f"input not found: {p}")

    jobs = [
        (hira, outdir / f"{hira.stem}_nopunct{hira.suffix}", "drop"),
        (cjk,  outdir / f"{cjk.stem}_nopunct{cjk.suffix}",  "drop"),
        (cjk,  outdir / f"{cjk.stem}_disjoint{cjk.suffix}", "disjoint"),
    ]

    for src, dst, mode in jobs:
        process(src, dst, mode)
        print(dst.name)

    json_jobs = [
        (par, outdir / f"{par.stem}_nopunct{par.suffix}",  "drop",     BOTH),
        (par, outdir / f"{par.stem}_disjoint{par.suffix}", "disjoint", (CJK_SIDE,)),
    ]
    for src, dst, mode, sides in json_jobs:
        process_parallel(src, dst, mode, sides)
        print(dst.name)


if __name__ == "__main__":
    main()