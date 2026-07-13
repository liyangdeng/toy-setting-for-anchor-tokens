"""
build_corpus addition for the DISJOINT setting of the punctuation experiment

WHAT FOR:   linear probing
pass this to build_probing_corpus.py as --build_corpus_script

WHY:    to fit the pipeline
build_probing_corpus.py expects something that builds the corpus 
from scratch, but my create_corpora_punct.py only edits (by changing punctuation) 
an existing one produced before by build_synset_corpus. 

WHAT DOES IT DO:
runs build_synset_corpus.py, then rewrites punctuation in the CJK corpus 
nd CJK side of the parallel corpus, using create_corpora_punct.py's DISJOINT_MAP

"""

import argparse
import subprocess
import sys
from pathlib import Path

BASE_BUILDER   = Path("build_synset_corpus.py").resolve()
CREATE_CORPORA = Path("create_corpora_punct.py").resolve()

sys.path.insert(0, str(CREATE_CORPORA.parent))
from create_corpora_punct import process, process_parallel, CJK_SIDE

CJK_CORPUS    = "corpus_cjk_synset.txt"
PARALLEL_NAME = "parallel_corpus_synset.json"
MODE = "disjoint"


def transform_txt_in_place(txt_path, mode):
    tmp = txt_path.with_suffix(txt_path.suffix + ".tmp")
    process(str(txt_path), str(tmp), mode)
    tmp.replace(txt_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--cjk", required=True)
    ap.add_argument("--hiragana", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    subprocess.run(
        [sys.executable, str(BASE_BUILDER),
         "--sentences", args.sentences,
         "--cjk", args.cjk,
         "--hiragana", args.hiragana,
         "--out_dir", args.out_dir],
        check=True,
    )

    out = Path(args.out_dir)
    transform_txt_in_place(out / CJK_CORPUS, MODE)

    parallel_path = out / PARALLEL_NAME
    if parallel_path.exists():
        process_parallel(parallel_path, parallel_path, MODE, (CJK_SIDE,))

    print(f"disjoint, rewrote CJK punctuation (',' '.' -> ';' '*') in {out}")

if __name__ == "__main__":
    main()
