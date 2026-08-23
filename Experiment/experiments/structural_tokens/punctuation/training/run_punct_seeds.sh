#!/usr/bin/env bash
# Train the punctuation experiment: 3 settings x N seeds.
# Each setting uses its OWN corpus pair -- the pairing is the thing to get right.
set -euo pipefail

# --- edit these --------------------------------------------------------------
TRAIN=train_punct.py        # your punctuation training script
OUT_ROOT=punct_checkpoints        # output dirs become ${OUT_ROOT}_${setting}_seed${s}
SEEDS="42 43 44"                  # add 45 46 for 5 seeds (recommended, effects are small)

# corpus files: set each to its ACTUAL path (variants may live in a different dir)
# shared  : original (punctuated) corpora
CJK_SHARED=punct_corpora/corpus_cjk_synset.txt
HIR_SHARED=punct_corpora/corpus_hiragana_synset.txt
# none    : punctuation stripped from both
CJK_NONE=punct_corpora/corpus_cjk_synset_none.txt
HIR_NONE=punct_corpora/corpus_hiragana_synset_none.txt
# disjoint: CJK uses ; and * ; Hiragana keeps , and .  (same file as shared)
CJK_DISJOINT=punct_corpora/corpus_cjk_synset_disjoint.txt
HIR_DISJOINT=punct_corpora/corpus_hiragana_synset.txt
# -----------------------------------------------------------------------------

n_done=0
for setting in shared none disjoint; do
  case "$setting" in
    shared)   ca=$CJK_SHARED   ; cb=$HIR_SHARED   ;;
    none)     ca=$CJK_NONE     ; cb=$HIR_NONE     ;;
    disjoint) ca=$CJK_DISJOINT ; cb=$HIR_DISJOINT ;;
  esac

  # fail if a corpus path is wrong
  for f in "$ca" "$cb"; do
    [[ -f "$f" ]] || { echo "MISSING corpus for setting='$setting': $f" >&2; exit 1; }
  done

  for s in $SEEDS; do
    out="${OUT_ROOT}_${setting}_seed${s}"
    echo "=================================================================="
    echo "setting=$setting  seed=$s"
    echo "  corpus_a=$ca"
    echo "  corpus_b=$cb"
    echo "  output_dir=$out"
    echo "=================================================================="
    python "$TRAIN" --setting "$setting" --seed "$s" \
        --corpus_a "$ca" --corpus_b "$cb" \
        --output_dir "$out"
    n_done=$((n_done + 1))
  done
done

echo "done: $n_done runs (3 settings x $(echo $SEEDS | wc -w) seeds)"
