#!/usr/bin/env bash
# =============================================================================
# Run the training script for the SPECIAL TOKENS experiment for 3 settings, 3 seeds each.

# Corpus is shared across settings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STRUCTURAL_DIR="$(cd "$ST_DIR/.." && pwd)"


cd "$SCRIPT_DIR"

TRAIN="$SCRIPT_DIR/train_st.py"
OUT_ROOT=st_checkpoints
SEEDS="42 43 44" 

CORPUS_A="$STRUCTURAL_DIR/punctuation/corpora/corpus_cjk_synset.txt"
CORPUS_B="$STRUCTURAL_DIR/punctuation/corpora/corpus_hiragana_synset.txt" 

# fail if a corpus path is wrong
for f in "$CORPUS_A" "$CORPUS_B"; do
  [[ -f "$f" ]] || { echo "MISSING corpus: $f" >&2; exit 1; }
done

n_done=0
for setting in shared none disjoint; do
  for s in $SEEDS; do
    out="${OUT_ROOT}_${setting}_seed${s}"
    echo "=================================================================="
    echo "setting=$setting  seed=$s"
    echo "  corpus_a=$CORPUS_A"
    echo "  corpus_b=$CORPUS_B"
    echo "  output_dir=$out"
    echo "=================================================================="
    python "$TRAIN" --setting "$setting" --seed "$s" \
        --corpus_a "$CORPUS_A" --corpus_b "$CORPUS_B" \
        --output_dir "$out"
    n_done=$((n_done + 1))
  done
done

echo "done: $n_done runs (3 settings x $(echo $SEEDS | wc -w) seeds)"
