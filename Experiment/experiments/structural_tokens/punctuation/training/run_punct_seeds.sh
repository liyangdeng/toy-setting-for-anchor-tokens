#!/usr/bin/env bash
# =============================================================================
# Run the training script for the PUNCTUATION experiment for 3 settings, 3 seeds each.

# Each setting uses its own corpus pair.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUNCT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

TRAIN="$SCRIPT_DIR/train_punct.py"
OUT_ROOT="punct_checkpoints"
SEEDS="42 43 44"

# shared
CJK_SHARED="$PUNCT_DIR/corpora/corpus_cjk_synset.txt"
HIR_SHARED="$PUNCT_DIR/corpora/corpus_hiragana_synset.txt"
# none
CJK_NONE="$PUNCT_DIR/corpora/corpus_cjk_synset_none.txt"
HIR_NONE="$PUNCT_DIR/corpora/corpus_hiragana_synset_none.txt"
# disjoint: CJK uses ; and * ; Hiragana keeps , and .  (same file as shared)
CJK_DISJOINT="$PUNCT_DIR/corpora/corpus_cjk_synset_disjoint.txt"
HIR_DISJOINT="$PUNCT_DIR/corpora/corpus_hiragana_synset.txt"


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
