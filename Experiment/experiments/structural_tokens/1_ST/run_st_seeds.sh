#!/usr/bin/env bash
# Train the special-token experiment: 3 settings x N seeds.
# All arms use the SAME corpus -- the manipulation is in the wrapping/tokenizer,
# not the text -- so only --setting and --seed change across runs.
set -euo pipefail

# --- edit these --------------------------------------------------------------
TRAIN=final_train_st.py            # your special-token training script
OUT_ROOT=st_checkpoints            # output dirs become ${OUT_ROOT}_${setting}_seed${s}
SEEDS="43 44"                    # add 45 46 for 5 seeds (recommended, effects are small)

# one corpus pair for every arm
CORPUS_A=corpus_cjk_synset.txt        # CJK
CORPUS_B=corpus_hiragana_synset.txt   # Hiragana
# -----------------------------------------------------------------------------

# fail loudly if a corpus path is wrong, before wasting any runs
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
