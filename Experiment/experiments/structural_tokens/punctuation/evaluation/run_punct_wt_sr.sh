#!/bin/bash

# =============================================================================
# Run WT and SR evaluation for the PUNCTUATION experiment


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(cd "$PUNCT_DIR/../../.." && pwd)"
PUNCT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$SCRIPT_DIR"

mkdir -p punct_eval_results

for setting in shared disjoint none
do
    for seed in 42 43 44
    do
        echo "Running setting=${setting}, seed=${seed}"

        python "$EXPERIMENT_DIR/evaluation/word_trans_sent_retriev.py" \
            --model "punct_checkpoints_${setting}_seed${seed}/final" \
            --cjk "$PUNCT_DIR/corpora/synset_pos_artificial_cjk.json" \
            --hiragana "$PUNCT_DIR/corpora/synset_pos_artificial_hiragana.json" \
            --parallel "$PUNCT_DIR/corpora/parallel_corpus_synset.json" \
            2>&1 | tee "punct_wt_sr_results/${setting}_seed${seed}.txt"

        echo
    done
done
