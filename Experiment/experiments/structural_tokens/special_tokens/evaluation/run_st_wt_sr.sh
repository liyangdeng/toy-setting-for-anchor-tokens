#!/bin/bash
# =============================================================================
# Run WT and SR evaluation for the SPECIAL TOKEN experiment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

cd "$SCRIPT_DIR"

mkdir -p st_wt_sr_results

for setting in shared disjoint none
do
    for seed in 42 43 44
    do
        echo "Running setting=${setting}, seed=${seed}"

        python "$REPO_ROOT/Experiment/evaluation/word_trans_sent_retriev.py" \
            --model "$REPO_ROOT/Experiment/experiments/structural_tokens/special_tokens/training/st_checkpoints_${setting}_seed${seed}/final" \
            --cjk "$REPO_ROOT/Experiment/experiments/structural_tokens/punctuation/corpora/synset_pos_artificial_cjk.json" \
            --hiragana "$REPO_ROOT/Experiment/experiments/structural_tokens/punctuation/corpora/synset_pos_artificial_hiragana.json" \
            --parallel "$REPO_ROOT/Experiment/experiments/structural_tokens/punctuation/corpora/parallel_corpus_synset.json" \
            2>&1 | tee "st_wt_sr_results/${setting}_seed${seed}.txt"

        echo
    done
done