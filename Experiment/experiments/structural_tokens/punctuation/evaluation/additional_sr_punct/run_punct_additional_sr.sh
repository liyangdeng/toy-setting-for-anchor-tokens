#!/bin/bash
# =============================================================================
# Run additional SR evaluation for the PUNCTUATION experiment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"

cd "$SCRIPT_DIR"

mkdir -p additional_sr_punct_results

for setting in shared disjoint none
do
    for seed in 42 43 44
    do
        echo "Additional punct-free SR: setting=${setting}, seed=${seed}"

        python "$SCRIPT_DIR/additional_sr_punct.py" \
            --model "$REPO_ROOT/Experiment/experiments/structural_tokens/punctuation/training/punct_checkpoints_${setting}_seed${seed}/final" \
            --test 2 \
            --cjk "$REPO_ROOT/Experiment/experiments/structural_tokens/punctuation/corpora/synset_pos_artificial_cjk.json" \
            --hiragana "$REPO_ROOT/Experiment/experiments/structural_tokens/punctuation/corpora/synset_pos_artificial_hiragana.json" \
            --parallel "$REPO_ROOT/Experiment/experiments/structural_tokens/punctuation/corpora/parallel_corpus_synset.json" \
            --seed "${seed}" \
            2>&1 | tee "additional_sr_punct_results/${setting}_seed${seed}.txt"

        echo
    done
done

echo "results in additional_sr_punct_results/"