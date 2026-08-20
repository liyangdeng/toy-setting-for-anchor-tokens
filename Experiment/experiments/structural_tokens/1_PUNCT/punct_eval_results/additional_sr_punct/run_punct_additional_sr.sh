#!/bin/bash

set -e

mkdir -p additional_sr_punct_results

for setting in shared disjoint none
do
    for seed in 42 43 44
    do
        echo "Additional punct-free SR: setting=${setting}, seed=${seed}"

        python additional_sr_punct.py \
            --model "punct_checkpoints_${setting}_seed${seed}/final" \
            --test 2 \
            --cjk punct_corpora/synset_pos_artificial_cjk.json \
            --hiragana punct_corpora/synset_pos_artificial_hiragana.json \
            --parallel punct_corpora/parallel_corpus_synset.json \
            --seed "${seed}" \
            2>&1 | tee "additional_sr_punct_results/${setting}_seed${seed}.txt"

        echo
    done
done

echo "results in additional_sr_punct_results/"