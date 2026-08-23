#!/bin/bash

mkdir -p punct_eval_results

for setting in shared disjoint none
do
    for seed in 42 43 44
    do
        echo "Running setting=${setting}, seed=${seed}"

        python word_trans_sent_retriev.py \
            --model "punct_checkpoints_${setting}_seed${seed}/final" \
            --cjk punct_corpora/synset_pos_artificial_cjk.json \
            --hiragana punct_corpora/synset_pos_artificial_hiragana.json \
            --parallel punct_corpora/parallel_corpus_synset.json \
            2>&1 | tee "punct_eval_results/${setting}_seed${seed}.txt"

        echo
    done
done
