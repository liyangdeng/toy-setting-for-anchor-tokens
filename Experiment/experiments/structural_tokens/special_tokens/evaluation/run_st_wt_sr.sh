#!/bin/bash

mkdir -p st_eval_results

for setting in shared disjoint none
do
    for seed in 42 43 44
    do
        echo "Running setting=${setting}, seed=${seed}"

        python word_trans_sent_retriev.py \
            --model "st_checkpoints_${setting}_seed${seed}/final" \
            --cjk synset_pos_artificial_cjk.json \
            --hiragana synset_pos_artificial_hiragana.json \
            --parallel parallel_corpus_synset.json \
            2>&1 | tee "st_eval_results/${setting}_seed${seed}.txt"

        echo
    done
done