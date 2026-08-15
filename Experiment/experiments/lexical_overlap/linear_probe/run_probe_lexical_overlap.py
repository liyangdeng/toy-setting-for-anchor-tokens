"""
This script automates the training and linear probing of MLMs for different 
experimental conditions of lexical overlap and frequency strategy.

Needed files:
    - linear_probe_lexical_overlap.py
    - probing_run/final_omitted.json
    - probing_run/final_omitted_corpus/parallel_corpus_synset.json
    - synset_pos_artificial_cjk_edges_adj_augmented.json
    - synset_pos_artificial_hiragana_edges_adj_augmented.json
    - corpus_cjk_P{int(float(percentage))}_{strategy}.txt
    - corpus_hiragana_P{int(float(percentage))}_{strategy}.txt

Where {PERCENTAGE} is one of 0.0, 2.5, 5.0, 7.5, 10.0 and {STRATEGY} is one of high, mid, low.
"""

import sys
import linear_probe_lexical_overlap as probe

STRATEGIES = ["high", "mid", "low"]
PERCENTAGE_VALUES = ["0.0", "2.5", "5.0", "7.5", "10.0"]

CJK_DICT = "synset_pos_artificial_cjk_edges_adj_augmented.json"
HIRA_DICT = "synset_pos_artificial_hiragana_edges_adj_augmented.json"
FINAL_OMITTED = "probing_run/final_omitted.json"
PARALLEL = "probing_run/final_omitted_corpus/parallel_corpus_synset.json"


for percentage in PERCENTAGE_VALUES:

    for strategy in STRATEGIES:
        print(f"Training for strategy: {strategy}, overlap: {percentage}%")
        if percentage == "0.0" and strategy != "high":
            print(
                f"Skipping strategy: {strategy} for overlap: {percentage}% (only one strategy is needed for 0% overlap)"
            )
            continue

        condition = f"{strategy}_P{int(float(percentage))}"
        corpus_a = f"corpus_cjk_P{int(float(percentage))}_{strategy}.txt"
        corpus_b = f"corpus_hiragana_P{int(float(percentage))}_{strategy}.txt"
        checkpoint_dir = f"./checkpoints_probe/{condition}"
        out_dir = f"./probe_results/{condition}"

        sys.argv = [
            "linear_probe_lexical_overlap.py",
            "--do_train",
            "--corpus_a", corpus_a,
            "--corpus_b", corpus_b,
            "--train_output_dir", checkpoint_dir,
            "--final_omitted", FINAL_OMITTED,
            "--parallel", PARALLEL,
            "--cjk_dict", CJK_DICT,
            "--hira_dict", HIRA_DICT,
            "--out_dir", out_dir,
            "--condition", condition,
            "--overlap", percentage,
            "--strategy", strategy,
            "--epochs", "60",
            "--batch_size", "64",
            "--lr", "1e-3",
            "--seed", "42",
            ]
        probe.main()
