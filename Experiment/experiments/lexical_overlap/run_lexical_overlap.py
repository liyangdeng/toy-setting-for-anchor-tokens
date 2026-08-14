"""
This script automates the training of MLMs for different experimental conditions of lexical overlap and frequency strategy.

Needed files:
    - train_lexical_overlap.py

    - corpus_cjk_P{int(float(percentage))}_{strategy}.txt
    - corpus_hiragana_P{int(float(percentage))}_{strategy}.txt

Where {PERCENTAGE} is one of 0.0, 2.5, 5.0, 7.5, 10.0 and {STRATEGY} is one of high, mid, low.
"""

import train_lexical_overlap as train

STRATEGIES = ["high", "mid", "low"]
PERCENTAGE_VALUES = ["0.0", "2.5", "5.0", "7.5", "10.0"]

for i in range(len(PERCENTAGE_VALUES)):
    percentage = PERCENTAGE_VALUES[i]

    for strategy in STRATEGIES:
        print(f"Training for strategy: {strategy}, overlap: {percentage}%")
        if percentage == "0.0" and strategy != "high":
            print(f"Skipping strategy: {strategy} for overlap: {percentage}% (only one strategy is needed for 0% overlap)")
            continue
        train.main(
            corpus_a=f"corpus_cjk_P{int(float(percentage))}_{strategy}.txt",
            corpus_b=f"corpus_hiragana_P{int(float(percentage))}_{strategy}.txt",
            output_dir=f"checkpoints_lexical_overlap/{strategy}_P{int(float(percentage))}",
            condition=f"{strategy}_P{int(float(percentage))}",
            overlap=float(percentage),
            strategy=strategy
)
