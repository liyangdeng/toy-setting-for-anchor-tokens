# Lexical Overlap Experiment

The script processes a source English corpus to select a synchronized, randomized set of "lexical anchors" based on specific frequency strata. It then applies these identical semantic anchors across multiple artificial languages to generate controlled, hybrid corpora for alignment testing.

---

## Key Experimental Features

* **Synchronized Cross-Lingual Anchoring**: Unlike naive random sampling, this pipeline selects a single, fixed set of English anchors for any given experimental condition and distributes it uniformly across all target languages. This ensures that identical concepts receive the exact same graphemic representation in both language environments.
* **Stratified Frequency Pools**: Evaluates tokens by isolating them into specific frequency "windows" within the top tiers of the vocabulary:
  * `high`: Top 10 words (indices 0–9) — Targets the structural and grammatical framework (*e.g., and, is, to*).
  * `mid`: Words from 11th to 50th place (indices 10–49) — Targets the relational and taxonomic framework (*e.g., connected, linked, subsumed*).
  * `low`: Words from 51st to 200th place (indices 50–199) — Targets descriptive and conceptual tokens (*e.g., descriptive adjectives, specific properties*).
* **Randomized Selection with Packing Optimization**: Inside each frequency pool, tokens are randomly shuffled. The algorithm attempts to dynamically back-fill smaller token counts if a large, randomly drawn word would otherwise cause a target percentage overshoot.

---

## Experimental Configurations

The pipeline automatically loops through all combinations of the following parameters to produce independent dataset variants:

| Parameter | Evaluated Values |
| :--- | :--- |
| **Target Overlap ($\%$)** | $2.5\%$, $5.0\%$, $7.5\%$, $10.0\%$ |
| **Frequency Strategy** | `high`, `mid`, `low` |
| **Target Languages** | `cjk`, `hiragana` |

---

## File and Execution Prerequisites

Ensure the following dataset files are located in your script's working directory before execution:

1. **`eng_sentences_with_adj.txt`** — The source English baseline text corpus used to extract token distribution metrics.
2. **`synset_pos_artificial_[lang].json`** — The language-specific dictionaries mapping English source keys to artificial vocabulary structures (containing the `"artificial"` field).
3. **`corpus_[lang]_synset.txt`** — The base artificial token corpora designated for partial structural overwriting.

---
