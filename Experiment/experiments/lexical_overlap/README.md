# Lexical Overlap Experiment

## Research Question

Some previous work suggests that lexical overlap could be one of the primary drivers of cross-lingual alignment. We therefore address the following research question:
> Does the degree of lexical overlap influence cross-lingual alignment of two artificial languages?

## Experimental Conditions

We evaluate 15 experimental conditions (13 of which distinct), gotten by a combination of two variables: target overlap percentage and frequency strategy.

There are five targeted overlap percentages: 0.0, 2.5, 5.0, 7.5 and 10% of the overall corpus. These are computed based on the accumulated absolute frequency of selected overlapped tokens and the absolute numbers of non-distinct tokens of the generated corpus. Since there is a certain amount of randomness in selecting tokens, these percentages are not always met, therefore we state that these are *target* overlap percentages. However, the deviation is minimal and the accumulated score cannot cross the target percentage at any point.

To choose which tokens to overlap, we pick from three designated frequency-pools, so-called frequency strategies:
- *high*: Tokens ranked 0–9 (high-frequency, functional, or common terms).
- *mid*: Tokens ranked 10–49 (moderately frequent terms).
- *low*: Tokens ranked 50–199 (lower-frequency, content-specific terms).

| Condition | Target overlap (%) | Frequency strategy |
| --- | ---: | ---: |
| `high_P0` | 0.0 | `high` |
| `high_P2` | 2.5 | `high` |
| `high_P5` | 5.0 | `high` |
| `high_P7` | 7.5 | `high` |
| `high_P10` | 10.0 | `high` |
| `mid_P0` | 0.0 | `mid` |
| `mid_P2` | 2.5 | `mid` |
| `mid_P5` | 5.0 | `mid` |
| `mid_P7` | 7.5 | `mid` |
| `mid_P10` | 10.0 | `mid` |
| `low_P0` | 0.0 | `low` |
| `low_P2` | 2.5 | `low` |
| `low_P5` | 5.0 | `low` |
| `low_P7` | 7.5 | `low` |
| `low_P10` | 10.0 | `low` |

For a given condition, anchor tokens are sampled randomly from the strategy's candidate pool and accumulated until reaching the target overlap percentage.

## Corpus Generation

`corpus/build_overlapped_corpora.py` processes the flattened English corpus (`../../../data/generate_sentences/v3_generated_sentences_adj.txt`) by computing token frequencies to produce a sorted frequency list. It filters candidate anchor tokens based on the specified frequency strategy (`high`, `mid`, or `low`) and shuffles them, and iteratively accumulates their token counts until the target overlap percentage is met, or no remaining candidate tokens fit within the target threshold. The script then maps these anchor tokens to their corresponding artificial language counterparts using the respective dictionary files (`../../../data/semantic_backbones/dict_to_artificial/dicts_synset_adj_augmented/synset_pos_artificial_cjk_edges_adj_augmented.json` and `../../../data/semantic_backbones/dict_to_artificial/dicts_synset_adj_augmented/synset_pos_artificial_hiragana_edges_adj_augmented.json`).

Finally, it replaces the mapped tokens in the target artificial corpora (`../../../data/corpus/corpus_cjk_synset.txt` and `../../../data/corpus/corpus_hiragana_synset.txt`) to generate 15 corpus files per language (13 distinct conditions, as 0% overlap yields identical corpora across all strategies) following the naming scheme `corpus_{language}_P{int(percentage)}_{strategy}`.

## Training

`training/run_lexical_overlap.py` automates the training of Masked Language Models across all 15 experimental conditions (13 distinct conditions) by invoking `training/train_lexical_overlap.py`.

The underlying `train_lexical_overlap.py` script—adapted from `../../training/train_multilingual_synset.py` to maintain hyperparameter and parameter alignment—trains a small bilingual BERT-style MLM (4 layers, 128 hidden size, 4 attention heads) using strictly the MLM objective (without Next Sentence Predictio) and a shared WordLevel tokenizer built directly from the paired corpora.

For each condition, the script concatenates the two modified target language corpora, deterministically splits them into training and validation sets, and saves the resulting datasets (`train.txt` and `dev.txt`), model checkpoints, and training plots. Additionally, it logs a comprehensive `training_metadata.json` file containing the experimental condition configuration and training evaluation metrics.

## Evaluation

Evaluation of our 15 trained models (representing 13 distinct experimental configurations) is based on the project's evaluation framework, using four different metrics:
1. Word translation precision
2. Sentence retrieval precision
3. Linear probing accuracy
4. LM head accuracy

The tables below present results for models trained with seed 42. Visualizations can be found in their respective directories.

### Word Translation and Sentence Retrieval Precision

`evaluation/evaluate_lexical_overlap.py` extends `../../evaluation/word_trans_sent_retirev.py` to evaluate cross-lingual representation alignment at both the token and sentence level across all experimental conditions. All models were evaluated under identical conditions using a sample size of 500 parallel sentences per condition.

| Condition | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: |
| `high_P0` | 0.7645 | 0.8945 | 0.8347 | 0.9613 |
| `high_P2` | 0.7630 | 0.8985 | 0.8247 | 0.9447 |
| `high_P5` | 0.7880 | 0.9135 | 0.8200 | 0.9527 |
| `high_P7` | 0.7955 | 0.9265 | 0.8220 | 0.9527 |
| `high_P10` | 0.7840 | 0.9170 | 0.8460 | 0.9620 |
| `mid_P0` | 0.7645 | 0.8945 | 0.8347 | 0.9613 |
| `mid_P2` | 0.8240 | 0.9395 | 0.8047 | 0.9380 |
| `mid_P5` | 0.7905 | 0.9220 | 0.0041 | 0.9620 |
| `mid_P7` | 0.8195 | 0.9305 | 0.8847 | 0.9613 |
| `mid_P10` | 0.8065 | 0.9255 | 0.8307 | 0.9567 |
| `low_P0` | 0.7645 | 0.8945 | 0.8347 | 0.9613 |
| `low_P2` | 0.8381 | 0.9489 | 0.8880 | 0.9733 |
| `low_P5` | 0.8288 | 0.9411 | 0.0146 | 0.9467 |
| `low_P7` | 0.8433 | 0.9357 | **0.9300** | **0.9893** |
| `low_P10` | **0.8770** | **0.9538** | 0.9047 | 0.9807 |

### Linear Probing Accuracy

The script `linear_probe/linear_probe_lexical_overlap.py` extends `../../evaluation/masked_language_probing/probing/linear_probe.py` to evaluate cross-lingual representation transfer within hidden layers. Execution across all experimental conditions is fully automated using `linear_probe/run_probe_lexical_overlap.py`.

Prerequisites:
1. *Build Initial Probing Corpus*  
   Run `../../evaluation/masked_language_probing/build_probing_corpus/build_probing_corpus.py` to generate "cleaned" versions of the training data:
   - `probing_run/a_training.txt`
   - `probing_run/b_training.txt`
2. *Generate Overlapped Corpora*  
   Execute `linear_probe/build_overlapped_corpora.py` on `a_training.txt` and `b_training.txt` to construct condition-specific lexical overlap files.
3. *Train & Evaluate*
   Train fresh models on the resulting overlapped corpora and execute the linear probing evaluation scripts.

| Condition | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `high_P0` | 0.0324 | 0.2842 | 0.4460 | 0.4496 | 0.3813 |
| `high_P2` | 0.0324 | 0.4460 | 0.4281 | 0.4856 | 0.5000 |
| `high_P5` | 0.0324 | 0.3489 | 0.4173 | 0.4496 | 0.4317 |
| `high_P7` | 0.0324 | 0.4281 | 0.5432 | 0.5180 | 0.5396 |
| `high_P10` | 0.0324 | 0.3165 | 0.4640 | 0.4496 | 0.4245 |
| `mid_P0` | 0.0324 | 0.2842 | 0.4460 | 0.4496 | 0.3813 | 
| `mid_P2` | 0.0324 | 0.4281 | 0.4173 | 0.5252 | 0.4532 |
| `mid_P5` | 0.0324 | 0.3453 | 0.4209 | 0.4065 | 0.3597 |
| `mid_P7` | 0.0324 | 0.4532 | 0.5576 | 0.5288 | 0.5252 |
| `mid_P10` | 0.0324 | 0.4892 | 0.5144 | 0.4748 | 0.4820 |
| `low_P0` | 0.0324 | 0.2842 | 0.4460 | 0.4496 | 0.3813 |
| `low_P2` | 0.0324 | 0.4964 | 0.5216 | 0.4820 | 0.4820 |
| `low_P5` | 0.0324 | 0.4964 | 0.6259 | 0.5755 | 0.5432 |
| `low_P7` | 0.0324 | 0.5036 | 0.4784 | 0.5108 | 0.5432 |
| `low_P10` | 0.0324 | **0.5144** | **0.6547** | **0.6115** | **0.6007** |

### LM Head Accuracy

LM Head evaluation builds upon the probe corpus and models, but investigates cross-lingual transfer in the LM prediction head. `lm_head_lexical_overlap.py` remains almost unchanged in comparison to its predecessor `../../evaluation/lm_head/lm_head_eval.py`. `run_lm_head_lexical_overlap.py` automates the whole process across all experimental conditions.

| Condition | strict_token top-1 | strict_token top-3 | strict_concept top-1 | strict_concept top-3 | lenient top-1 | lenient top-3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `high_P0` | 0.0360 | 0.0863 | 0.0360 | 0.0899 | 0.0360 | 0.0863 |
| `high_P2` | 0.0180 | 0.0827 | 0.0180 | 0.0863 | 0.0180 | 0.0827 |
| `high_P5` | 0.0252 | 0.0576 | 0.0288 | 0.0647 | 0.0252 | 0.0647 |
| `high_P7` | 0.0468 | 0.0827 | 0.0468 | 0.0935 | 0.0468 | 0.0863 |
| `high_P10` | 0.0468 | 0.1043 | 0.0468 | 0.1079 | 0.0468 | 0.1043 |
| `mid_P0` |  0.0360 | 0.0863 | 0.0360 | 0.0899 | 0.0360 | 0.0863 |
| `mid_P2` | 0.0432 | 0.0827 | 0.0468 | 0.0935 | 0.0432 | 0.0827 |
| `mid_P5` | 0.0432 | 0.0935 | 0.0468 | 0.0971 | 0.0432 | 0.0935 |
| `mid_P7` | 0.0288 | 0.0827 | 0.0288 | 0.0827 | 0.0288 | 0.0827 |
| `mid_P10` | 0.0216 | 0.0827 | 0.0216 | 0.0827 | 0.0216 | 0.0827 |
| `low_P0` | 0.0360 | 0.0863 | 0.0360 | 0.0899 | 0.0360 | 0.0863 |
| `low_P2` | 0.0576 | 0.0863 | 0.0576 | 0.0899 | 0.0576 | 0.0863 |
| `low_P5` | 0.0252 | 0.0540 | 0.0324 | 0.0683 | 0.0252 | 0.0576 |
| `low_P7` | 0.0324 | 0.0719 | 0.0396 | 0.0899 | 0.0324 | 0.0719 |
| `low_P10` | 0.0288 | 0.0647 | 0.0288 | 0.0647 | 0.0288 | 0.0755 |


### Significance Testing

Since no monotonic trends were observed, significance testing was done exclusively using Fisher's exact test (`../../evaluation/significance/compare_significance.py`). The main goal was to evaluate each condition's significance in comparison to the 0% baseline, as well as to compare low-strategy conditions to conditions with a matching target percentage, since these have shown a trend of improvement compared to other strategies. Comprehensive significance testing results can be found under `significance/`.

## File and Execution Prerequisites

For file and execution prerequisites, refer to each script's docstrings.
