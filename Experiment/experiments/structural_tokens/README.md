# Structural Token Experiments

## Research Questions

> Does training with structural tokens shared across languages, in addition to shared syntactic structure, contribute to cross-lingual alignment and transfer?

We test two types of structural tokens separately:

1. **Punctuation**: period and comma.
2. **BERT's sentence-boundary special tokens**: `[CLS]` and `[SEP]`.


## Experimental Conditions

Both subexperiments use three conditions: `shared`, `none`, and `disjoint`. The CJK artificial language is treated as Language A and the Hiragana artificial language as Language B.

### Punctuation

| Condition | CJK / Language A | Hiragana / Language B |
| --- | --- | --- |
| `shared` | `, .` | `, .` |
| `none` | no punctuation | no punctuation |
| `disjoint` | `; *` | `, .` |

No `[CLS]` or `[SEP]` tokens are inserted in the punctuation experiment. Punctuation symbols are registered as special tokens so that they are not selected for MLM masking.

### Special Tokens

| Condition | CJK / Language A | Hiragana / Language B |
| --- | --- | --- |
| `shared` | `[CLS] ... [SEP]` | `[CLS] ... [SEP]` |
| `none` | no boundary tokens | no boundary tokens |
| `disjoint` | `[BEG] ... [END]` | `[CLS] ... [SEP]` |

The same punctuation-containing CJK and Hiragana corpora are used in all special-token conditions. `[MASK]` remains shared across both languages in all conditions.

## Corpus Preparation

The base corpora and artificial-language dictionaries used by the structural-token experiment are stored under:

```text
punctuation/corpora/
```

The relevant files include:

```text
corpus_cjk_synset.txt
corpus_hiragana_synset.txt
parallel_corpus_synset.json
synset_pos_artificial_cjk.json
synset_pos_artificial_hiragana.json
```

### Punctuation Corpora

`punctuation/corpora/create_corpora_punct.py` derives the two additional punctuation conditions from the standard CJK and Hiragana corpora.

For the `none` condition, it removes commas and periods from both languages and from both sides of the parallel corpus. For the `disjoint` condition, it rewrites punctuation on the CJK side:

```text
,  ->  ;
.  ->  *
```

The script can be run as:

```bash
python punctuation/corpora/create_corpora_punct.py \
    --hiragana PATH/corpus_hiragana_synset.txt \
    --cjk PATH/corpus_cjk_synset.txt \
    --parallel PATH/parallel_corpus_synset.json \
    --outdir punctuation/corpora
```

For linear probing, two bridge scripts are used:

```text
punctuation/corpora/punct_to_lp_bridge_nopunct.py
punctuation/corpora/punct_to_lp_bridge_disjoint.py
```

## Training

### Punctuation Training

`punctuation/training/train_punct.py` can be run for all three conditions and three seeds using:

```bash
bash punctuation/training/run_punct_seeds.sh
```

### Special-Token Training

`special_tokens/training/train_st.py` makes sure that in the `disjoint` condition, CJK is encoded as `[BEG] $A [END]`, while Hiragana is encoded as `[CLS] $A [SEP]`.

All three conditions and seeds can be automated with:

```bash
bash special_tokens/training/run_st_seeds.sh
```

Each final checkpoint additionally contains:

```text
special_config.json
```

This records the boundary-token policy used for each language. The special-token probing and LM-head evaluation scripts read this file to reproduce the correct special tokens.

## Evaluation

The structural-token experiments use four evaluation metrics:

1. Word Translation precision.
2. Sentence Retrieval precision.
3. Linear probing accuracy.
4. LM-head transfer accuracy.

Word Translation and Sentence Retrieval are run for all three seeds. Linear probing and LM-head evaluation are run one seed at a time.

### Word Translation and Sentence Retrieval

Both experiments use the common evaluation script:

```text
Experiment/evaluation/word_trans_sent_retriev.py
```

All punctuation models can be evaluated with:

```bash
bash punctuation/evaluation/run_punct_wt_sr.sh
```

Results are written to:

```text
punctuation/evaluation/punct_wt_sr_results/
```

All special-token models:

```bash
bash special_tokens/evaluation/run_st_wt_sr.sh
```

Results are written to:

```text
special_tokens/evaluation/st_wt_sr_results/
```

The result directories contain scripts for extracting the values and computing mean ± standard deviation.

For punctuation:

```bash
python punctuation/evaluation/punct_wt_sr_results/summarise_wt_sr_seeds.py \
    punctuation/evaluation/punct_wt_sr_results \
    --output punctuation_wt_sr_summary.csv
```

For special tokens:

```bash
python special_tokens/evaluation/st_wt_sr_results/summarise_wt_sr_seeds.py \
    special_tokens/evaluation/st_wt_sr_results \
    --output st_wt_sr_summary.csv
```

### WT/SR Results

Top-1 scores, reported as mean ± sample standard deviation across seeds 42, 43, and 44:

| Experiment | Metric | `shared` | `none` | `disjoint` |
| --- | --- | ---: | ---: | ---: |
| Punctuation | Word Translation | 0.7387 ± 0.0110 | 0.7457 ± 0.0273 | 0.4577 ± 0.0774 |
| Punctuation | Sentence Retrieval | 0.8313 ± 0.0631 | 0.8313 ± 0.0561 | 0.4880 ± 0.0908 |
| Special tokens | Word Translation | 0.7848 ± 0.0302 | 0.7953 ± 0.0284 | 0.6877 ± 0.0262 |
| Special tokens | Sentence Retrieval | 0.8287 ± 0.0559 | 0.8673 ± 0.0378 | 0.8160 ± 0.0495 |

### Additional Punctuation-Free Sentence Retrieval

Sentence Retrieval evaluation includes punctuation in the mean-pooled sentence representations.

An additional control evaluation, just in case, removes:

```text
, . ; *
```

before sentence vectors are calculated. This makes the sentence-retrieval input more directly comparable with the special-token experiment, where [CLS] and [SEP] are not part of the static sentence representation.


```bash
bash punctuation/evaluation/additional_sr_punct/run_punct_additional_sr.sh
```

The outputs are written to:

```text
punctuation/evaluation/additional_sr_punct/additional_sr_punct_results/
```

## Linear Probing

### Punctuation Linear Probe


```bash
bash punctuation/evaluation/probing_lmhead/run_punct_probe.sh
```

Builds separate probing corpora for `shared`, `none`, and `disjoint`, trains a treatment model for each condition, and runs the linear probe.

Results are written under:

```text
punctuation/evaluation/probing_lmhead/probe_results/
```

Each result directory contains:

```text
layerwise_accuracy.csv
layerwise_accuracy_per_relation.csv
layerwise_accuracy.png
```

### Special-Token Linear Probe


```bash
bash special_tokens/evaluation/probing_lmhead/run_st_probe.sh
```

Here, the evaluator is:

```text
special_tokens/evaluation/probing_lmhead/linear_probe_special_final.py
```

Reads `special_config.json` required for the `disjoint` condition.

Results are written under:

```text
special_tokens/evaluation/probing_lmhead/probe_results/
```

## LM-Head Evaluation

### Punctuation LM Head

Run:

```bash
bash punctuation/evaluation/probing_lmhead/run_punct_lmhead.sh
```

Results are written to:

```text
punctuation/evaluation/probing_lmhead/lm_head_results/
```

### Special-Token LM Head

Run:

```bash
bash special_tokens/evaluation/probing_lmhead/run_st_lmhead.sh
```

This uses:

```text
special_tokens/evaluation/probing_lmhead/lm_head_eval_special.py
```

Each LM-head result directory contains:

```text
lm_head_accuracy.csv
lm_head_diagnostics.csv
strict_token_hits.csv
```

## Significance Testing

```text
significance/run_significance_from_results.sh
```

collects the results and passes the hits/misses to `compare_significance.py`.

The comparisons are:

```text
shared > disjoint
none   > disjoint
shared vs none
```

Results are stored in:

```text
significance/significance_results.txt
```