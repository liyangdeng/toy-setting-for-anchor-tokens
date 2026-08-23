# Structural Tokens and Cross-Lingual Alignment

Investigates how shared structural tokens (punctuation and BERT's special [CLS] and [SEP]) affect cross-lingual alignment.

Most part are controlled through the provided shell scripts.
Unless changed in the .sh scripts, the main experiments use seeds `42`, `43`, and `44`.

---

# 1. Punctuation experiment

## Research Question

> Does training with punctuation shared across languages, in addition to shared syntactic structure, contribute to cross-lingual alignment and transfer?

No `[CLS]` or `[SEP]` tokens are used in this experiment.

## Experimental conditions

| Setting | CJK | Hiragana |
| --- | --- | --- |
| `shared` | `, .` | `, .` |
| `none` | no punctuation | no punctuation |
| `disjoint` | `; *` | `, .` |

Punctuation symbols are registered as special tokens where relevant so that they are not selected for MLM masking, to match the second (special token) experiment.

## 1.1 Corpus preparation

The punctuation corpora are generated using: 

The training script expects separate corpus pairs for the three experimental conditions.

The Hiragana corpus in the `disjoint` condition uses the same `, .` punctuation as the `shared` condition.

---

## 1.2 Train all punctuation models

From `structural_tokens/punctuation`, run: 
```bash
bash run_punct_seeds.sh
```

A model is saved for every condition and seed:

```text
punct_checkpoints_shared_seed42/final/
punct_checkpoints_shared_seed43/final/
punct_checkpoints_shared_seed44/final/
...
punct_checkpoints_disjoint_seed44/final/
```

Each run stores its train/dev split, Hugging Face checkpoints and a loss curve.

---

## 1.3 Word translation and sentence retrieval

Static cross-lingual alignment is evaluated with:

```text
word_trans_sent_retriev.py (from the 'evaluation' dir of the project)
```

To evaluate all punctuation models:

```bash
bash punctuation/run_punct_eval.sh
```

Results are written to:

```text
punct_eval_results/shared_seed42.txt
punct_eval_results/shared_seed43.txt
...
punct_eval_results/disjoint_seed44.txt
```

---

## 1.4 Average WT/SR results across seeds

Use:

```bash
python punct_eval_results/summarise_wt_sr_seeds.py \
    punct_eval_results \
    --output punctuation_wt_sr_summary.csv
```

The script extracts top-1 Word-Translation and Sentence-Retrieval precision and reports mean ± sample SD across training seeds.

The individual seed values are written to:

```text
punct_eval_results/wt_sr_seed_values.csv
```

---

## 1.5 Punctuation-free Sentence Retrieval

Sentence Retrieval evaluation includes punctuation in the mean-pooled sentence representations.

An additional control evaluation, just in case, removes:

```text
, . ; *
```

before sentence vectors are calculated. This makes the sentence-retrieval input more directly comparable with the special-token experiment, where [CLS] and [SEP] are not part of the static sentence representation.

The relevant scripts are:

```text
additional_sr_punct.py
run_punct_additional_sr.sh
summarise_additional_sr_punct.py
```

To run:

```bash
bash PATH/run_punct_additional_sr.sh
```

and summarise it with:

```bash
python PATH/summarise_additional_sr_punct.py
```

---

# 2. Special-token experiment

## Research question
The special-token experiment manipulates BERT's sentence-boundary tokens [CLS] and [SEP]. 

> Does training with [CLS]/[SEP] shared across languages, in addition to shared syntactic structure, contribute to cross-lingual alignment and transfer?

The same CJK and Hiragana corpus files are used in all conditions. The corpus includes punctuation. `[MASK]` remains shared in every condition.

## Experimental conditions

| Setting | CJK | Hiragana |
| --- | --- | --- |
| `shared` | `[CLS] ... [SEP]` | `[CLS] ... [SEP]` |
| `none` | no boundary tokens | no boundary tokens |
| `disjoint` | `[BEG] ... [END]` | `[CLS] ... [SEP]` |


---

## 2.1 Train all special-token models

From `structural_tokens/st`, run: 

```bash
bash st_run_st_seeds.sh
```

Set the corpus locations in the script:

```bash
CORPUS_A=PATH
CORPUS_B=PATH
```

The outputs are saved to:

```text
st_checkpoints_shared_seed42/final/
st_checkpoints_none_seed42/final/
st_checkpoints_disjoint_seed42/final/
...
```

Each final special-token checkpoint additionally contains:

```text
special_config.json
```

This records the boundary-token policy used for each language. The special-token probing and LM-head evaluation scripts read this file to reproduce the correct special tokens.

---

## 2.2 Word Translation and Sentence Retrieval

```bash
bash st/run_st_eval.sh
```

```text
synset_pos_artificial_cjk.json       # PATH
synset_pos_artificial_hiragana.json  # PATH
parallel_corpus_synset.json          # PATH
```

Results are written to:

```text
st_eval_results/
```

Average across seeds with:

```bash
python st_eval_results/summarise_wt_sr_seeds.py \
    st_eval_results \
    --output st_wt_sr_summary.csv
```

---

# 3. Linear probing

Expect several files from the corpus-generation pipeline, such as:

```text
PATH/build_probing_corpus.py
PATH/deprived_triples.json
PATH/omitted_triples.json
PATH/probe_manifest.json
PATH/v3_generate_sentences.py
PATH/build_synset_corpus.py
PATH/train_monolingual_synset.py
PATH/grammar_templates_adj.py
PATH/synset_pos_artificial_cjk.json
PATH/synset_pos_artificial_hiragana.json
```
## 3.1 Punctuation linear probe

The punctuation experiment uses additional setting-specific corpus-building bridges:

```text
PATH/punct_to_lp_bridge_nopunct.py
PATH/punct_to_lp_bridge_disjoint.py
```

```bash
bash punctuation/probing/run_punct_probe.sh
```

The probe is expected at:

```text
PATH/linear_probe.py
```

Results contain:

```text
layerwise_accuracy.csv
layerwise_accuracy_per_relation.csv
layerwise_accuracy.png
```

---

## 3.2 Special-token linear probe

From `structural_tokens/st`, run: 

```bash
bash probing/run_st_probe.sh
```
The probe is implemented in:

```text
probing/linear_probe_special_final.py
```
The script differs from the one in the 'evaluation' dir because it adds boundary special tokens.

Results are written as:

```text
layerwise_accuracy.csv
layerwise_accuracy_per_relation.csv
layerwise_accuracy.png
```
each in their corresponding folder, e.g. probing/

---

# 4. MLM-head evaluation

---

## 4.1 Punctuation LM-head evaluation


```bash
bash punctuation/probing/run_lmhead_punct.sh
```

The evaluator itself can be found in:

```text
PATH/lm_head_eval.py
```

---

## 4.2 Special-token LM-head evaluation

```bash
bash st/probing/run_lmhead_st.sh
```

This uses:

```text
probing/lm_head_eval_special.py
```

Reads `special_config.json` before constructing the Language B prompts. This is important for the `disjoint` condition. The tokenizer saved by `final_train_st.py` retains the CJK/A-side `[BEG] ... [END]` template, while Hiragana was trained with `[CLS] ... [SEP]`. The evaluator restores the correct Hiragana template before calculating predictions.

Results:

```text
lm_head_accuracy.csv
lm_head_diagnostics.csv
strict_token_hits.csv
```

`lm_head_diagnostics.csv` additionally reports:

```text
source_token_rate
right_concept_wrong_lang
```

which measure the tendency of the model to predict Language A tokens when prompted in Language B.

---

# 5. Perplexity

If final train and development perplexities need to be collected, run:

```bash
python collect_perplexity.py \
    --glob '1_PUNCT/punct_checkpoints_*' '1_ST/st_checkpoints_*' \
    --out all_perplexity.txt
```

---

# 6. Significance tests

---
