# Structural Tokens and Cross-Lingual Alignment

Investigates how shared structural tokens (punctuation and BERT's special [CLS] and [SEP]) affect cross-lingual alignment.

Most part are controlled through the provided shell scripts.
Unless changed in the .sh scripts, the main experiments use seeds `42`, `43`, and `44`.

---

# 1. Punctuation experiment

The punctuation experiment tests whether punctuation shared across languages contributes to cross-lingual alignment.

No `[CLS]` or `[SEP]` tokens are used in this experiment.

The three conditions are:

| Setting | CJK | Hiragana |
| --- | --- | --- |
| `shared` | `, .` | `, .` |
| `none` | no punctuation | no punctuation |
| `disjoint` | `; *` | `, .` |

Punctuation symbols are registered as special tokens where relevant so that they are not selected as MLM masking targets, to match the second (special token) experiment.

## 1.1 Corpus preparation

The punctuation corpora are generated using:

The training script expects separate corpus pairs for the three experimental conditions.

The Hiragana corpus in the `disjoint` condition uses the same `, .` punctuation as the `shared` condition.

---

## 1.2 Train all punctuation models

```bash
bash punctuation/run_punct_seeds.sh
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

The script extracts top-1 word-translation and sentence-retrieval precision from each evaluation log and reports:

```text
mean ± sample SD
```

across training seeds.

It also writes the individual seed values to:

```text
punct_eval_results/wt_sr_seed_values.csv
```

---

## 1.5 Punctuation-free sentence retrieval

The primary punctuation sentence-retrieval evaluation includes punctuation in the mean-pooled sentence representations.

An additional control evaluation, just in case, removes:

```text
, . ; *
```

before sentence vectors are calculated. This makes the sentence-retrieval input more directly comparable with the special-token experiment, where [CLS] and [SEP]are not part of the static sentence representation.

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

The special-token experiment manipulates sentence-boundary tokens. The corpus text identical, including punctuation.

`[MASK]` remains shared in every condition.

The conditions are:

| Setting | CJK | Hiragana |
| --- | --- | --- |
| `shared` | `[CLS] ... [SEP]` | `[CLS] ... [SEP]` |
| `none` | no boundary tokens | no boundary tokens |
| `disjoint` | `[BEG] ... [END]` | `[CLS] ... [SEP]` |

Unlike the punctuation experiment, the same CJK and Hiragana corpus files are used for all three arms.

---

## 2.1 Train all special-token models

Set the corpus locations in the script:

```bash
CORPUS_A=PATH
CORPUS_B=PATH
```

Training is handled by:

```bash
bash st_run_st_seeds.sh
```

The outputs follow the same naming scheme:

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

This records the boundary-token policy used for each artificial language. The special-token probing and LM-head evaluation scripts read this file to reproduce the correct special tokens at evaluation time.

---

## 2.2 Word translation and sentence retrieval

```bash
bash st/run_st_eval.sh
```

```text
synset_pos_artificial_cjk.json       # PATH
synset_pos_artificial_hiragana.json  # PATH
parallel_corpus_synset.json          # PATH
```

The evaluation again uses the static embedding matrix.

Consequently, `[CLS]`, `[SEP]`, `[BEG]`, and `[END]` are not themselves included in the static word-translation or sentence-retrieval representations. The experiment therefore tests the effect of exposure to the different settings during training.

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

Run from `1_ST/`:

```bash
cd 1_ST
bash probing/run_st_probe.sh
```
Here, the probe is implemented in:

```text
probing/linear_probe_special_final.py
```
The script differs from the one in the 'evaluation' dir because it adds boundary special tokens.

Results are again written as:

```text
layerwise_accuracy.csv
layerwise_accuracy_per_relation.csv
layerwise_accuracy.png
```

---

# 4. MLM-head evaluation

---

## 4.1 Punctuation LM-head evaluation


```bash
bash punctuation/probing/run_lmhead_punct.sh
```

The evaluator is expected at:

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

The special-token evaluator reads `special_config.json` before constructing the Language B prompts.

This is important for the `disjoint` condition. The tokenizer saved by `final_train_st.py` retains the CJK/A-side `[BEG] ... [END]` template, while Hiragana was trained with `[CLS] ... [SEP]`. The evaluator restores the correct Hiragana template before calculating MLM predictions.

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

# 5. Development perplexity

Training scripts print final train and development perplexities. Perplexities can also be collected automatically from saved Hugging Face trainer states.

From the repository root:

```bash
python collect_perplexity.py \
    --glob '1_PUNCT/punct_checkpoints_*' \
    --out punct_perplexity.txt
```

For the special-token experiment:

```bash
python collect_perplexity.py \
    --glob '1_ST/st_checkpoints_*' \
    --out st_perplexity.txt
```

Both experiments can also be processed together:

```bash
python collect_perplexity.py \
    --glob '1_PUNCT/punct_checkpoints_*' '1_ST/st_checkpoints_*' \
    --out all_perplexity.txt
```

The script reports every individual run followed by mean ± standard deviation for each condition.

---

# 6. Results configuration

`results_config.json` contains the results used for downstream comparison and statistical testing.

The current conventions are:

```text
word translation   -> top-1 precision
sentence retrieval -> top-1 precision
linear probe       -> selected Transformer-layer accuracy
LM head            -> strict_token top-1 accuracy
```

For word translation and sentence retrieval, results are stored separately for each training seed.

For probe and LM-head results, `n` denotes the number of evaluated triples. For WT/SR it denotes the number of evaluated aligned pairs.

Any downstream significance-analysis script can be placed at:

```text
PATH
```

and configured to read `results_config.json`.

---

# Execution order

## Punctuation

From `punctuation/`:

```bash
# 1. Train shared / none / disjoint models over all seeds
bash run_punct_seeds.sh

# 2. Static word translation + sentence retrieval
bash run_punct_eval.sh

# 3. Summarise WT/SR across seeds
python punct_eval_results/summarise_wt_sr_seeds.py \
    punct_eval_results \
    --output punctuation_wt_sr_summary.csv

# 4. Build probing data, train treatment models and run linear probe
bash probing/run_punct_probe.sh

# 5. Evaluate LM head
bash probing/run_lmhead_punct.sh

# 6. Optional punctuation-free SR control
bash PATH/run_punct_additional_sr.sh
python PATH/summarise_additional_sr_punct.py
```

## Special tokens

From `1_ST/`:

```bash
# 1. Train shared / none / disjoint models over all seeds
bash run_st_seeds.sh

# 2. Static word translation + sentence retrieval
bash run_st_eval.sh

# 3. Summarise WT/SR across seeds
python st_eval_results/summarise_wt_sr_seeds.py \
    st_eval_results \
    --output st_wt_sr_summary.csv

# 4. Build probing data, train treatment models and run linear probe
bash probing/run_st_probe.sh

# 5. Evaluate the trained MLM head
bash probing/run_lmhead_st.sh
```