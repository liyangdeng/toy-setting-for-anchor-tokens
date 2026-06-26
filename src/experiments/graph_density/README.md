# Experiment Plan: Effect of Knowledge Graph Density on Cross-lingual Alignment

## Research Question

This experiment tests whether a denser semantic graph provides stronger
cross-lingual alignment signals for two artificial languages with distinct
surface vocabularies.

In this project, artificial languages are generated from semantic triples
extracted from WordNet and ConceptNet. The underlying semantic structure is
shared, while surface forms are replaced by artificial tokens.

The main question is:

> Does higher semantic graph density improve cross-lingual alignment, after
> controlling for training size and number of unique facts?

## Core Idea

Models are trained on corpora generated from pruned versions of the same full
knowledge graph. The full graph for this experiment is:

```text
data/semantic_backbones/edges_adj.json
```

All density conditions should be derived by pruning/subsampling edges from this
same full graph, not by rebuilding separate graphs. This keeps the concept
inventory, token dictionaries, grammar templates, and graph source consistent.
The node universe is the set of unique endpoints in `edges_adj.json`.

High-density graphs naturally contain more triples. If high-density models
perform better, the improvement could come from either:

1. More unique semantic facts.
2. Higher graph density / richer semantic neighborhoods.
3. More total sentence diversity.

To separate these factors, the experiment includes medium- and high-density
control conditions with the same number of unique training triples as the
low-density condition.

## Experimental Conditions

The current full graph has 7,181 edges and 2,042 endpoint nodes. The exact
counts should be computed from the input graph during data preparation, but the
expected condition sizes are:

| Condition | Graph source | Candidate edges | Training unique triples | Purpose |
| --- | --- | ---: | ---: | --- |
| `low_40` | 40% pruned full graph | about 2,872 | about 2,872 | Low-density baseline |
| `low_40_relation` | 40% relation-matched full graph | about 2,872 | about 2,872 | Low density with original relation proportions prioritized |
| `medium_60_full` | 60% pruned full graph | about 4,309 | about 4,309 | Medium density plus more facts |
| `medium_60_relation` | 60% relation-matched full graph | about 4,309 | about 4,309 | Medium density with original relation proportions prioritized |
| `medium_60_control` | 60% pruned candidate graph | about 4,309 | about 2,872 | Medium-density control with low-sized fact set |
| `medium_60_relation_control` | 60% pruned candidate graph | about 4,309 | about 2,872 | Medium-density control prioritizing relation proportions |
| `high_80_full` | 80% pruned full graph | about 5,745 | about 5,745 | High density plus more facts |
| `high_80_relation` | 80% relation-matched full graph | about 5,745 | about 5,745 | High density with original relation proportions prioritized |
| `high_80_control` | 80% pruned candidate graph | about 5,745 | about 2,872 | High-density control with low-sized fact set |
| `high_80_relation_control` | 80% pruned candidate graph | about 5,745 | about 2,872 | High-density control prioritizing relation proportions |

## Pruning Strategy

Low-, medium-, and high-density graphs are produced by pruning edges from the
same full graph.

The preferred pruning strategy is coverage-first edge sampling:

1. First select an edge cover so every concept node appears in at least one
   retained edge.
2. Then add edges until the target density is reached.
3. During the fill step, try to preserve relation and `source_type`
   distributions as much as possible.
4. Node coverage has higher priority than relation-distribution matching.
5. Record graph statistics in `metadata.json`.

The `*_relation` variants use relation-first sampling instead. They first match
the original graph's relation/source-type proportions as closely as possible,
then repair isolated nodes by edge replacement while keeping the target edge
count fixed. These variants are useful when relation distribution should be a
primary control.

## Corpus Generation

For each condition and seed, generate two artificial languages:

- Language A: CJK artificial tokens.
- Language B: Hiragana artificial tokens.

Use the same grammar, word order, and sentence-generation parameters across all
conditions. The default word order for this experiment is:

```text
s1 = 0
s2 = 1
order = SVO
```

Use the existing artificial-token dictionaries:

The corpus builder should produce:

## Training Protocol

For each condition, train a small BERT-style masked language model from scratch
on the mixed CJK + Hiragana corpus.

Training controls:

| Factor | Setting |
| --- | --- |
| Objective | Masked Language Modeling |
| Model architecture | Same small BERT encoder for all runs |
| Tokenizer | Fixed artificial vocabulary, no subword tokenization |
| Word order | Same in both languages |
| Surface overlap | No shared content tokens between CJK and Hiragana |
| Epoch sampling | Randomly sample the same number of sentences as the `low_40` corpus size from each condition corpus |
| Epoch count | Same for all runs |
| Seeds | 3 seeds per condition |

The per-epoch sentence budget is not fixed in advance. After the `low_40`
corpus is generated, its total sentence count becomes the matched per-epoch
sampling budget for all conditions.

Important: matching the number of sampled sentences per epoch controls training
updates per epoch, but it does not by itself control the total number of unique
facts seen across training. That is why all control variants fix the unique
training-triple pool to the same size as `low_40`.

Recommended seeds:

```text
42, 43, 44
```

## Evaluation

Evaluate whether graph density improves cross-lingual alignment and knowledge
transfer.

Primary evaluations:

1. Word translation precision.
2. Sentence retrieval precision.
3. Cross-lingual fact probing, if probe data is prepared for the condition.

Report mean and standard deviation across seeds.

## Full Graph Control: Seed 42

A 100% full-graph control was added after the 40/60/80 density runs. This
condition uses the complete `data/semantic_backbones/edges_adj.json` graph, but
keeps the per-epoch sampled sentence count matched to the smallest low-density
corpus. This makes it a full-graph coverage comparison without increasing the
number of training examples per epoch.

Current status: only seed 42 has been trained. Seeds 43 and 44 have generated
corpora available, but their models are not included in the numbers below.

### Full Graph Seed 42 Corpus

| Field | Value |
| --- | ---: |
| Run directory | `experiments/graph_density/full_graph_seed_42` |
| Full graph edges | 7,181 |
| CJK corpus sentences | 50,003 |
| Hiragana corpus sentences | 50,003 |

The CJK and Hiragana corpora were generated from the same PCFG sentence set.

### Full Graph Seed 42 Training

| Model | Epochs | Epoch sample size | Total sampled train sentences | Dev sentences | Vocab size | Steps/epoch | Total steps | Train perplexity | Dev perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multilingual CJK+Hiragana | 60 | 38,532 | 2,311,920 | 10,000 | 4,515 | 603 | 36,124 | 4.4662 | 5.0136 |
| Monolingual Hiragana | 60 | 19,266 | 1,155,960 | 5,000 | 2,260 | 302 | 18,062 | 4.6806 | 5.3011 |

The full-graph seed 42 models have not yet been added to the evaluation tables
below, which report the earlier 3-seed density experiments.

### Full Graph Seed 42 Evaluation

The full-graph seed 42 models were evaluated with the same repository scripts
used for the density experiments. These are single-seed results, so they should
be compared to the per-seed rows below rather than to the 3-seed means.

Multilingual alignment, using `evaluation/word_trans_sent_retriev.py` with
`--n_sample 500`:

| Condition | Seed | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_graph` | 42 | 0.5970 | 0.8090 | 0.6680 | 0.9080 |

Monolingual Hiragana MLM accuracy, using `evaluation/accuracy.py`:

| Condition | Seed | Top-1 | Top-5 | MRR | Masked tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `full_graph` | 42 | 0.4387 | 0.8548 | 0.6153 | 3766 |

## Visualizations

Static visualizations are generated by:

```text
experiments/graph_density/visualize_experiment_results.py
```

The visualizations combine the completed 3-seed density experiments with the
single-seed `full_graph` control. The 40/60/80 density conditions are shown as
mean +/- sample standard deviation across seeds 42, 43, and 44. The full graph
control is shown as seed 42 only, so it is drawn as a single-point control.

Generated chart files:

```text
experiments/graph_density/visualizations/multilingual_alignment_top1.png
experiments/graph_density/visualizations/multilingual_alignment_top5.png
experiments/graph_density/visualizations/monolingual_hiragana_mlm.png
experiments/graph_density/visualizations/seed42_condition_comparison.png
experiments/graph_density/visualizations/full_graph_delta_vs_high80_relation.png
```

Chart-ready CSV summaries:

```text
experiments/graph_density/visualizations/multilingual_summary.csv
experiments/graph_density/visualizations/monolingual_summary.csv
```

## Evaluation Results

Evaluation was run with the existing repository scripts:

- `evaluation/word_trans_sent_retriev.py` for multilingual word translation and sentence retrieval.
- `evaluation/accuracy.py` for monolingual Hiragana MLM top-k accuracy and MRR.

All results below are mean +/- sample standard deviation across seeds 42, 43, and 44. Sentence retrieval uses `--n_sample 500`.

### Multilingual Alignment

| Condition | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: |
| `low_40` | 0.1477 +/- 0.0240 | 0.2810 +/- 0.0343 | 0.1547 +/- 0.0352 | 0.3493 +/- 0.0546 |
| `medium_60_full` | 0.1630 +/- 0.0634 | 0.3118 +/- 0.1023 | 0.2120 +/- 0.0647 | 0.4440 +/- 0.1121 |
| `high_80_full` | 0.2635 +/- 0.0068 | 0.4773 +/- 0.0206 | 0.3480 +/- 0.0990 | 0.6440 +/- 0.1119 |
| `low_40_relation` | 0.2830 +/- 0.0918 | 0.4772 +/- 0.1122 | 0.3120 +/- 0.1368 | 0.5893 +/- 0.1615 |
| `medium_60_relation` | 0.4097 +/- 0.0539 | 0.6220 +/- 0.0560 | 0.4313 +/- 0.0966 | 0.7133 +/- 0.0546 |
| `high_80_relation` | 0.5580 +/- 0.0412 | 0.7605 +/- 0.0581 | 0.6127 +/- 0.0269 | 0.8507 +/- 0.0192 |

### Monolingual Hiragana MLM Accuracy

| Condition | MLM top-1 | MLM top-5 | MRR | Masked tokens |
| --- | ---: | ---: | ---: | ---: |
| `low_40` | 0.5159 +/- 0.0203 | 0.9187 +/- 0.0055 | 0.6844 +/- 0.0171 | 1376.0000 +/- 26.8887 |
| `medium_60_full` | 0.4678 +/- 0.0112 | 0.9059 +/- 0.0053 | 0.6498 +/- 0.0052 | 2218.3333 +/- 60.5007 |
| `high_80_full` | 0.4572 +/- 0.0221 | 0.8878 +/- 0.0051 | 0.6342 +/- 0.0158 | 3026.3333 +/- 74.4468 |
| `low_40_relation` | 0.5394 +/- 0.0035 | 0.9255 +/- 0.0106 | 0.7048 +/- 0.0029 | 1272.3333 +/- 37.0720 |
| `medium_60_relation` | 0.4934 +/- 0.0154 | 0.9066 +/- 0.0037 | 0.6676 +/- 0.0095 | 2073.6667 +/- 9.4516 |
| `high_80_relation` | 0.4757 +/- 0.0180 | 0.8841 +/- 0.0047 | 0.6469 +/- 0.0106 | 2872.3333 +/- 31.3422 |

### Per-Seed Multilingual Results

| Condition | Seed | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low_40` | 42 | 0.1635 | 0.2990 | 0.1260 | 0.3400 |
| `low_40` | 43 | 0.1595 | 0.3025 | 0.1940 | 0.4080 |
| `low_40` | 44 | 0.1200 | 0.2415 | 0.1440 | 0.3000 |
| `medium_60_full` | 42 | 0.1025 | 0.2095 | 0.1660 | 0.3500 |
| `medium_60_full` | 43 | 0.2290 | 0.4140 | 0.2860 | 0.5680 |
| `medium_60_full` | 44 | 0.1575 | 0.3120 | 0.1840 | 0.4140 |
| `high_80_full` | 42 | 0.2700 | 0.5010 | 0.4480 | 0.7420 |
| `high_80_full` | 43 | 0.2565 | 0.4680 | 0.2500 | 0.5220 |
| `high_80_full` | 44 | 0.2640 | 0.4630 | 0.3460 | 0.6680 |
| `low_40_relation` | 42 | 0.1790 | 0.3505 | 0.1540 | 0.4040 |
| `low_40_relation` | 43 | 0.3175 | 0.5170 | 0.3920 | 0.7000 |
| `low_40_relation` | 44 | 0.3525 | 0.5640 | 0.3900 | 0.6640 |
| `medium_60_relation` | 42 | 0.4045 | 0.6275 | 0.3880 | 0.6680 |
| `medium_60_relation` | 43 | 0.4660 | 0.6750 | 0.5420 | 0.7740 |
| `medium_60_relation` | 44 | 0.3585 | 0.5635 | 0.3640 | 0.6980 |
| `high_80_relation` | 42 | 0.5105 | 0.6955 | 0.6320 | 0.8680 |
| `high_80_relation` | 43 | 0.5845 | 0.8075 | 0.6240 | 0.8540 |
| `high_80_relation` | 44 | 0.5790 | 0.7785 | 0.5820 | 0.8300 |

### Per-Seed Monolingual Results

| Condition | Seed | Top-1 | Top-5 | MRR | Masked tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low_40` | 42 | 0.4932 | 0.9124 | 0.6658 | 1393 |
| `low_40` | 43 | 0.5323 | 0.9227 | 0.6993 | 1345 |
| `low_40` | 44 | 0.5223 | 0.9209 | 0.6881 | 1390 |
| `medium_60_full` | 42 | 0.4685 | 0.9077 | 0.6523 | 2188 |
| `medium_60_full` | 43 | 0.4786 | 0.8999 | 0.6533 | 2288 |
| `medium_60_full` | 44 | 0.4562 | 0.9101 | 0.6439 | 2179 |
| `high_80_full` | 42 | 0.4321 | 0.8832 | 0.6163 | 2946 |
| `high_80_full` | 43 | 0.4740 | 0.8933 | 0.6463 | 3093 |
| `high_80_full` | 44 | 0.4655 | 0.8868 | 0.6401 | 3040 |
| `low_40_relation` | 42 | 0.5398 | 0.9358 | 0.7082 | 1230 |
| `low_40_relation` | 43 | 0.5427 | 0.9146 | 0.7032 | 1288 |
| `low_40_relation` | 44 | 0.5358 | 0.9261 | 0.7031 | 1299 |
| `medium_60_relation` | 42 | 0.4760 | 0.9069 | 0.6576 | 2063 |
| `medium_60_relation` | 43 | 0.4993 | 0.9027 | 0.6686 | 2077 |
| `medium_60_relation` | 44 | 0.5050 | 0.9101 | 0.6766 | 2081 |
| `high_80_relation` | 42 | 0.4747 | 0.8800 | 0.6449 | 2867 |
| `high_80_relation` | 43 | 0.4942 | 0.8830 | 0.6584 | 2906 |
| `high_80_relation` | 44 | 0.4582 | 0.8892 | 0.6375 | 2844 |
