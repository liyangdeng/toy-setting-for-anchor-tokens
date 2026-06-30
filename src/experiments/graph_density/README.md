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

## V3 Adjective-Variant Corpora

The new v3 corpus generation run is stored separately from the earlier
graph-density data:

```text
experiments/graph_density/v3_adj_variants/
```

This run uses `data/generate_sentences/v3_generate_sentences.py`, which adds
adjective minimal-pair variants on top of the earlier adjective-aware PCFG
generation. The generated conditions are:

```text
low_40
medium_60_full
high_80_full
low_40_relation
medium_60_relation
high_80_relation
full_graph
```

Each condition has seeds 42, 43, and 44. The corpus metadata and relation
counts are summarized in:

```text
experiments/graph_density/v3_adj_variants/v3_adj_variant_corpus_report.md
experiments/graph_density/v3_adj_variants/v3_adj_variant_corpus_summary.json
```

The matched training sample sizes for this v3 run should be:

```text
monolingual epoch sample size = 17,797
multilingual epoch sample size = 35,594
```

Start all v3 training runs with:

```bash
conda activate SWP
bash experiments/graph_density/run_v3_training.sh
```

The launcher skips already completed models by default. Use `FORCE=1` to rerun
completed outputs, `RUN_MONOLINGUAL=0` to train only multilingual models, or
`CUDA_VISIBLE_DEVICES=0` to select a GPU.

For the no-downsample follow-up on 60%, 80%, and full-graph conditions, use:

```bash
conda activate SWP
bash experiments/graph_density/run_v3_full_corpus_training.sh
```

This launcher does not pass `--epoch_sample_size`, so each epoch uses the full
train split of that condition's corpus. Outputs are written separately to
`model_multilingual_fulltrain` and `model_mono_hiragana_fulltrain`.

## V3 Adjective-Variant Performance Results

Evaluation uses the same repository scripts as the earlier graph-density experiments:

- `evaluation/word_trans_sent_retriev.py` for multilingual word translation and sentence retrieval.
- `evaluation/accuracy.py` for monolingual Hiragana MLM top-k accuracy and MRR.

All aggregate rows are mean +/- sample standard deviation across seeds 42, 43, and 44.
Sentence retrieval uses `--n_sample 500`.

## Performance Comparison

- Best multilingual word top-1 is `full_graph` (0.5255), essentially tied with `high_80_relation` (0.5192).
- Best multilingual sentence top-1 is `high_80_relation` (0.4640), slightly above `full_graph` (0.4560).
- Relation-controlled pruning is consistently stronger than the corresponding coverage-first/full-density condition for multilingual alignment: +0.1727 word top-1 at low density, +0.1797 at medium density, and +0.3297 at high density.
- Monolingual Hiragana MLM accuracy moves in the opposite direction: lower-density relation-controlled corpora are easiest, with `low_40_relation` giving the best MLM top-1 (0.5501) and MRR (0.7138).
- The full graph is a strong multilingual control but not the best monolingual MLM setting: compared with `high_80_relation`, it is +0.0063 on word top-1, -0.0080 on sentence top-1, and -0.0238 on monolingual MLM top-1.

## Multilingual Alignment

| Condition | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: |
| `low_40` | 0.1255 +/- 0.0354 | 0.2460 +/- 0.0693 | 0.1507 +/- 0.0450 | 0.3513 +/- 0.0833 |
| `medium_60_full` | 0.1702 +/- 0.0199 | 0.3418 +/- 0.0203 | 0.1693 +/- 0.0336 | 0.4100 +/- 0.0452 |
| `high_80_full` | 0.1895 +/- 0.0956 | 0.3587 +/- 0.1414 | 0.1973 +/- 0.1094 | 0.4347 +/- 0.1824 |
| `low_40_relation` | 0.2982 +/- 0.0595 | 0.4933 +/- 0.0713 | 0.2580 +/- 0.0730 | 0.5453 +/- 0.1139 |
| `medium_60_relation` | 0.3498 +/- 0.0782 | 0.5690 +/- 0.0928 | 0.2907 +/- 0.0965 | 0.5793 +/- 0.1299 |
| `high_80_relation` | 0.5192 +/- 0.0423 | 0.7323 +/- 0.0311 | 0.4640 +/- 0.0262 | 0.7447 +/- 0.0200 |
| `full_graph` | 0.5255 +/- 0.0697 | 0.7433 +/- 0.0572 | 0.4560 +/- 0.0450 | 0.7553 +/- 0.0270 |

## Monolingual Hiragana MLM Accuracy

| Condition | MLM top-1 | MLM top-5 | MRR | Masked tokens |
| --- | ---: | ---: | ---: | ---: |
| `low_40` | 0.5004 +/- 0.0170 | 0.9206 +/- 0.0007 | 0.6752 +/- 0.0083 | 1356.3333 +/- 12.8582 |
| `medium_60_full` | 0.4737 +/- 0.0101 | 0.8881 +/- 0.0015 | 0.6482 +/- 0.0054 | 2153.6667 +/- 21.0792 |
| `high_80_full` | 0.4633 +/- 0.0101 | 0.8758 +/- 0.0093 | 0.6354 +/- 0.0085 | 2989.3333 +/- 44.0606 |
| `low_40_relation` | 0.5501 +/- 0.0168 | 0.9413 +/- 0.0085 | 0.7138 +/- 0.0107 | 1299.6667 +/- 71.4586 |
| `medium_60_relation` | 0.5048 +/- 0.0008 | 0.9119 +/- 0.0159 | 0.6761 +/- 0.0024 | 2060.0000 +/- 9.8489 |
| `high_80_relation` | 0.4929 +/- 0.0057 | 0.8829 +/- 0.0018 | 0.6578 +/- 0.0038 | 2901.0000 +/- 9.5394 |
| `full_graph` | 0.4691 +/- 0.0067 | 0.8597 +/- 0.0031 | 0.6351 +/- 0.0057 | 3835.3333 +/- 80.0895 |

## Per-Seed Multilingual Results

| Condition | Seed | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low_40` | 42 | 0.1625 | 0.3140 | 0.1860 | 0.4100 |
| `low_40` | 43 | 0.1220 | 0.2485 | 0.1660 | 0.3880 |
| `low_40` | 44 | 0.0920 | 0.1755 | 0.1000 | 0.2560 |
| `medium_60_full` | 42 | 0.1475 | 0.3210 | 0.1400 | 0.3720 |
| `medium_60_full` | 43 | 0.1785 | 0.3430 | 0.2060 | 0.4600 |
| `medium_60_full` | 44 | 0.1845 | 0.3615 | 0.1620 | 0.3980 |
| `high_80_full` | 42 | 0.0795 | 0.1960 | 0.0720 | 0.2300 |
| `high_80_full` | 43 | 0.2525 | 0.4520 | 0.2740 | 0.5800 |
| `high_80_full` | 44 | 0.2365 | 0.4280 | 0.2460 | 0.4940 |
| `low_40_relation` | 42 | 0.2295 | 0.4125 | 0.1760 | 0.4240 |
| `low_40_relation` | 43 | 0.3340 | 0.5200 | 0.3160 | 0.6500 |
| `low_40_relation` | 44 | 0.3310 | 0.5475 | 0.2820 | 0.5620 |
| `medium_60_relation` | 42 | 0.3375 | 0.5665 | 0.2400 | 0.5540 |
| `medium_60_relation` | 43 | 0.4335 | 0.6630 | 0.4020 | 0.7200 |
| `medium_60_relation` | 44 | 0.2785 | 0.4775 | 0.2300 | 0.4640 |
| `high_80_relation` | 42 | 0.5395 | 0.7485 | 0.4880 | 0.7600 |
| `high_80_relation` | 43 | 0.4705 | 0.6965 | 0.4360 | 0.7220 |
| `high_80_relation` | 44 | 0.5475 | 0.7520 | 0.4680 | 0.7520 |
| `full_graph` | 42 | 0.5475 | 0.7570 | 0.4580 | 0.7560 |
| `full_graph` | 43 | 0.4475 | 0.6805 | 0.4100 | 0.7280 |
| `full_graph` | 44 | 0.5815 | 0.7925 | 0.5000 | 0.7820 |

## Per-Seed Monolingual Results

| Condition | Seed | Top-1 | Top-5 | MRR | Masked tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low_40` | 42 | 0.5189 | 0.9208 | 0.6844 | 1351 |
| `low_40` | 43 | 0.4967 | 0.9212 | 0.6729 | 1371 |
| `low_40` | 44 | 0.4855 | 0.9198 | 0.6683 | 1347 |
| `medium_60_full` | 42 | 0.4666 | 0.8865 | 0.6471 | 2141 |
| `medium_60_full` | 43 | 0.4853 | 0.8884 | 0.6541 | 2178 |
| `medium_60_full` | 44 | 0.4692 | 0.8894 | 0.6434 | 2142 |
| `high_80_full` | 42 | 0.4522 | 0.8656 | 0.6256 | 2992 |
| `high_80_full` | 43 | 0.4657 | 0.8838 | 0.6406 | 2944 |
| `high_80_full` | 44 | 0.4720 | 0.8780 | 0.6400 | 3032 |
| `low_40_relation` | 42 | 0.5474 | 0.9510 | 0.7149 | 1224 |
| `low_40_relation` | 43 | 0.5681 | 0.9370 | 0.7240 | 1366 |
| `low_40_relation` | 44 | 0.5348 | 0.9358 | 0.7026 | 1309 |
| `medium_60_relation` | 42 | 0.5048 | 0.9212 | 0.6788 | 2068 |
| `medium_60_relation` | 43 | 0.5056 | 0.8936 | 0.6741 | 2049 |
| `medium_60_relation` | 44 | 0.5041 | 0.9210 | 0.6753 | 2063 |
| `high_80_relation` | 42 | 0.4893 | 0.8810 | 0.6561 | 2900 |
| `high_80_relation` | 43 | 0.4995 | 0.8846 | 0.6622 | 2911 |
| `high_80_relation` | 44 | 0.4900 | 0.8831 | 0.6552 | 2892 |
| `full_graph` | 42 | 0.4614 | 0.8566 | 0.6286 | 3877 |
| `full_graph` | 43 | 0.4738 | 0.8628 | 0.6393 | 3886 |
| `full_graph` | 44 | 0.4721 | 0.8597 | 0.6375 | 3743 |

Raw logs and parsed JSON are stored in:

```text
experiments/graph_density/v3_adj_variants/evaluation_results/
```

## V3 Full-Corpus Performance Results

Full-corpus runs do not use per-epoch downsampling. Each aggregate row is mean +/- sample standard deviation across seeds 42, 43, and 44.
Sentence retrieval uses `--n_sample 500`.

## Multilingual Alignment

| Condition | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: |
| `medium_60_full` | 0.1703 +/- 0.0573 | 0.3307 +/- 0.0746 | 0.1927 +/- 0.0763 | 0.4280 +/- 0.1114 |
| `high_80_full` | 0.3855 +/- 0.0439 | 0.5853 +/- 0.0477 | 0.3693 +/- 0.0515 | 0.6640 +/- 0.0772 |
| `medium_60_relation` | 0.5177 +/- 0.0974 | 0.7203 +/- 0.0862 | 0.5100 +/- 0.0399 | 0.7913 +/- 0.0197 |
| `high_80_relation` | 0.6540 +/- 0.0416 | 0.8298 +/- 0.0251 | 0.6473 +/- 0.0314 | 0.8867 +/- 0.0356 |
| `full_graph` | 0.7150 +/- 0.0277 | 0.8680 +/- 0.0114 | 0.7047 +/- 0.0219 | 0.9013 +/- 0.0162 |

## Monolingual Hiragana MLM Accuracy

| Condition | MLM top-1 | MLM top-5 | MRR | Masked tokens |
| --- | ---: | ---: | ---: | ---: |
| `medium_60_full` | 0.4849 +/- 0.0048 | 0.9064 +/- 0.0071 | 0.6584 +/- 0.0033 | 2153.6667 +/- 21.0792 |
| `high_80_full` | 0.4970 +/- 0.0128 | 0.9130 +/- 0.0037 | 0.6664 +/- 0.0068 | 2989.3333 +/- 44.0606 |
| `medium_60_relation` | 0.5335 +/- 0.0056 | 0.9302 +/- 0.0098 | 0.7000 +/- 0.0031 | 2060.0000 +/- 9.8489 |
| `high_80_relation` | 0.5213 +/- 0.0021 | 0.9200 +/- 0.0048 | 0.6879 +/- 0.0046 | 2901.0000 +/- 9.5394 |
| `full_graph` | 0.5141 +/- 0.0056 | 0.9101 +/- 0.0068 | 0.6793 +/- 0.0064 | 3835.3333 +/- 80.0895 |

Raw logs and parsed JSON are stored in:

```text
experiments/graph_density/v3_adj_variants/evaluation_results_fulltrain/
```

## V3 Downsample vs Full-Corpus Training Comparison

This combines the original downsampled v3 runs with the full-corpus follow-up on 60%, 80%, and full-graph conditions.
Low-density rows are included for the downsampled experiment only, because no full-corpus low-density runs were trained.
All rows are mean +/- sample standard deviation across seeds 42, 43, and 44.

## Multilingual Alignment

| Condition | Mode | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | --- | ---: | ---: | ---: | ---: |
| `low_40` | `downsample` | 0.1255 +/- 0.0354 | 0.2460 +/- 0.0693 | 0.1507 +/- 0.0450 | 0.3513 +/- 0.0833 |
| `medium_60_full` | `downsample` | 0.1702 +/- 0.0199 | 0.3418 +/- 0.0203 | 0.1693 +/- 0.0336 | 0.4100 +/- 0.0452 |
| `medium_60_full` | `full_corpus` | 0.1703 +/- 0.0573 | 0.3307 +/- 0.0746 | 0.1927 +/- 0.0763 | 0.4280 +/- 0.1114 |
| `high_80_full` | `downsample` | 0.1895 +/- 0.0956 | 0.3587 +/- 0.1414 | 0.1973 +/- 0.1094 | 0.4347 +/- 0.1824 |
| `high_80_full` | `full_corpus` | 0.3855 +/- 0.0439 | 0.5853 +/- 0.0477 | 0.3693 +/- 0.0515 | 0.6640 +/- 0.0772 |
| `low_40_relation` | `downsample` | 0.2982 +/- 0.0595 | 0.4933 +/- 0.0713 | 0.2580 +/- 0.0730 | 0.5453 +/- 0.1139 |
| `medium_60_relation` | `downsample` | 0.3498 +/- 0.0782 | 0.5690 +/- 0.0928 | 0.2907 +/- 0.0965 | 0.5793 +/- 0.1299 |
| `medium_60_relation` | `full_corpus` | 0.5177 +/- 0.0974 | 0.7203 +/- 0.0862 | 0.5100 +/- 0.0399 | 0.7913 +/- 0.0197 |
| `high_80_relation` | `downsample` | 0.5192 +/- 0.0423 | 0.7323 +/- 0.0311 | 0.4640 +/- 0.0262 | 0.7447 +/- 0.0200 |
| `high_80_relation` | `full_corpus` | 0.6540 +/- 0.0416 | 0.8298 +/- 0.0251 | 0.6473 +/- 0.0314 | 0.8867 +/- 0.0356 |
| `full_graph` | `downsample` | 0.5255 +/- 0.0697 | 0.7433 +/- 0.0572 | 0.4560 +/- 0.0450 | 0.7553 +/- 0.0270 |
| `full_graph` | `full_corpus` | 0.7150 +/- 0.0277 | 0.8680 +/- 0.0114 | 0.7047 +/- 0.0219 | 0.9013 +/- 0.0162 |

## Monolingual Hiragana MLM Accuracy

| Condition | Mode | MLM top-1 | MLM top-5 | MRR |
| --- | --- | ---: | ---: | ---: |
| `low_40` | `downsample` | 0.5004 +/- 0.0170 | 0.9206 +/- 0.0007 | 0.6752 +/- 0.0083 |
| `medium_60_full` | `downsample` | 0.4737 +/- 0.0101 | 0.8881 +/- 0.0015 | 0.6482 +/- 0.0054 |
| `medium_60_full` | `full_corpus` | 0.4849 +/- 0.0048 | 0.9064 +/- 0.0071 | 0.6584 +/- 0.0033 |
| `high_80_full` | `downsample` | 0.4633 +/- 0.0101 | 0.8758 +/- 0.0093 | 0.6354 +/- 0.0085 |
| `high_80_full` | `full_corpus` | 0.4970 +/- 0.0128 | 0.9130 +/- 0.0037 | 0.6664 +/- 0.0068 |
| `low_40_relation` | `downsample` | 0.5501 +/- 0.0168 | 0.9413 +/- 0.0085 | 0.7138 +/- 0.0107 |
| `medium_60_relation` | `downsample` | 0.5048 +/- 0.0008 | 0.9119 +/- 0.0159 | 0.6761 +/- 0.0024 |
| `medium_60_relation` | `full_corpus` | 0.5335 +/- 0.0056 | 0.9302 +/- 0.0098 | 0.7000 +/- 0.0031 |
| `high_80_relation` | `downsample` | 0.4929 +/- 0.0057 | 0.8829 +/- 0.0018 | 0.6578 +/- 0.0038 |
| `high_80_relation` | `full_corpus` | 0.5213 +/- 0.0021 | 0.9200 +/- 0.0048 | 0.6879 +/- 0.0046 |
| `full_graph` | `downsample` | 0.4691 +/- 0.0067 | 0.8597 +/- 0.0031 | 0.6351 +/- 0.0057 |
| `full_graph` | `full_corpus` | 0.5141 +/- 0.0056 | 0.9101 +/- 0.0068 | 0.6793 +/- 0.0064 |

## Full-Corpus Delta

| Condition | Word top-1 | Sentence top-1 | MLM top-1 | MRR |
| --- | ---: | ---: | ---: | ---: |
| `medium_60_full` | +0.0002 | +0.0233 | +0.0112 | +0.0102 |
| `high_80_full` | +0.1960 | +0.1720 | +0.0337 | +0.0310 |
| `medium_60_relation` | +0.1678 | +0.2193 | +0.0287 | +0.0239 |
| `high_80_relation` | +0.1348 | +0.1833 | +0.0284 | +0.0300 |
| `full_graph` | +0.1895 | +0.2487 | +0.0450 | +0.0442 |

Chart files are stored in:

```text
experiments/graph_density/v3_adj_variants/visualizations_training_mode/
```

Density-style line charts:

- `training_mode_multilingual_alignment_top1_lines.png`
- `training_mode_multilingual_alignment_top5_lines.png`
- `training_mode_monolingual_lines.png`

Mode comparison bar and delta charts:

- `training_mode_multilingual_top1.png`
- `training_mode_multilingual_top5.png`
- `training_mode_monolingual_mlm.png`
- `training_mode_word_top1_delta.png`
- `training_mode_mlm_top1_delta.png`


## V3 Adjective-Variant Performance Visualizations

Static charts using the same structure as the earlier graph-density
visualizations are stored in:

```text
experiments/graph_density/v3_adj_variants/visualizations/
```

Generated chart files:

- `multilingual_alignment_top1.png`
- `multilingual_alignment_top5.png`
- `monolingual_hiragana_mlm.png`
- `seed42_condition_comparison.png`
- `full_graph_delta_vs_high80_relation.png`

Chart-ready CSV summaries:

```text
experiments/graph_density/v3_adj_variants/visualizations/multilingual_summary.csv
experiments/graph_density/v3_adj_variants/visualizations/monolingual_summary.csv
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
