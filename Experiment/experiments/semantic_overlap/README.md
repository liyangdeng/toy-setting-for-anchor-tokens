# Semantic Overlap Experiment

## Research Question

This experiment tests whether increasing the semantic overlap between two
artificial-language corpora improves cross-lingual alignment, while keeping the
per-language graph size, relation distribution, grammar, tokenizer, and model
training setup fixed.

The main question is:

> If two languages describe increasingly overlapping subsets of the same
> semantic graph, do their separately named artificial tokens become easier to
> align?

## Source Graph

All conditions are sampled from the same graph:

```text
data/semantic_backbones/edges_adj.json
```

The raw graph has 7,181 edges and 2,042 endpoint nodes. After de-duplicating
edge identities for sampling, the experiment uses 7,174 unique candidate edges.

Each language receives 2,870 edges, about 40% of the unique edge inventory.
This edge budget is fixed across all overlap conditions.

## Experimental Conditions

For each seed and each overlap condition, two language-specific subgraphs are
created:

- L1: CJK artificial-token language.
- L2: Hiragana artificial-token language.

The overlap ratio controls how many of the 2,870 edges are shared by both
languages.

| Condition | Target overlap | Shared edges | L1-only edges | L2-only edges | L1 total | L2 total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `overlap_000` | 0% | 0 | 2,870 | 2,870 | 2,870 | 2,870 |
| `overlap_025` | 25% | 718 | 2,152 | 2,152 | 2,870 | 2,870 |
| `overlap_050` | 50% | 1,435 | 1,435 | 1,435 | 2,870 | 2,870 |
| `overlap_075` | 75% | 2,152 | 718 | 718 | 2,870 | 2,870 |
| `overlap_100` | 100% | 2,870 | 0 | 0 | 2,870 | 2,870 |

The `overlap_100` condition is not the full graph. It is a 40% sampled subgraph
where L1 and L2 receive the same sampled edge set.

## Sampling Method

Sampling is stratified by `(relation, source_type)` so that each L1/L2 subgraph
closely follows the full graph's semantic relation and source-type
distribution.

For each seed:

1. Compute the requested shared-edge count from the 2,870-edge language budget.
2. Sample the shared pool by relation/source-type strata.
3. Sample L1-only and L2-only pools from the remaining edges, also by strata.
4. Build `edges_l1.json` as `shared + l1_only`.
5. Build `edges_l2.json` as `shared + l2_only`.
6. Write graph statistics and sampling diagnostics to `metadata.json`.

Because relation distribution is prioritized, the sampler does not force every
full-graph node to appear in every subgraph. Nodes with no retained edges are
treated as outside that language's corpus for that condition.

Across seeds 42, 43, and 44, the maximum relation/source-type distribution
deviation from the target is below 0.001 for every condition.

## Graph Statistics

Mean values across seeds 42, 43, and 44:

| Condition | Actual overlap | L1 active nodes | L2 active nodes | L1 isolated nodes | L2 isolated nodes | Max distribution delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `overlap_000` | 0.0000 | 1,725.3 | 1,717.3 | 316.7 | 324.7 | 0.00049 |
| `overlap_025` | 0.2502 | 1,721.0 | 1,725.3 | 321.0 | 316.7 | 0.00055 |
| `overlap_050` | 0.5000 | 1,732.0 | 1,744.0 | 310.0 | 298.0 | 0.00055 |
| `overlap_075` | 0.7498 | 1,728.3 | 1,723.0 | 313.7 | 319.0 | 0.00035 |
| `overlap_100` | 1.0000 | 1,723.0 | 1,723.0 | 319.0 | 319.0 | 0.00021 |

Isolated nodes are counted relative to the 2,042-node full-graph universe. They
are not emitted into the generated corpora because sentence generation only
uses retained edges.

## Corpus Generation

Corpora are generated with the existing repository sentence generator:

```text
data/generate_sentences/v3_generate_sentences.py
```

The semantic-overlap shell scripts then build exact corpus files with:

```text
experiments/graph_density/build_corpus_exact.py
```

For seed 42, each condition generates roughly 18k CJK sentences and 18k
Hiragana sentences. The recommended multilingual per-epoch sample size is
36,162 sentences, based on the smallest combined corpus.

## Training Protocol

Training uses a small multilingual BERT-style MLM model trained from scratch on
the mixed CJK + Hiragana corpus for each condition.

The experiment-specific training script is:

```text
experiments/semantic_overlap/train_semantic_overlap_multilingual.py
```

The script is adapted from:

```text
src/training/train_multilingual_synset.py
```
## Evaluation

Evaluate whether graph density improves cross-lingual alignment and knowledge
transfer.

Primary evaluations:

1. Word translation precision.
2. Sentence retrieval precision.
3. Monolingual MLM accuracy
4. Linear probing accuracy on each layer.

## Evaluation Results

All results below are mean +/- sample standard deviation across seeds 42, 43, and 44. Sentence retrieval uses `--n_sample 500`.

All results are visualized in [visualizations](visualizations) for multilingual alignment and monolingual accuracy, [probing visualizations](linear_probe_evaluation/visualizations/).

### Multilingual Alignment

| Condition | Overlap | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `overlap_000` | 0% | 0.0071 +/- 0.0013 | 0.0291 +/- 0.0044 | N/A | N/A |
| `overlap_025` | 25% | 0.0210 +/- 0.0060 | 0.0614 +/- 0.0107 | 0.0300 +/- 0.0106 | 0.1380 +/- 0.0330 |
| `overlap_050` | 50% | 0.0420 +/- 0.0093 | 0.1188 +/- 0.0185 | 0.0467 +/- 0.0103 | 0.1460 +/- 0.0180 |
| `overlap_075` | 75% | 0.1273 +/- 0.0210 | 0.2617 +/- 0.0354 | 0.0767 +/- 0.0099 | 0.2327 +/- 0.0190 |
| `overlap_100` | 100% | 0.3523 +/- 0.0988 | 0.5553 +/- 0.1081 | 0.1740 +/- 0.0470 | 0.3807 +/- 0.0559 |

### Monolingual Hiragana MLM Accuracy

| Condition | Overlap | MLM top-1 | MLM top-5 | MLM MRR |
| --- | ---: | ---: | ---: | ---: |
| `overlap_000` | 0% | 0.5516 +/- 0.0023 | 0.9345 +/- 0.0030 | 0.7126 +/- 0.0015 |
| `overlap_025` | 25% | 0.5452 +/- 0.0128 | 0.9391 +/- 0.0050 | 0.7096 +/- 0.0095 |
| `overlap_050` | 50% | 0.5469 +/- 0.0098 | 0.9329 +/- 0.0078 | 0.7091 +/- 0.0040 |
| `overlap_075` | 75% | 0.5405 +/- 0.0079 | 0.9390 +/- 0.0034 | 0.7077 +/- 0.0046 |
| `overlap_100` | 100% | 0.5439 +/- 0.0108 | 0.9346 +/- 0.0024 | 0.7091 +/- 0.0060 |

### Semantic Overlap Masked-Language Probe Summary (entity)

| condition | seeds | usable probes | best acc | final acc | embedding acc | mean best layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_000 | 3 | 2753.0 | 0.0566 +/- 0.0043 | 0.0455 +/- 0.0029 | 0.0143 +/- 0.0018 | 2.7 |
| overlap_025 | 3 | 2064.0 | 0.0743 +/- 0.0093 | 0.0594 +/- 0.0123 | 0.0155 +/- 0.0015 | 1.7 |
| overlap_050 | 3 | 1377.0 | 0.1150 +/- 0.0197 | 0.1017 +/- 0.0215 | 0.0198 +/- 0.0008 | 1.7 |
| overlap_075 | 3 | 689.0 | 0.2279 +/- 0.0080 | 0.1959 +/- 0.0233 | 0.0227 +/- 0.0030 | 2.3 |

#### Per-Run Summary

| seed | condition | usable probes | best layer | best acc | final acc | embedding acc |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | overlap_000 | 2753 | 2 | 0.0585 | 0.0421 | 0.0160 |
| 43 | overlap_000 | 2753 | 3 | 0.0516 | 0.0472 | 0.0145 |
| 44 | overlap_000 | 2753 | 3 | 0.0596 | 0.0472 | 0.0124 |
| 42 | overlap_025 | 2064 | 2 | 0.0770 | 0.0572 | 0.0155 |
| 43 | overlap_025 | 2064 | 1 | 0.0819 | 0.0727 | 0.0141 |
| 44 | overlap_025 | 2064 | 2 | 0.0640 | 0.0484 | 0.0170 |
| 42 | overlap_050 | 1377 | 2 | 0.1373 | 0.1264 | 0.0203 |
| 43 | overlap_050 | 1377 | 1 | 0.1002 | 0.0915 | 0.0189 |
| 44 | overlap_050 | 1377 | 2 | 0.1075 | 0.0871 | 0.0203 |
| 42 | overlap_075 | 689 | 2 | 0.2293 | 0.1713 | 0.0261 |
| 43 | overlap_075 | 689 | 3 | 0.2351 | 0.2177 | 0.0218 |
| 44 | overlap_075 | 689 | 2 | 0.2192 | 0.1988 | 0.0203 |

## Token Embedding Distance Probe

A post-hoc embedding probe was run for seed 42 with cosine distance between
true CJK-Hiragana token pairs. Only synsets that appear as endpoints in
`edges_adj.json` are included.

Graph endpoint nodes used for filtering: 2,042. Valid graph-token pairs present
in every model: 994.

| Pair group | 0% | 25% | 50% | 75% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Close at 100% | 0.7534 | 0.7093 | 0.6401 | 0.6993 | 0.3183 |
| Random graph true pairs | 0.7638 | 0.7568 | 0.6913 | 0.6445 | 0.5004 |
| Far at 100% | 0.8527 | 0.7727 | 0.7363 | 0.6786 | 0.8529 |

Cosine distance is lower when embeddings are closer. The close-at-100% group
shows a large drop in distance at 100% overlap, consistent with stronger
cross-lingual alignment for some true synset pairs.


