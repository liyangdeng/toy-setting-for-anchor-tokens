# V3 Downsample vs Full-Corpus Training Comparison

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
