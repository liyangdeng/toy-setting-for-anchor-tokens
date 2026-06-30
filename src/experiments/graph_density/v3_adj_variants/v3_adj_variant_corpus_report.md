# Graph Density V3 Adjective-Variant Corpus Report

This report is for the new v3 sentence generator with adjective minimal-pair variants.
It is intentionally separated from the earlier graph-density corpus report.

## Generation Parameters

| Parameter | Value |
| --- | ---: |
| `s1` | 0 |
| `s2` | 1 |
| `s3` | 0 |
| `adj_when_available` | 0.8 |
| `adj_variant_prob` | 0.3 |
| `max_adj_variants` | 3 |
| `n_samples` | 8 |

## Full Graph

| Field | Value |
| --- | ---: |
| Edges | 7181 |
| Endpoint nodes | 2042 |
| HasQuality edges | 292 |

## Full Graph Source-Type Distribution

| Source type | Edges | Percent |
| --- | ---: | ---: |
| `conceptnet` | 4710 | 65.59% |
| `wordnet` | 2405 | 33.49% |
| `repair` | 56 | 0.78% |
| `adjustment` | 10 | 0.14% |

## Corpus Summary By Seed

| Condition | Seed | Strategy | Edges | Active nodes | Isolated nodes | HasQuality | Records | Sentences/lang | Missing CJK | Missing Hiragana |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `low_40` | 42 | coverage | 2872 | 2042 | 0 | 45 | 2827 | 19268 | 0 | 0 |
| `low_40` | 43 | coverage | 2872 | 2042 | 0 | 40 | 2832 | 19266 | 0 | 0 |
| `low_40` | 44 | coverage | 2872 | 2042 | 0 | 38 | 2834 | 19315 | 0 | 0 |
| `medium_60_full` | 42 | coverage | 4309 | 2042 | 0 | 45 | 4264 | 30503 | 0 | 0 |
| `medium_60_full` | 43 | coverage | 4309 | 2042 | 0 | 40 | 4269 | 30543 | 0 | 0 |
| `medium_60_full` | 44 | coverage | 4309 | 2042 | 0 | 38 | 4271 | 30535 | 0 | 0 |
| `high_80_full` | 42 | coverage | 5745 | 2042 | 0 | 45 | 5700 | 41361 | 0 | 0 |
| `high_80_full` | 43 | coverage | 5745 | 2042 | 0 | 80 | 5665 | 41115 | 0 | 0 |
| `high_80_full` | 44 | coverage | 5745 | 2042 | 0 | 73 | 5672 | 41156 | 0 | 0 |
| `low_40_relation` | 42 | relation | 2872 | 2042 | 0 | 115 | 2757 | 17797 | 0 | 0 |
| `low_40_relation` | 43 | relation | 2872 | 2042 | 0 | 114 | 2758 | 17893 | 0 | 0 |
| `low_40_relation` | 44 | relation | 2872 | 2042 | 0 | 115 | 2757 | 17886 | 0 | 0 |
| `medium_60_relation` | 42 | relation | 4309 | 2042 | 0 | 175 | 4134 | 28551 | 0 | 0 |
| `medium_60_relation` | 43 | relation | 4309 | 2042 | 0 | 175 | 4134 | 28428 | 0 | 0 |
| `medium_60_relation` | 44 | relation | 4309 | 2042 | 0 | 174 | 4135 | 28447 | 0 | 0 |
| `high_80_relation` | 42 | relation | 5745 | 2042 | 0 | 234 | 5511 | 39346 | 0 | 0 |
| `high_80_relation` | 43 | relation | 5745 | 2042 | 0 | 234 | 5511 | 39179 | 0 | 0 |
| `high_80_relation` | 44 | relation | 5745 | 2042 | 0 | 234 | 5511 | 39129 | 0 | 0 |
| `full_graph` | 42 | coverage | 7181 | 2042 | 0 | 292 | 6889 | 50215 | 0 | 0 |
| `full_graph` | 43 | coverage | 7181 | 2042 | 0 | 292 | 6889 | 50207 | 0 | 0 |
| `full_graph` | 44 | coverage | 7181 | 2042 | 0 | 292 | 6889 | 50206 | 0 | 0 |

## Experiment Group Summary

| Condition | Seeds | Sampling | Candidate fraction | Edges | Active nodes | Isolated nodes | HasQuality mean | Avg degree mean | Max degree range | Sentences/lang mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `low_40` | 42,43,44 | `coverage` | 0.40 | 2872,2872,2872 | 2042,2042,2042 | 0,0,0 | 41.0 | 2.427 | 12-12 | 19283.0 |
| `medium_60_full` | 42,43,44 | `coverage` | 0.60 | 4309,4309,4309 | 2042,2042,2042 | 0,0,0 | 41.0 | 3.663 | 13-13 | 30527.0 |
| `high_80_full` | 42,43,44 | `coverage` | 0.80 | 5745,5745,5745 | 2042,2042,2042 | 0,0,0 | 66.0 | 4.376 | 13-13 | 41210.7 |
| `low_40_relation` | 42,43,44 | `relation` | 0.40 | 2872,2872,2872 | 2042,2042,2042 | 0,0,0 | 114.7 | 2.570 | 11-11 | 17858.7 |
| `medium_60_relation` | 42,43,44 | `relation` | 0.60 | 4309,4309,4309 | 2042,2042,2042 | 0,0,0 | 174.7 | 3.557 | 13-14 | 28475.3 |
| `high_80_relation` | 42,43,44 | `relation` | 0.80 | 5745,5745,5745 | 2042,2042,2042 | 0,0,0 | 234.0 | 4.435 | 15-18 | 39218.0 |
| `full_graph` | 42,43,44 | `coverage` | 1.00 | 7181,7181,7181 | 2042,2042,2042 | 0,0,0 | 292.0 | 5.239 | 19-19 | 50209.3 |

## Full Graph Relation Distribution

| Relation | Edges | Percent |
| --- | ---: | ---: |
| `RelatedTo` | 1300 | 18.10% |
| `HasContext` | 1193 | 16.61% |
| `hyponym` | 1034 | 14.40% |
| `AtLocation` | 944 | 13.15% |
| `hypernym` | 867 | 12.07% |
| `HasQuality` | 292 | 4.07% |
| `UsedFor` | 250 | 3.48% |
| `HasProperty` | 152 | 2.12% |
| `Antonym` | 136 | 1.89% |
| `SimilarTo` | 113 | 1.57% |
| `CapableOf` | 112 | 1.56% |
| `part_meronym` | 94 | 1.31% |
| `part_holonym` | 91 | 1.27% |
| `member_holonym` | 71 | 0.99% |
| `instance_hypernym` | 67 | 0.93% |
| `member_meronym` | 67 | 0.93% |
| `instance_hyponym` | 64 | 0.89% |
| `HasPrerequisite` | 56 | 0.78% |
| `DistinctFrom` | 41 | 0.57% |
| `HasSubevent` | 38 | 0.53% |
| `Causes` | 37 | 0.52% |
| `MadeOf` | 26 | 0.36% |
| `NotHasProperty` | 23 | 0.32% |
| `MotivatedByGoal` | 22 | 0.31% |
| `ReceivesAction` | 21 | 0.29% |
| `Desires` | 20 | 0.28% |
| `CausesDesire` | 13 | 0.18% |
| `HasFirstSubevent` | 9 | 0.13% |
| `HasLastSubevent` | 7 | 0.10% |
| `NotDesires` | 6 | 0.08% |
| `substance_holonym` | 5 | 0.07% |
| `substance_meronym` | 5 | 0.07% |
| `CreatedBy` | 2 | 0.03% |
| `DefinedAs` | 1 | 0.01% |
| `LocatedNear` | 1 | 0.01% |
| `NotCapableOf` | 1 | 0.01% |

## Relation Distribution by Experiment Group

Counts are averaged across seeds 42, 43, and 44. Percent is mean count divided by mean edge count for that condition.

### `low_40`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 1296.0 | 45.13% | 18.10% |
| `hyponym` | 658.0 | 22.91% | 14.40% |
| `HasContext` | 230.3 | 8.02% | 16.61% |
| `hypernym` | 190.0 | 6.62% | 12.07% |
| `AtLocation` | 114.3 | 3.98% | 13.15% |
| `HasQuality` | 41.0 | 1.43% | 4.07% |
| `CapableOf` | 39.7 | 1.38% | 1.56% |
| `SimilarTo` | 39.0 | 1.36% | 1.57% |
| `UsedFor` | 37.7 | 1.31% | 3.48% |
| `Antonym` | 33.7 | 1.17% | 1.89% |
| `HasProperty` | 26.0 | 0.91% | 2.12% |
| `instance_hypernym` | 24.0 | 0.84% | 0.93% |
| `member_meronym` | 22.0 | 0.77% | 0.93% |
| `member_holonym` | 20.3 | 0.71% | 0.99% |
| `part_holonym` | 16.0 | 0.56% | 1.27% |
| `Causes` | 11.0 | 0.38% | 0.52% |
| `part_meronym` | 10.3 | 0.36% | 1.31% |
| `HasSubevent` | 9.0 | 0.31% | 0.53% |
| `DistinctFrom` | 8.7 | 0.30% | 0.57% |
| `instance_hyponym` | 8.3 | 0.29% | 0.89% |
| `HasPrerequisite` | 7.3 | 0.26% | 0.78% |
| `ReceivesAction` | 7.3 | 0.26% | 0.29% |
| `MotivatedByGoal` | 5.0 | 0.17% | 0.31% |
| `NotDesires` | 4.0 | 0.14% | 0.08% |
| `MadeOf` | 3.0 | 0.10% | 0.36% |
| `HasFirstSubevent` | 2.3 | 0.08% | 0.13% |
| `Desires` | 2.0 | 0.07% | 0.28% |
| `CreatedBy` | 1.0 | 0.03% | 0.03% |
| `DefinedAs` | 1.0 | 0.03% | 0.01% |
| `LocatedNear` | 1.0 | 0.03% | 0.01% |
| `NotCapableOf` | 1.0 | 0.03% | 0.01% |
| `substance_holonym` | 0.7 | 0.02% | 0.07% |
| `CausesDesire` | 0.3 | 0.01% | 0.18% |
| `NotHasProperty` | 0.3 | 0.01% | 0.32% |
| `substance_meronym` | 0.3 | 0.01% | 0.07% |

### `medium_60_full`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 1296.0 | 30.08% | 18.10% |
| `HasContext` | 1192.0 | 27.66% | 16.61% |
| `hyponym` | 1028.0 | 23.86% | 14.40% |
| `AtLocation` | 219.7 | 5.10% | 13.15% |
| `hypernym` | 190.0 | 4.41% | 12.07% |
| `HasQuality` | 41.0 | 0.95% | 4.07% |
| `CapableOf` | 39.7 | 0.92% | 1.56% |
| `SimilarTo` | 39.0 | 0.91% | 1.57% |
| `UsedFor` | 37.7 | 0.87% | 3.48% |
| `Antonym` | 33.7 | 0.78% | 1.89% |
| `HasProperty` | 26.0 | 0.60% | 2.12% |
| `instance_hypernym` | 24.0 | 0.56% | 0.93% |
| `member_meronym` | 22.0 | 0.51% | 0.93% |
| `member_holonym` | 20.3 | 0.47% | 0.99% |
| `part_holonym` | 16.0 | 0.37% | 1.27% |
| `Causes` | 11.0 | 0.26% | 0.52% |
| `part_meronym` | 10.3 | 0.24% | 1.31% |
| `HasSubevent` | 9.0 | 0.21% | 0.53% |
| `DistinctFrom` | 8.7 | 0.20% | 0.57% |
| `instance_hyponym` | 8.3 | 0.19% | 0.89% |
| `HasPrerequisite` | 7.3 | 0.17% | 0.78% |
| `ReceivesAction` | 7.3 | 0.17% | 0.29% |
| `MotivatedByGoal` | 5.0 | 0.12% | 0.31% |
| `NotDesires` | 4.0 | 0.09% | 0.08% |
| `MadeOf` | 3.0 | 0.07% | 0.36% |
| `HasFirstSubevent` | 2.3 | 0.05% | 0.13% |
| `Desires` | 2.0 | 0.05% | 0.28% |
| `CreatedBy` | 1.0 | 0.02% | 0.03% |
| `DefinedAs` | 1.0 | 0.02% | 0.01% |
| `LocatedNear` | 1.0 | 0.02% | 0.01% |
| `NotCapableOf` | 1.0 | 0.02% | 0.01% |
| `substance_holonym` | 0.7 | 0.02% | 0.07% |
| `CausesDesire` | 0.3 | 0.01% | 0.18% |
| `NotHasProperty` | 0.3 | 0.01% | 0.32% |
| `substance_meronym` | 0.3 | 0.01% | 0.07% |

### `high_80_full`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 1296.0 | 22.56% | 18.10% |
| `HasContext` | 1192.0 | 20.75% | 16.61% |
| `hyponym` | 1028.0 | 17.89% | 14.40% |
| `AtLocation` | 944.0 | 16.43% | 13.15% |
| `hypernym` | 866.0 | 15.07% | 12.07% |
| `HasQuality` | 66.0 | 1.15% | 4.07% |
| `UsedFor` | 48.3 | 0.84% | 3.48% |
| `CapableOf` | 39.7 | 0.69% | 1.56% |
| `SimilarTo` | 39.0 | 0.68% | 1.57% |
| `Antonym` | 33.7 | 0.59% | 1.89% |
| `HasProperty` | 26.0 | 0.45% | 2.12% |
| `instance_hypernym` | 24.0 | 0.42% | 0.93% |
| `member_meronym` | 22.0 | 0.38% | 0.93% |
| `member_holonym` | 20.3 | 0.35% | 0.99% |
| `part_holonym` | 16.0 | 0.28% | 1.27% |
| `Causes` | 11.0 | 0.19% | 0.52% |
| `part_meronym` | 10.3 | 0.18% | 1.31% |
| `HasSubevent` | 9.0 | 0.16% | 0.53% |
| `DistinctFrom` | 8.7 | 0.15% | 0.57% |
| `instance_hyponym` | 8.3 | 0.15% | 0.89% |
| `HasPrerequisite` | 7.3 | 0.13% | 0.78% |
| `ReceivesAction` | 7.3 | 0.13% | 0.29% |
| `MotivatedByGoal` | 5.0 | 0.09% | 0.31% |
| `NotDesires` | 4.0 | 0.07% | 0.08% |
| `MadeOf` | 3.0 | 0.05% | 0.36% |
| `HasFirstSubevent` | 2.3 | 0.04% | 0.13% |
| `Desires` | 2.0 | 0.03% | 0.28% |
| `CreatedBy` | 1.0 | 0.02% | 0.03% |
| `DefinedAs` | 1.0 | 0.02% | 0.01% |
| `LocatedNear` | 1.0 | 0.02% | 0.01% |
| `NotCapableOf` | 1.0 | 0.02% | 0.01% |
| `substance_holonym` | 0.7 | 0.01% | 0.07% |
| `CausesDesire` | 0.3 | 0.01% | 0.18% |
| `NotHasProperty` | 0.3 | 0.01% | 0.32% |
| `substance_meronym` | 0.3 | 0.01% | 0.07% |

### `low_40_relation`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 518.0 | 18.04% | 18.10% |
| `HasContext` | 481.3 | 16.76% | 16.61% |
| `hyponym` | 412.0 | 14.35% | 14.40% |
| `AtLocation` | 378.0 | 13.16% | 13.15% |
| `hypernym` | 346.7 | 12.07% | 12.07% |
| `HasQuality` | 114.7 | 3.99% | 4.07% |
| `UsedFor` | 102.0 | 3.55% | 3.48% |
| `HasProperty` | 60.3 | 2.10% | 2.12% |
| `Antonym` | 54.0 | 1.88% | 1.89% |
| `CapableOf` | 48.7 | 1.69% | 1.56% |
| `SimilarTo` | 46.0 | 1.60% | 1.57% |
| `part_meronym` | 38.0 | 1.32% | 1.31% |
| `part_holonym` | 36.0 | 1.25% | 1.27% |
| `member_holonym` | 27.7 | 0.96% | 0.99% |
| `member_meronym` | 27.0 | 0.94% | 0.93% |
| `instance_hypernym` | 26.0 | 0.91% | 0.93% |
| `instance_hyponym` | 25.0 | 0.87% | 0.89% |
| `HasPrerequisite` | 22.0 | 0.77% | 0.78% |
| `Causes` | 18.0 | 0.63% | 0.52% |
| `DistinctFrom` | 17.0 | 0.59% | 0.57% |
| `HasSubevent` | 14.0 | 0.49% | 0.53% |
| `MadeOf` | 9.0 | 0.31% | 0.36% |
| `MotivatedByGoal` | 8.0 | 0.28% | 0.31% |
| `NotHasProperty` | 8.0 | 0.28% | 0.32% |
| `ReceivesAction` | 8.0 | 0.28% | 0.29% |
| `Desires` | 7.0 | 0.24% | 0.28% |
| `CausesDesire` | 4.0 | 0.14% | 0.18% |
| `NotDesires` | 3.7 | 0.13% | 0.08% |
| `HasFirstSubevent` | 3.0 | 0.10% | 0.13% |
| `HasLastSubevent` | 2.0 | 0.07% | 0.10% |
| `substance_holonym` | 2.0 | 0.07% | 0.07% |
| `substance_meronym` | 2.0 | 0.07% | 0.07% |
| `CreatedBy` | 1.3 | 0.05% | 0.03% |
| `NotCapableOf` | 1.0 | 0.03% | 0.01% |
| `DefinedAs` | 0.7 | 0.02% | 0.01% |

### `medium_60_relation`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 780.0 | 18.10% | 18.10% |
| `HasContext` | 718.0 | 16.66% | 16.61% |
| `hyponym` | 620.3 | 14.40% | 14.40% |
| `AtLocation` | 566.0 | 13.14% | 13.15% |
| `hypernym` | 521.3 | 12.10% | 12.07% |
| `HasQuality` | 174.7 | 4.05% | 4.07% |
| `UsedFor` | 151.3 | 3.51% | 3.48% |
| `HasProperty` | 90.0 | 2.09% | 2.12% |
| `Antonym` | 80.0 | 1.86% | 1.89% |
| `SimilarTo` | 69.0 | 1.60% | 1.57% |
| `CapableOf` | 68.3 | 1.59% | 1.56% |
| `part_meronym` | 56.0 | 1.30% | 1.31% |
| `part_holonym` | 55.0 | 1.28% | 1.27% |
| `member_holonym` | 43.3 | 1.01% | 0.99% |
| `instance_hypernym` | 40.0 | 0.93% | 0.93% |
| `member_meronym` | 40.0 | 0.93% | 0.93% |
| `instance_hyponym` | 38.0 | 0.88% | 0.89% |
| `HasPrerequisite` | 33.0 | 0.77% | 0.78% |
| `DistinctFrom` | 25.0 | 0.58% | 0.57% |
| `Causes` | 24.0 | 0.56% | 0.52% |
| `HasSubevent` | 23.0 | 0.53% | 0.53% |
| `MadeOf` | 16.0 | 0.37% | 0.36% |
| `NotHasProperty` | 14.0 | 0.32% | 0.32% |
| `MotivatedByGoal` | 13.0 | 0.30% | 0.31% |
| `ReceivesAction` | 13.0 | 0.30% | 0.29% |
| `Desires` | 11.0 | 0.26% | 0.28% |
| `CausesDesire` | 6.0 | 0.14% | 0.18% |
| `HasFirstSubevent` | 4.0 | 0.09% | 0.13% |
| `NotDesires` | 3.3 | 0.08% | 0.08% |
| `HasLastSubevent` | 3.0 | 0.07% | 0.10% |
| `substance_holonym` | 3.0 | 0.07% | 0.07% |
| `substance_meronym` | 3.0 | 0.07% | 0.07% |
| `CreatedBy` | 1.0 | 0.02% | 0.03% |
| `LocatedNear` | 1.0 | 0.02% | 0.01% |
| `NotCapableOf` | 1.0 | 0.02% | 0.01% |
| `DefinedAs` | 0.3 | 0.01% | 0.01% |

### `high_80_relation`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 1040.0 | 18.10% | 18.10% |
| `HasContext` | 956.0 | 16.64% | 16.61% |
| `hyponym` | 828.3 | 14.42% | 14.40% |
| `AtLocation` | 755.0 | 13.14% | 13.15% |
| `hypernym` | 694.3 | 12.09% | 12.07% |
| `HasQuality` | 234.0 | 4.07% | 4.07% |
| `UsedFor` | 200.0 | 3.48% | 3.48% |
| `HasProperty` | 121.0 | 2.11% | 2.12% |
| `Antonym` | 106.0 | 1.85% | 1.89% |
| `SimilarTo` | 91.0 | 1.58% | 1.57% |
| `CapableOf` | 89.0 | 1.55% | 1.56% |
| `part_meronym` | 75.0 | 1.31% | 1.31% |
| `part_holonym` | 73.0 | 1.27% | 1.27% |
| `member_holonym` | 57.3 | 1.00% | 0.99% |
| `instance_hypernym` | 54.0 | 0.94% | 0.93% |
| `member_meronym` | 54.0 | 0.94% | 0.93% |
| `instance_hyponym` | 51.0 | 0.89% | 0.89% |
| `HasPrerequisite` | 45.0 | 0.78% | 0.78% |
| `DistinctFrom` | 33.0 | 0.57% | 0.57% |
| `Causes` | 30.0 | 0.52% | 0.52% |
| `HasSubevent` | 30.0 | 0.52% | 0.53% |
| `MadeOf` | 21.0 | 0.37% | 0.36% |
| `MotivatedByGoal` | 18.0 | 0.31% | 0.31% |
| `NotHasProperty` | 18.0 | 0.31% | 0.32% |
| `Desires` | 16.0 | 0.28% | 0.28% |
| `ReceivesAction` | 16.0 | 0.28% | 0.29% |
| `CausesDesire` | 9.0 | 0.16% | 0.18% |
| `HasFirstSubevent` | 7.0 | 0.12% | 0.13% |
| `HasLastSubevent` | 6.0 | 0.10% | 0.10% |
| `NotDesires` | 5.0 | 0.09% | 0.08% |
| `substance_holonym` | 4.0 | 0.07% | 0.07% |
| `substance_meronym` | 4.0 | 0.07% | 0.07% |
| `CreatedBy` | 1.0 | 0.02% | 0.03% |
| `DefinedAs` | 1.0 | 0.02% | 0.01% |
| `LocatedNear` | 1.0 | 0.02% | 0.01% |
| `NotCapableOf` | 1.0 | 0.02% | 0.01% |

### `full_graph`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 1300.0 | 18.10% | 18.10% |
| `HasContext` | 1193.0 | 16.61% | 16.61% |
| `hyponym` | 1034.0 | 14.40% | 14.40% |
| `AtLocation` | 944.0 | 13.15% | 13.15% |
| `hypernym` | 867.0 | 12.07% | 12.07% |
| `HasQuality` | 292.0 | 4.07% | 4.07% |
| `UsedFor` | 250.0 | 3.48% | 3.48% |
| `HasProperty` | 152.0 | 2.12% | 2.12% |
| `Antonym` | 136.0 | 1.89% | 1.89% |
| `SimilarTo` | 113.0 | 1.57% | 1.57% |
| `CapableOf` | 112.0 | 1.56% | 1.56% |
| `part_meronym` | 94.0 | 1.31% | 1.31% |
| `part_holonym` | 91.0 | 1.27% | 1.27% |
| `member_holonym` | 71.0 | 0.99% | 0.99% |
| `instance_hypernym` | 67.0 | 0.93% | 0.93% |
| `member_meronym` | 67.0 | 0.93% | 0.93% |
| `instance_hyponym` | 64.0 | 0.89% | 0.89% |
| `HasPrerequisite` | 56.0 | 0.78% | 0.78% |
| `DistinctFrom` | 41.0 | 0.57% | 0.57% |
| `HasSubevent` | 38.0 | 0.53% | 0.53% |
| `Causes` | 37.0 | 0.52% | 0.52% |
| `MadeOf` | 26.0 | 0.36% | 0.36% |
| `NotHasProperty` | 23.0 | 0.32% | 0.32% |
| `MotivatedByGoal` | 22.0 | 0.31% | 0.31% |
| `ReceivesAction` | 21.0 | 0.29% | 0.29% |
| `Desires` | 20.0 | 0.28% | 0.28% |
| `CausesDesire` | 13.0 | 0.18% | 0.18% |
| `HasFirstSubevent` | 9.0 | 0.13% | 0.13% |
| `HasLastSubevent` | 7.0 | 0.10% | 0.10% |
| `NotDesires` | 6.0 | 0.08% | 0.08% |
| `substance_holonym` | 5.0 | 0.07% | 0.07% |
| `substance_meronym` | 5.0 | 0.07% | 0.07% |
| `CreatedBy` | 2.0 | 0.03% | 0.03% |
| `DefinedAs` | 1.0 | 0.01% | 0.01% |
| `LocatedNear` | 1.0 | 0.01% | 0.01% |
| `NotCapableOf` | 1.0 | 0.01% | 0.01% |

## Source-Type Distribution by Experiment Group

Counts are averaged across seeds 42, 43, and 44.

### `low_40`

| Source type | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `conceptnet` | 1862.7 | 64.86% | 65.59% |
| `wordnet` | 958.3 | 33.37% | 33.49% |
| `repair` | 50.0 | 1.74% | 0.78% |
| `adjustment` | 1.0 | 0.03% | 0.14% |

### `medium_60_full`

| Source type | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `conceptnet` | 2929.7 | 67.99% | 65.59% |
| `wordnet` | 1328.3 | 30.83% | 33.49% |
| `repair` | 50.0 | 1.16% | 0.78% |
| `adjustment` | 1.0 | 0.02% | 0.14% |

### `high_80_full`

| Source type | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `conceptnet` | 3689.7 | 64.22% | 65.59% |
| `wordnet` | 2004.3 | 34.89% | 33.49% |
| `repair` | 50.0 | 0.87% | 0.78% |
| `adjustment` | 1.0 | 0.02% | 0.14% |

### `low_40_relation`

| Source type | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `conceptnet` | 1864.7 | 64.93% | 65.59% |
| `wordnet` | 957.3 | 33.33% | 33.49% |
| `repair` | 47.3 | 1.65% | 0.78% |
| `adjustment` | 2.7 | 0.09% | 0.14% |

### `medium_60_relation`

| Source type | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `conceptnet` | 2813.7 | 65.30% | 65.59% |
| `wordnet` | 1443.7 | 33.50% | 33.49% |
| `repair` | 48.3 | 1.12% | 0.78% |
| `adjustment` | 3.3 | 0.08% | 0.14% |

### `high_80_relation`

| Source type | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `conceptnet` | 3763.0 | 65.50% | 65.59% |
| `wordnet` | 1926.0 | 33.52% | 33.49% |
| `repair` | 50.0 | 0.87% | 0.78% |
| `adjustment` | 6.0 | 0.10% | 0.14% |

### `full_graph`

| Source type | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `conceptnet` | 4710.0 | 65.59% | 65.59% |
| `wordnet` | 2405.0 | 33.49% | 33.49% |
| `repair` | 56.0 | 0.78% | 0.78% |
| `adjustment` | 10.0 | 0.14% | 0.14% |

## Mean Sentences Per Language

| Condition | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| `low_40` | 19283.0 | 19266 | 19315 |
| `medium_60_full` | 30527.0 | 30503 | 30543 |
| `high_80_full` | 41210.7 | 41115 | 41361 |
| `low_40_relation` | 17858.7 | 17797 | 17893 |
| `medium_60_relation` | 28475.3 | 28428 | 28551 |
| `high_80_relation` | 39218.0 | 39129 | 39346 |
| `full_graph` | 50209.3 | 50206 | 50215 |

## Recommended Epoch Sample Size

Use the smallest low-density single-language corpus as the matched
per-epoch sample budget for the next training step.

| Setting | Value |
| --- | ---: |
| Source condition | `low_40_relation` |
| Source seed | 42 |
| Monolingual epoch sample size | 17797 |
| Multilingual epoch sample size | 35594 |


## Relation Counts

Full relation distributions for every generated graph are stored in:

```text
experiments/graph_density/v3_adj_variants/v3_adj_variant_corpus_summary.json
```
