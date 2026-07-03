# Graph Density Experiment Report - edges_adj Full Graph

## Source Graph

This report uses `data/semantic_backbones/edges_adj.json` as the full graph.
Earlier generated experiment groups used `data/semantic_backbones/kg_noun_only/edges.json`; those generated group directories were deleted and regenerated from `edges_adj.json`.
The node universe for this report is all unique endpoints in `edges_adj.json`, because `edges_adj.json` contains 42 adjective endpoints that are not present in the existing `synsets.json` files.

| Graph | Edges | Nodes |
| --- | ---: | ---: |
| full `edges_adj` | 7181 | 2042 |

## Experiment Group Summary

| Condition | Seeds | Sampling | Control | Candidate fraction | Training fraction | Edges | Active nodes | Isolated nodes | Avg degree mean | Max degree range |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `low_40` | 42,43,44 | `coverage` | `none` | 0.40 | 0.40 | 2872,2872,2872 | 2042,2042,2042 | 0,0,0 | 2.427 | 12-12 |
| `low_40_relation` | 42,43,44 | `relation` | `none` | 0.40 | 0.40 | 2872,2872,2872 | 2042,2042,2042 | 0,0,0 | 2.570 | 11-11 |
| `medium_60_full` | 42,43,44 | `coverage` | `none` | 0.60 | 0.60 | 4309,4309,4309 | 2042,2042,2042 | 0,0,0 | 3.663 | 13-13 |
| `medium_60_relation` | 42,43,44 | `relation` | `none` | 0.60 | 0.60 | 4309,4309,4309 | 2042,2042,2042 | 0,0,0 | 3.557 | 13-14 |
| `high_80_full` | 42,43,44 | `coverage` | `none` | 0.80 | 0.80 | 5745,5745,5745 | 2042,2042,2042 | 0,0,0 | 4.376 | 13-13 |
| `high_80_relation` | 42,43,44 | `relation` | `none` | 0.80 | 0.80 | 5745,5745,5745 | 2042,2042,2042 | 0,0,0 | 4.435 | 15-18 |


## Relation Distribution by Experiment Group

Counts are averaged across seeds 42, 43, and 44. Percent is mean count divided by mean edge count for that condition.

### `low_40`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 1296.0 | 45.13% | 18.10% |
| `HasContext` | 230.3 | 8.02% | 16.61% |
| `hyponym` | 658.0 | 22.91% | 14.40% |
| `AtLocation` | 114.3 | 3.98% | 13.15% |
| `hypernym` | 190.0 | 6.62% | 12.07% |
| `HasQuality` | 41.0 | 1.43% | 4.07% |
| `UsedFor` | 37.7 | 1.31% | 3.48% |
| `HasProperty` | 26.0 | 0.91% | 2.12% |
| `Antonym` | 33.7 | 1.17% | 1.89% |
| `SimilarTo` | 39.0 | 1.36% | 1.57% |
| `CapableOf` | 39.7 | 1.38% | 1.56% |
| `part_meronym` | 10.3 | 0.36% | 1.31% |
| `part_holonym` | 16.0 | 0.56% | 1.27% |
| `member_holonym` | 20.3 | 0.71% | 0.99% |
| `instance_hypernym` | 24.0 | 0.84% | 0.93% |
| `member_meronym` | 22.0 | 0.77% | 0.93% |
| `instance_hyponym` | 8.3 | 0.29% | 0.89% |
| `HasPrerequisite` | 7.3 | 0.26% | 0.78% |
| `DistinctFrom` | 8.7 | 0.30% | 0.57% |
| `HasSubevent` | 9.0 | 0.31% | 0.53% |
| `Causes` | 11.0 | 0.38% | 0.52% |
| `MadeOf` | 3.0 | 0.10% | 0.36% |
| `NotHasProperty` | 0.3 | 0.01% | 0.32% |
| `MotivatedByGoal` | 5.0 | 0.17% | 0.31% |
| `ReceivesAction` | 7.3 | 0.26% | 0.29% |
| `Desires` | 2.0 | 0.07% | 0.28% |
| `CausesDesire` | 0.3 | 0.01% | 0.18% |
| `HasFirstSubevent` | 2.3 | 0.08% | 0.13% |
| `NotDesires` | 4.0 | 0.14% | 0.08% |
| `substance_holonym` | 0.7 | 0.02% | 0.07% |
| `substance_meronym` | 0.3 | 0.01% | 0.07% |
| `CreatedBy` | 1.0 | 0.03% | 0.03% |
| `DefinedAs` | 1.0 | 0.03% | 0.01% |
| `LocatedNear` | 1.0 | 0.03% | 0.01% |
| `NotCapableOf` | 1.0 | 0.03% | 0.01% |

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
| `SimilarTo` | 46.0 | 1.60% | 1.57% |
| `CapableOf` | 48.7 | 1.69% | 1.56% |
| `part_meronym` | 38.0 | 1.32% | 1.31% |
| `part_holonym` | 36.0 | 1.25% | 1.27% |
| `member_holonym` | 27.7 | 0.96% | 0.99% |
| `instance_hypernym` | 26.0 | 0.91% | 0.93% |
| `member_meronym` | 27.0 | 0.94% | 0.93% |
| `instance_hyponym` | 25.0 | 0.87% | 0.89% |
| `HasPrerequisite` | 22.0 | 0.77% | 0.78% |
| `DistinctFrom` | 17.0 | 0.59% | 0.57% |
| `HasSubevent` | 14.0 | 0.49% | 0.53% |
| `Causes` | 18.0 | 0.63% | 0.52% |
| `MadeOf` | 9.0 | 0.31% | 0.36% |
| `NotHasProperty` | 8.0 | 0.28% | 0.32% |
| `MotivatedByGoal` | 8.0 | 0.28% | 0.31% |
| `ReceivesAction` | 8.0 | 0.28% | 0.29% |
| `Desires` | 7.0 | 0.24% | 0.28% |
| `CausesDesire` | 4.0 | 0.14% | 0.18% |
| `HasFirstSubevent` | 3.0 | 0.10% | 0.13% |
| `HasLastSubevent` | 2.0 | 0.07% | 0.10% |
| `NotDesires` | 3.7 | 0.13% | 0.08% |
| `substance_holonym` | 2.0 | 0.07% | 0.07% |
| `substance_meronym` | 2.0 | 0.07% | 0.07% |
| `CreatedBy` | 1.3 | 0.05% | 0.03% |
| `DefinedAs` | 0.7 | 0.02% | 0.01% |
| `NotCapableOf` | 1.0 | 0.03% | 0.01% |

### `medium_60_full`

| Relation | Mean edges | Percent | Full graph percent |
| --- | ---: | ---: | ---: |
| `RelatedTo` | 1296.0 | 30.08% | 18.10% |
| `HasContext` | 1192.0 | 27.66% | 16.61% |
| `hyponym` | 1028.0 | 23.86% | 14.40% |
| `AtLocation` | 219.7 | 5.10% | 13.15% |
| `hypernym` | 190.0 | 4.41% | 12.07% |
| `HasQuality` | 41.0 | 0.95% | 4.07% |
| `UsedFor` | 37.7 | 0.87% | 3.48% |
| `HasProperty` | 26.0 | 0.60% | 2.12% |
| `Antonym` | 33.7 | 0.78% | 1.89% |
| `SimilarTo` | 39.0 | 0.91% | 1.57% |
| `CapableOf` | 39.7 | 0.92% | 1.56% |
| `part_meronym` | 10.3 | 0.24% | 1.31% |
| `part_holonym` | 16.0 | 0.37% | 1.27% |
| `member_holonym` | 20.3 | 0.47% | 0.99% |
| `instance_hypernym` | 24.0 | 0.56% | 0.93% |
| `member_meronym` | 22.0 | 0.51% | 0.93% |
| `instance_hyponym` | 8.3 | 0.19% | 0.89% |
| `HasPrerequisite` | 7.3 | 0.17% | 0.78% |
| `DistinctFrom` | 8.7 | 0.20% | 0.57% |
| `HasSubevent` | 9.0 | 0.21% | 0.53% |
| `Causes` | 11.0 | 0.26% | 0.52% |
| `MadeOf` | 3.0 | 0.07% | 0.36% |
| `NotHasProperty` | 0.3 | 0.01% | 0.32% |
| `MotivatedByGoal` | 5.0 | 0.12% | 0.31% |
| `ReceivesAction` | 7.3 | 0.17% | 0.29% |
| `Desires` | 2.0 | 0.05% | 0.28% |
| `CausesDesire` | 0.3 | 0.01% | 0.18% |
| `HasFirstSubevent` | 2.3 | 0.05% | 0.13% |
| `NotDesires` | 4.0 | 0.09% | 0.08% |
| `substance_holonym` | 0.7 | 0.02% | 0.07% |
| `substance_meronym` | 0.3 | 0.01% | 0.07% |
| `CreatedBy` | 1.0 | 0.02% | 0.03% |
| `DefinedAs` | 1.0 | 0.02% | 0.01% |
| `LocatedNear` | 1.0 | 0.02% | 0.01% |
| `NotCapableOf` | 1.0 | 0.02% | 0.01% |

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
| `HasSubevent` | 23.0 | 0.53% | 0.53% |
| `Causes` | 24.0 | 0.56% | 0.52% |
| `MadeOf` | 16.0 | 0.37% | 0.36% |
| `NotHasProperty` | 14.0 | 0.32% | 0.32% |
| `MotivatedByGoal` | 13.0 | 0.30% | 0.31% |
| `ReceivesAction` | 13.0 | 0.30% | 0.29% |
| `Desires` | 11.0 | 0.26% | 0.28% |
| `CausesDesire` | 6.0 | 0.14% | 0.18% |
| `HasFirstSubevent` | 4.0 | 0.09% | 0.13% |
| `HasLastSubevent` | 3.0 | 0.07% | 0.10% |
| `NotDesires` | 3.3 | 0.08% | 0.08% |
| `substance_holonym` | 3.0 | 0.07% | 0.07% |
| `substance_meronym` | 3.0 | 0.07% | 0.07% |
| `CreatedBy` | 1.0 | 0.02% | 0.03% |
| `DefinedAs` | 0.3 | 0.01% | 0.01% |
| `LocatedNear` | 1.0 | 0.02% | 0.01% |
| `NotCapableOf` | 1.0 | 0.02% | 0.01% |

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
| `HasProperty` | 26.0 | 0.45% | 2.12% |
| `Antonym` | 33.7 | 0.59% | 1.89% |
| `SimilarTo` | 39.0 | 0.68% | 1.57% |
| `CapableOf` | 39.7 | 0.69% | 1.56% |
| `part_meronym` | 10.3 | 0.18% | 1.31% |
| `part_holonym` | 16.0 | 0.28% | 1.27% |
| `member_holonym` | 20.3 | 0.35% | 0.99% |
| `instance_hypernym` | 24.0 | 0.42% | 0.93% |
| `member_meronym` | 22.0 | 0.38% | 0.93% |
| `instance_hyponym` | 8.3 | 0.15% | 0.89% |
| `HasPrerequisite` | 7.3 | 0.13% | 0.78% |
| `DistinctFrom` | 8.7 | 0.15% | 0.57% |
| `HasSubevent` | 9.0 | 0.16% | 0.53% |
| `Causes` | 11.0 | 0.19% | 0.52% |
| `MadeOf` | 3.0 | 0.05% | 0.36% |
| `NotHasProperty` | 0.3 | 0.01% | 0.32% |
| `MotivatedByGoal` | 5.0 | 0.09% | 0.31% |
| `ReceivesAction` | 7.3 | 0.13% | 0.29% |
| `Desires` | 2.0 | 0.03% | 0.28% |
| `CausesDesire` | 0.3 | 0.01% | 0.18% |
| `HasFirstSubevent` | 2.3 | 0.04% | 0.13% |
| `NotDesires` | 4.0 | 0.07% | 0.08% |
| `substance_holonym` | 0.7 | 0.01% | 0.07% |
| `substance_meronym` | 0.3 | 0.01% | 0.07% |
| `CreatedBy` | 1.0 | 0.02% | 0.03% |
| `DefinedAs` | 1.0 | 0.02% | 0.01% |
| `LocatedNear` | 1.0 | 0.02% | 0.01% |
| `NotCapableOf` | 1.0 | 0.02% | 0.01% |

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
| `HasSubevent` | 30.0 | 0.52% | 0.53% |
| `Causes` | 30.0 | 0.52% | 0.52% |
| `MadeOf` | 21.0 | 0.37% | 0.36% |
| `NotHasProperty` | 18.0 | 0.31% | 0.32% |
| `MotivatedByGoal` | 18.0 | 0.31% | 0.31% |
| `ReceivesAction` | 16.0 | 0.28% | 0.29% |
| `Desires` | 16.0 | 0.28% | 0.28% |
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


### Generated Graph-Density Corpora

| Condition | Seeds | Sentence count by seed | Mean sentences | Mean CJK vocab | Mean Hiragana vocab | Missing token counts |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `low_40` | 42,43,44 | 19268,19266,19315 | 19283.0 | 2196.7 | 2196.7 | 0/0, 0/0, 0/0 |
| `medium_60_full` | 42,43,44 | 30503,30543,30535 | 30527.0 | 2215.3 | 2215.3 | 0/0, 0/0, 0/0 |
| `high_80_full` | 42,43,44 | 41361,41105,41152 | 41206.0 | 2228.0 | 2228.0 | 0/0, 0/0, 0/0 |
| `low_40_relation` | 42,43,44 | 17790,17873,17871 | 17844.7 | 2234.3 | 2234.3 | 0/0, 0/0, 0/0 |
| `medium_60_relation` | 42,43,44 | 28503,28386,28409 | 28432.7 | 2245.0 | 2245.0 | 0/0, 0/0, 0/0 |
| `high_80_relation` | 42,43,44 | 39246,39081,39024 | 39117.0 | 2250.3 | 2250.3 | 0/0, 0/0, 0/0 |
