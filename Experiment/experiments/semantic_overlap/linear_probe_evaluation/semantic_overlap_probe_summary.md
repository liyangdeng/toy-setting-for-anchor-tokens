# Semantic Overlap Masked-Language Probe Summary (entity)

Within-condition diagnostic: each condition uses its own L1-only probe set, so cross-condition trends are suggestive rather than a controlled identical-item comparison.

## Condition Summary

| condition | seeds | usable probes | best acc | final acc | embedding acc | mean best layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overlap_000 | 3 | 2753.0 | 0.0566 +/- 0.0043 | 0.0455 +/- 0.0029 | 0.0143 +/- 0.0018 | 2.7 |
| overlap_025 | 3 | 2064.0 | 0.0743 +/- 0.0093 | 0.0594 +/- 0.0123 | 0.0155 +/- 0.0015 | 1.7 |
| overlap_050 | 3 | 1377.0 | 0.1150 +/- 0.0197 | 0.1017 +/- 0.0215 | 0.0198 +/- 0.0008 | 1.7 |
| overlap_075 | 3 | 689.0 | 0.2279 +/- 0.0080 | 0.1959 +/- 0.0233 | 0.0227 +/- 0.0030 | 2.3 |

## Per-Run Summary

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

## Strongest Relations

Top relations by mean accuracy at each run's best overall layer, requiring all three seeds.

### overlap_000

| relation | acc | n |
| --- | ---: | ---: |
| NotHasProperty | 0.3333 +/- 0.1111 | 9.0 |
| MadeOf | 0.3000 +/- 0.1732 | 10.0 |
| HasLastSubevent | 0.2222 +/- 0.3849 | 3.0 |
| NotDesires | 0.1667 +/- 0.2887 | 2.0 |
| substance_holonym | 0.1667 +/- 0.2887 | 2.0 |
| CapableOf | 0.1408 +/- 0.0463 | 45.0 |
| DistinctFrom | 0.1250 +/- 0.0625 | 16.0 |
| ReceivesAction | 0.1250 +/- 0.2165 | 8.0 |
| HasProperty | 0.1149 +/- 0.0199 | 58.0 |
| HasPrerequisite | 0.1061 +/- 0.0946 | 22.0 |

### overlap_025

| relation | acc | n |
| --- | ---: | ---: |
| HasFirstSubevent | 0.3333 +/- 0.3334 | 3.0 |
| HasLastSubevent | 0.3333 +/- 0.2887 | 2.0 |
| substance_meronym | 0.3333 +/- 0.5774 | 1.0 |
| MadeOf | 0.2083 +/- 0.1909 | 8.0 |
| HasProperty | 0.1926 +/- 0.0559 | 45.0 |
| CapableOf | 0.1919 +/- 0.0350 | 33.0 |
| NotHasProperty | 0.1905 +/- 0.2182 | 7.0 |
| HasPrerequisite | 0.1875 +/- 0.1654 | 16.0 |
| CausesDesire | 0.1667 +/- 0.1443 | 4.0 |
| UsedFor | 0.1511 +/- 0.0601 | 75.0 |

### overlap_050

| relation | acc | n |
| --- | ---: | ---: |
| HasFirstSubevent | 0.3333 +/- 0.5774 | 2.0 |
| NotDesires | 0.3333 +/- 0.5774 | 1.0 |
| MadeOf | 0.2667 +/- 0.2309 | 5.0 |
| NotHasProperty | 0.2667 +/- 0.1155 | 5.0 |
| instance_hypernym | 0.2381 +/- 0.0412 | 14.0 |
| HasProperty | 0.2111 +/- 0.0770 | 30.0 |
| DistinctFrom | 0.2083 +/- 0.1443 | 8.0 |
| instance_hyponym | 0.2051 +/- 0.1601 | 13.0 |
| part_holonym | 0.1852 +/- 0.1156 | 18.0 |
| part_meronym | 0.1755 +/- 0.1520 | 19.0 |

### overlap_075

| relation | acc | n |
| --- | ---: | ---: |
| NotDesires | 1.0000 +/- 0.0000 | 1.0 |
| HasLastSubevent | 0.6667 +/- 0.5774 | 1.0 |
| DistinctFrom | 0.5833 +/- 0.1443 | 4.0 |
| MadeOf | 0.5555 +/- 0.3849 | 3.0 |
| HasPrerequisite | 0.5333 +/- 0.1155 | 5.0 |
| HasProperty | 0.5111 +/- 0.1018 | 15.0 |
| MotivatedByGoal | 0.5000 +/- 0.5000 | 2.0 |
| ReceivesAction | 0.5000 +/- 0.5000 | 2.0 |
| part_meronym | 0.4667 +/- 0.1528 | 10.0 |
| member_holonym | 0.4286 +/- 0.0000 | 7.0 |
