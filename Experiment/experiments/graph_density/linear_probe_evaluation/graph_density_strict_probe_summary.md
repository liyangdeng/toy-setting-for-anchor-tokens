# Graph Density Strict Entity Probe Summary

Strict transfer setup: probe triples are retained in CJK/A training and removed from Hiragana/B training. Accuracy is triple-level OR-aggregated over B renderings.

## Condition Summary

| condition | strategy | density | seeds | usable probes | best acc | final acc | embedding acc | mean best layer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low_40 | coverage | 40% | 3 | 88.7 | 0.2973 +/- 0.0323 | 0.2728 +/- 0.0167 | 0.0423 +/- 0.0109 | 2.3 |
| low_40_relation | relation | 40% | 3 | 168.7 | 0.2804 +/- 0.1774 | 0.2429 +/- 0.1484 | 0.0237 +/- 0.0061 | 2.7 |
| medium_60_full | coverage | 60% | 3 | 76.0 | 0.3444 +/- 0.1368 | 0.3191 +/- 0.1457 | 0.0463 +/- 0.0213 | 2.3 |
| medium_60_relation | relation | 60% | 3 | 244.0 | 0.5095 +/- 0.0689 | 0.4647 +/- 0.0122 | 0.0273 +/- 0.0042 | 3.0 |
| high_80_full | coverage | 80% | 3 | 63.0 | 0.6606 +/- 0.1359 | 0.6228 +/- 0.1248 | 0.0704 +/- 0.0242 | 2.3 |
| high_80_relation | relation | 80% | 3 | 210.0 | 0.6958 +/- 0.0659 | 0.6592 +/- 0.0593 | 0.0191 +/- 0.0083 | 2.3 |
| full_graph | coverage | 100% | 3 | 169.7 | 0.8091 +/- 0.0487 | 0.7920 +/- 0.0286 | 0.0337 +/- 0.0051 | 3.0 |

## Per-Run Summary

| condition | seed | usable probes | best layer | best acc | final acc | embedding acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low_40 | 42 | 106 | 1 | 0.2642 | 0.2547 | 0.0377 |
| low_40 | 43 | 87 | 3 | 0.2989 | 0.2759 | 0.0345 |
| low_40 | 44 | 73 | 3 | 0.3288 | 0.2877 | 0.0548 |
| low_40_relation | 42 | 166 | 3 | 0.1807 | 0.1566 | 0.0241 |
| low_40_relation | 43 | 171 | 2 | 0.1754 | 0.1579 | 0.0175 |
| low_40_relation | 44 | 169 | 3 | 0.4852 | 0.4142 | 0.0296 |
| medium_60_full | 42 | 90 | 1 | 0.2111 | 0.1778 | 0.0222 |
| medium_60_full | 43 | 64 | 3 | 0.4844 | 0.4688 | 0.0625 |
| medium_60_full | 44 | 74 | 3 | 0.3378 | 0.3108 | 0.0541 |
| medium_60_relation | 42 | 249 | 3 | 0.4659 | 0.4578 | 0.0321 |
| medium_60_relation | 43 | 247 | 3 | 0.4737 | 0.4575 | 0.0243 |
| medium_60_relation | 44 | 236 | 3 | 0.5890 | 0.4788 | 0.0254 |
| high_80_full | 42 | 94 | 3 | 0.5106 | 0.4787 | 0.0426 |
| high_80_full | 43 | 49 | 2 | 0.7755 | 0.6939 | 0.0816 |
| high_80_full | 44 | 46 | 2 | 0.6957 | 0.6957 | 0.0870 |
| high_80_relation | 42 | 208 | 3 | 0.7356 | 0.7019 | 0.0144 |
| high_80_relation | 43 | 209 | 2 | 0.7321 | 0.6842 | 0.0287 |
| high_80_relation | 44 | 213 | 2 | 0.6197 | 0.5915 | 0.0141 |
| full_graph | 42 | 152 | 3 | 0.8618 | 0.8224 | 0.0395 |
| full_graph | 43 | 165 | 2 | 0.8000 | 0.7879 | 0.0303 |
| full_graph | 44 | 192 | 4 | 0.7656 | 0.7656 | 0.0312 |

## Strongest Relations

Top relations by mean accuracy at each run's best overall layer, requiring all three seeds and mean n >= 5.

### low_40

| relation | acc | n |
| --- | ---: | ---: |
| hyponym | 0.4500 +/- 0.1000 | 20.0 |
| AtLocation | 0.3000 +/- 0.1000 | 20.0 |
| RelatedTo | 0.2616 +/- 0.2483 | 15.3 |
| HasContext | 0.2544 +/- 0.0506 | 19.7 |

### low_40_relation

| relation | acc | n |
| --- | ---: | ---: |
| HasProperty | 0.4279 +/- 0.1873 | 18.7 |
| hyponym | 0.4219 +/- 0.1552 | 19.7 |
| UsedFor | 0.3669 +/- 0.3687 | 19.0 |
| HasContext | 0.3000 +/- 0.1323 | 20.0 |
| RelatedTo | 0.2500 +/- 0.0833 | 17.3 |
| AtLocation | 0.2333 +/- 0.2754 | 20.0 |
| hypernym | 0.2097 +/- 0.1553 | 15.7 |
| Antonym | 0.1611 +/- 0.1512 | 19.3 |

### medium_60_full

| relation | acc | n |
| --- | ---: | ---: |
| AtLocation | 0.3939 +/- 0.1714 | 19.3 |
| HasContext | 0.3513 +/- 0.1114 | 18.3 |
| RelatedTo | 0.3318 +/- 0.2323 | 15.0 |
| hyponym | 0.3293 +/- 0.1260 | 17.0 |

### medium_60_relation

| relation | acc | n |
| --- | ---: | ---: |
| HasProperty | 0.6652 +/- 0.0896 | 14.7 |
| AtLocation | 0.6342 +/- 0.1941 | 19.3 |
| UsedFor | 0.6166 +/- 0.1797 | 17.3 |
| CapableOf | 0.5994 +/- 0.0879 | 19.0 |
| hyponym | 0.5912 +/- 0.0255 | 18.0 |
| RelatedTo | 0.5699 +/- 0.1152 | 16.3 |
| instance_hypernym | 0.5228 +/- 0.1661 | 14.0 |
| hypernym | 0.5056 +/- 0.1686 | 10.7 |

### high_80_full

| relation | acc | n |
| --- | ---: | ---: |
| RelatedTo | 0.8399 +/- 0.1322 | 13.3 |
| AtLocation | 0.6384 +/- 0.1694 | 18.3 |
| HasContext | 0.6223 +/- 0.0786 | 18.3 |

### high_80_relation

| relation | acc | n |
| --- | ---: | ---: |
| HasProperty | 0.8308 +/- 0.1422 | 12.7 |
| CapableOf | 0.8298 +/- 0.0349 | 19.7 |
| part_meronym | 0.8120 +/- 0.0118 | 16.0 |
| UsedFor | 0.7222 +/- 0.1470 | 18.0 |
| HasContext | 0.7186 +/- 0.0758 | 19.0 |
| Antonym | 0.6941 +/- 0.1262 | 15.7 |
| AtLocation | 0.6842 +/- 0.1579 | 19.0 |
| HasPrerequisite | 0.6586 +/- 0.0304 | 17.7 |

### full_graph

| relation | acc | n |
| --- | ---: | ---: |
| part_meronym | 0.9333 +/- 0.0577 | 10.3 |
| UsedFor | 0.8918 +/- 0.0461 | 18.0 |
| HasProperty | 0.8857 +/- 0.1030 | 12.0 |
| AtLocation | 0.8372 +/- 0.0930 | 18.3 |
| HasPrerequisite | 0.8218 +/- 0.0772 | 15.0 |
| HasContext | 0.7971 +/- 0.1147 | 18.0 |
| RelatedTo | 0.7740 +/- 0.1322 | 11.7 |
| Antonym | 0.7691 +/- 0.1174 | 16.3 |
