# Lexical Overlap Experiment

---
## Research Question
---

## Experimental Conditions

The pipeline automatically loops through all combinations of the following parameters to produce independent dataset variants:

| Parameter | Evaluated Values |
| :--- | :--- |
| **Target Overlap ($\%$)** | $2.5\%$, $5.0\%$, $7.5\%$, $10.0\%$ |
| **Frequency Strategy** | `high`, `mid`, `low` |
| **Target Languages** | `cjk`, `hiragana` |

---
## Corpus Generation
---
## Training
---
## Evaluation

### Word Translation and Sentence Retrieval Precision
| Condition | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |
| --- | ---: | ---: | ---: | ---: |
| `high_P0` | 0.7645 | 0.8945 | 0.8347 | 0.9613 |
| `high_P2` | 0.7630 | 0.8985 | 0.8247 | 0.9447 |
| `high_P5` | 0.7880 | 0.9135 | 0.8200 | 0.9527 |
| `high_P7` | 0.7955 | 0.9265 | 0.8220 | 0.9527 |
| `high_P10` | 0.7840 | 0.9170 | 0.8460 | 0.9620 |
| `mid_P0` | 0.7645 | 0.8945 | 0.8347 | 0.9613 |
| `mid_P2` | 0.8240 | 0.9395 | 0.8047 | 0.9380 |
| `mid_P5` | 0.7905 | 0.9220 | 0.0041 | 0.9620 |
| `mid_P7` | 0.8195 | 0.9305 | 0.8847 | 0.9613 |
| `mid_P10` | 0.8065 | 0.9255 | 0.8307 | 0.9567 |
| `low_P0` | 0.7645 | 0.8945 | 0.8347 | 0.9613 |
| `low_P2` | 0.8381 | 0.9489 | 0.8880 | 0.9733 |
| `low_P5` | 0.8288 | 0.9411 | 0.0146 | 0.9467 |
| `low_P7` | 0.8433 | 0.9357 | 0.9300 | 0.9893 |
| `low_P10` | 0.8770 | 0.9538 | 0.9047 | 0.9807 |

### Linear Probing Accuracy
| Condition | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Layer 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `high_P0` | 0.0324 | 0.2842 | 0.4460 | 0.4496 | 0.3813 |
| `high_P2` | 0.0324 | 0.4460 | 0.4281 | 0.4856 | 0.5000 |
| `high_P5` | 0.0324 | 0.3489 | 0.4173 | 0.4496 | 0.4317 |
| `high_P7` | 0.0324 | 0.4281 | 0.5432 | 0.5180 | 0.5396 |
| `high_P10` | 0.0324 | 0.3165 | 0.4640 | 0.4496 | 0.4245 |
| `mid_P0` | 0.0324 | 0.2842 | 0.4460 | 0.4496 | 0.3813 | 
| `mid_P2` | 0.0324 | 0.4281 | 0.4173 | 0.5252 | 0.4532 |
| `mid_P5` | 0.0324 | 0.3453 | 0.4209 | 0.4065 | 0.3597 |
| `mid_P7` | 0.0324 | 0.4532 | 0.5576 | 0.5288 | 0.5252 |
| `mid_P10` | 0.0324 | 0.4892 | 0.5144 | 0.4748 | 0.4820 |
| `low_P0` | 0.0324 | 0.2842 | 0.4460 | 0.4496 | 0.3813 |
| `low_P2` | 0.0324 | 0.4964 | 0.5216 | 0.4820 | 0.4820 |
| `low_P5` | 0.0324 | 0.4964 | 0.6259 | 0.5755 | 0.5432 |
| `low_P7` | 0.0324 | 0.5036 | 0.4784 | 0.5108 | 0.5432 |
| `low_P10` | 0.0324 | 0.5144 | 0.6547 | 0.6115 | 0.6007 |

### LM Head strict_token Accuracy
| Condition | Top-1 | Top-3 |
| --- | ---: | ---: |
| `high_P0` | 0.0360 | 0.0863 |
| `high_P2` | 0.0180 | 0.0827 |
| `high_P5` | 0.0252 | 0.0576 |
| `high_P7` | 0.0468 | 0.0827 |
| `high_P10` | 0.0468 | 0.1043 |
| `mid_P0` |  0.0360 | 0.0863 | 
| `mid_P2` | 0.0432 | 0.0827 |
| `mid_P5` | 0.0432 | 0.0935 |
| `mid_P7` | 0.0288 | 0.0827 |
| `mid_P10` | 0.0216 | 0.0827 |
| `low_P0` | 0.0360 | 0.0863 |
| `low_P2` | 0.0576 | 0.0863 |
| `low_P5` | 0.0252 | 0.0540 |
| `low_P7` | 0.0324 | 0.0719 |
| `low_P10` | 0.0288 | 0.0647 |

---

## Significance Testing
---

## File and Execution Prerequisites

Ensure the following dataset files are located in your script's working directory before execution:

1. **`eng_sentences_with_adj.txt`** — The source English baseline text corpus used to extract token distribution metrics.
2. **`synset_pos_artificial_[lang].json`** — The language-specific dictionaries mapping English source keys to artificial vocabulary structures (containing the `"artificial"` field).
3. **`corpus_[lang]_synset.txt`** — The base artificial token corpora designated for partial structural overwriting.

---
