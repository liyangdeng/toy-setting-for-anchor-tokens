# Anchor Necessity Experiment

## Research Question

Cross-lingual transfer in this project usually depends on anchor tokens: a
small number of surface forms that both languages share. This experiment removes
them and tests whether transfer still happens when no tokens are shared at all.

> With no shared tokens at all, can two languages still align through structural
> (word-order) similarity alone, or is a shared anchor actually necessary?

To isolate structure, the two languages share **zero** tokens, and we only vary
how similar their word order is.

## The Two Languages

Both languages are built from the same semantic triples. Only the surface script
and the word order differ.

| Language | Role | Script | Word order | Held-out facts |
| --- | --- | --- | --- | --- |
| A | source | CJK | fixed at `111` | none (trains on every fact) |
| B | target | Hiragana | one of four settings (below) | held out from B, then probed on B |

A always sees the full graph. A fact is removed from B's training data, so if the
model can still fill it in on the B side, that knowledge could only have come
from A. That is the transfer we are testing for.

The CJK and Hiragana token ranges are disjoint, and the sentence generator strips
punctuation, so A and B never share a single token. That is what makes every
condition a **zero-anchor** setting.

## Word-Order Settings

Word order is set by three binary switches:

- `s1` sets the subject / predicate order
- `s2` sets the object / verb order
- `s3` sets the noun / adjective order

Language A is fixed at `111`. Language B is run at four settings, each one step
further from A. "Structural similarity" is just how many of the three switches
still match A.

| B setting | `s1` | `s2` | `s3` | Switches matching A | Structural similarity to A |
| --- | --- | --- | --- | :---: | :---: |
| `000` | NP VP | OV | A N | 0 / 3 | 0% |
| `001` | NP VP | OV | N A | 1 / 3 | 33% |
| `011` | NP VP | VO | N A | 2 / 3 | 67% |
| `111` | VP NP | VO | N A | 3 / 3 | 100% |

At `111` the two languages have the exact same word order. At `000` they share
nothing at all, not the tokens and not the order. Running all four settings
shows how much transfer remains as the structural similarity is removed.

## Corpus Generation

Two builders drive this:

- `corpus/build_different_parallel_corpus.py` builds the full parallel corpus
  (nothing held out), used for word translation and sentence retrieval.
- `results/build_probing_corpus_necessity.py` builds the held-out corpus for the
  probing metrics. It runs in stages:

  ```bash
  python build_probing_corpus_necessity.py --stage select
  python build_probing_corpus_necessity.py --stage fastgen
  python build_probing_corpus_necessity.py --stage mono   --cfg 000
  python build_probing_corpus_necessity.py --stage filter --cfg 000
  python build_probing_corpus_necessity.py --stage assemble --cfg 000
  ```

  `select` and `fastgen` run once. `mono`, `filter` and `assemble` run once per
  B setting. `mono` trains a Hiragana-only model, and the `filter` step throws
  out any held-out fact that model can already guess on its own, so what stays is
  a clean test of transfer.

`results/run_pipeline.sh` runs the per-setting stages, trains the bilingual
models, and calls the probe and LM-head evaluations for all four settings.

## Evaluation

The four settings are compared with the shared evaluation scripts in
`Experiment/evaluation/`:

1. Word translation precision
2. Sentence retrieval precision
3. Linear probing accuracy per layer (mask-entity)
4. LM-head accuracy (strict top-1)

## Results

Result tables and plots for all four settings are in [results](results):
`word_trans_sent_retrieval.csv`, `lm_head_accuracy.csv`, the per-setting probe
curves in `mask_entity_csv/`, and `significance.csv` for the significance tests.
