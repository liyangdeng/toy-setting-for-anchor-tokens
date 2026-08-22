# Anchor Necessity Experiment

## Research Question

Cross-lingual transfer in this project usually relies on anchor tokens: a few
surface forms shared between the two languages that give the model a place to
tie their representations together. This experiment removes that crutch and asks
whether transfer can still happen without it.

> With no shared tokens at all, can two languages still align through structural
> (word-order) similarity alone, or is a shared anchor actually necessary?

To isolate structure, the two languages share **zero** tokens, and we vary only
how similar their word order is.

## The Two Languages

Both languages are built from the same semantic triples; only the surface script
and the word order differ.

| Language | Role | Script | Word order | Held-out facts |
| --- | --- | --- | --- | --- |
| A | source | CJK | fixed at `111` | none — trains on every fact |
| B | target | Hiragana | one of four settings (below) | held out from B, then probed on B |

A always sees the full graph. A fact is removed from B's training data, so if the
model can still fill it in on the B side, that knowledge can only have come from
A — i.e. it transferred across the two languages.

Because the CJK and Hiragana token ranges are disjoint and the sentence generator
strips punctuation, A and B never share a single token. This is what makes every
condition a **zero-anchor** setting.

## Word-Order Settings

Word order is set by three binary switches:

- `s1` — subject / predicate order
- `s2` — object / verb order
- `s3` — noun / adjective order

Language A is fixed at `111`. Language B is run at four settings, each one step
further from A. "Structural similarity" is just how many of the three switches
match A.

| B setting | `s1` | `s2` | `s3` | Switches matching A | Structural similarity to A |
| --- | --- | --- | --- | :---: | :---: |
| `000` | NP VP | OV | A N | 0 / 3 | 0% |
| `001` | NP VP | OV | N A | 1 / 3 | 33% |
| `011` | NP VP | VO | N A | 2 / 3 | 67% |
| `111` | VP NP | VO | N A | 3 / 3 | 100% |

At `111` the two languages have identical word order; at `000` they share nothing
at all — neither tokens nor structure. Sweeping across the four settings shows how
much transfer survives as the only remaining similarity is taken away.

## Corpus Generation

Two builders live in this folder:

- `corpus/build_different_parallel_corpus.py` — the full parallel corpus
  (nothing held out), used for word translation and sentence retrieval.
- `probing/build_probing_corpus_necessity.py` — the held-out corpus for the
  probing metrics, run in stages:

  ```bash
  python build_probing_corpus_necessity.py --stage select
  python build_probing_corpus_necessity.py --stage fastgen
  python build_probing_corpus_necessity.py --stage mono   --cfg 000
  python build_probing_corpus_necessity.py --stage filter --cfg 000
  python build_probing_corpus_necessity.py --stage assemble --cfg 000
  ```

  `select` and `fastgen` run once. `mono`, `filter`, and `assemble` run once per
  B setting. `mono` trains a Hiragana-only model and the `filter` step drops any
  held-out fact that model can already guess on its own, so what remains is a
  clean test of transfer.

`probing/run_overnight.sh` runs the per-setting stages, trains the bilingual
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
