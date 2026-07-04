# Masked Token Probing

Tests cross-lingual transfer directly: hold a fact out of one language's
training data entirely, then ask a bilingual model to fill in the blank for
that fact in the language that never saw it. If it gets it right, the fact
must have transferred from the other language.

Two independent tracks, sharing the same pipeline:

- **mask-ENTITY** — `a r [MASK]` → predict the target entity `b`
- **mask-RELATION** — `a [MASK] b` → predict the relation-phrase tokens
  between (not necessarily "between" in the sentence — see below) the two
  entities

## Directory layout

```
masked_language_probing/
  build_probing_corpus/
    select_probe_triples.py    # stage 0: pick which triples to hold out
    build_probing_corpus.py    # stage 1: orchestrator (see below)
    probe_manifest.json        # stage-0 output, kept in the repo
    edges_probing.json         # input KG edges for this experiment
../probing/
    linear_probe.py            # stage 3: the actual probe (separate dir, not moved on purpose)
```

`linear_probe.py` lives one level up, in `Experiment/evaluation/probing/`,
because it's shared machinery (it can also just train a bilingual model from
scratch with `--do_train`). Everything else here is specific to this
experiment.

## Pipeline

### Stage 0 — pick probe triples

```bash
cd build_probing_corpus
python select_probe_triples.py \
    --input  ../../../../data/generate_sentences/v3_generated_sentences_adj.json \
    --n_per_relation 50 --min_total 100 --min_degree 2 \
    --direction B \
    --out probe_manifest.json \
    --out_omitted omitted_triples.json \
    --out_deprived deprived_triples.json
```

Produces, per relation, a fixed number of held-out triples for each track
(entity + relation), subject to the filter rules described in the script's
docstring (unique-target / unique-relation + minimum degree, so the held-out
fact stays "grounded" elsewhere in the graph). `probe_manifest.json` is the
only file that keeps each triple's **track** tag — `omitted_triples.json` /
`deprived_triples.json` are deduped across both tracks into plain edge-lists,
so anything downstream that needs to tell entity-probes from relation-probes
apart has to go back to the manifest (`build_probing_corpus.py` does this via
`load_manifest_tracks`).

### Stage 1 — build the training corpora (`build_probing_corpus.py`)

This is the orchestrator. Re-run it from scratch for every model you train —
the sentence generator is a PCFG, so every run produces different concrete
sentences, and it trains a fresh mono-B model internally as part of the
filtering step.

```bash
python build_probing_corpus.py \
    --deprived_triples deprived_triples.json \
    --omitted_triples  omitted_triples.json \
    --probe_manifest   probe_manifest.json \
    --gen_script          .../data/generate_sentences/v3_generate_sentences.py \
    --build_corpus_script .../data/corpus/build_synset_corpus.py \
    --mono_train_script   .../train_monolingual_synset.py \
    --grammar   .../data/grammar/grammar_templates_adj.py \
    --cjk_dict  .../synset_pos_artificial_cjk_edges_adj_augmented.json \
    --hira_dict .../synset_pos_artificial_hiragana_edges_adj_augmented.json \
    --out_dir ./probing_run_1
```

`--gen_script` and `--build_corpus_script` are required with no default —
pick the exact versions matching your current grammar/switches.
`--build_corpus_script` specifically **must** be a punctuation-aware version
(`build_synset_corpus.py`), not the older `build_corpus.py`, or sentences get
silently corrupted.

What it does, in order:

1. Generate sentences for the deprived (KG minus held-out triples) and
   omitted (just the held-out triples) triple sets.
2. Build CJK/Hiragana corpora for both.
3. Train a **mono-B** model (Hiragana-only) on the deprived-B corpus.
4. **Leakage filter** (same direction for both tracks): for every held-out
   triple, mask the relevant span in its B sentence(s) and ask mono-B to fill
   it in. If mono-B alone gets it right in *any* of that triple's sentences,
   drop the triple — it doesn't need cross-lingual transfer to be guessed, so
   it isn't a clean probe.
     - mask-ENTITY: mask the single target-entity token.
     - mask-RELATION: mask the whole relation-phrase span, located by
       `strip_entities_and_match` — see "How mask-RELATION works" below.
5. **Per-relation floor**: if a relation has too few survivors after the
   leakage filter (`--min_survivors`), drop the whole relation.
6. Write `final_omitted.json` — the surviving probes, each tagged
   `"track": "entity"` or `"track": "relation"`.
7. Regenerate sentences for `final_omitted.json` **fresh** (new PCFG draw,
   not reusing step 1's sentences) and build its corpus — this is the
   parallel file `linear_probe.py` needs.
8. Assemble the treatment model's training corpora:
   - `a_training.txt` = deprived-A + final_omitted-A (Language A sees
     everything)
   - `b_training.txt` = deprived-B only (Language B never sees the held-out
     facts — that's the whole point)

### Stage 2 — train the bilingual model(s)

Not part of this script. Either train separately with
`train_multilingual_synset.py` on `a_training.txt`/`b_training.txt`, or let
`linear_probe.py --do_train` do it inline (see below). If you also want a
"no-anchor floor" comparison model, train it the same way on a version of the
corpora with no shared/anchor tokens.

### Stage 3 — probe (`linear_probe.py`)

```bash
python ../probing/linear_probe.py \
    --track entity \            # or: relation
    --model_dir ./checkpoints_treatment/final \    # or --do_train, see script docstring
    --final_omitted probing_run_1/final_omitted.json \
    --parallel probing_run_1/final_omitted_corpus/parallel_corpus_synset.json \
    --cjk_dict  .../synset_pos_artificial_cjk_edges_adj_augmented.json \
    --hira_dict .../synset_pos_artificial_hiragana_edges_adj_augmented.json \
    --gen_script .../v3_generate_sentences.py --grammar .../grammar_templates_adj.py \  # --track relation only
    --out_dir ./probe_results_treatment
```

`--gen_script`/`--grammar` are only needed for `--track relation` (to rebuild
the candidate template sets — see below). `--final_omitted` is filtered
internally by `--track`, so the same `final_omitted.json` (with both tracks
mixed in) works for both runs.

For every layer, this fits a logistic-regression classifier on Language-A
`[MASK]`-position representations and evaluates it on Language-B's — if a
classifier trained only on A can still read the fact off B's representation,
that's evidence of transfer. Outputs `layerwise_accuracy.csv`,
`layerwise_accuracy_per_relation.csv`, and a PNG plot per run.

## How mask-RELATION works

The relation-phrase in this grammar always renders as one of two fixed
shapes: a single verb (`relates_to`, TRANS branch, 1 token) or an
AUX+ADJP+PP triple (`is related to`, COP branch, 3 tokens). `build_probing_corpus.build_relation_templates` parses `grammar_templates_adj.py`
directly (not sampled) to enumerate every possible realization per relation,
bucketed by token length — this is the exact, complete candidate/"gold" set.

`strip_entities_and_match` then locates the two entity spans in a rendered
sentence (each possibly with one adjacent adjective, recognized as "a
neighboring token that isn't in the grammar's own vocabulary"), strips them
plus the trailing period, and checks whether what's left matches one of the
enumerated candidates **exactly**. This works regardless of word order
(switches can put the relation phrase before, between, or after the two
entities) and naturally rejects complex/coordinated sentences (their leftover
tokens either aren't contiguous or don't exactly match any candidate) —
`linear_probe.py` and the leakage filter both reuse this one function so the
definition of "a valid simple-sentence rendering" never drifts between them.

Because the label for mask-RELATION is "which specific template did this
rendering draw" (the PCFG redraws the VP independently per sentence, so two
renderings of the same triple can realize different templates), it's a
**per-rendering** label, not a per-triple constant like mask-ENTITY's concept
id. `linear_probe.py` handles this with a single global template-id space
across all relations (mirroring entity track's `concept_to_id`), and reports
per-relation numbers as a post-hoc groupby, not a separately-trained
classifier.

## A few things to keep in sync

- **Accuracy unit is the triple, not the sentence.** A triple can have
  several candidate sentence renderings; a triple counts as "hit" (leakage
  filter) or "correctly transferred" (linear probe) if **any** of its
  renderings hits. Final accuracy = hit triples / total triples. Training the
  linear-probe classifier likewise uses every triple's every A-rendering as a
  separate row under the same label.
- **Hidden-state extraction mean-pools over however many `[MASK]` positions
  are present.** mask-ENTITY always has exactly one, so this is a no-op
  there; mask-RELATION can have 1 or 3, pooled into the same fixed-size
  vector so both tracks (and both branch lengths) share one classifier per
  layer.
- **`--gen_script`/`--build_corpus_script`/`--grammar` must all be the exact
  versions used to build the corpus you're probing.** The PCFG output (and
  therefore what "deprived"/"omitted" concretely render to, and which
  candidate templates are valid) depends on the exact grammar file version.
- If you regenerate sentences after a grammar-file change (e.g. the
  `VP -> TRANS [0.4]` escaped-bracket typo fixed on 2026-07-04, which had
  silently dropped the single-verb TRANS realization for 30/35 relations),
  existing checkpoints/corpora built on the old grammar are no longer the
  same distribution — regenerate them too if you want a fair comparison.
