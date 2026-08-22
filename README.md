# A Toy Setting for the Anchor Token Hypothesis

This repository contains a controlled artificial-language setup for studying the
anchor token hypothesis.

The project builds a synthetic semantic graph, generates artificial-language
sentences from that graph, trains small masked-language models from scratch, and
evaluates monolingual learning, cross-lingual alignment, and transfer.

## Repository Overview

- [`data/`](data/)  
  Data generation and corpus resources.
  - [`data/semantic_backbones/`](data/semantic_backbones/) - WordNet +
    ConceptNet semantic graph.
  - [`data/grammar/`](data/grammar/) - PCFG grammar templates.
  - [`data/generate_sentences/`](data/generate_sentences/) - sentence
    generation scripts and outputs.
  - [`data/corpus/`](data/corpus/) - CJK/Hiragana corpora and parallel corpora.
- [`Experiment/`](Experiment/)  
  Training, evaluation, and controlled experiments.
  - [`Experiment/training/`](Experiment/training/) - monolingual and
    multilingual MLM training.
  - [`Experiment/evaluation/`](Experiment/evaluation/) - alignment, probing,
    accuracy, and significance evaluation.
  - [`Experiment/experiments/`](Experiment/experiments/) - graph density,
    semantic overlap, lexical overlap, and control experiments.
- [`requirements.txt`](requirements.txt) - Python dependencies.
- [`README.md`](README.md) - this overview.

## Main Components

### 1. Semantic Backbone

The semantic graph starts from WordNet noun synsets and is expanded with selected
ConceptNet relations. The default backbone is designed to be connected,
degree-controlled, and easy to sample for controlled experiments.

Current graph resources are in:

- `data/semantic_backbones/edges_adj.json`
- `data/semantic_backbones/README.md`

### 2. Artificial Language Generation

The project uses PCFG grammar templates to generate sentences over the semantic
graph. It supports artificial token inventories such as CJK-style tokens and
Hiragana-style tokens, plus different syntactic configurations.

Important files:

- `data/grammar/grammar_templates_adj.py`
- `data/generate_sentences/v3_generate_sentences.py`
- `data/generate_sentences/v3_generated_sentences_adj.json`

### 3. Corpus Construction

Generated graph sentences are converted into model-training corpora, including
monolingual corpora and parallel CJK/Hiragana corpora.

Important files:

- `data/corpus/build_synset_corpus.py`
- `data/corpus/corpus_cjk_synset.txt`
- `data/corpus/corpus_hiragana_synset.txt`
- `data/corpus/parallel_corpus_synset.json`

### 4. Model Training

Training scripts build small BERT-style masked-language models from scratch for
monolingual and multilingual settings.

Important files:

- `Experiment/training/train_monolingual_synset.py`
- `Experiment/training/train_multilingual_synset.py`

### 5. Experiments and Evaluation

The repository includes experiments for semantic overlap, graph density, lexical
overlap, punctuation/special-token controls, and masked-language probing.

Key directories:

- `Experiment/experiments/semantic_overlap/`
- `Experiment/experiments/graph_density/`
- `Experiment/experiments/lexical_overlap/`
- `Experiment/evaluation/masked_language_probing/`
- `Experiment/evaluation/significance/`

Common evaluation targets include:

- word translation precision
- sentence retrieval precision
- monolingual MLM accuracy
- layerwise masked-language probing
- significance testing across experimental conditions

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are NLTK, PyTorch, Transformers, Tokenizers, Datasets,
Accelerate, and Matplotlib.

## Typical Workflow

1. Build or inspect the semantic backbone in `data/semantic_backbones/`.
2. Generate artificial-language sentences with scripts in
   `data/generate_sentences/`.
3. Build CJK/Hiragana training corpora with `data/corpus/build_synset_corpus.py`.
4. Train monolingual or multilingual MLM models with scripts in
   `Experiment/training/`.
5. Run alignment, probing, and significance evaluations under
   `Experiment/evaluation/` and `Experiment/experiments/`.

For experiment-specific details, see the README files inside each subdirectory.

## Notes

- Large checkpoints and generated experiment outputs are not documented here;
  check the relevant experiment directory for expected inputs and outputs.
- The root README is intended as a high-level map. Method details, sampling
  rules, and result tables belong in the component-level README files.
