# A Toy Setting for the Anchor Token Hypothesis
## Artificial Language creation
### Semantic structure

We build a degree-controlled noun concept backbone from WordNet, enriches it with ConceptNet relations and virtual adjective properties, and provides a controlled, connected, extensible semantic backbone for constructing our artificial language. More information in [corresponding README](data/semantic_backbones/README.md).

### Syntactic structure
**1. PCFG production rules** and the per-relation verb-phrase templates live in
`data/grammar/grammar_templates_adj.py` (adjective-aware; `grammar_templates.py`
is the no-adjective variant). They are not run directly — the generator imports
them via `--grammar`. Each relation gets four VP templates, with
copula + adjective + preposition bound together via subcategorised
non-terminals (e.g. ADJ_1/PREP_1) so invalid combinations like is + with
can't be generated.

**2. The main sentence generator.** Word order is chosen per run by the switches
`--s1` (S/VP order), `--s2` (O/V order), `--s3` (N/A order):
      ```bash
      python3 data/generate_sentences/v3_generate_sentences.py \
        --edges data/semantic_backbones/edges_adj.json \
        --grammar data/grammar/grammar_templates_adj.py \
        --s1 0 --s2 1 --s3 0 \
        --output v3_generated_sentences_adj.json
      ```
Since NLTK's PCFG module ships no
sentence sampler, the sampler is implemented here from scratch.

Output: **v3_generated_sentences_adj.json** — one word-order config per file. The committed corpus (s1=0, s2=1, s3=0) holds 6891 triples / 50,215 sentences.

