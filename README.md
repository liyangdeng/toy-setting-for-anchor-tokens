# A Toy Setting for the Anchor Token Hypothesis
## Artificial Language creation
### Semantic structure

We build a degree-controlled noun concept backbone from WordNet, enriches it with ConceptNet relations and virtual adjective properties, and provides a controlled, connected, extensible semantic backbone for constructing our artificial language. More information in [corresponding README](data/semantic_backbones/README.md).

### Syntactic structure
**1. PCFG production rules** and the per-relation verb-phrase templates:
      ```bash
      python3 grammar/grammar_templates.py
      ```  
Each relation gets four VP templates, with
copula + adjective + preposition bound together via subcategorised
non-terminals (e.g. ADJ_1/PREP_1) so invalid combinations like is + with
can't be generated.
      
**2. The main sentence generator.** Applies switches to generate the configurations: SVO, SOV, VSO, OVS
      ```bash
      python3 generate_sentences/generate_pcfg_sentences.py
      ```      
Since NLTK's PCFG module ships no
sentence sampler, the sampler is implemented here from scratch.

Output: **generated_sentences.json** with 2174 triples × 3 sentences × 4 configs = **26,088 sentences**

