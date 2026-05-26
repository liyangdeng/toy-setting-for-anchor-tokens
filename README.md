# A Toy Setting for the Anchor Token Hypothesis
## Artificial Language creation
### Semantic structure

1. Build the initial 512-concept WordNet inventory.
   ```bash
   python3 build_wordnet_synset_inventory.py
   ```
2. Expand with ConceptNet relations: IsA, PartOf, RelatedTo, Synonym, …
*(requires assertions.csv from https://conceptnet.io ??)*
   ```bash
   python3 expand_conceptnet_relations_for_synsets.py
   ```
3. Merge WordNet and ConceptNet edges into the combined edge set.
   ```bash
   python3 merge_conceptnet_edges_with_wordnet.py
   ```
Output: **combined_synset_edges.json** with **2174 triples over 33 relations**

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

