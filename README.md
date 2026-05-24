# Toy Setting for Anchor Tokens

## Sentence Generation from Semantic Triples

After building the 512-concept WordNet/ConceptNet synset inventory and the
combined edge set, the next step is to generate sentences expressing each
semantic triple `(source, relation, target)`.

### Entity Treatment

Each synset carries a part-of-speech tag (`n`, `v`, `a`) that reflects its
**semantic** origin in WordNet. For sentence generation purposes, however,
every entity is treated uniformly as an **NP** — its syntactic role in the
generated sentence is always that of a noun phrase, regardless of whether the
underlying synset is nominal, verbal, or adjectival. This avoids the need for
morphological inflection and keeps the focus on structural variation rather
than lexical form.

---

## Relation Statistics

The combined edge set contains **2174 triples** across **33 relation types**:

| Relation | Count | Source |
|---|---|---|
| RelatedTo | 798 | ConceptNet |
| hyponym | 317 | WordNet |
| hypernym | 317 | WordNet |
| part_holonym | 89 | WordNet |
| part_meronym | 89 | WordNet |
| IsA | 87 | ConceptNet |
| Synonym | 77 | ConceptNet |
| similar_to | 70 | WordNet |
| MannerOf | 52 | ConceptNet |
| also_see | 46 | WordNet |
| DerivedFrom | 40 | ConceptNet |
| SimilarTo | 35 | ConceptNet |
| PartOf | 29 | ConceptNet |
| AtLocation | 29 | ConceptNet |
| UsedFor | 14 | ConceptNet |
| substance_meronym | 14 | WordNet |
| substance_holonym | 14 | WordNet |
| member_meronym | 10 | WordNet |
| member_holonym | 10 | WordNet |
| Antonym | 9 | ConceptNet |
| HasA | 5 | ConceptNet |
| MadeOf | 4 | ConceptNet |
| FormOf | 3 | ConceptNet |
| DistinctFrom | 3 | ConceptNet |
| entailment | 3 | WordNet |
| CapableOf | 2 | ConceptNet |
| HasSubevent | 2 | ConceptNet |
| HasProperty | 1 | ConceptNet |
| Entails | 1 | ConceptNet |
| HasLastSubevent | 1 | ConceptNet |
| Desires | 1 | ConceptNet |
| CreatedBy | 1 | ConceptNet |
| HasPrerequisite | 1 | ConceptNet |

---

## Grammar Templates

Each relation is assigned a set of **PCFG production rules** defined in
`generate_sentences/grammar.py`. The grammar decomposes each VP into
independently reusable constituents:

```
S  -> NP_src VP NP_tgt

VP -> V                    (transitive verb)
VP -> V_cop ADJ PREP       (copula + predicative adjective + preposition)
```

ADJ and PREP are bound together via subcategorised non-terminals (e.g.
`ADJ_1`/`PREP_1`) to prevent invalid combinations such as `is` + `with`
without a corresponding adjective.

Each relation has **4 VP templates**. For each triple, **3 sentences are
sampled without replacement** (retry on duplicate). Because NLTK's PCFG module
does not include a sentence generator, a custom sampler is implemented in
`generate_sentences/generate_pcfg_sentences.py`.

---

## Switch Mechanism

Following White & Cotterell (2021), structural variation is controlled by
binary switches rather than by enumerating templates manually.

We originally considered four switch dimensions — S, VP, PP, NP — but PP and
NP switches govern the internal ordering of modifiers *within* an entity
phrase. Since entities here are bare lemmas with no additional modifiers, there
is no constituent to reorder inside the NP or PP. PP and NP switches are
therefore deferred and only **S switch** and **VP switch** are active.

| Switch | Value 0 | Value 1 |
|---|---|---|
| S switch | NP VP (subject first) | VP NP (predicate first) |
| VP switch | NP V (head-final, SOV-like) | V NP (head-initial, SVO-like) |

The two switches combine to produce **4 language configurations**:

| S | VP | Word order | Natural language analogy |
|---|---|---|---|
| 0 | 0 | SOV | Japanese, Turkish |
| 0 | 1 | SVO | English, Chinese |
| 1 | 0 | OVS | Hixkaryana |
| 1 | 1 | VOS | Malagasy |

### Structural Similarity

With two binary switches, similarity between any two configurations is
measured as `1 - Hamming(a, b) / 2`:

| Config A | Config B | Similarity |
|---|---|---|
| any | identical | 100% |
| SOV | OVS or SVO | 50% |
| SVO | VOS or SOV | 50% |
| SOV | VOS | 0% |
| SVO | OVS | 0% |

---

## Output

The generator produces one JSON file per run containing all four
configurations. Total output: **2174 triples × 3 sentences × 4 configs =
26,088 sentences**.

### Sample output

**SVO** (`S=0, VP=1`)
```
abdomen.n.01 --IsA--> inside.n.01
belly is classified as inside
belly represents inside
belly instantiates inside

abdomen.n.01 --IsA--> sac.n.04
stomach is categorized as sac
stomach is classified as sac
stomach instantiates sac
```

**SOV** (`S=0, VP=0`)
```
abdomen.n.01 --IsA--> inside.n.01
belly inside is categorized as
belly inside instantiates
belly inside is classified as

abdomen.n.01 --IsA--> sac.n.04
stomach sac is classified as
stomach sac instantiates
stomach sac is categorized as
```

**VOS** (`S=1, VP=1`)
```
abdomen.n.01 --IsA--> inside.n.01
instantiates inside belly
is categorized as inside belly
represents inside belly

abdomen.n.01 --IsA--> sac.n.04
is classified as sac stomach
is categorized as sac stomach
represents sac stomach
```

**OVS** (`S=1, VP=0`)
```
abdomen.n.01 --IsA--> inside.n.01
inside is classified as belly
inside represents belly
inside is categorized as belly

abdomen.n.01 --IsA--> sac.n.04
sac instantiates stomach
sac is categorized as stomach
sac is classified as stomach
```

---

## Files

```
generate_sentences/
    generate_pcfg_sentences.py   main generator + switch logic + sampler
data/
    combined_synset_edges.json   2174 semantic triples (input)
    generated_sentences.json     generated sentences for all 4 configs (output)
```
