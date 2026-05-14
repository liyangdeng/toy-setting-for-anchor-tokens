# 512 Concept WordNet Synset Inventory

This pipeline builds a 512-concept WordNet synset inventory for experiments that
need a compact concept graph with controlled part-of-speech balance and limited
semantic complexity.

The script is:

```bash
python3 build_wordnet_synset_inventory.py
```

## Default Target

The default inventory contains 512 synsets:

```text
noun synsets:      400
adjective synsets: 60
verb synsets:      52
```

These counts satisfy the requested ranges:

```text
noun synsets:      350-400
adjective synsets: 40-60
verb synsets:      30-60
```

The exact defaults are intentionally noun-heavy because concrete WordNet noun
regions are broader and usually provide cleaner concept coverage than verbs and
adjectives.

## Method

The pipeline keeps the original synset-level sampling method:

```text
root BFS expansion
-> surface-form filtering
-> polysemy filtering
-> candidate edge construction
-> min/max degree filtering
-> connected-component filtering
-> POS/root-balanced sampling
```

Each retained concept is still a WordNet synset, not a surface word:

```text
dog.n.01 != dog.n.03
bank.n.01 != bank.n.09
```

## Semantic Complexity Control

The script filters by candidate-graph degree:

```text
min_degree = 1
max_degree = 16
```

`min_degree` removes isolated concepts. `max_degree` removes concepts that have
too many WordNet relations inside the candidate graph, which helps avoid overly
central or semantically complex concepts.

Disable the upper bound with:

```bash
python3 build_wordnet_synset_inventory.py --max-degree -1
```

Use a stricter cap with:

```bash
python3 build_wordnet_synset_inventory.py --max-degree 12
```

## Default Command

Run:

```bash
python3 build_wordnet_synset_inventory.py
```

This is equivalent to:

```bash
python3 build_wordnet_synset_inventory.py \
  --target-total 512 \
  --noun-count 400 \
  --adjective-count 60 \
  --verb-count 52 \
  --min-degree 1 \
  --max-degree 16 \
  --component-mode per-root \
  --inventory-output wordnet_concept_synsets.json \
  --edges-output wordnet_concept_edges.json \
  --metadata-output wordnet_concept_metadata.json
```

## Outputs

The pipeline writes three ordinary JSON files.

### `wordnet_concept_edges.json`

An array of all final semantic edges:

```json
[
  {
    "source": "dog.n.01",
    "relation": "hypernym",
    "target": "canine.n.02"
  }
]
```

### `wordnet_concept_synsets.json`

An array of final synsets and their descriptions:

```json
[
  {
    "synset_id": "dog.n.01",
    "pos": "noun",
    "wordnet_pos": "n",
    "primary_lemma": "dog",
    "lemmas": ["dog"],
    "definition": "...",
    "examples": [],
    "root_sources": ["animal.n.01"],
    "min_bfs_depth": 2,
    "candidate_degree": 6,
    "final_degree": 3,
    "lemma_sense_counts": {
      "dog": {
        "total": 8,
        "same_pos": 7
      }
    },
    "relations": {
      "hypernym": ["canine.n.02"]
    }
  }
]
```

### `wordnet_concept_metadata.json`

An object containing the sampling configuration, filter settings, counts,
distribution summary, relation counts, and warnings.

Important fields include:

```text
metadata.target_total
metadata.target_counts
metadata.target_count_ranges
metadata.graph_filter
counts.final_synset_count
counts.final_edge_count
summary.final_average_degree
summary.final_isolated_synset_count
summary.distribution
summary.warnings
```

## Notes

If the final inventory contains fewer than 512 synsets, relax one of these
constraints:

```text
increase --max-degree
decrease --min-degree
switch --component-mode from per-root to per-pos
increase BFS depth
relax polysemy thresholds
```

For reproducibility, keep `--seed` fixed.

## ConceptNet Relation Expansion

After building the 512 WordNet synset inventory, expand it with ConceptNet
lemma-level relations:

```bash
python3 expand_conceptnet_relations_for_synsets.py
```

Default inputs:

```text
wordnet_concept_synsets.json
wordnet_concept_edges.json
assertions.csv
```

Default outputs:

```text
conceptnet_expanded_edges.json
conceptnet_expanded_metadata.json
```

The expansion script:

1. Collects every lemma from the current synset inventory.
2. Projects existing WordNet synset edges into ConceptNet-style lemma triples.
3. Streams the local ConceptNet `assertions.csv` file once.
4. Keeps English ConceptNet edges whose source and target are both current
   lemmas.
5. Skips ConceptNet triples already implied by the projected WordNet edges.
6. Writes new edges in the same simple edge format:

```json
[
  {
    "source": "abdomen",
    "relation": "RelatedTo",
    "target": "belly"
  }
]
```

WordNet relations are mapped to ConceptNet-style relations before duplicate
checking. Examples:

```text
hypernym / instance_hypernym -> IsA
hyponym / instance_hyponym   -> reverse IsA
part_meronym                 -> reverse PartOf and HasA
part_holonym                 -> PartOf and reverse HasA
substance_meronym            -> MadeOf
similar_to / verb_group      -> SimilarTo
also_see                     -> RelatedTo
entailment                   -> Entails
cause                        -> Causes
```

The default ConceptNet relation whitelist is broad and includes lexical
relations such as `RelatedTo`, `Synonym`, and `DerivedFrom`, plus commonsense
relations such as `IsA`, `PartOf`, `HasA`, `HasProperty`, and `UsedFor`.

Use repeated or comma-separated `--relation` arguments to narrow the expansion:

```bash
python3 expand_conceptnet_relations_for_synsets.py \
  --relation IsA,PartOf,HasA,MadeOf,HasProperty,UsedFor,CapableOf,AtLocation
```

## Merging ConceptNet Edges Back To Synsets

`conceptnet_expanded_edges.json` is lemma-level, while
`wordnet_concept_edges.json` is synset-level. Do not merge them directly.
First assign each ConceptNet lemma endpoint to a synset.

For annotator-facing instructions, see
`README_CONCEPTNET_SYNSET_ANNOTATION.md`.

Prepare automatic mappings and manual review files:

```bash
python3 merge_conceptnet_edges_with_synsets.py prepare
```

This writes:

```text
conceptnet_auto_synset_edges.json
conceptnet_synset_review.json
conceptnet_synset_review.csv
conceptnet_synset_review_metadata.json
```

The script automatically converts ConceptNet edges where both lemmas map to one
unique synset. Edges involving ambiguous lemmas are written to the review files.
Each review item includes candidate synsets, definitions, examples, and lemmas.

For manual review, fill these fields in either the JSON or CSV file:

```text
selected_source_synset
selected_target_synset
decision
```

Use:

```text
decision = accept
```

to include the edge, or:

```text
decision = reject
```

to skip it.

Generate the combined synset-level edge file with only automatic ConceptNet
assignments:

```bash
python3 merge_conceptnet_edges_with_synsets.py finalize
```

After manual review, include accepted review decisions:

```bash
python3 merge_conceptnet_edges_with_synsets.py finalize \
  --review-decisions conceptnet_synset_review.csv
```

The output files are:

```text
combined_synset_edges.json
combined_synset_edges_metadata.json
```

`combined_synset_edges.json` keeps synset ids as `source` and `target`. WordNet
edges are preserved unchanged with `source_type = wordnet`; ConceptNet edges use
synset ids plus the original lemma evidence:

```json
{
  "source": "abdomen.n.01",
  "relation": "RelatedTo",
  "target": "blind.a.01",
  "source_lemma": "abdomen",
  "target_lemma": "blind",
  "source_type": "conceptnet"
}
```
