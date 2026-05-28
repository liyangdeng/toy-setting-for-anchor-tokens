# Semantic Backbone

## Method

The pipeline has four stages:

```text
WordNet noun backbone
-> ConceptNet expansion
-> low-degree repair
-> metadata validation
-> Virtual adjective expansion
```

### 1. WordNet noun backbone

The backbone starts from configured WordNet noun roots and expands with BFS.
The default roots are:

```text
person.n.01
animal.n.01
plant.n.02
artifact.n.01
food.n.01
body_part.n.01
location.n.01
```

Only noun synsets are sampled. The base WordNet structure uses concrete
WordNet structural relations:

```text
hypernym
hyponym
instance_hypernym
instance_hyponym
part_meronym
member_meronym
substance_meronym
part_holonym
member_holonym
substance_holonym
```

The base structure deliberately excludes `also_see` and `similar_to`.

Each lemma can appear in at most one sampled node. If a candidate synset would
reuse an existing lemma, it is skipped.

The WordNet stage caps each node at 6 unique neighbors.

### 2. ConceptNet expansion

ConceptNet expansion uses lemmas from the selected nodes to find additional
edges and ConceptNet lemma nodes.

The script excludes WordNet-like or lexical relations such as:

```text
IsA
PartOf
HasA
Synonym
FormOf
DerivedFrom
EtymologicallyRelatedTo
EtymologicallyDerivedFrom
MannerOf
```

It prefers more specific ConceptNet relations before broader relations. Examples
include:

```text
AtLocation
UsedFor
CapableOf
HasProperty
MadeOf
Causes
HasPrerequisite
HasSubevent
ReceivesAction
Desires
MotivatedByGoal
HasContext
```

`RelatedTo` is handled separately with its own count and ratio limits. The
default output keeps `RelatedTo` below roughly 20% of all edges.

The ConceptNet stage caps each node at 12 unique neighbors.

### 3. Low-degree repair

After WordNet and ConceptNet expansion, nodes with degree below 2 are candidates
for repair.

Repair searches ConceptNet for a useful external lemma and adds a new
`conceptnet_lemma` node connected only to the low-degree node. Repair nodes do
not recursively participate in further repair. This keeps the node count under
control.

The default repair budget is intentionally small:

```text
max_repair_nodes = 50
```

Node-count control has priority over fully fixing every low-degree node.

### 4. Validation

The metadata reports:

```text
node count
edge count
relation counts
source type counts
degree histogram
RelatedTo count and ratio
duplicate lemma count
duplicate edge count
self-loop count
missing endpoint count
degree cap violations
```

## Current Default Output

Current generated summary:

```text
nodes = 2000
  wordnet_synsets = 1200
  conceptnet_lemma_nodes = 800

edges = 6790
  wordnet = 2358
  conceptnet = 4382
  repair = 50

degree_max = 12
RelatedTo = 1300 / 6790 = 19.15%
duplicate_lemma_count = 0
duplicate_edge_count = 0
self_loop_count = 0
missing_endpoint_edge_count = 0
```

## Output Files

### `synsets.json`

Despite the filename, this file contains both WordNet synsets and ConceptNet
lemma nodes.

WordNet synset node:

```json
{
  "id": "dog.n.01",
  "node_type": "wordnet_synset",
  "synset_id": "dog.n.01",
  "pos": "noun",
  "primary_lemma": "dog",
  "lemmas": ["dog", "domestic dog"],
  "definition": "...",
  "examples": [],
  "root_sources": ["animal.n.01"],
  "min_wordnet_depth": 2
}
```

ConceptNet lemma node:

```json
{
  "id": "conceptnet:eating:1257",
  "node_type": "conceptnet_lemma",
  "synset_id": null,
  "pos": null,
  "primary_lemma": "eating",
  "lemmas": ["eating"],
  "definition": null,
  "examples": [],
  "root_sources": [],
  "min_wordnet_depth": null,
  "repair_for": null
}
```

The numeric suffix in IDs such as `conceptnet:eating:1257` is only a unique
identifier. It has no semantic meaning.

### `edges.json`

Each edge is a simple directed relation:

```json
{
  "source": "dog.n.01",
  "relation": "AtLocation",
  "target": "conceptnet:home:1421",
  "source_type": "conceptnet"
}
```

`source_type` is one of:

```text
wordnet
conceptnet
repair
```

Degree caps are computed using unique neighbors, not raw edge count. This means
the same node pair can have multiple different relation edges without increasing
the neighbor degree more than once.

### `metadata.json`

`metadata.json` records the build parameters, stage summaries, and validation
results. Use this file to check whether a generated backbone satisfies the
expected graph constraints.

## Usage

Run the default build:

```bash
python build_semantic_backbone.py
```

Useful parameters:

```bash
python build_semantic_backbone.py \
  --target-wordnet-nodes 1200 \
  --target-total-nodes 2000 \
  --target-total-edges 8000 \
  --wordnet-max-degree 6 \
  --conceptnet-max-degree 12 \
  --relatedto-max-ratio 0.2 \
  --relatedto-max-count 1300 \
  --max-repair-nodes 50 \
  --output-dir semantic_backbones/rebuilt
```

## Virtual Adjective Expansio


The default configuration creates:

```text
virtual_adjective_count = 200
links_per_adjective = 3-6
relation = HasProperty
```

Each artificial adjective is a node:

```json
{
  "id": "virtual_adj:0001",
  "node_type": "virtual_adjective",
  "synset_id": null,
  "pos": "adjective",
  "primary_lemma": "ꀀꂊꃁ",
  "lemmas": ["ꀀꂊꃁ"],
  "definition": null,
  "examples": [],
  "root_sources": [],
  "min_wordnet_depth": null,
  "repair_for": null
}
```

Each virtual adjective edge points from an existing concept to the adjective:

```json
{
  "source": "dog.n.01",
  "relation": "HasProperty",
  "target": "virtual_adj:0001",
  "source_type": "virtual_adjective"
}
```

This means:

```text
concept --HasProperty--> artificial adjective
```

The current generated virtual-adjective output has:

```text
nodes = 2200
edges = 7700
virtual_adjective_count = 200
virtual_adjective_edges = 910
degree_distribution = {3: 46, 4: 49, 5: 54, 6: 51}
```

