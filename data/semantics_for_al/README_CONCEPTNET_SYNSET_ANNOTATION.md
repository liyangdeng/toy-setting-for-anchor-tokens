# ConceptNet-to-WordNet Synset Annotation Guide

This guide explains how to annotate ConceptNet lemma-level edges so they can be
merged into the WordNet synset-level concept graph.

## Goal

ConceptNet edges connect surface lemmas:

```text
abdomen RelatedTo abdominal
```

The WordNet concept graph connects synsets:

```text
abdomen.n.01 RelatedTo abdominal.n.01
```

Annotation decides which WordNet synset each ConceptNet lemma refers to.

## Files

Annotators should use one of these files:

```text
conceptnet_synset_review.csv
conceptnet_synset_review.json
```

The CSV is easier for spreadsheet editing. The JSON preserves structure better
for programmatic review. Both contain the same annotation task.

## What Each Row Means

Each row is one ConceptNet lemma edge that could not be mapped automatically.

Important fields:

```text
edge_id
source_lemma
relation
target_lemma
source_options
target_options
selected_source_synset
selected_target_synset
decision
notes
```

Example:

```text
edge_id: abdomen|RelatedTo|abdominal
source_lemma: abdomen
relation: RelatedTo
target_lemma: abdominal
```

This means ConceptNet contains:

```text
abdomen RelatedTo abdominal
```

The annotator must choose which WordNet synset best matches `abdomen` and which
WordNet synset best matches `abdominal`.

## Why There Are Multiple Options

A lemma can appear in more than one WordNet synset.

For example, the lemma `abdomen` may appear in:

```text
abdomen.n.01
definition: the region of the body of a vertebrate between the thorax and the pelvis

abdominal_cavity.n.01
definition: the cavity containing the major viscera; in mammals it is separated from the thorax by the diaphragm
```

ConceptNet only gives the lemma `abdomen`, so the annotation must choose the
intended synset using definitions, examples, POS, and neighboring lemmas.

## Annotation Fields To Fill

For each row, fill:

```text
selected_source_synset
selected_target_synset
decision
```

Use a synset id exactly as shown in the options, for example:

```text
selected_source_synset = abdomen.n.01
selected_target_synset = abdominal.n.01
decision = accept
```

If the edge should not be included:

```text
decision = reject
```

Use `notes` only when a case is unclear or needs discussion.

## Decision Rules

Choose `accept` when both conditions are true:

```text
1. The selected source synset matches the source lemma in this relation.
2. The selected target synset matches the target lemma in this relation.
```

Choose `reject` when any of these are true:

```text
The lemma is too ambiguous to choose confidently.
The ConceptNet relation seems wrong for all available synset options.
The best source and target choices would be the same synset.
The relation only connects two surface forms of the same concept.
The edge is too vague to be useful.
```

## Relation Interpretation

Use the ConceptNet relation to help choose the sense.

Common relations:

```text
IsA          source is a kind of target
PartOf       source is part of target
HasA         source has target as a part or attribute
MadeOf       source is made of target
HasProperty  source has target as a property
UsedFor      source is used for target
AtLocation   source is typically found at target
CapableOf    source can do target
RelatedTo    broad relatedness
Synonym      same or very similar meaning
Antonym      opposite meaning
DerivedFrom  lexical derivation
FormOf       inflectional or lexical form
SimilarTo    similar meaning
```

For `RelatedTo`, be conservative. Accept only when the relation is clearly
useful for the intended concept graph.

For `Synonym`, accept only if the two selected synsets are genuinely the same
or near-equivalent concept. If they are merely related words, reject.

For `DerivedFrom` and `FormOf`, accept only if lexical relations are useful for
the downstream task. Otherwise reject them.

## Recommended Workflow

1. Open `conceptnet_synset_review.csv`.
2. Read `source_lemma`, `relation`, and `target_lemma`.
3. Compare `source_options` definitions and choose one source synset.
4. Compare `target_options` definitions and choose one target synset.
5. If the edge is semantically valid, set `decision` to `accept`.
6. If not valid or too unclear, set `decision` to `reject`.
7. Add a short `notes` entry for difficult cases.

## Examples

### Accept

```text
source_lemma: abdomen
relation: RelatedTo
target_lemma: abdominal

selected_source_synset: abdomen.n.01
selected_target_synset: abdominal.n.01
decision: accept
```

Reason: the body-region sense of `abdomen` is clearly related to the abdominal
muscle/body-part sense.

### Reject

```text
source_lemma: food
relation: RelatedTo
target_lemma: plant
```

Reject if the available senses make the relation too broad or unclear for the
concept graph.

### Same Synset

```text
source_lemma: abdomen
relation: Synonym
target_lemma: belly
```

If both lemmas point to `abdomen.n.01`, reject the row because it would create a
self-loop in the synset graph.

## Final Merge

After annotation is complete, run:

```bash
python3 merge_conceptnet_edges_with_synsets.py finalize \
  --review-decisions conceptnet_synset_review.csv
```

This writes:

```text
combined_synset_edges.json
combined_synset_edges_metadata.json
```

The combined edge file keeps synset ids:

```json
{
  "source": "abdomen.n.01",
  "relation": "RelatedTo",
  "target": "abdominal.n.01",
  "source_lemma": "abdomen",
  "target_lemma": "abdominal",
  "source_type": "conceptnet_reviewed"
}
```

## Quality Standard

Prefer fewer high-confidence edges over many uncertain edges.

When uncertain, reject and leave a note.
