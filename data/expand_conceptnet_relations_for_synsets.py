import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_ALLOWED_RELATIONS = [
    "Antonym",
    "AtLocation",
    "CapableOf",
    "Causes",
    "CreatedBy",
    "DerivedFrom",
    "Desires",
    "DistinctFrom",
    "Entails",
    "FormOf",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPrerequisite",
    "HasProperty",
    "HasSubevent",
    "IsA",
    "MadeOf",
    "MannerOf",
    "MotivatedByGoal",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SimilarTo",
    "Synonym",
    "UsedFor",
]


WORDNET_TO_CONCEPTNET = {
    "hypernym": [("forward", "IsA")],
    "hyponym": [("reverse", "IsA")],
    "instance_hypernym": [("forward", "IsA")],
    "instance_hyponym": [("reverse", "IsA")],
    "part_meronym": [("reverse", "PartOf"), ("forward", "HasA")],
    "member_meronym": [("reverse", "PartOf"), ("forward", "HasA")],
    "substance_meronym": [("forward", "MadeOf")],
    "part_holonym": [("forward", "PartOf"), ("reverse", "HasA")],
    "member_holonym": [("forward", "PartOf"), ("reverse", "HasA")],
    "substance_holonym": [("reverse", "MadeOf")],
    "attribute": [("forward", "HasProperty")],
    "similar_to": [("forward", "SimilarTo"), ("reverse", "SimilarTo")],
    "also_see": [("forward", "RelatedTo"), ("reverse", "RelatedTo")],
    "verb_group": [("forward", "SimilarTo"), ("reverse", "SimilarTo")],
    "entailment": [("forward", "Entails")],
    "cause": [("forward", "Causes")],
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def normalize_lemma(raw: str) -> str:
    return raw.replace("_", " ").lower()


def load_synsets(path: Path) -> Tuple[Dict[str, List[str]], Set[str]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array of synset rows.")

    lemmas_by_synset: Dict[str, List[str]] = {}
    vocabulary: Set[str] = set()
    for row in rows:
        sid = row.get("synset_id")
        lemmas = row.get("lemmas")
        if not isinstance(sid, str) or not isinstance(lemmas, list):
            raise ValueError(f"{path} contains a row without synset_id/lemmas.")
        normalized = []
        for lemma in lemmas:
            if not isinstance(lemma, str):
                continue
            word = normalize_lemma(lemma)
            normalized.append(word)
            vocabulary.add(word)
        lemmas_by_synset[sid] = sorted(set(normalized))

    return lemmas_by_synset, vocabulary


def load_wordnet_edges(path: Path) -> List[Dict[str, str]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array of edges.")
    edges: List[Dict[str, str]] = []
    for row in rows:
        source = row.get("source")
        relation = row.get("relation")
        target = row.get("target")
        if not all(isinstance(value, str) for value in (source, relation, target)):
            raise ValueError(f"{path} contains an edge without source/relation/target.")
        edges.append({"source": source, "relation": relation, "target": target})
    return edges


def normalize_conceptnet_node(node: str) -> Optional[str]:
    parts = node.strip().split("/")
    if len(parts) < 4 or parts[1] != "c" or parts[2] != "en":
        return None
    return normalize_lemma(parts[3])


def parse_weight(raw: str) -> float:
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError:
        return 1.0
    weight = metadata.get("weight", 1.0)
    try:
        return float(weight)
    except (TypeError, ValueError):
        return 1.0


def parse_conceptnet_line(line: str) -> Optional[Tuple[str, str, str, float]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        return None

    relation = parts[1].split("/")[-1]
    source = normalize_conceptnet_node(parts[2])
    target = normalize_conceptnet_node(parts[3])
    if source is None or target is None:
        return None

    weight = parse_weight(parts[4]) if len(parts) >= 5 else 1.0
    return source, relation, target, weight


def parse_relation_list(values: Optional[List[str]]) -> Set[str]:
    if not values:
        return set(DEFAULT_ALLOWED_RELATIONS)

    relations: Set[str] = set()
    for value in values:
        for relation in value.split(","):
            relation = relation.strip()
            if relation:
                relations.add(relation)
    return relations


def mapped_wordnet_triples(
    wordnet_edges: Iterable[Dict[str, str]],
    lemmas_by_synset: Dict[str, List[str]],
) -> Set[Tuple[str, str, str]]:
    triples: Set[Tuple[str, str, str]] = set()

    for edge in wordnet_edges:
        mappings = WORDNET_TO_CONCEPTNET.get(edge["relation"], [])
        source_lemmas = lemmas_by_synset.get(edge["source"], [])
        target_lemmas = lemmas_by_synset.get(edge["target"], [])
        for direction, conceptnet_relation in mappings:
            for source_lemma in source_lemmas:
                for target_lemma in target_lemmas:
                    if source_lemma == target_lemma:
                        continue
                    if direction == "forward":
                        triples.add((source_lemma, conceptnet_relation, target_lemma))
                    elif direction == "reverse":
                        triples.add((target_lemma, conceptnet_relation, source_lemma))

    return triples


def iter_new_conceptnet_edges(
    conceptnet_path: Path,
    vocabulary: Set[str],
    allowed_relations: Set[str],
    existing_triples: Set[Tuple[str, str, str]],
    min_weight: float,
) -> Iterable[Dict[str, Any]]:
    seen: Set[Tuple[str, str, str]] = set()

    with conceptnet_path.open(encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_conceptnet_line(line)
            if parsed is None:
                continue
            source, relation, target, weight = parsed
            if relation not in allowed_relations:
                continue
            if weight < min_weight:
                continue
            if source == target:
                continue
            if source not in vocabulary or target not in vocabulary:
                continue

            key = (source, relation, target)
            if key in existing_triples or key in seen:
                continue

            seen.add(key)
            yield {
                "source": source,
                "relation": relation,
                "target": target,
                "weight": weight,
            }


def strip_weights(edges: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "source": edge["source"],
            "relation": edge["relation"],
            "target": edge["target"],
        }
        for edge in edges
    ]


def summarize(
    vocabulary: Set[str],
    wordnet_projected_triples: Set[Tuple[str, str, str]],
    weighted_edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    degree = Counter()
    for edge in weighted_edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1

    return {
        "lemma_count": len(vocabulary),
        "projected_wordnet_conceptnet_triple_count": len(wordnet_projected_triples),
        "new_conceptnet_edge_count": len(weighted_edges),
        "new_relation_counts": dict(
            sorted(Counter(edge["relation"] for edge in weighted_edges).items())
        ),
        "covered_lemma_count": len(degree),
        "uncovered_lemma_count": len(vocabulary) - len(degree),
        "average_new_edge_degree": round(sum(degree.values()) / len(vocabulary), 4)
        if vocabulary
        else 0,
    }


def build_expansion(args: argparse.Namespace) -> Dict[str, Any]:
    lemmas_by_synset, vocabulary = load_synsets(args.synsets)
    wordnet_edges = load_wordnet_edges(args.wordnet_edges)
    allowed_relations = parse_relation_list(args.relation)
    wordnet_projected_triples = mapped_wordnet_triples(
        wordnet_edges=wordnet_edges,
        lemmas_by_synset=lemmas_by_synset,
    )
    weighted_edges = sorted(
        iter_new_conceptnet_edges(
            conceptnet_path=args.conceptnet,
            vocabulary=vocabulary,
            allowed_relations=allowed_relations,
            existing_triples=wordnet_projected_triples,
            min_weight=args.min_weight,
        ),
        key=lambda edge: (edge["source"], edge["relation"], edge["target"]),
    )
    edges = strip_weights(weighted_edges)

    metadata = {
        "metadata": {
            "conceptnet_path": str(args.conceptnet),
            "synsets_path": str(args.synsets),
            "wordnet_edges_path": str(args.wordnet_edges),
            "allowed_relations": sorted(allowed_relations),
            "min_weight": args.min_weight,
            "wordnet_to_conceptnet_relation_map": WORDNET_TO_CONCEPTNET,
            "strategy": (
                "Collect all lemmas from the current synset inventory, project "
                "existing WordNet synset edges into ConceptNet-style lemma triples, "
                "then stream ConceptNet assertions once and keep English lemma pairs "
                "whose source and target are both in the current lemma vocabulary. "
                "Triples already implied by projected WordNet edges are skipped."
            ),
        },
        "summary": summarize(vocabulary, wordnet_projected_triples, weighted_edges),
        "vocabulary": sorted(vocabulary),
        "relations": sorted({edge["relation"] for edge in weighted_edges}),
    }
    return {
        "edges": edges,
        "weighted_edges": weighted_edges,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand the current WordNet synset inventory with new lemma-level "
            "ConceptNet relations that are not already implied by mapped WordNet edges."
        )
    )
    parser.add_argument("--conceptnet", type=Path, default=Path("assertions.csv"))
    parser.add_argument("--synsets", type=Path, default=Path("wordnet_concept_synsets.json"))
    parser.add_argument(
        "--wordnet-edges",
        type=Path,
        default=Path("wordnet_concept_edges.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("conceptnet_expanded_edges.json"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("conceptnet_expanded_metadata.json"),
    )
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument(
        "--relation",
        action="append",
        help=(
            "Allowed ConceptNet relation. Can be repeated or comma-separated. "
            "Defaults to a broad English commonsense and lexical relation whitelist."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_expansion(args)
    write_json(args.output, data["edges"])
    write_json(args.metadata_output, data["metadata"])

    summary = data["metadata"]["summary"]
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata_output}")
    print(f"lemma_count={summary['lemma_count']}")
    print(
        "projected_wordnet_conceptnet_triples="
        f"{summary['projected_wordnet_conceptnet_triple_count']}"
    )
    print(f"new_conceptnet_edges={summary['new_conceptnet_edge_count']}")
    print(f"covered_lemmas={summary['covered_lemma_count']}")


if __name__ == "__main__":
    main()
