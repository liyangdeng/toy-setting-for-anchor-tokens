import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_synsets(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array.")

    synsets_by_id: Dict[str, Dict[str, Any]] = {}
    lemma_to_synsets: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        sid = row.get("synset_id")
        lemmas = row.get("lemmas")
        if not isinstance(sid, str) or not isinstance(lemmas, list):
            raise ValueError(f"{path} contains a row without synset_id/lemmas.")
        synsets_by_id[sid] = row
        for lemma in lemmas:
            if isinstance(lemma, str):
                lemma_to_synsets[lemma.lower()].append(sid)

    return synsets_by_id, {
        lemma: sorted(set(sids))
        for lemma, sids in lemma_to_synsets.items()
    }


def load_edges(path: Path) -> List[Dict[str, str]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array.")

    edges: List[Dict[str, str]] = []
    for row in rows:
        source = row.get("source")
        relation = row.get("relation")
        target = row.get("target")
        if not all(isinstance(value, str) for value in (source, relation, target)):
            raise ValueError(f"{path} contains an edge without source/relation/target.")
        edges.append({"source": source, "relation": relation, "target": target})
    return edges


def synset_option(synset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "synset_id": synset["synset_id"],
        "pos": synset.get("pos"),
        "primary_lemma": synset.get("primary_lemma"),
        "lemmas": synset.get("lemmas", []),
        "definition": synset.get("definition"),
        "examples": synset.get("examples", []),
        "root_sources": synset.get("root_sources", []),
    }


def edge_key(edge: Dict[str, str]) -> str:
    return f"{edge['source']}|{edge['relation']}|{edge['target']}"


def resolve_edge(
    edge: Dict[str, str],
    lemma_to_synsets: Dict[str, List[str]],
) -> Optional[Tuple[str, str, str]]:
    source_options = lemma_to_synsets.get(edge["source"].lower(), [])
    target_options = lemma_to_synsets.get(edge["target"].lower(), [])
    if len(source_options) == 1 and len(target_options) == 1:
        source = source_options[0]
        target = target_options[0]
        if source != target:
            return source, edge["relation"], target
    return None


def is_unambiguous_self_loop(
    edge: Dict[str, str],
    lemma_to_synsets: Dict[str, List[str]],
) -> bool:
    source_options = lemma_to_synsets.get(edge["source"].lower(), [])
    target_options = lemma_to_synsets.get(edge["target"].lower(), [])
    return (
        len(source_options) == 1
        and len(target_options) == 1
        and source_options[0] == target_options[0]
    )


def build_review_items(
    conceptnet_edges: List[Dict[str, str]],
    synsets_by_id: Dict[str, Dict[str, Any]],
    lemma_to_synsets: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    auto_edges: List[Dict[str, Any]] = []
    review_items: List[Dict[str, Any]] = []
    skipped_self_loops: List[Dict[str, str]] = []

    for edge in conceptnet_edges:
        if is_unambiguous_self_loop(edge, lemma_to_synsets):
            skipped_self_loops.append(edge)
            continue

        resolved = resolve_edge(edge, lemma_to_synsets)
        if resolved is not None:
            source, relation, target = resolved
            auto_edges.append({
                "source": source,
                "relation": relation,
                "target": target,
                "source_lemma": edge["source"],
                "target_lemma": edge["target"],
                "source_type": "conceptnet",
            })
            continue

        source_options = lemma_to_synsets.get(edge["source"].lower(), [])
        target_options = lemma_to_synsets.get(edge["target"].lower(), [])
        review_items.append({
            "edge_id": edge_key(edge),
            "source_lemma": edge["source"],
            "relation": edge["relation"],
            "target_lemma": edge["target"],
            "source_options": [
                synset_option(synsets_by_id[sid])
                for sid in source_options
            ],
            "target_options": [
                synset_option(synsets_by_id[sid])
                for sid in target_options
            ],
            "selected_source_synset": "",
            "selected_target_synset": "",
            "decision": "pending",
            "notes": "",
        })

    return auto_edges, review_items, skipped_self_loops


def write_review_csv(path: Path, review_items: Iterable[Dict[str, Any]]) -> None:
    fieldnames = [
        "edge_id",
        "source_lemma",
        "relation",
        "target_lemma",
        "source_options",
        "target_options",
        "selected_source_synset",
        "selected_target_synset",
        "decision",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in review_items:
            writer.writerow({
                "edge_id": item["edge_id"],
                "source_lemma": item["source_lemma"],
                "relation": item["relation"],
                "target_lemma": item["target_lemma"],
                "source_options": format_options(item["source_options"]),
                "target_options": format_options(item["target_options"]),
                "selected_source_synset": item["selected_source_synset"],
                "selected_target_synset": item["selected_target_synset"],
                "decision": item["decision"],
                "notes": item["notes"],
            })


def format_options(options: List[Dict[str, Any]]) -> str:
    return " || ".join(
        f"{option['synset_id']}: {option.get('definition')}"
        for option in options
    )


def load_review_decisions(path: Path) -> Dict[str, Dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return {
                row["edge_id"]: row
                for row in csv.DictReader(handle)
            }

    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON array of review items.")
    return {
        row["edge_id"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("edge_id"), str)
    }


def accepted_review_edges(
    review_items: List[Dict[str, Any]],
    decisions: Dict[str, Dict[str, str]],
    synsets_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    accepted: List[Dict[str, Any]] = []
    for item in review_items:
        decision = decisions.get(item["edge_id"], item)
        status = str(decision.get("decision", "")).strip().lower()
        if status not in {"accept", "accepted", "keep"}:
            continue

        source = str(decision.get("selected_source_synset", "")).strip()
        target = str(decision.get("selected_target_synset", "")).strip()
        if source not in synsets_by_id or target not in synsets_by_id:
            continue
        if source == target:
            continue

        accepted.append({
            "source": source,
            "relation": item["relation"],
            "target": target,
            "source_lemma": item["source_lemma"],
            "target_lemma": item["target_lemma"],
            "source_type": "conceptnet_reviewed",
        })
    return accepted


def wordnet_edges_with_source_type(edges: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "source": edge["source"],
            "relation": edge["relation"],
            "target": edge["target"],
            "source_type": "wordnet",
        }
        for edge in edges
    ]


def dedupe_edges(edges: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for edge in edges:
        source_type = edge.get("source_type", "unknown")
        key = (edge["source"], edge["relation"], edge["target"], source_type)
        merged[key] = edge
    return sorted(
        merged.values(),
        key=lambda edge: (
            edge["source"],
            edge["relation"],
            edge["target"],
            edge.get("source_type", ""),
        ),
    )


def summarize_edges(edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "edge_count": len(edges),
        "relation_counts": dict(sorted(Counter(edge["relation"] for edge in edges).items())),
        "source_type_counts": dict(
            sorted(Counter(edge.get("source_type", "unknown") for edge in edges).items())
        ),
    }


def prepare(args: argparse.Namespace) -> None:
    synsets_by_id, lemma_to_synsets = load_synsets(args.synsets)
    conceptnet_edges = load_edges(args.conceptnet_edges)
    auto_edges, review_items, skipped_self_loops = build_review_items(
        conceptnet_edges=conceptnet_edges,
        synsets_by_id=synsets_by_id,
        lemma_to_synsets=lemma_to_synsets,
    )

    write_json(args.auto_output, auto_edges)
    write_json(args.review_output, review_items)
    write_review_csv(args.review_csv_output, review_items)
    metadata = {
        "metadata": {
            "synsets_path": str(args.synsets),
            "conceptnet_edges_path": str(args.conceptnet_edges),
            "review_instructions": (
                "For each pending ConceptNet lemma edge, inspect source_options and "
                "target_options. Fill selected_source_synset and selected_target_synset "
                "with synset ids, then set decision to accept. Use reject to skip."
            ),
        },
        "summary": {
            "conceptnet_edge_count": len(conceptnet_edges),
            "auto_resolved_edge_count": len(auto_edges),
            "manual_review_edge_count": len(review_items),
            "skipped_unambiguous_self_loop_count": len(skipped_self_loops),
            "ambiguous_lemma_count": sum(1 for sids in lemma_to_synsets.values() if len(sids) > 1),
        },
    }
    write_json(args.metadata_output, metadata)

    print(f"Wrote {args.auto_output}")
    print(f"Wrote {args.review_output}")
    print(f"Wrote {args.review_csv_output}")
    print(f"Wrote {args.metadata_output}")
    print(f"auto_resolved_edges={len(auto_edges)}")
    print(f"manual_review_edges={len(review_items)}")
    print(f"skipped_unambiguous_self_loops={len(skipped_self_loops)}")


def finalize(args: argparse.Namespace) -> None:
    synsets_by_id, lemma_to_synsets = load_synsets(args.synsets)
    wordnet_edges = load_edges(args.wordnet_edges)
    conceptnet_edges = load_edges(args.conceptnet_edges)
    auto_edges, review_items, skipped_self_loops = build_review_items(
        conceptnet_edges=conceptnet_edges,
        synsets_by_id=synsets_by_id,
        lemma_to_synsets=lemma_to_synsets,
    )
    reviewed_edges: List[Dict[str, Any]] = []
    if args.review_decisions is not None:
        decisions = load_review_decisions(args.review_decisions)
        reviewed_edges = accepted_review_edges(review_items, decisions, synsets_by_id)

    combined_edges = dedupe_edges(
        wordnet_edges_with_source_type(wordnet_edges)
        + auto_edges
        + reviewed_edges
    )
    write_json(args.combined_output, combined_edges)
    metadata = {
        "metadata": {
            "synsets_path": str(args.synsets),
            "wordnet_edges_path": str(args.wordnet_edges),
            "conceptnet_edges_path": str(args.conceptnet_edges),
            "review_decisions_path": str(args.review_decisions)
            if args.review_decisions
            else None,
            "merge_strategy": (
                "Keep WordNet synset edges unchanged. Convert unambiguous ConceptNet "
                "lemma edges to synset edges automatically. Add ambiguous ConceptNet "
                "edges only when the review file marks them accepted with selected "
                "source and target synsets."
            ),
        },
        "summary": {
            "wordnet_edge_count": len(wordnet_edges),
            "auto_conceptnet_synset_edge_count": len(auto_edges),
            "reviewed_conceptnet_synset_edge_count": len(reviewed_edges),
            "pending_review_edge_count": len(review_items) - len(reviewed_edges),
            "skipped_unambiguous_self_loop_count": len(skipped_self_loops),
            "combined": summarize_edges(combined_edges),
        },
    }
    write_json(args.metadata_output, metadata)

    print(f"Wrote {args.combined_output}")
    print(f"Wrote {args.metadata_output}")
    print(f"combined_edges={len(combined_edges)}")
    print(f"auto_conceptnet_edges={len(auto_edges)}")
    print(f"reviewed_conceptnet_edges={len(reviewed_edges)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare manual synset assignment for lemma-level ConceptNet edges, "
            "then merge WordNet and reviewed ConceptNet edges into one synset graph."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--synsets", type=Path, default=Path("wordnet_concept_synsets.json"))
    prepare_parser.add_argument(
        "--conceptnet-edges",
        type=Path,
        default=Path("conceptnet_expanded_edges.json"),
    )
    prepare_parser.add_argument(
        "--auto-output",
        type=Path,
        default=Path("conceptnet_auto_synset_edges.json"),
    )
    prepare_parser.add_argument(
        "--review-output",
        type=Path,
        default=Path("conceptnet_synset_review.json"),
    )
    prepare_parser.add_argument(
        "--review-csv-output",
        type=Path,
        default=Path("conceptnet_synset_review.csv"),
    )
    prepare_parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("conceptnet_synset_review_metadata.json"),
    )
    prepare_parser.set_defaults(func=prepare)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--synsets", type=Path, default=Path("wordnet_concept_synsets.json"))
    finalize_parser.add_argument(
        "--wordnet-edges",
        type=Path,
        default=Path("wordnet_concept_edges.json"),
    )
    finalize_parser.add_argument(
        "--conceptnet-edges",
        type=Path,
        default=Path("conceptnet_expanded_edges.json"),
    )
    finalize_parser.add_argument("--review-decisions", type=Path)
    finalize_parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path("combined_synset_edges.json"),
    )
    finalize_parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("combined_synset_edges_metadata.json"),
    )
    finalize_parser.set_defaults(func=finalize)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
