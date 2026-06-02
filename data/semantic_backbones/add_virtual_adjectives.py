import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


DEFAULT_SYNSETS = Path("semantic_backbones/kg_noun_only/synsets.json")
DEFAULT_EDGES = Path("semantic_backbones/kg_noun_only/edges.json")
DEFAULT_METADATA = Path("semantic_backbones/kg_noun_only/metadata.json")
DEFAULT_OUTPUT_DIR = Path("semantic_backbones/kg_with_virtual_adj")
DEFAULT_UNICODE_START = 0xA000
DEFAULT_UNICODE_END = 0xA4C6


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_codepoint(raw: str) -> int:
    try:
        return int(raw, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid codepoint {raw!r}. Use decimal or hex like 0xA000."
        ) from exc


def unicode_pool(start: int, end: int) -> List[str]:
    if start > end:
        raise ValueError("--unicode-start must be <= --unicode-end.")
    return [
        chr(codepoint)
        for codepoint in range(start, end + 1)
        if 0 <= codepoint <= 0x10FFFF and not 0xD800 <= codepoint <= 0xDFFF
    ]


def clipped_normal_int(
    rng: random.Random,
    mean: float,
    std: float,
    minimum: int,
    maximum: int,
) -> int:
    return max(minimum, min(maximum, round(rng.gauss(mean, std))))


def generate_tokens(
    count: int,
    existing_lemmas: Set[str],
    seed: int,
    unicode_start: int,
    unicode_end: int,
    min_length: int,
    max_length: int,
    mean_length: float,
    std_length: float,
) -> List[str]:
    rng = random.Random(seed)
    characters = unicode_pool(unicode_start, unicode_end)
    if not characters:
        raise ValueError("Selected Unicode range is empty.")

    tokens: Set[str] = set()
    attempts = 0
    max_attempts = max(count * 200, 2000)
    while len(tokens) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Could not generate enough unique adjective tokens.")
        length = clipped_normal_int(
            rng=rng,
            mean=mean_length,
            std=std_length,
            minimum=min_length,
            maximum=max_length,
        )
        token = "".join(rng.choice(characters) for _ in range(length))
        if token not in existing_lemmas:
            tokens.add(token)
    return sorted(tokens)


def neighbors_from_edges(nodes: Iterable[Dict[str, Any]], edges: Iterable[Dict[str, Any]]) -> Dict[str, Set[str]]:
    neighbors: Dict[str, Set[str]] = defaultdict(set)
    for node in nodes:
        neighbors[node["id"]]
    for edge in edges:
        neighbors[edge["source"]].add(edge["target"])
        neighbors[edge["target"]].add(edge["source"])
    return neighbors


def load_array(path: Path, name: str) -> List[Dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{name} must be a JSON array: {path}")
    return rows


def select_concepts(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    rng: random.Random,
) -> List[str]:
    neighbors = neighbors_from_edges(nodes, edges)
    concept_ids = [
        node["id"]
        for node in nodes
        if node.get("node_type") in {"wordnet_synset", "conceptnet_lemma"}
    ]
    concept_ids.sort(key=lambda node_id: (len(neighbors[node_id]), node_id))
    low = concept_ids[: max(1, len(concept_ids) // 4)]
    middle = concept_ids[len(concept_ids) // 4 : len(concept_ids) * 3 // 4]
    high = concept_ids[len(concept_ids) * 3 // 4 :]

    weighted = low * 2 + middle * 3 + high
    rng.shuffle(weighted)
    return weighted


def build_virtual_adjective_nodes(tokens: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "id": f"virtual_adj:{index + 1:04d}",
            "node_type": "virtual_adjective",
            "synset_id": None,
            "pos": "adjective",
            "primary_lemma": token,
            "lemmas": [token],
            "definition": None,
            "examples": [],
            "root_sources": [],
            "min_wordnet_depth": None,
            "repair_for": None,
        }
        for index, token in enumerate(tokens)
    ]


def add_virtual_adjectives(args: argparse.Namespace) -> Dict[str, Any]:
    nodes = load_array(args.synsets, "synsets")
    edges = load_array(args.edges, "edges")
    metadata = read_json(args.metadata) if args.metadata.exists() else {}

    existing_node_ids = {node["id"] for node in nodes}
    existing_lemmas = {
        lemma
        for node in nodes
        for lemma in node.get("lemmas", [])
        if isinstance(lemma, str)
    }
    tokens = generate_tokens(
        count=args.adjective_count,
        existing_lemmas=existing_lemmas,
        seed=args.seed,
        unicode_start=args.unicode_start,
        unicode_end=args.unicode_end,
        min_length=args.token_min_length,
        max_length=args.token_max_length,
        mean_length=args.token_mean_length,
        std_length=args.token_std_length,
    )
    adjective_nodes = build_virtual_adjective_nodes(tokens)

    rng = random.Random(args.seed)
    concept_pool = select_concepts(nodes, edges, rng)
    if len(set(concept_pool)) < args.max_links_per_adjective:
        raise ValueError("Not enough concepts to attach virtual adjectives.")

    new_edges: List[Dict[str, str]] = []
    concept_cursor = 0
    for adjective in adjective_nodes:
        link_count = rng.randint(args.min_links_per_adjective, args.max_links_per_adjective)
        chosen: Set[str] = set()
        while len(chosen) < link_count:
            candidate = concept_pool[concept_cursor % len(concept_pool)]
            concept_cursor += 1
            if candidate not in existing_node_ids:
                continue
            chosen.add(candidate)

        for concept_id in sorted(chosen):
            new_edges.append({
                "source": concept_id,
                "relation": "HasProperty",
                "target": adjective["id"],
                "source_type": "virtual_adjective",
            })

    link_counts = Counter(edge["target"] for edge in new_edges)
    output_nodes = nodes + adjective_nodes
    output_edges = edges + new_edges
    new_metadata = {
        **metadata,
        "virtual_adjectives": {
            "script": "add_virtual_adjectives.py",
            "input_synsets": str(args.synsets),
            "input_edges": str(args.edges),
            "seed": args.seed,
            "adjective_count": args.adjective_count,
            "min_links_per_adjective": args.min_links_per_adjective,
            "max_links_per_adjective": args.max_links_per_adjective,
            "relation": "HasProperty",
            "source_type": "virtual_adjective",
            "unicode_start": args.unicode_start,
            "unicode_end": args.unicode_end,
            "token_min_length": args.token_min_length,
            "token_max_length": args.token_max_length,
            "added_edge_count": len(new_edges),
            "link_count_distribution": dict(sorted(Counter(link_counts.values()).items())),
        },
        "virtual_adjective_summary": summarize(output_nodes, output_edges, adjective_nodes, new_edges),
    }

    return {
        "nodes": output_nodes,
        "edges": output_edges,
        "metadata": new_metadata,
        "adjective_nodes": adjective_nodes,
        "adjective_edges": new_edges,
    }


def summarize(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    adjective_nodes: List[Dict[str, Any]],
    adjective_edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    neighbors = neighbors_from_edges(nodes, edges)
    adjective_ids = {node["id"] for node in adjective_nodes}
    adjective_degrees = {
        node_id: len(neighbors[node_id])
        for node_id in adjective_ids
    }
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "virtual_adjective_count": len(adjective_nodes),
        "virtual_adjective_edge_count": len(adjective_edges),
        "virtual_adjective_degree_min": min(adjective_degrees.values()) if adjective_degrees else 0,
        "virtual_adjective_degree_max": max(adjective_degrees.values()) if adjective_degrees else 0,
        "virtual_adjective_degree_counts": dict(sorted(Counter(adjective_degrees.values()).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add artificial adjective nodes connected with HasProperty edges."
    )
    parser.add_argument("--synsets", type=Path, default=DEFAULT_SYNSETS)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adjective-count", type=int, default=200)
    parser.add_argument("--min-links-per-adjective", type=int, default=3)
    parser.add_argument("--max-links-per-adjective", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--unicode-start", type=parse_codepoint, default=DEFAULT_UNICODE_START)
    parser.add_argument("--unicode-end", type=parse_codepoint, default=DEFAULT_UNICODE_END)
    parser.add_argument("--token-min-length", type=int, default=2)
    parser.add_argument("--token-max-length", type=int, default=6)
    parser.add_argument("--token-mean-length", type=float, default=3.8)
    parser.add_argument("--token-std-length", type=float, default=1.1)
    args = parser.parse_args()
    if args.min_links_per_adjective <= 0:
        raise ValueError("--min-links-per-adjective must be positive.")
    if args.min_links_per_adjective > args.max_links_per_adjective:
        raise ValueError("--min-links-per-adjective must be <= --max-links-per-adjective.")
    return args


def main() -> None:
    args = parse_args()
    data = add_virtual_adjectives(args)
    write_json(args.output_dir / "synsets.json", data["nodes"])
    write_json(args.output_dir / "edges.json", data["edges"])
    write_json(args.output_dir / "metadata.json", data["metadata"])
    summary = data["metadata"]["virtual_adjective_summary"]
    print(f"Wrote {args.output_dir / 'synsets.json'}")
    print(f"Wrote {args.output_dir / 'edges.json'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")
    print(f"virtual_adjectives={summary['virtual_adjective_count']}")
    print(f"virtual_adjective_edges={summary['virtual_adjective_edge_count']}")
    print(
        "virtual_adjective_degree_range="
        f"{summary['virtual_adjective_degree_min']}..{summary['virtual_adjective_degree_max']}"
    )


if __name__ == "__main__":
    main()
