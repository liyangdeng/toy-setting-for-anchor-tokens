import argparse
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote


DEFAULT_ROOTS = [
    "person.n.01",
    "animal.n.01",
    "plant.n.02",
    "artifact.n.01",
    "food.n.01",
    "body_part.n.01",
    "location.n.01",
]

WORDNET_RELATIONS = [
    "hypernym",
    "hyponym",
    "instance_hypernym",
    "instance_hyponym",
    "part_meronym",
    "member_meronym",
    "substance_meronym",
    "part_holonym",
    "member_holonym",
    "substance_holonym",
]

WORDNET_RELATION_PRIORITY = {
    "hypernym": 0,
    "hyponym": 1,
    "instance_hypernym": 2,
    "instance_hyponym": 3,
    "part_meronym": 4,
    "member_meronym": 5,
    "substance_meronym": 6,
    "part_holonym": 7,
    "member_holonym": 8,
    "substance_holonym": 9,
}

CONCEPTNET_RELATION_PRIORITY = {
    "Causes": 0,
    "CausesDesire": 1,
    "Entails": 2,
    "HasPrerequisite": 3,
    "HasFirstSubevent": 4,
    "HasSubevent": 5,
    "HasLastSubevent": 6,
    "CapableOf": 7,
    "NotCapableOf": 8,
    "ReceivesAction": 9,
    "UsedFor": 10,
    "MadeOf": 11,
    "HasProperty": 12,
    "NotHasProperty": 13,
    "AtLocation": 14,
    "LocatedNear": 15,
    "Desires": 16,
    "NotDesires": 17,
    "MotivatedByGoal": 18,
    "CreatedBy": 19,
    "DefinedAs": 20,
    "DistinctFrom": 21,
    "Antonym": 22,
    "SimilarTo": 23,
    "HasContext": 24,
    "RelatedTo": 100,
}

EXCLUDED_CONCEPTNET_RELATIONS = {
    "IsA",
    "PartOf",
    "HasA",
    "Synonym",
    "FormOf",
    "DerivedFrom",
    "EtymologicallyRelatedTo",
    "EtymologicallyDerivedFrom",
    "MannerOf",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_wordnet() -> Tuple[Any, Dict[str, Callable[[Any], Sequence[Any]]]]:
    try:
        from nltk.corpus import wordnet as wn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: nltk. Install it with `python3 -m pip install nltk`."
        ) from exc

    try:
        wn.synset("entity.n.01")
    except LookupError as exc:
        raise SystemExit(
            "Missing NLTK WordNet data. Run `python3 -m nltk.downloader wordnet omw-1.4`."
        ) from exc

    return wn, {
        "hypernym": lambda s: s.hypernyms(),
        "hyponym": lambda s: s.hyponyms(),
        "instance_hypernym": lambda s: s.instance_hypernyms(),
        "instance_hyponym": lambda s: s.instance_hyponyms(),
        "part_meronym": lambda s: s.part_meronyms(),
        "member_meronym": lambda s: s.member_meronyms(),
        "substance_meronym": lambda s: s.substance_meronyms(),
        "part_holonym": lambda s: s.part_holonyms(),
        "member_holonym": lambda s: s.member_holonyms(),
        "substance_holonym": lambda s: s.substance_holonyms(),
    }


def normalize_lemma(raw: str) -> str:
    return unquote(raw).replace("_", " ").lower().strip()


def clean_lemma(raw: str) -> Optional[str]:
    lemma = normalize_lemma(raw)
    tokens = lemma.split()
    if not tokens or len(tokens) > 3:
        return None
    if any(not token.isalpha() or token != token.lower() for token in tokens):
        return None
    return " ".join(tokens)


def synset_id(synset: Any) -> str:
    return synset.name()


def noun_synset(synset: Any) -> bool:
    return synset.pos() == "n"


def acceptable_lemmas(synset: Any) -> List[str]:
    lemmas: List[str] = []
    seen: Set[str] = set()
    for lemma in synset.lemma_names():
        cleaned = clean_lemma(lemma)
        if cleaned is not None and cleaned not in seen:
            lemmas.append(cleaned)
            seen.add(cleaned)
    return lemmas


def build_neighbor_cache(
    extractors: Dict[str, Callable[[Any], Sequence[Any]]],
) -> Callable[[Any], Dict[str, List[Any]]]:
    cache: Dict[str, Dict[str, List[Any]]] = {}

    def neighbors_by_relation(synset: Any) -> Dict[str, List[Any]]:
        sid = synset_id(synset)
        if sid in cache:
            return cache[sid]
        related: Dict[str, List[Any]] = {}
        for relation in WORDNET_RELATIONS:
            try:
                targets = [target for target in extractors[relation](synset) if noun_synset(target)]
            except Exception:
                targets = []
            if targets:
                related[relation] = sorted(targets, key=synset_id)
        cache[sid] = related
        return related

    return neighbors_by_relation


def wordnet_connectivity_score(
    synset: Any,
    neighbors_by_relation: Callable[[Any], Dict[str, List[Any]]],
) -> int:
    return sum(len(targets) for targets in neighbors_by_relation(synset).values())


def edge_key(source: str, relation: str, target: str) -> Tuple[str, str, str]:
    return source, relation, target


def pair_key(source: str, target: str) -> Tuple[str, str]:
    return tuple(sorted((source, target)))


def add_edge(
    edges: List[Dict[str, str]],
    seen_edges: Set[Tuple[str, str, str]],
    seen_pairs: Set[Tuple[str, str]],
    neighbors: Dict[str, Set[str]],
    source: str,
    relation: str,
    target: str,
    source_type: str,
    max_degree: int,
    one_edge_per_pair: bool = True,
) -> bool:
    if source == target:
        return False
    key = edge_key(source, relation, target)
    if key in seen_edges:
        return False
    pkey = pair_key(source, target)
    if one_edge_per_pair and pkey in seen_pairs:
        return False
    new_source_neighbor = target not in neighbors[source]
    new_target_neighbor = source not in neighbors[target]
    if new_source_neighbor and len(neighbors[source]) >= max_degree:
        return False
    if new_target_neighbor and len(neighbors[target]) >= max_degree:
        return False

    edges.append({
        "source": source,
        "relation": relation,
        "target": target,
        "source_type": source_type,
    })
    seen_edges.add(key)
    seen_pairs.add(pkey)
    neighbors[source].add(target)
    neighbors[target].add(source)
    return True


def add_wordnet_node(
    nodes: Dict[str, Dict[str, Any]],
    used_lemmas: Set[str],
    synset: Any,
    root_id: str,
    depth: int,
) -> bool:
    sid = synset_id(synset)
    if sid in nodes:
        nodes[sid]["root_sources"] = sorted(set(nodes[sid]["root_sources"]) | {root_id})
        nodes[sid]["min_wordnet_depth"] = min(nodes[sid]["min_wordnet_depth"], depth)
        return True

    lemmas = acceptable_lemmas(synset)
    if not lemmas or any(lemma in used_lemmas for lemma in lemmas):
        return False

    nodes[sid] = {
        "id": sid,
        "node_type": "wordnet_synset",
        "synset_id": sid,
        "pos": "noun",
        "primary_lemma": lemmas[0],
        "lemmas": lemmas,
        "definition": synset.definition(),
        "examples": list(synset.examples()),
        "root_sources": [root_id],
        "min_wordnet_depth": depth,
    }
    used_lemmas.update(lemmas)
    return True


def remove_wordnet_node(
    nodes: Dict[str, Dict[str, Any]],
    used_lemmas: Set[str],
    node_id: str,
) -> None:
    node = nodes.pop(node_id, None)
    if node is None:
        return
    for lemma in node.get("lemmas", []):
        used_lemmas.discard(lemma)


def build_wordnet_backbone(args: argparse.Namespace) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]]:
    wn, extractors = load_wordnet()
    neighbors_by_relation = build_neighbor_cache(extractors)
    roots = [wn.synset(root_id) for root_id in args.root]

    nodes: Dict[str, Dict[str, Any]] = {}
    used_lemmas: Set[str] = set()
    edges: List[Dict[str, str]] = []
    seen_edges: Set[Tuple[str, str, str]] = set()
    seen_pairs: Set[Tuple[str, str]] = set()
    graph_neighbors: Dict[str, Set[str]] = defaultdict(set)
    queue = deque()

    for root in roots:
        if not noun_synset(root):
            raise ValueError(f"{synset_id(root)} is not a noun synset.")
        if add_wordnet_node(nodes, used_lemmas, root, synset_id(root), 0):
            queue.append((root, synset_id(root), 0))

    visited: Set[Tuple[str, str]] = set()
    rng = random.Random(args.seed)
    while queue and len(nodes) < args.target_wordnet_nodes:
        current, root_id, depth = queue.popleft()
        current_id = synset_id(current)
        visit_key = (root_id, current_id)
        if visit_key in visited:
            continue
        visited.add(visit_key)
        if len(graph_neighbors[current_id]) >= args.wordnet_max_degree:
            continue
        if depth >= args.wordnet_max_depth:
            continue

        relation_items = sorted(
            neighbors_by_relation(current).items(),
            key=lambda item: WORDNET_RELATION_PRIORITY[item[0]],
        )
        for relation, targets in relation_items:
            shuffled = list(targets)
            rng.shuffle(shuffled)
            shuffled.sort(
                key=lambda target: (
                    -wordnet_connectivity_score(target, neighbors_by_relation),
                    synset_id(target),
                )
            )
            for target in shuffled:
                if len(graph_neighbors[current_id]) >= args.wordnet_max_degree:
                    break
                if len(nodes) >= args.target_wordnet_nodes:
                    break
                target_id = synset_id(target)
                if not noun_synset(target):
                    continue
                added_node = False
                if target_id not in nodes:
                    if not add_wordnet_node(nodes, used_lemmas, target, root_id, depth + 1):
                        continue
                    added_node = True
                if add_edge(
                    edges=edges,
                    seen_edges=seen_edges,
                    seen_pairs=seen_pairs,
                    neighbors=graph_neighbors,
                    source=current_id,
                    relation=relation,
                    target=target_id,
                    source_type="wordnet",
                    max_degree=args.wordnet_max_degree,
                ):
                    queue.append((target, root_id, depth + 1))
                elif added_node:
                    remove_wordnet_node(nodes, used_lemmas, target_id)

    selected_ids = set(nodes)
    for current_id in sorted(selected_ids):
        if len(graph_neighbors[current_id]) >= args.wordnet_max_degree:
            continue
        current = wn.synset(current_id)
        relation_items = sorted(
            neighbors_by_relation(current).items(),
            key=lambda item: WORDNET_RELATION_PRIORITY[item[0]],
        )
        for relation, targets in relation_items:
            sorted_targets = sorted(
                (target for target in targets if synset_id(target) in selected_ids),
                key=lambda target: (
                    -wordnet_connectivity_score(target, neighbors_by_relation),
                    synset_id(target),
                ),
            )
            for target in sorted_targets:
                if len(graph_neighbors[current_id]) >= args.wordnet_max_degree:
                    break
                add_edge(
                    edges=edges,
                    seen_edges=seen_edges,
                    seen_pairs=seen_pairs,
                    neighbors=graph_neighbors,
                    source=current_id,
                    relation=relation,
                    target=synset_id(target),
                    source_type="wordnet",
                    max_degree=args.wordnet_max_degree,
                    one_edge_per_pair=False,
                )

    metadata = {
        "requested_roots": args.root,
        "wordnet_relation_whitelist": WORDNET_RELATIONS,
        "excluded_from_wordnet_backbone": ["also_see", "similar_to"],
        "target_wordnet_nodes": args.target_wordnet_nodes,
        "wordnet_max_depth": args.wordnet_max_depth,
        "wordnet_max_degree": args.wordnet_max_degree,
        "wordnet_node_count": len(nodes),
        "wordnet_edge_count": len(edges),
    }
    return nodes, edges, metadata


def normalize_conceptnet_node(node: str) -> Optional[str]:
    parts = node.strip().split("/")
    if len(parts) < 4 or parts[1] != "c" or parts[2] != "en":
        return None
    return clean_lemma(parts[3])


def parse_weight(raw: str) -> float:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 1.0
    try:
        return float(data.get("weight", 1.0))
    except (TypeError, ValueError):
        return 1.0


def parse_conceptnet_line(line: str) -> Optional[Tuple[str, str, str, float, str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 4:
        return None
    relation = parts[1].split("/")[-1]
    source = normalize_conceptnet_node(parts[2])
    target = normalize_conceptnet_node(parts[3])
    if source is None or target is None:
        return None
    metadata = parts[4] if len(parts) >= 5 else "{}"
    return source, relation, target, parse_weight(metadata), metadata


def lemma_to_node_map(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for node_id, node in nodes.items():
        for lemma in node.get("lemmas", []):
            mapping[lemma] = node_id
    return mapping


def conceptnet_relation_allowed(relation: str, allow_related_to: bool = True) -> bool:
    if relation in EXCLUDED_CONCEPTNET_RELATIONS:
        return False
    if relation == "RelatedTo":
        return allow_related_to
    return relation in CONCEPTNET_RELATION_PRIORITY


def iter_conceptnet_candidates(
    conceptnet_path: Path,
    lemma_to_node: Dict[str, str],
    min_weight: float,
    exclude_wordnet_dataset: bool,
) -> Iterable[Dict[str, Any]]:
    with conceptnet_path.open(encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_conceptnet_line(line)
            if parsed is None:
                continue
            source_lemma, relation, target_lemma, weight, raw_metadata = parsed
            if weight < min_weight:
                continue
            if exclude_wordnet_dataset and "wordnet" in raw_metadata.lower():
                continue
            if not conceptnet_relation_allowed(relation):
                continue
            source_node = lemma_to_node.get(source_lemma)
            target_node = lemma_to_node.get(target_lemma)
            if source_node is None and target_node is None:
                continue
            if source_node is not None and target_node is not None and source_node == target_node:
                continue
            yield {
                "source": source_node,
                "relation": relation,
                "target": target_node,
                "source_lemma": source_lemma,
                "target_lemma": target_lemma,
                "weight": weight,
            }


def expand_with_conceptnet(
    args: argparse.Namespace,
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, str]],
) -> Dict[str, Any]:
    graph_neighbors = neighbors_from_edges(nodes, edges)
    seen_edges = {edge_key(edge["source"], edge["relation"], edge["target"]) for edge in edges}
    seen_pairs = {pair_key(edge["source"], edge["target"]) for edge in edges}
    lemma_to_node = lemma_to_node_map(nodes)
    existing_lemmas = set(lemma_to_node)
    expansion_node_limit = max(len(nodes), args.target_total_nodes - args.max_repair_nodes)

    candidates = list(iter_conceptnet_candidates(
        conceptnet_path=args.conceptnet,
        lemma_to_node=lemma_to_node,
        min_weight=args.min_weight,
        exclude_wordnet_dataset=args.exclude_wordnet_dataset,
    ))
    external_counts = Counter(
        candidate["source_lemma"] if candidate["source"] is None else candidate["target_lemma"]
        for candidate in candidates
        if (candidate["source"] is None) != (candidate["target"] is None)
    )

    def expansion_candidate_key(edge: Dict[str, Any]) -> Tuple[Any, ...]:
        both_known = edge["source"] is not None and edge["target"] is not None
        external_lemma = ""
        if edge["source"] is None:
            external_lemma = edge["source_lemma"]
        elif edge["target"] is None:
            external_lemma = edge["target_lemma"]
        return (
            0 if both_known else 1,
            -external_counts.get(external_lemma, 0),
            CONCEPTNET_RELATION_PRIORITY.get(edge["relation"], 99),
            -edge["weight"],
            len(graph_neighbors[edge["source"]]) if edge["source"] is not None else 0,
            len(graph_neighbors[edge["target"]]) if edge["target"] is not None else 0,
            edge["source"] or edge["source_lemma"],
            edge["target"] or edge["target_lemma"],
            edge["relation"],
        )

    candidates.sort(
        key=expansion_candidate_key
    )

    added = 0
    added_nodes = 0
    related_to_added = 0
    related_to_limit = min(
        args.relatedto_max_count,
        int(args.target_total_edges * args.relatedto_max_ratio),
    )
    for candidate in candidates:
        if len(edges) >= args.target_total_edges:
            break
        if candidate["relation"] == "RelatedTo":
            if related_to_added >= related_to_limit:
                continue
        source = candidate["source"]
        target = candidate["target"]
        created_lemma: Optional[str] = None
        created_node: Optional[str] = None
        if source is None and target is None:
            continue
        if source is None:
            if candidate["source_lemma"] not in lemma_to_node:
                if candidate["source_lemma"] in existing_lemmas or len(nodes) >= expansion_node_limit:
                    continue
                source = add_conceptnet_node(
                    nodes=nodes,
                    lemma_to_node=lemma_to_node,
                    graph_neighbors=graph_neighbors,
                    lemma=candidate["source_lemma"],
                    node_type="conceptnet_lemma",
                    prefix="conceptnet",
                    index=len(nodes) + 1,
                )
                existing_lemmas.add(candidate["source_lemma"])
                added_nodes += 1
                created_lemma = candidate["source_lemma"]
                created_node = source
            else:
                source = lemma_to_node[candidate["source_lemma"]]
        if target is None:
            if candidate["target_lemma"] not in lemma_to_node:
                if candidate["target_lemma"] in existing_lemmas or len(nodes) >= expansion_node_limit:
                    continue
                target = add_conceptnet_node(
                    nodes=nodes,
                    lemma_to_node=lemma_to_node,
                    graph_neighbors=graph_neighbors,
                    lemma=candidate["target_lemma"],
                    node_type="conceptnet_lemma",
                    prefix="conceptnet",
                    index=len(nodes) + 1,
                )
                existing_lemmas.add(candidate["target_lemma"])
                added_nodes += 1
                created_lemma = candidate["target_lemma"]
                created_node = target
            else:
                target = lemma_to_node[candidate["target_lemma"]]
        if source is None or target is None or source == target:
            continue
        added_edge = add_edge(
            edges=edges,
            seen_edges=seen_edges,
            seen_pairs=seen_pairs,
            neighbors=graph_neighbors,
            source=source,
            relation=candidate["relation"],
            target=target,
            source_type="conceptnet",
            max_degree=args.conceptnet_max_degree,
            one_edge_per_pair=False,
        )
        if added_edge:
            added += 1
            if candidate["relation"] == "RelatedTo":
                related_to_added += 1
        elif created_node is not None and created_lemma is not None:
            nodes.pop(created_node, None)
            graph_neighbors.pop(created_node, None)
            lemma_to_node.pop(created_lemma, None)
            existing_lemmas.discard(created_lemma)
            added_nodes -= 1

    second_pass_candidates = [
        candidate
        for candidate in iter_conceptnet_candidates(
            conceptnet_path=args.conceptnet,
            lemma_to_node=lemma_to_node,
            min_weight=args.min_weight,
            exclude_wordnet_dataset=args.exclude_wordnet_dataset,
        )
        if candidate["source"] is not None and candidate["target"] is not None
    ]
    second_pass_candidates.sort(
        key=lambda edge: (
            CONCEPTNET_RELATION_PRIORITY.get(edge["relation"], 99),
            -edge["weight"],
            len(graph_neighbors[edge["source"]]) + len(graph_neighbors[edge["target"]]),
            edge["source"],
            edge["target"],
            edge["relation"],
        )
    )

    second_pass_added = 0
    for candidate in second_pass_candidates:
        if len(edges) >= args.target_total_edges:
            break
        if candidate["relation"] == "RelatedTo" and related_to_added >= related_to_limit:
            continue
        added_edge = add_edge(
            edges=edges,
            seen_edges=seen_edges,
            seen_pairs=seen_pairs,
            neighbors=graph_neighbors,
            source=candidate["source"],
            relation=candidate["relation"],
            target=candidate["target"],
            source_type="conceptnet",
            max_degree=args.conceptnet_max_degree,
            one_edge_per_pair=False,
        )
        if added_edge:
            second_pass_added += 1
            if candidate["relation"] == "RelatedTo":
                related_to_added += 1

    return {
        "conceptnet_candidate_count": len(candidates),
        "conceptnet_edge_count": added,
        "conceptnet_second_pass_candidate_count": len(second_pass_candidates),
        "conceptnet_second_pass_edge_count": second_pass_added,
        "conceptnet_total_edge_count": added + second_pass_added,
        "conceptnet_node_count": added_nodes,
        "conceptnet_expansion_node_limit": expansion_node_limit,
        "relatedto_edge_count": related_to_added,
        "relatedto_limit": related_to_limit,
        "conceptnet_max_degree": args.conceptnet_max_degree,
        "excluded_conceptnet_relations": sorted(EXCLUDED_CONCEPTNET_RELATIONS),
        "relation_priority": CONCEPTNET_RELATION_PRIORITY,
        "exclude_wordnet_dataset": args.exclude_wordnet_dataset,
        "min_weight": args.min_weight,
    }


def neighbors_from_edges(
    nodes: Dict[str, Dict[str, Any]],
    edges: Iterable[Dict[str, str]],
) -> Dict[str, Set[str]]:
    neighbors: Dict[str, Set[str]] = defaultdict(set)
    for node_id in nodes:
        neighbors[node_id]
    for edge in edges:
        neighbors[edge["source"]].add(edge["target"])
        neighbors[edge["target"]].add(edge["source"])
    return neighbors


def conceptnet_node_id(lemma: str, index: int) -> str:
    slug = lemma.replace(" ", "_")
    return f"conceptnet:{slug}:{index}"


def add_conceptnet_node(
    nodes: Dict[str, Dict[str, Any]],
    lemma_to_node: Dict[str, str],
    graph_neighbors: Dict[str, Set[str]],
    lemma: str,
    node_type: str,
    prefix: str,
    index: int,
    repair_for: Optional[str] = None,
) -> str:
    node_id = f"{prefix}:{lemma.replace(' ', '_')}:{index}"
    while node_id in nodes:
        index += 1
        node_id = f"{prefix}:{lemma.replace(' ', '_')}:{index}"
    nodes[node_id] = {
        "id": node_id,
        "node_type": node_type,
        "synset_id": None,
        "pos": None,
        "primary_lemma": lemma,
        "lemmas": [lemma],
        "definition": None,
        "examples": [],
        "root_sources": [],
        "min_wordnet_depth": None,
        "repair_for": repair_for,
    }
    lemma_to_node[lemma] = node_id
    graph_neighbors[node_id]
    return node_id


def repair_low_degree_nodes(
    args: argparse.Namespace,
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, str]],
) -> Dict[str, Any]:
    graph_neighbors = neighbors_from_edges(nodes, edges)
    low_nodes = [
        node_id
        for node_id in sorted(nodes)
        if len(graph_neighbors[node_id]) < args.low_degree_target
    ]
    if not low_nodes:
        return {
            "initial_low_degree_count": 0,
            "repair_node_count": 0,
            "repair_edge_count": 0,
            "remaining_low_degree_count": 0,
        }

    low_lemma_to_node: Dict[str, str] = {}
    for node_id in low_nodes:
        for lemma in nodes[node_id].get("lemmas", []):
            low_lemma_to_node.setdefault(lemma, node_id)

    existing_lemmas = set(lemma_to_node_map(nodes))
    lemma_to_node = lemma_to_node_map(nodes)
    seen_edges = {edge_key(edge["source"], edge["relation"], edge["target"]) for edge in edges}
    seen_pairs = {pair_key(edge["source"], edge["target"]) for edge in edges}
    candidates_by_node: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    related_to_limit = min(
        args.relatedto_max_count,
        int(args.target_total_edges * args.relatedto_max_ratio),
    )
    related_to_count = sum(1 for edge in edges if edge["relation"] == "RelatedTo")

    with args.conceptnet.open(encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_conceptnet_line(line)
            if parsed is None:
                continue
            source_lemma, relation, target_lemma, weight, raw_metadata = parsed
            if weight < args.min_weight:
                continue
            if args.exclude_wordnet_dataset and "wordnet" in raw_metadata.lower():
                continue
            if not conceptnet_relation_allowed(relation):
                continue
            if source_lemma in low_lemma_to_node and target_lemma not in existing_lemmas:
                candidates_by_node[low_lemma_to_node[source_lemma]].append({
                    "external_lemma": target_lemma,
                    "relation": relation,
                    "direction": "forward",
                    "weight": weight,
                })
            if target_lemma in low_lemma_to_node and source_lemma not in existing_lemmas:
                candidates_by_node[low_lemma_to_node[target_lemma]].append({
                    "external_lemma": source_lemma,
                    "relation": relation,
                    "direction": "reverse",
                    "weight": weight,
                })

    for candidates in candidates_by_node.values():
        candidates.sort(
            key=lambda item: (
                CONCEPTNET_RELATION_PRIORITY.get(item["relation"], 99),
                -item["weight"],
                item["external_lemma"],
            )
        )

    repair_nodes = 0
    repair_edges = 0
    used_repair_lemmas: Set[str] = set()
    for node_id in low_nodes:
        while (
            len(graph_neighbors[node_id]) < args.low_degree_target
            and repair_nodes < args.max_repair_nodes
        ):
            candidate = next(
                (
                    item
                    for item in candidates_by_node.get(node_id, [])
                    if item["external_lemma"] not in existing_lemmas
                    and item["external_lemma"] not in used_repair_lemmas
                ),
                None,
            )
            if candidate is None:
                break

            if candidate["relation"] == "RelatedTo" and related_to_count >= related_to_limit:
                used_repair_lemmas.add(candidate["external_lemma"])
                continue
            repair_id = add_conceptnet_node(
                nodes=nodes,
                lemma_to_node=lemma_to_node,
                graph_neighbors=graph_neighbors,
                lemma=candidate["external_lemma"],
                node_type="conceptnet_lemma",
                prefix="repair",
                index=repair_nodes + 1,
                repair_for=node_id,
            )
            used_repair_lemmas.add(candidate["external_lemma"])
            existing_lemmas.add(candidate["external_lemma"])
            source = node_id if candidate["direction"] == "forward" else repair_id
            target = repair_id if candidate["direction"] == "forward" else node_id
            if add_edge(
                edges=edges,
                seen_edges=seen_edges,
                seen_pairs=seen_pairs,
                neighbors=graph_neighbors,
                source=source,
                relation=candidate["relation"],
                target=target,
                source_type="repair",
                max_degree=args.conceptnet_max_degree,
            ):
                repair_nodes += 1
                repair_edges += 1
                if candidate["relation"] == "RelatedTo":
                    related_to_count += 1
            else:
                nodes.pop(repair_id, None)
                lemma_to_node.pop(candidate["external_lemma"], None)
                break

    final_neighbors = neighbors_from_edges(nodes, edges)
    remaining_low = sum(
        1
        for node_id, node in nodes.items()
        if node.get("node_type") != "conceptnet_lemma"
        and len(final_neighbors[node_id]) < args.low_degree_target
    )
    return {
        "initial_low_degree_count": len(low_nodes),
        "repair_node_count": repair_nodes,
        "repair_edge_count": repair_edges,
        "max_repair_nodes": args.max_repair_nodes,
        "remaining_low_degree_count": remaining_low,
    }


def validate_graph(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    neighbors = neighbors_from_edges(nodes, edges)
    degree = {node_id: len(neighbor_ids) for node_id, neighbor_ids in neighbors.items()}
    lemmas = [
        lemma
        for node in nodes.values()
        for lemma in node.get("lemmas", [])
    ]
    edge_keys = [edge_key(edge["source"], edge["relation"], edge["target"]) for edge in edges]
    self_loops = [edge for edge in edges if edge["source"] == edge["target"]]
    missing_endpoint_edges = [
        edge for edge in edges if edge["source"] not in nodes or edge["target"] not in nodes
    ]
    relatedto_count = sum(1 for edge in edges if edge["relation"] == "RelatedTo")
    wordnet_degree_violations = [
        node_id
        for node_id, value in degree.items()
        if nodes[node_id].get("node_type") == "wordnet_synset"
        and value > args.conceptnet_max_degree
    ]

    return {
        "node_count": len(nodes),
        "wordnet_synset_count": sum(1 for node in nodes.values() if node["node_type"] == "wordnet_synset"),
        "conceptnet_lemma_node_count": sum(1 for node in nodes.values() if node["node_type"] == "conceptnet_lemma"),
        "edge_count": len(edges),
        "relation_counts": dict(sorted(Counter(edge["relation"] for edge in edges).items())),
        "source_type_counts": dict(sorted(Counter(edge["source_type"] for edge in edges).items())),
        "degree": {
            "min": min(degree.values()) if degree else 0,
            "max": max(degree.values()) if degree else 0,
            "mean": round(sum(degree.values()) / len(degree), 4) if degree else 0,
            "histogram": dict(sorted(Counter(degree.values()).items())),
            "nodes_below_2": sum(
                1
                for node_id, value in degree.items()
                if nodes[node_id].get("node_type") != "conceptnet_lemma" and value < 2
            ),
        },
        "relatedto_count": relatedto_count,
        "relatedto_ratio": round(relatedto_count / len(edges), 4) if edges else 0,
        "duplicate_lemma_count": len(lemmas) - len(set(lemmas)),
        "duplicate_edge_count": len(edge_keys) - len(set(edge_keys)),
        "self_loop_count": len(self_loops),
        "missing_endpoint_edge_count": len(missing_endpoint_edges),
        "degree_over_conceptnet_max_count": sum(
            1 for value in degree.values() if value > args.conceptnet_max_degree
        ),
        "wordnet_degree_over_conceptnet_max_count": len(wordnet_degree_violations),
    }


def parse_list(values: Optional[List[str]], default: List[str]) -> List[str]:
    if not values:
        return list(default)
    parsed: List[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a noun-first semantic backbone from WordNet and ConceptNet."
    )
    parser.add_argument("--conceptnet", type=Path, default=Path("assertions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("semantic_backbones/rebuilt"))
    parser.add_argument("--target-wordnet-nodes", type=int, default=1200)
    parser.add_argument("--target-total-nodes", type=int, default=2000)
    parser.add_argument("--target-total-edges", type=int, default=8000)
    parser.add_argument("--wordnet-max-depth", type=int, default=8)
    parser.add_argument("--wordnet-max-degree", type=int, default=6)
    parser.add_argument("--conceptnet-max-degree", type=int, default=12)
    parser.add_argument("--low-degree-target", type=int, default=2)
    parser.add_argument("--max-repair-nodes", type=int, default=50)
    parser.add_argument("--relatedto-max-ratio", type=float, default=0.2)
    parser.add_argument("--relatedto-max-count", type=int, default=1300)
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-wordnet-dataset", dest="exclude_wordnet_dataset", action="store_false")
    parser.set_defaults(exclude_wordnet_dataset=True)
    parser.add_argument("--root", action="append", help="WordNet noun root. Repeat or comma-separate.")
    args = parser.parse_args()
    args.root = parse_list(args.root, DEFAULT_ROOTS)
    return args


def main() -> None:
    args = parse_args()
    nodes, edges, wordnet_metadata = build_wordnet_backbone(args)
    conceptnet_metadata = expand_with_conceptnet(args, nodes, edges)
    repair_metadata = repair_low_degree_nodes(args, nodes, edges)
    validation = validate_graph(nodes, edges, args)

    metadata = {
        "parameters": {
            "conceptnet_path": str(args.conceptnet),
            "seed": args.seed,
            "target_total_edges": args.target_total_edges,
            "relatedto_max_ratio": args.relatedto_max_ratio,
            "relatedto_max_count": args.relatedto_max_count,
            "low_degree_target": args.low_degree_target,
        },
        "wordnet": wordnet_metadata,
        "conceptnet": conceptnet_metadata,
        "repair": repair_metadata,
        "validation": validation,
    }

    ordered_nodes = [nodes[node_id] for node_id in sorted(nodes)]
    ordered_edges = sorted(edges, key=lambda edge: (edge["source"], edge["relation"], edge["target"]))
    write_json(args.output_dir / "synsets.json", ordered_nodes)
    write_json(args.output_dir / "edges.json", ordered_edges)
    write_json(args.output_dir / "metadata.json", metadata)

    print(f"Wrote {args.output_dir / 'synsets.json'}")
    print(f"Wrote {args.output_dir / 'edges.json'}")
    print(f"Wrote {args.output_dir / 'metadata.json'}")
    print(f"nodes={validation['node_count']}")
    print(f"edges={validation['edge_count']}")
    print(f"degree_max={validation['degree']['max']}")
    print(f"nodes_below_2={validation['degree']['nodes_below_2']}")


if __name__ == "__main__":
    main()
