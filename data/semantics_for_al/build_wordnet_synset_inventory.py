import argparse
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_NOUN_ROOTS = [
    "animal.n.01",
    "plant.n.02",
    "artifact.n.01",
    "food.n.01",
    "body_part.n.01",
    "location.n.01",
]

DEFAULT_VERB_ROOTS = [
    "move.v.02",
    "change.v.01",
    "consume.v.02",
    "communicate.v.02",
    "make.v.03",
    "touch.v.01",
]

DEFAULT_ADJ_ROOTS = [
    "hot.a.01",
    "cold.a.01",
    "large.a.01",
    "small.a.01",
    "new.a.01",
    "old.a.01",
    "good.a.01",
    "bad.a.01",
]

DEFAULT_TARGET_TOTAL = 512
DEFAULT_NOUN_COUNT = 400
DEFAULT_VERB_COUNT = 52
DEFAULT_ADJECTIVE_COUNT = 60

DEFAULT_RELATIONS = [
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
    "attribute",
    "similar_to",
    "also_see",
    "verb_group",
    "entailment",
    "cause",
]

POS_LABELS = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
}


def load_wordnet() -> Tuple[Any, Dict[str, Callable[[Any], Sequence[Any]]]]:
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ModuleNotFoundError as exc:
        if exc.name == "nltk":
            raise SystemExit(
                "Missing dependency: nltk. Install it with `python3 -m pip install nltk`."
            ) from exc
        raise

    try:
        wn.synsets("dog")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    relation_extractors = {
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
        "attribute": lambda s: s.attributes(),
        "similar_to": lambda s: s.similar_tos(),
        "also_see": lambda s: s.also_sees(),
        "verb_group": lambda s: s.verb_groups(),
        "entailment": lambda s: s.entailments(),
        "cause": lambda s: s.causes(),
    }
    return wn, relation_extractors


def synset_id(synset: Any) -> str:
    return synset.name()


def normalized_pos(synset_or_pos: Any) -> str:
    pos = synset_or_pos.pos() if hasattr(synset_or_pos, "pos") else str(synset_or_pos)
    if pos in ("a", "s"):
        return "adjective"
    if pos == "n":
        return "noun"
    if pos == "v":
        return "verb"
    return pos


def parse_list(values: Optional[List[str]], defaults: Iterable[str]) -> List[str]:
    if not values:
        return list(defaults)

    parsed: List[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def parse_relation_list(values: Optional[List[str]]) -> Set[str]:
    return set(parse_list(values, DEFAULT_RELATIONS))


def build_neighbor_cache(
    relation_extractors: Dict[str, Callable[[Any], Sequence[Any]]],
    allowed_relations: Set[str],
) -> Callable[[Any], Dict[str, Set[Any]]]:
    cache: Dict[str, Dict[str, Set[Any]]] = {}

    def neighbors_by_relation(synset: Any) -> Dict[str, Set[Any]]:
        sid = synset_id(synset)
        if sid in cache:
            return cache[sid]

        related: Dict[str, Set[Any]] = {}
        for relation in sorted(allowed_relations):
            extractor = relation_extractors.get(relation)
            if extractor is None:
                continue
            try:
                targets = set(extractor(synset))
            except Exception:
                targets = set()
            if targets:
                related[relation] = targets

        cache[sid] = related
        return related

    return neighbors_by_relation


def acceptable_surface_lemmas(synset: Any) -> List[str]:
    accepted: List[str] = []
    seen: Set[str] = set()

    for lemma in synset.lemmas():
        raw = lemma.name()
        if raw != raw.lower():
            continue
        if not raw.isalpha():
            continue
        if raw in seen:
            continue
        seen.add(raw)
        accepted.append(raw)

    return accepted


def same_pos_senses(wn: Any, lemma: str, pos_label: str) -> List[Any]:
    if pos_label == "adjective":
        return [
            sense
            for sense in wn.synsets(lemma)
            if normalized_pos(sense) == "adjective"
        ]
    if pos_label == "noun":
        return wn.synsets(lemma, pos="n")
    if pos_label == "verb":
        return wn.synsets(lemma, pos="v")
    return []


def lemma_sense_counts(wn: Any, lemma: str, pos_label: str) -> Dict[str, int]:
    return {
        "total": len(wn.synsets(lemma)),
        "same_pos": len(same_pos_senses(wn, lemma, pos_label)),
    }


def is_high_polysemy(
    wn: Any,
    lemma: str,
    pos_label: str,
    max_total_senses: int,
    max_same_pos_senses: int,
) -> bool:
    counts = lemma_sense_counts(wn, lemma, pos_label)
    return (
        counts["total"] > max_total_senses
        or counts["same_pos"] > max_same_pos_senses
    )


def target_counts_from_ratios(
    target_total: int,
    noun_ratio: float,
    verb_ratio: float,
    adjective_ratio: float,
) -> Dict[str, int]:
    ratios = {
        "noun": noun_ratio,
        "verb": verb_ratio,
        "adjective": adjective_ratio,
    }
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("At least one POS ratio must be positive.")

    exact = {
        pos: target_total * ratio / total_ratio
        for pos, ratio in ratios.items()
    }
    counts = {pos: int(value) for pos, value in exact.items()}
    remaining = target_total - sum(counts.values())
    remainders = sorted(
        ((exact[pos] - counts[pos], pos) for pos in counts),
        reverse=True,
    )

    for _, pos in remainders[:remaining]:
        counts[pos] += 1

    return counts


def target_counts_from_explicit_counts(
    target_total: int,
    noun_count: Optional[int],
    verb_count: Optional[int],
    adjective_count: Optional[int],
    noun_ratio: float,
    verb_ratio: float,
    adjective_ratio: float,
) -> Dict[str, int]:
    explicit_counts = {
        "noun": noun_count,
        "verb": verb_count,
        "adjective": adjective_count,
    }
    provided = {
        pos: count
        for pos, count in explicit_counts.items()
        if count is not None
    }
    if not provided:
        return target_counts_from_ratios(
            target_total=target_total,
            noun_ratio=noun_ratio,
            verb_ratio=verb_ratio,
            adjective_ratio=adjective_ratio,
        )

    if len(provided) != 3:
        missing = sorted(pos for pos, count in explicit_counts.items() if count is None)
        raise ValueError(
            "When using explicit POS counts, provide all three counts. "
            f"Missing: {missing}."
        )

    counts = {pos: int(count) for pos, count in provided.items()}
    if any(count < 0 for count in counts.values()):
        raise ValueError("Explicit POS counts must be non-negative.")
    if sum(counts.values()) != target_total:
        raise ValueError(
            "Explicit POS counts must sum to target_total. "
            f"Got {sum(counts.values())}, expected {target_total}."
        )
    return counts


def load_roots(
    wn: Any,
    noun_roots: List[str],
    verb_roots: List[str],
    adjective_roots: List[str],
) -> Dict[str, List[Any]]:
    roots_by_pos: Dict[str, List[Any]] = {
        "noun": [],
        "verb": [],
        "adjective": [],
    }
    requested = {
        "noun": noun_roots,
        "verb": verb_roots,
        "adjective": adjective_roots,
    }

    for pos_label, root_ids in requested.items():
        for root_id in root_ids:
            try:
                root = wn.synset(root_id)
            except Exception as exc:
                raise ValueError(f"Unknown root synset: {root_id}") from exc
            if normalized_pos(root) != pos_label:
                raise ValueError(
                    f"Root {root_id} has POS {normalized_pos(root)}, expected {pos_label}."
                )
            roots_by_pos[pos_label].append(root)

    return roots_by_pos


def bfs_candidates_for_pos(
    roots: List[Any],
    pos_label: str,
    max_depth: int,
    neighbors_by_relation: Callable[[Any], Dict[str, Set[Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Set[str]], Dict[str, int]]:
    candidates: Dict[str, Any] = {}
    root_sources: Dict[str, Set[str]] = defaultdict(set)
    min_depths: Dict[str, int] = {}
    queue = deque()

    for root in roots:
        rid = synset_id(root)
        queue.append((root, rid, 0))

    visited_by_root: Set[Tuple[str, str]] = set()
    while queue:
        current, root_id, depth = queue.popleft()
        sid = synset_id(current)
        key = (root_id, sid)
        if key in visited_by_root:
            continue
        visited_by_root.add(key)

        if normalized_pos(current) != pos_label:
            continue

        candidates[sid] = current
        root_sources[sid].add(root_id)
        min_depths[sid] = min(depth, min_depths.get(sid, depth))

        if depth >= max_depth:
            continue

        for targets in neighbors_by_relation(current).values():
            for target in targets:
                if normalized_pos(target) == pos_label:
                    queue.append((target, root_id, depth + 1))

    return candidates, root_sources, min_depths


def build_bfs_candidates(
    roots_by_pos: Dict[str, List[Any]],
    max_depths: Dict[str, int],
    neighbors_by_relation: Callable[[Any], Dict[str, Set[Any]]],
) -> Tuple[Dict[str, Any], Dict[str, Set[str]], Dict[str, int]]:
    candidates: Dict[str, Any] = {}
    root_sources: Dict[str, Set[str]] = defaultdict(set)
    min_depths: Dict[str, int] = {}

    for pos_label, roots in roots_by_pos.items():
        pos_candidates, pos_root_sources, pos_min_depths = bfs_candidates_for_pos(
            roots=roots,
            pos_label=pos_label,
            max_depth=max_depths[pos_label],
            neighbors_by_relation=neighbors_by_relation,
        )
        candidates.update(pos_candidates)
        for sid, roots_for_sid in pos_root_sources.items():
            root_sources[sid].update(roots_for_sid)
        for sid, depth in pos_min_depths.items():
            min_depths[sid] = min(depth, min_depths.get(sid, depth))

    return candidates, root_sources, min_depths


def filter_surface_forms(
    candidates: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[str]], Dict[str, str]]:
    kept: Dict[str, Any] = {}
    lemmas_by_synset: Dict[str, List[str]] = {}
    primary_lemma_by_synset: Dict[str, str] = {}

    for sid, synset in candidates.items():
        lemmas = acceptable_surface_lemmas(synset)
        if not lemmas:
            continue
        kept[sid] = synset
        lemmas_by_synset[sid] = lemmas
        primary_lemma_by_synset[sid] = lemmas[0]

    return kept, lemmas_by_synset, primary_lemma_by_synset


def filter_polysemy(
    wn: Any,
    candidates: Dict[str, Any],
    lemmas_by_synset: Dict[str, List[str]],
    primary_lemma_by_synset: Dict[str, str],
    max_total_senses: int,
    max_senses_by_pos: Dict[str, int],
    ambiguous_lemma_fraction: float,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Dict[str, int]]]]:
    kept: Dict[str, Any] = {}
    sense_counts: Dict[str, Dict[str, Dict[str, int]]] = {}

    for sid, synset in candidates.items():
        pos_label = normalized_pos(synset)
        max_same_pos_senses = max_senses_by_pos[pos_label]
        lemma_counts = {
            lemma: lemma_sense_counts(wn, lemma, pos_label)
            for lemma in lemmas_by_synset[sid]
        }
        sense_counts[sid] = lemma_counts

        primary = primary_lemma_by_synset[sid]
        primary_counts = lemma_counts[primary]
        primary_is_high = (
            primary_counts["total"] > max_total_senses
            or primary_counts["same_pos"] > max_same_pos_senses
        )
        if primary_is_high:
            continue

        ambiguous_count = sum(
            1
            for lemma in lemmas_by_synset[sid]
            if (
                lemma_counts[lemma]["total"] > max_total_senses
                or lemma_counts[lemma]["same_pos"] > max_same_pos_senses
            )
        )
        ambiguous_fraction = ambiguous_count / len(lemmas_by_synset[sid])
        if ambiguous_fraction >= ambiguous_lemma_fraction:
            continue

        kept[sid] = synset

    return kept, sense_counts


def build_edges(
    candidates: Dict[str, Any],
    neighbors_by_relation: Callable[[Any], Dict[str, Set[Any]]],
) -> List[Dict[str, str]]:
    candidate_ids = set(candidates)
    edges: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for sid in sorted(candidates):
        synset = candidates[sid]
        for relation, targets in neighbors_by_relation(synset).items():
            for target in targets:
                tid = synset_id(target)
                key = (sid, relation, tid)
                if tid in candidate_ids and key not in seen and sid != tid:
                    edges.append({
                        "source": sid,
                        "relation": relation,
                        "target": tid,
                    })
                    seen.add(key)

    return edges


def undirected_neighbors(edges: Iterable[Dict[str, str]]) -> Dict[str, Set[str]]:
    neighbors: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        neighbors[source].add(target)
        neighbors[target].add(source)
    return neighbors


def filter_by_degree(
    candidates: Dict[str, Any],
    edges: List[Dict[str, str]],
    min_degree: int,
    max_degree: Optional[int],
) -> Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, int]]:
    neighbors = undirected_neighbors(edges)
    degree = {sid: len(neighbors.get(sid, set())) for sid in candidates}
    kept_ids = {
        sid
        for sid, value in degree.items()
        if value >= min_degree and (max_degree is None or value <= max_degree)
    }
    kept_candidates = {
        sid: synset
        for sid, synset in candidates.items()
        if sid in kept_ids
    }
    kept_edges = [
        edge
        for edge in edges
        if edge["source"] in kept_ids and edge["target"] in kept_ids
    ]
    return kept_candidates, kept_edges, degree


def largest_connected_component(
    candidates: Dict[str, Any],
    edges: List[Dict[str, str]],
) -> Tuple[Dict[str, Any], List[Dict[str, str]], Set[str]]:
    if not candidates:
        return {}, [], set()

    neighbors = undirected_neighbors(edges)
    unseen = set(candidates)
    components: List[Set[str]] = []

    while unseen:
        start = min(unseen)
        queue = deque([start])
        component: Set[str] = set()
        unseen.remove(start)

        while queue:
            current = queue.popleft()
            component.add(current)
            for neighbor in neighbors.get(current, set()):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)

        components.append(component)

    largest = max(components, key=lambda item: (len(item), sorted(item)[0]))
    kept_candidates = {
        sid: synset
        for sid, synset in candidates.items()
        if sid in largest
    }
    kept_edges = [
        edge
        for edge in edges
        if edge["source"] in largest and edge["target"] in largest
    ]
    return kept_candidates, kept_edges, largest


def largest_connected_component_by_pos(
    candidates: Dict[str, Any],
    edges: List[Dict[str, str]],
) -> Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, int]]:
    kept_ids: Set[str] = set()
    component_sizes: Dict[str, int] = {}

    for pos_label in ("noun", "verb", "adjective"):
        pos_candidates = {
            sid: synset
            for sid, synset in candidates.items()
            if normalized_pos(synset) == pos_label
        }
        pos_edges = [
            edge
            for edge in edges
            if edge["source"] in pos_candidates and edge["target"] in pos_candidates
        ]
        _, _, pos_lcc_ids = largest_connected_component(pos_candidates, pos_edges)
        kept_ids.update(pos_lcc_ids)
        component_sizes[pos_label] = len(pos_lcc_ids)

    kept_candidates = {
        sid: synset
        for sid, synset in candidates.items()
        if sid in kept_ids
    }
    kept_edges = [
        edge
        for edge in edges
        if edge["source"] in kept_ids and edge["target"] in kept_ids
    ]
    return kept_candidates, kept_edges, component_sizes


def largest_connected_component_by_root(
    candidates: Dict[str, Any],
    edges: List[Dict[str, str]],
    root_sources: Dict[str, Set[str]],
) -> Tuple[Dict[str, Any], List[Dict[str, str]], Dict[str, int]]:
    kept_ids: Set[str] = set()
    component_sizes: Dict[str, int] = {}
    root_names = sorted({root for roots in root_sources.values() for root in roots})

    for root in root_names:
        root_candidates = {
            sid: synset
            for sid, synset in candidates.items()
            if root in root_sources.get(sid, set())
        }
        root_edges = [
            edge
            for edge in edges
            if edge["source"] in root_candidates and edge["target"] in root_candidates
        ]
        _, _, root_lcc_ids = largest_connected_component(root_candidates, root_edges)
        kept_ids.update(root_lcc_ids)
        component_sizes[root] = len(root_lcc_ids)

    kept_candidates = {
        sid: synset
        for sid, synset in candidates.items()
        if sid in kept_ids
    }
    kept_edges = [
        edge
        for edge in edges
        if edge["source"] in kept_ids and edge["target"] in kept_ids
    ]
    return kept_candidates, kept_edges, component_sizes


def choose_root_bucket(root_sources: Dict[str, Set[str]], sid: str) -> str:
    roots = sorted(root_sources.get(sid, []))
    return roots[0] if roots else "__unknown__"


def sample_synsets_by_pos(
    candidates: Dict[str, Any],
    edges: List[Dict[str, str]],
    target_counts: Dict[str, int],
    degree: Dict[str, int],
    root_sources: Dict[str, Set[str]],
    min_depths: Dict[str, int],
    seed: int,
) -> Set[str]:
    rng = random.Random(seed)
    selected: Set[str] = set()
    graph_neighbors = undirected_neighbors(edges)

    for pos_label in ("noun", "verb", "adjective"):
        pos_ids = [
            sid
            for sid, synset in candidates.items()
            if normalized_pos(synset) == pos_label
        ]
        buckets: Dict[str, List[str]] = defaultdict(list)
        for sid in pos_ids:
            buckets[choose_root_bucket(root_sources, sid)].append(sid)

        for bucket_ids in buckets.values():
            rng.shuffle(bucket_ids)
            bucket_ids.sort(
                key=lambda sid: (
                    -degree.get(sid, 0),
                    min_depths.get(sid, 999),
                    candidates[sid].lemma_names()[0],
                    sid,
                )
            )

        bucket_names = sorted(
            buckets,
            key=lambda name: (-len(buckets[name]), name),
        )
        target_count = target_counts[pos_label]
        offsets = Counter()
        selected_pos_ids: Set[str] = set()

        while len(selected_pos_ids) < target_count:
            added = False
            for bucket_name in bucket_names:
                remaining_ids = [
                    sid
                    for sid in buckets[bucket_name][offsets[bucket_name]:]
                    if sid not in selected_pos_ids
                ]
                if not remaining_ids:
                    continue

                remaining_ids.sort(
                    key=lambda sid: (
                        -len(graph_neighbors.get(sid, set()) & selected_pos_ids),
                        -degree.get(sid, 0),
                        min_depths.get(sid, 999),
                        candidates[sid].lemma_names()[0],
                        sid,
                    )
                )
                chosen = remaining_ids[0]
                selected.add(chosen)
                selected_pos_ids.add(chosen)
                buckets[bucket_name].remove(chosen)
                added = True
                if len(selected_pos_ids) >= target_count:
                    break
            if not added:
                break

    return selected


def degree_from_edges(node_ids: Set[str], edges: Iterable[Dict[str, str]]) -> Dict[str, int]:
    neighbors = undirected_neighbors(edges)
    return {sid: len(neighbors.get(sid, set())) for sid in node_ids}


def relations_for_nodes(
    node_ids: Set[str],
    edges: Iterable[Dict[str, str]],
) -> Dict[str, Dict[str, List[str]]]:
    relations: Dict[str, Dict[str, List[str]]] = {
        sid: defaultdict(list)
        for sid in node_ids
    }
    for edge in edges:
        if edge["source"] in node_ids and edge["target"] in node_ids:
            relations[edge["source"]][edge["relation"]].append(edge["target"])

    return {
        sid: {
            relation: sorted(targets)
            for relation, targets in relation_map.items()
        }
        for sid, relation_map in relations.items()
    }


def build_inventory_rows(
    selected_ids: Set[str],
    candidates: Dict[str, Any],
    lemmas_by_synset: Dict[str, List[str]],
    primary_lemma_by_synset: Dict[str, str],
    root_sources: Dict[str, Set[str]],
    min_depths: Dict[str, int],
    candidate_degree: Dict[str, int],
    final_degree: Dict[str, int],
    final_relations: Dict[str, Dict[str, List[str]]],
    sense_counts: Dict[str, Dict[str, Dict[str, int]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for sid in sorted(selected_ids):
        synset = candidates[sid]
        rows.append({
            "synset_id": sid,
            "pos": normalized_pos(synset),
            "wordnet_pos": synset.pos(),
            "primary_lemma": primary_lemma_by_synset[sid],
            "lemmas": lemmas_by_synset[sid],
            "definition": synset.definition(),
            "examples": synset.examples(),
            "root_sources": sorted(root_sources.get(sid, [])),
            "min_bfs_depth": min_depths.get(sid),
            "candidate_degree": candidate_degree.get(sid, 0),
            "final_degree": final_degree.get(sid, 0),
            "lemma_sense_counts": sense_counts.get(sid, {}),
            "relations": final_relations.get(sid, {}),
        })

    return rows


def summarize_distribution(
    selected_ids: Set[str],
    candidates: Dict[str, Any],
    root_sources: Dict[str, Set[str]],
) -> Dict[str, Any]:
    pos_counts = Counter(normalized_pos(candidates[sid]) for sid in selected_ids)
    root_counts = Counter(
        choose_root_bucket(root_sources, sid)
        for sid in selected_ids
    )
    dominant_root = None
    if root_counts:
        root, count = root_counts.most_common(1)[0]
        dominant_root = {
            "root": root,
            "count": count,
            "share": round(count / len(selected_ids), 4) if selected_ids else 0,
        }

    return {
        "pos_counts": dict(sorted(pos_counts.items())),
        "root_counts": dict(sorted(root_counts.items())),
        "dominant_root": dominant_root,
    }


def build_wordnet_synset_inventory(args: argparse.Namespace) -> Dict[str, Any]:
    wn, relation_extractors = load_wordnet()
    allowed_relations = parse_relation_list(args.relation)
    unknown_relations = allowed_relations - set(relation_extractors)
    if unknown_relations:
        raise ValueError(f"Unknown WordNet relations: {sorted(unknown_relations)}")

    neighbors_by_relation = build_neighbor_cache(relation_extractors, allowed_relations)
    roots_by_pos = load_roots(
        wn=wn,
        noun_roots=parse_list(args.noun_root, DEFAULT_NOUN_ROOTS),
        verb_roots=parse_list(args.verb_root, DEFAULT_VERB_ROOTS),
        adjective_roots=parse_list(args.adjective_root, DEFAULT_ADJ_ROOTS),
    )
    target_counts = target_counts_from_explicit_counts(
        target_total=args.target_total,
        noun_count=args.noun_count,
        verb_count=args.verb_count,
        adjective_count=args.adjective_count,
        noun_ratio=args.noun_ratio,
        verb_ratio=args.verb_ratio,
        adjective_ratio=args.adjective_ratio,
    )

    max_depths = {
        "noun": args.max_noun_depth,
        "verb": args.max_verb_depth,
        "adjective": args.max_adjective_depth,
    }
    candidates, root_sources, min_depths = build_bfs_candidates(
        roots_by_pos=roots_by_pos,
        max_depths=max_depths,
        neighbors_by_relation=neighbors_by_relation,
    )
    initial_candidate_count = len(candidates)

    surface_candidates, lemmas_by_synset, primary_lemma_by_synset = filter_surface_forms(
        candidates
    )
    after_surface_count = len(surface_candidates)

    max_senses_by_pos = {
        "noun": args.max_noun_senses,
        "verb": args.max_verb_senses,
        "adjective": args.max_adjective_senses,
    }
    polysemy_candidates, sense_counts = filter_polysemy(
        wn=wn,
        candidates=surface_candidates,
        lemmas_by_synset=lemmas_by_synset,
        primary_lemma_by_synset=primary_lemma_by_synset,
        max_total_senses=args.max_total_senses,
        max_senses_by_pos=max_senses_by_pos,
        ambiguous_lemma_fraction=args.ambiguous_lemma_fraction,
    )
    after_polysemy_count = len(polysemy_candidates)

    candidate_edges = build_edges(polysemy_candidates, neighbors_by_relation)
    degree_filtered_candidates, degree_filtered_edges, pre_degree = filter_by_degree(
        candidates=polysemy_candidates,
        edges=candidate_edges,
        min_degree=args.min_degree,
        max_degree=args.max_degree,
    )
    after_degree_count = len(degree_filtered_candidates)

    lcc_candidates: Dict[str, Any] = {}
    lcc_edges: List[Dict[str, str]] = []
    lcc_ids: Set[str] = set()
    per_pos_lcc_sizes: Dict[str, int] = {}
    per_root_lcc_sizes: Dict[str, int] = {}
    if args.component_mode == "global":
        lcc_candidates, lcc_edges, lcc_ids = largest_connected_component(
            candidates=degree_filtered_candidates,
            edges=degree_filtered_edges,
        )
    if args.component_mode == "per-pos":
        lcc_candidates, lcc_edges, per_pos_lcc_sizes = largest_connected_component_by_pos(
            candidates=degree_filtered_candidates,
            edges=degree_filtered_edges,
        )
        lcc_ids = set(lcc_candidates)
    if args.component_mode == "per-root":
        lcc_candidates, lcc_edges, per_root_lcc_sizes = largest_connected_component_by_root(
            candidates=degree_filtered_candidates,
            edges=degree_filtered_edges,
            root_sources=root_sources,
        )
        lcc_ids = set(lcc_candidates)
    lcc_degree = degree_from_edges(set(lcc_candidates), lcc_edges)

    selected_ids = sample_synsets_by_pos(
        candidates=lcc_candidates,
        edges=lcc_edges,
        target_counts=target_counts,
        degree=lcc_degree,
        root_sources=root_sources,
        min_depths=min_depths,
        seed=args.seed,
    )
    final_edges = [
        edge
        for edge in lcc_edges
        if edge["source"] in selected_ids and edge["target"] in selected_ids
    ]
    final_degree = degree_from_edges(selected_ids, final_edges)
    final_relations = relations_for_nodes(selected_ids, final_edges)
    inventory_rows = build_inventory_rows(
        selected_ids=selected_ids,
        candidates=lcc_candidates,
        lemmas_by_synset=lemmas_by_synset,
        primary_lemma_by_synset=primary_lemma_by_synset,
        root_sources=root_sources,
        min_depths=min_depths,
        candidate_degree=lcc_degree,
        final_degree=final_degree,
        final_relations=final_relations,
        sense_counts=sense_counts,
    )

    requested_roots = {
        pos: [synset_id(root) for root in roots]
        for pos, roots in roots_by_pos.items()
    }
    distribution = summarize_distribution(selected_ids, lcc_candidates, root_sources)
    relation_counts = Counter(edge["relation"] for edge in final_edges)
    warnings: List[str] = []
    if len(selected_ids) < args.target_total:
        warnings.append(
            "Final inventory is smaller than target_total because the filtered LCC "
            "did not contain enough synsets for one or more POS quotas."
        )
    dominant_root = distribution["dominant_root"]
    if dominant_root and dominant_root["share"] > args.max_root_share:
        warnings.append(
            f"Root distribution is narrow: {dominant_root['root']} covers "
            f"{dominant_root['share']:.2%} of final synsets."
        )

    report = {
        "metadata": {
            "seed": args.seed,
            "target_total": args.target_total,
            "target_counts": target_counts,
            "target_count_ranges": {
                "noun": [350, 400],
                "adjective": [40, 60],
                "verb": [30, 60],
            },
            "requested_roots": requested_roots,
            "max_depths": max_depths,
            "allowed_relations": sorted(allowed_relations),
            "surface_form_filter": (
                "Keep only lowercase single-token alphabetic WordNet lemma forms; "
                "drop spaces, underscores, hyphens, digits, symbols, and uppercase forms."
            ),
            "polysemy_filter": {
                "max_total_senses": args.max_total_senses,
                "max_same_pos_senses": max_senses_by_pos,
                "ambiguous_lemma_fraction": args.ambiguous_lemma_fraction,
            },
            "graph_filter": {
                "min_degree": args.min_degree,
                "max_degree": args.max_degree,
                "component_mode": args.component_mode,
            },
        },
        "counts": {
            "initial_candidates": initial_candidate_count,
            "after_surface_filter": after_surface_count,
            "after_polysemy_filter": after_polysemy_count,
            "candidate_edge_count": len(candidate_edges),
            "after_degree_filter": after_degree_count,
            "largest_component_size": len(lcc_ids),
            "per_pos_largest_component_sizes": per_pos_lcc_sizes,
            "per_root_largest_component_sizes": per_root_lcc_sizes,
            "largest_component_edge_count": len(lcc_edges),
            "final_synset_count": len(selected_ids),
            "final_edge_count": len(final_edges),
        },
        "summary": {
            "final_average_degree": round(sum(final_degree.values()) / len(selected_ids), 4)
            if selected_ids
            else 0,
            "final_isolated_synset_count": sum(
                1 for sid in selected_ids if final_degree.get(sid, 0) == 0
            ),
            "final_relation_counts": dict(sorted(relation_counts.items())),
            "distribution": distribution,
            "warnings": warnings,
        },
    }

    return {
        "inventory": inventory_rows,
        "edges": sorted(final_edges, key=lambda item: (item["source"], item["relation"], item["target"])),
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a WordNet synset-level concept inventory from configured roots, "
            "BFS expansion, surface-form filtering, polysemy filtering, graph "
            "degree filtering, largest connected component extraction, and POS-"
            "balanced sampling."
        )
    )
    parser.add_argument("--target-total", type=int, default=DEFAULT_TARGET_TOTAL)
    parser.add_argument("--noun-count", type=int)
    parser.add_argument("--verb-count", type=int)
    parser.add_argument("--adjective-count", type=int)
    parser.add_argument("--noun-ratio", type=float, default=0.5)
    parser.add_argument("--verb-ratio", type=float, default=0.25)
    parser.add_argument("--adjective-ratio", type=float, default=0.25)
    parser.add_argument("--max-noun-depth", type=int, default=3)
    parser.add_argument("--max-verb-depth", type=int, default=3)
    parser.add_argument("--max-adjective-depth", type=int, default=4)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument(
        "--max-degree",
        type=int,
        default=16,
        help=(
            "Drop candidate synsets whose undirected candidate-graph degree is "
            "above this value. Use a negative value to disable the cap."
        ),
    )
    parser.add_argument("--max-total-senses", type=int, default=10)
    parser.add_argument("--max-noun-senses", type=int, default=5)
    parser.add_argument("--max-verb-senses", type=int, default=4)
    parser.add_argument("--max-adjective-senses", type=int, default=3)
    parser.add_argument("--ambiguous-lemma-fraction", type=float, default=1.0)
    parser.add_argument("--max-root-share", type=float, default=0.5)
    parser.add_argument(
        "--component-mode",
        choices=["per-root", "per-pos", "global"],
        default="per-root",
        help=(
            "Use the largest connected component inside each root domain, inside "
            "each POS, or force one global LCC. Per-root is the default because "
            "the requested root domains are intentionally diverse."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noun-root",
        action="append",
        help="Noun root synset id. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--verb-root",
        action="append",
        help="Verb root synset id. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--adjective-root",
        action="append",
        help="Adjective root synset id. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--relation",
        action="append",
        help="Allowed WordNet relation. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        default=Path("wordnet_concept_synsets.json"),
    )
    parser.add_argument(
        "--edges-output",
        type=Path,
        default=Path("wordnet_concept_edges.json"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("wordnet_concept_metadata.json"),
    )
    args = parser.parse_args()
    explicit_pos_counts = (
        args.noun_count,
        args.verb_count,
        args.adjective_count,
    )
    if all(count is None for count in explicit_pos_counts) and args.target_total == DEFAULT_TARGET_TOTAL:
        args.noun_count = DEFAULT_NOUN_COUNT
        args.verb_count = DEFAULT_VERB_COUNT
        args.adjective_count = DEFAULT_ADJECTIVE_COUNT
    if args.max_degree is not None and args.max_degree < 0:
        args.max_degree = None
    return args


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    data = build_wordnet_synset_inventory(args)

    write_json(args.inventory_output, data["inventory"])
    write_json(args.edges_output, data["edges"])
    write_json(args.metadata_output, data["report"])

    report = data["report"]
    summary = report["summary"]
    print(f"Wrote {args.inventory_output}")
    print(f"Wrote {args.edges_output}")
    print(f"Wrote {args.metadata_output}")
    print(f"final_synset_count={report['counts']['final_synset_count']}")
    print(f"final_edge_count={report['counts']['final_edge_count']}")
    print(f"final_average_degree={summary['final_average_degree']}")
    print(f"final_isolated_synsets={summary['final_isolated_synset_count']}")
    for warning in summary["warnings"]:
        print(f"warning={warning}")


if __name__ == "__main__":
    main()
