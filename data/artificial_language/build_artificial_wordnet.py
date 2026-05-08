import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import nltk
from nltk.corpus import wordnet as wn
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_BASE_WORDS = [
    "person", "man", "woman", "child", "family", "body", "head", "hand",
    "eye", "mouth", "food", "water", "fire", "earth", "air", "sun", "moon",
    "star", "day", "night", "animal", "dog", "cat", "bird", "fish", "tree",
    "plant", "seed", "leaf", "root", "stone", "mountain", "river", "sea",
    "rain", "snow", "wind", "house", "room", "door", "road", "tool", "knife",
    "clothing", "name", "word", "language", "sound", "music", "number",
    "one", "two", "many", "thing", "place", "time", "life", "death", "birth",
    "love", "fear", "anger", "joy", "pain", "good", "bad", "big", "small",
    "long", "short", "hot", "cold", "new", "old", "near", "far", "eat",
    "drink", "sleep", "walk", "run", "see", "hear", "speak", "know", "think",
    "give", "take", "make", "come", "go", "sit", "stand", "open", "close",
    "black", "white", "red", "green", "blue", "yellow",
]

# Unicode range chosen to be in the "Yi Syllables" block
DEFAULT_UNICODE_START = 0xA000
DEFAULT_UNICODE_END = 0xA4C6

DEFAULT_WORD_LENGTH_MEAN = 4.5
DEFAULT_WORD_LENGTH_STD = 1.25
DEFAULT_WORD_LENGTH_MIN = 2
DEFAULT_WORD_LENGTH_MAX = 9


def load_wordnet() -> Tuple[Any, Dict[str, Callable[[Any], Sequence[Any]]]]:
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


def read_base_words(path: Optional[Path]) -> List[str]:
    if path is None:
        return DEFAULT_BASE_WORDS

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def first_sense_for_word(wn: Any, word: str) -> Optional[Any]:
    senses = wn.synsets(word)
    if not senses:
        return None

    noun_senses = [sense for sense in senses if sense.pos() == "n"]
    return noun_senses[0] if noun_senses else senses[0]


def choose_base_synsets(
    wn: Any,
    base_size: int,
    seed: int,
    base_words_path: Optional[Path],
) -> Set[Any]:
    rng = random.Random(seed)
    selected: List[Any] = []
    seen: Set[str] = set()

    for word in read_base_words(base_words_path):
        sense = first_sense_for_word(wn, word)
        if sense is None or synset_id(sense) in seen:
            continue
        selected.append(sense)
        seen.add(synset_id(sense))

    if len(selected) < base_size:
        all_senses = list(wn.all_synsets())
        rng.shuffle(all_senses)
        for sense in all_senses:
            if synset_id(sense) in seen:
                continue
            selected.append(sense)
            seen.add(synset_id(sense))
            if len(selected) >= base_size:
                break

    return set(selected[:base_size])


def build_neighbor_cache(
    relation_extractors: Dict[str, Callable[[Any], Sequence[Any]]],
) -> Tuple[Callable[[Any], Dict[str, Set[Any]]], Callable[[Any], Set[Any]]]:
    cache: Dict[str, Dict[str, Set[Any]]] = {}

    def neighbors_by_relation(synset: Any) -> Dict[str, Set[Any]]:
        sid = synset_id(synset)
        if sid in cache:
            return cache[sid]

        related: Dict[str, Set[Any]] = {}
        for relation, extractor in relation_extractors.items():
            try:
                targets = set(extractor(synset))
            except Exception:
                targets = set()
            if targets:
                related[relation] = targets

        cache[sid] = related
        return related

    def all_neighbors(synset: Any) -> Set[Any]:
        neighbors: Set[Any] = set()
        for targets in neighbors_by_relation(synset).values():
            neighbors.update(targets)
        return neighbors

    return neighbors_by_relation, all_neighbors


def expand_candidates(
    anchors: Iterable[Any],
    all_neighbors: Callable[[Any], Set[Any]],
    excluded_ids: Set[str],
) -> Set[Any]:
    candidates: Set[Any] = set()
    for anchor in anchors:
        for neighbor in all_neighbors(anchor):
            if synset_id(neighbor) not in excluded_ids:
                candidates.add(neighbor)
    return candidates


def connectivity_score(
    candidate: Any,
    all_neighbors: Callable[[Any], Set[Any]],
    selected_ids: Set[str],
    candidate_ids: Set[str],
) -> int:
    selected_neighbor_count = 0
    candidate_neighbor_count = 0

    for neighbor in all_neighbors(candidate):
        nid = synset_id(neighbor)
        if nid in selected_ids:
            selected_neighbor_count += 1
        elif nid in candidate_ids:
            candidate_neighbor_count += 1

    return selected_neighbor_count * 5 + candidate_neighbor_count


def select_dense_layer(
    candidates: Set[Any],
    already_selected: Set[Any],
    all_neighbors: Callable[[Any], Set[Any]],
    size: int,
    seed: int,
    batch_size: int,
) -> Set[Any]:
    rng = random.Random(seed)
    chosen: Set[Any] = set()
    remaining: Dict[str, Any] = {synset_id(sense): sense for sense in candidates}
    selected_ids = {synset_id(sense) for sense in already_selected}

    while remaining and len(chosen) < size:
        candidate_ids = set(remaining)
        scored = [
            (
                connectivity_score(
                    candidate=sense,
                    all_neighbors=all_neighbors,
                    selected_ids=selected_ids,
                    candidate_ids=candidate_ids,
                ),
                synset_id(sense),
                sense,
            )
            for sense in remaining.values()
        ]
        rng.shuffle(scored)
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

        take = min(batch_size, size - len(chosen), len(scored))
        for _, _, sense in scored[:take]:
            chosen.add(sense)
            selected_ids.add(synset_id(sense))
            remaining.pop(synset_id(sense), None)

    return chosen


def sample_dense_subgraph(
    wn: Any,
    all_neighbors: Callable[[Any], Set[Any]],
    base_size: int,
    layer1_size: int,
    layer2_size: int,
    seed: int,
    base_words_path: Optional[Path],
    batch_size: int,
) -> Tuple[Set[Any], Set[Any], Set[Any]]:
    base = choose_base_synsets(wn, base_size, seed, base_words_path)

    layer1_candidates = expand_candidates(
        anchors=base,
        all_neighbors=all_neighbors,
        excluded_ids={synset_id(sense) for sense in base},
    )
    layer1 = select_dense_layer(
        candidates=layer1_candidates,
        already_selected=base,
        all_neighbors=all_neighbors,
        size=layer1_size,
        seed=seed + 1,
        batch_size=batch_size,
    )

    selected_after_layer1 = base | layer1
    layer2_candidates = expand_candidates(
        anchors=selected_after_layer1,
        all_neighbors=all_neighbors,
        excluded_ids={synset_id(sense) for sense in selected_after_layer1},
    )
    layer2 = select_dense_layer(
        candidates=layer2_candidates,
        already_selected=selected_after_layer1,
        all_neighbors=all_neighbors,
        size=layer2_size,
        seed=seed + 2,
        batch_size=batch_size,
    )

    return base, layer1, layer2


def unicode_pool(start: int, end: int) -> List[str]:
    if start > end:
        raise ValueError("--unicode-start must be <= --unicode-end.")
    return [chr(codepoint) for codepoint in range(start, end + 1)]


def normal_word_length(
    rng: random.Random,
    mean: float,
    std: float,
    minimum: int,
    maximum: int,
) -> int:
    if minimum <= 0:
        raise ValueError("--word-length-min must be positive.")
    if minimum > maximum:
        raise ValueError("--word-length-min must be <= --word-length-max.")
    if std <= 0:
        raise ValueError("--word-length-std must be positive.")

    sampled = round(rng.gauss(mean, std))
    return max(minimum, min(maximum, sampled))


def generate_artificial_words(
    count: int,
    seed: int,
    unicode_start: int,
    unicode_end: int,
    word_length_mean: float,
    word_length_std: float,
    word_length_min: int,
    word_length_max: int,
) -> List[str]:
    rng = random.Random(seed)
    characters = unicode_pool(unicode_start, unicode_end)
    words: Set[str] = set()

    while len(words) < count:
        length = normal_word_length(
            rng=rng,
            mean=word_length_mean,
            std=word_length_std,
            minimum=word_length_min,
            maximum=word_length_max,
        )
        words.add("".join(rng.choice(characters) for _ in range(length)))

    return sorted(words)


def build_edges(
    sampled: Set[Any],
    neighbors_by_relation: Callable[[Any], Dict[str, Set[Any]]],
) -> List[Dict[str, str]]:
    sampled_ids = {synset_id(sense) for sense in sampled}
    edges: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for source in sorted(sampled, key=synset_id):
        for relation, targets in neighbors_by_relation(source).items():
            for target in targets:
                key = (synset_id(source), relation, synset_id(target))
                if synset_id(target) in sampled_ids and key not in seen:
                    edges.append({
                        "source_original": synset_id(source),
                        "relation": relation,
                        "target_original": synset_id(target),
                    })
                    seen.add(key)

    return edges


def node_record(synset: Any, artificial_word: str, layer: str) -> Dict[str, Any]:
    return {
        "id": artificial_word,
        "original_id": synset_id(synset),
        "layer": layer,
        "pos": synset.pos(),
        "original_lemmas": [lemma.name() for lemma in synset.lemmas()],
        "definition": synset.definition(),
        "examples": synset.examples(),
    }


def graph_summary(nodes: List[Dict[str, Any]], edges: List[Dict[str, str]]) -> Dict[str, Any]:
    degree = Counter()
    relation_counts = Counter(edge["relation"] for edge in edges)

    for edge in edges:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1

    node_count = len(nodes)
    possible_directed_edges = node_count * (node_count - 1)
    layer_counts = Counter(str(node["layer"]) for node in nodes)

    return {
        "node_count": node_count,
        "edge_count": len(edges),
        "average_degree": round(sum(degree.values()) / node_count, 4) if node_count else 0,
        "directed_density": round(len(edges) / possible_directed_edges, 8)
        if possible_directed_edges
        else 0,
        "isolated_node_count": sum(1 for node in nodes if degree[node["id"]] == 0),
        "layer_counts": dict(sorted(layer_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


def build_artificial_wordnet(args: argparse.Namespace) -> Dict[str, Any]:
    wn, relation_extractors = load_wordnet()
    neighbors_by_relation, all_neighbors = build_neighbor_cache(relation_extractors)

    base, layer1, layer2 = sample_dense_subgraph(
        wn=wn,
        all_neighbors=all_neighbors,
        base_size=args.base_size,
        layer1_size=args.layer1_size,
        layer2_size=args.layer2_size,
        seed=args.seed,
        base_words_path=args.base_words,
        batch_size=args.batch_size,
    )

    sampled = base | layer1 | layer2
    sorted_sampled = sorted(sampled, key=synset_id)
    artificial_words = generate_artificial_words(
        count=len(sorted_sampled),
        seed=args.seed + 3,
        unicode_start=args.unicode_start,
        unicode_end=args.unicode_end,
        word_length_mean=args.word_length_mean,
        word_length_std=args.word_length_std,
        word_length_min=args.word_length_min,
        word_length_max=args.word_length_max,
    )

    original_to_artificial = {
        synset_id(sense): artificial_word
        for sense, artificial_word in zip(sorted_sampled, artificial_words)
    }
    artificial_to_original = {
        artificial_word: original_id
        for original_id, artificial_word in original_to_artificial.items()
    }

    layer_by_original = {
        **{synset_id(sense): "base" for sense in base},
        **{synset_id(sense): "layer1" for sense in layer1},
        **{synset_id(sense): "layer2" for sense in layer2},
    }

    original_edges = build_edges(sampled, neighbors_by_relation)
    artificial_edges = [
        {
            "source": original_to_artificial[edge["source_original"]],
            "relation": edge["relation"],
            "target": original_to_artificial[edge["target_original"]],
            "source_original": edge["source_original"],
            "target_original": edge["target_original"],
        }
        for edge in original_edges
    ]

    nodes = [
        node_record(
            synset=sense,
            artificial_word=original_to_artificial[synset_id(sense)],
            layer=layer_by_original[synset_id(sense)],
        )
        for sense in sorted_sampled
    ]

    return {
        "metadata": {
            "seed": args.seed,
            "requested": {
                "base_size": args.base_size,
                "layer1_size": args.layer1_size,
                "layer2_size": args.layer2_size,
            },
            "actual": {
                "base_size": len(base),
                "layer1_size": len(layer1),
                "layer2_size": len(layer2),
                "total_size": len(sampled),
            },
            "sampling_strategy": (
                "Start with basic WordNet synsets, expand relation-neighbor candidates, "
                "then greedily keep candidates with high connectivity to selected nodes "
                "and to the candidate pool."
            ),
            "relations_used": sorted(relation_extractors),
            "artificial_word_generator": {
                "type": "random_unicode_characters",
                "unicode_start": hex(args.unicode_start),
                "unicode_end": hex(args.unicode_end),
                "length_distribution": "normal",
                "word_length_mean": args.word_length_mean,
                "word_length_std": args.word_length_std,
                "word_length_min": args.word_length_min,
                "word_length_max": args.word_length_max,
                "unique_words": True,
            },
        },
        "summary": graph_summary(nodes, artificial_edges),
        "nodes": nodes,
        "edges": artificial_edges,
        "mappings": {
            "original_to_artificial": original_to_artificial,
            "artificial_to_original": artificial_to_original,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dense artificial-language WordNet sampled from NLTK WordNet."
    )
    parser.add_argument("--base-size", type=int, default=100)
    parser.add_argument("--layer1-size", type=int, default=5000)
    parser.add_argument("--layer2-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--base-words", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artificial_wordnet_unicode.json"))
    parser.add_argument("--unicode-start", type=lambda value: int(value, 0), default=DEFAULT_UNICODE_START)
    parser.add_argument("--unicode-end", type=lambda value: int(value, 0), default=DEFAULT_UNICODE_END)
    parser.add_argument("--word-length-mean", type=float, default=DEFAULT_WORD_LENGTH_MEAN)
    parser.add_argument("--word-length-std", type=float, default=DEFAULT_WORD_LENGTH_STD)
    parser.add_argument("--word-length-min", type=int, default=DEFAULT_WORD_LENGTH_MIN)
    parser.add_argument("--word-length-max", type=int, default=DEFAULT_WORD_LENGTH_MAX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_artificial_wordnet(args)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = data["summary"]
    print(f"Wrote {args.output}")
    print(f"nodes={summary['node_count']}")
    print(f"edges={summary['edge_count']}")
    print(f"average_degree={summary['average_degree']}")
    print(f"isolated_nodes={summary['isolated_node_count']}")


if __name__ == "__main__":
    main()
