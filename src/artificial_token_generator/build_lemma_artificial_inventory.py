#!/usr/bin/env python3
"""Build a lemma-keyed POS inventory with unique artificial Unicode tokens."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from extract_grammar_terminals import base_pos, load_grammar, parse_grammar


DEFAULT_SYNSETS_FILE = Path("semantic_backbones/kg_noun_only/synsets.json")
DEFAULT_GRAMMAR_FILE = Path("grammar/grammar_templates_extended.py")

type CodepointRange = tuple[int, int]
type PosIndex = dict[str, set[str]]
type JsonInventory = dict[str, dict[str, object]]

PRESETS: dict[str, list[CodepointRange]] = {
    "yi": [(0xA000, 0xA48C)],
    "hiragana": [(0x3041, 0x3096)],
    "cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "cjk": [(0x4E00, 0x9FFF)],
}


def chars_for_preset(name: str) -> list[str]:
    try:
        ranges = PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Choose one of: {choices}") from exc

    return [
        chr(codepoint)
        for start, end in ranges
        for codepoint in range(start, end + 1)
    ]


def sample_length(
    rng: random.Random,
    *,
    mean: float,
    stddev: float,
    min_length: int,
    max_length: int,
) -> int:
    if stddev <= 0:
        raise ValueError("stddev must be positive")
    if min_length < 1:
        raise ValueError("min_length must be at least 1")
    if max_length < min_length:
        raise ValueError("max_length must be greater than or equal to min_length")

    sampled = round(rng.gauss(mean, stddev))
    return min(max(sampled, min_length), max_length)


def random_token(rng: random.Random, alphabet: list[str], length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def collect_synset_lemmas(synsets_path: Path) -> PosIndex:
    synsets = json.loads(synsets_path.read_text(encoding="utf-8"))
    if not isinstance(synsets, list):
        raise ValueError(f"{synsets_path} must contain a JSON list")

    index: PosIndex = defaultdict(set)
    for item in synsets:
        if not isinstance(item, dict):
            continue
        for lemma in item.get("lemmas", []):
            if isinstance(lemma, str) and lemma.strip():
                index[lemma.strip()].add("N")
    return index


def collect_grammar_terms(grammar_path: Path) -> PosIndex:
    grammar = load_grammar(grammar_path)
    terminals, _inline = parse_grammar(grammar)

    index: PosIndex = defaultdict(set)
    for terminal, pos_map in terminals.items():
        for pos in pos_map:
            index[terminal].add(base_pos(pos))
    return index


def merge_pos_indexes(*indexes: PosIndex) -> PosIndex:
    merged: PosIndex = defaultdict(set)
    for index in indexes:
        for key, pos_values in index.items():
            merged[key].update(pos_values)
    return merged


def attach_artificial_tokens(
    index: PosIndex,
    *,
    preset: str,
    length_mean: float,
    length_stddev: float,
    min_length: int,
    max_length: int,
    seed: int,
) -> JsonInventory:
    alphabet = chars_for_preset(preset)
    if not alphabet:
        raise ValueError(f"Preset {preset!r} has no usable characters")

    rng = random.Random(seed)
    used: set[str] = set()
    inventory: JsonInventory = {}

    for lemma, pos_values in sorted(index.items()):
        while True:
            length = sample_length(
                rng,
                mean=length_mean,
                stddev=length_stddev,
                min_length=min_length,
                max_length=max_length,
            )
            artificial = random_token(rng, alphabet, length)
            if artificial not in used:
                break

        used.add(artificial)
        inventory[lemma] = {
            "POS": sorted(pos_values),
            "artificial": artificial,
        }

    return inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build lemma -> POS/artificial-token JSON from synset lemmas and grammar terminals."
    )
    parser.add_argument("--synsets", type=Path, default=DEFAULT_SYNSETS_FILE)
    parser.add_argument("--grammar", type=Path, default=DEFAULT_GRAMMAR_FILE)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path. Default: synsets directory / lemma_pos_artificial_<preset>.json",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="cjk",
        help="Unicode range preset to use. Default: cjk",
    )
    parser.add_argument("--length-mean", type=float, default=3.0)
    parser.add_argument("--length-stddev", type=float, default=1.0)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    synset_index = collect_synset_lemmas(args.synsets)
    grammar_index = collect_grammar_terms(args.grammar)
    merged = merge_pos_indexes(synset_index, grammar_index)
    inventory = attach_artificial_tokens(
        merged,
        preset=args.preset,
        length_mean=args.length_mean,
        length_stddev=args.length_stddev,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
    )

    output = args.output or args.synsets.with_name(f"lemma_pos_artificial_{args.preset}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(inventory)} lemma mappings to {output}")
    print(f"  synset lemmas: {len(synset_index)}")
    print(f"  grammar terms: {len(grammar_index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
