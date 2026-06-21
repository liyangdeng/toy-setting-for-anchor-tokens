#!/usr/bin/env python3
"""Build a synset-keyed POS inventory with unique artificial Unicode tokens."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from src.artificial_token_generator.extract_grammar_terminals import base_pos, load_grammar, parse_grammar


DEFAULT_SYNSETS_FILE = Path("semantic_backbones/kg_noun_only/synsets.json")
DEFAULT_GRAMMAR_FILE = Path("grammar/grammar_templates_extended.py")

type CodepointRange = tuple[int, int]
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


def synset_key(item: dict[str, object]) -> str:
    key = item.get("synset_id") or item.get("id")
    if not isinstance(key, str) or not key:
        raise ValueError(f"Synset item has no usable synset_id/id: {item}")
    return key


def collect_synset_entries(synsets_path: Path) -> JsonInventory:
    synsets = json.loads(synsets_path.read_text(encoding="utf-8"))
    if not isinstance(synsets, list):
        raise ValueError(f"{synsets_path} must contain a JSON list")

    inventory: JsonInventory = {}
    for item in synsets:
        if not isinstance(item, dict):
            continue
        key = synset_key(item)
        inventory[key] = {
            "POS": ["N"],
            "source": "synsets",
            "id": item.get("id"),
            "synset_id": item.get("synset_id"),
            "primary_lemma": item.get("primary_lemma"),
            "lemmas": item.get("lemmas", []),
        }
    return inventory


def collect_grammar_entries(grammar_path: Path) -> JsonInventory:
    grammar = load_grammar(grammar_path)
    terminals, _inline = parse_grammar(grammar)

    pos_by_terminal: dict[str, set[str]] = defaultdict(set)
    for terminal, pos_map in terminals.items():
        for pos in pos_map:
            pos_by_terminal[terminal].add(base_pos(pos))

    return {
        terminal: {
            "POS": sorted(pos_values),
            "source": "grammar",
            "id": terminal,
            "synset_id": None,
            "primary_lemma": terminal,
            "lemmas": [terminal],
        }
        for terminal, pos_values in sorted(pos_by_terminal.items())
    }


def merge_entries(*inventories: JsonInventory) -> JsonInventory:
    merged: JsonInventory = {}
    for inventory in inventories:
        for key, value in inventory.items():
            if key in merged:
                raise ValueError(f"Duplicate synset/grammar key: {key!r}")
            merged[key] = value
    return {key: merged[key] for key in sorted(merged)}


def attach_artificial_tokens(
    inventory: JsonInventory,
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
    mapped: JsonInventory = {}

    for key, entry in sorted(inventory.items()):
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
        mapped[key] = {
            **entry,
            "artificial": artificial,
        }

    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build synset/grammar-form -> POS/artificial-token JSON."
    )
    parser.add_argument("--synsets", type=Path, default=DEFAULT_SYNSETS_FILE)
    parser.add_argument("--grammar", type=Path, default=DEFAULT_GRAMMAR_FILE)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path. Default: synsets directory / synset_pos_artificial_<preset>.json",
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

    synset_entries = collect_synset_entries(args.synsets)
    grammar_entries = collect_grammar_entries(args.grammar)
    merged = merge_entries(synset_entries, grammar_entries)
    inventory = attach_artificial_tokens(
        merged,
        preset=args.preset,
        length_mean=args.length_mean,
        length_stddev=args.length_stddev,
        min_length=args.min_length,
        max_length=args.max_length,
        seed=args.seed,
    )

    output = args.output or args.synsets.with_name(f"synset_pos_artificial_{args.preset}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(inventory)} synset/grammar mappings to {output}")
    print(f"  synset entries: {len(synset_entries)}")
    print(f"  grammar original-form entries: {len(grammar_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
