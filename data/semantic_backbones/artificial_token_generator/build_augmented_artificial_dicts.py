#!/usr/bin/env python3
"""Augment artificial-token dictionaries with edge endpoints missing from them."""

import argparse
import json
import random
from pathlib import Path


PRESETS = {
    "hiragana": [(0x3041, 0x3096)],
    "cjk": [(0x4E00, 0x9FFF)],
}


def chars_for_preset(name):
    return [chr(codepoint) for start, end in PRESETS[name] for codepoint in range(start, end + 1)]


def random_token(rng, alphabet, used, min_length=2, max_length=6):
    while True:
        length = rng.randint(min_length, max_length)
        token = "".join(rng.choice(alphabet) for _ in range(length))
        if token not in used:
            used.add(token)
            return token


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, default=Path("data/semantic_backbones/edges_adj.json"))
    parser.add_argument("--base-cjk", type=Path, default=Path("data/semantic_backbones/dict_to_artificial/synset_pos_artificial_cjk.json"))
    parser.add_argument("--base-hiragana", type=Path, default=Path("data/semantic_backbones/dict_to_artificial/synset_pos_artificial_hiragana.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/graph_density/dicts"))
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    edges = read_json(args.edges)
    endpoints = sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})
    outputs = []
    for preset, base_path in [("cjk", args.base_cjk), ("hiragana", args.base_hiragana)]:
        dictionary = read_json(base_path)
        used = {entry["artificial"] for entry in dictionary.values()}
        rng = random.Random(args.seed + (0 if preset == "cjk" else 1))
        alphabet = chars_for_preset(preset)
        added = []
        for endpoint in endpoints:
            if endpoint in dictionary:
                continue
            pos = ["ADJ"] if endpoint.startswith("a_") else ["N"]
            dictionary[endpoint] = {
                "POS": pos,
                "source": "edge_endpoint_augmented",
                "id": endpoint,
                "synset_id": None,
                "primary_lemma": endpoint,
                "lemmas": [endpoint],
                "artificial": random_token(rng, alphabet, used),
            }
            added.append(endpoint)
        out_path = args.out_dir / f"synset_pos_artificial_{preset}_edges_adj_augmented.json"
        write_json(out_path, dictionary)
        outputs.append({
            "preset": preset,
            "base": str(base_path),
            "output": str(out_path),
            "base_entries": len(read_json(base_path)),
            "output_entries": len(dictionary),
            "added_entries": len(added),
            "added_keys": added,
        })

    write_json(args.out_dir / "metadata.json", {
        "edges": str(args.edges),
        "endpoint_count": len(endpoints),
        "outputs": outputs,
    })
    for output in outputs:
        print(f"{output['preset']}: added {output['added_entries']} -> {output['output']}")


if __name__ == "__main__":
    main()
