#!/usr/bin/env python3
"""Build artificial corpora from generated synset/id sentences using exact tokens."""

import argparse
import json
import re
from pathlib import Path


TOKEN_RE = re.compile(r"^([,.;:!?]*)(.*?)([,.;:!?]*)$")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def artificial_map(dictionary):
    return {key: value["artificial"] for key, value in dictionary.items()}


def replace_token(token, mapping, missing):
    match = TOKEN_RE.match(token)
    if not match:
        missing.add(token)
        return token
    _prefix, core, _suffix = match.groups()
    if not core:
        return ""
    if core in mapping:
        return mapping[core]
    missing.add(core)
    return core


def replace_sentence(sentence, mapping, missing):
    tokens = []
    for token in sentence.split():
        replaced = replace_token(token, mapping, missing)
        if replaced:
            tokens.append(replaced)
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--cjk", type=Path, required=True)
    parser.add_argument("--hiragana", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data = read_json(args.sentences)
    cjk = artificial_map(read_json(args.cjk))
    hira = artificial_map(read_json(args.hiragana))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cjk_lines = []
    hira_lines = []
    parallel = []
    missing_cjk = set()
    missing_hira = set()

    for result in data["results"]:
        cjk_sents = [replace_sentence(sentence, cjk, missing_cjk) for sentence in result["sentences"]]
        hira_sents = [replace_sentence(sentence, hira, missing_hira) for sentence in result["sentences"]]
        cjk_lines.extend(cjk_sents)
        hira_lines.extend(hira_sents)
        parallel.append({
            "source": result["source"],
            "relation": result["relation"],
            "target": result["target"],
            "lang_a": cjk_sents,
            "lang_b": hira_sents,
        })

    (args.out_dir / "corpus_cjk.txt").write_text("\n".join(cjk_lines), encoding="utf-8")
    (args.out_dir / "corpus_hiragana.txt").write_text("\n".join(hira_lines), encoding="utf-8")
    (args.out_dir / "parallel_corpus_synset.json").write_text(
        json.dumps(parallel, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "sentences": str(args.sentences),
        "cjk_dictionary": str(args.cjk),
        "hiragana_dictionary": str(args.hiragana),
        "records": len(data["results"]),
        "cjk_sentences": len(cjk_lines),
        "hiragana_sentences": len(hira_lines),
        "missing_cjk_count": len(missing_cjk),
        "missing_hiragana_count": len(missing_hira),
        "missing_cjk_sample": sorted(missing_cjk)[:50],
        "missing_hiragana_sample": sorted(missing_hira)[:50],
    }
    (args.out_dir / "corpus_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Sentences per language: {len(cjk_lines)}")
    print(f"Missing CJK tokens: {len(missing_cjk)}")
    print(f"Missing Hiragana tokens: {len(missing_hira)}")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
