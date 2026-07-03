#!/usr/bin/env python3
"""Evaluate seed-42 semantic-overlap models and create tables/plots."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-semantic-overlap-eval")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = ROOT / "experiments" / "semantic_overlap" / "v1_seed_42"
EVAL_DIR = ROOT / "experiments" / "semantic_overlap" / "evaluation_seed42"
WORD_SCRIPT = ROOT / "evaluation" / "word_trans_sent_retriev.py"
ACCURACY_SCRIPT = ROOT / "evaluation" / "accuracy.py"
CJK_DICT = (
    ROOT
    / "data"
    / "semantic_backbones"
    / "dict_to_artificial"
    / "dicts_synset_adj_augmented"
    / "synset_pos_artificial_cjk_edges_adj_augmented.json"
)
HIRAGANA_DICT = (
    ROOT
    / "data"
    / "semantic_backbones"
    / "dict_to_artificial"
    / "dicts_synset_adj_augmented"
    / "synset_pos_artificial_hiragana_edges_adj_augmented.json"
)
CONDITIONS = [
    "overlap_000",
    "overlap_025",
    "overlap_050",
    "overlap_075",
    "overlap_100",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(cmd, *, input_text=None, log_path=None):
    print("Running:", " ".join(str(part) for part in cmd))
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=ROOT,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    return result.stdout


def triple_key(record):
    return record["source"], record["relation"], record["target"]


def build_shared_parallel(condition_dir, output_path):
    l1_parallel = read_json(condition_dir / "corpus_build_l1" / "parallel_corpus_synset.json")
    l2_parallel = read_json(condition_dir / "corpus_build_l2" / "parallel_corpus_synset.json")
    l1_by_key = {triple_key(record): record for record in l1_parallel}
    l2_by_key = {triple_key(record): record for record in l2_parallel}
    shared = []
    for key in sorted(set(l1_by_key) & set(l2_by_key)):
        l1 = l1_by_key[key]
        l2 = l2_by_key[key]
        shared.append({
            "source": key[0],
            "relation": key[1],
            "target": key[2],
            "lang_a": l1["lang_a"],
            "lang_b": l2["lang_b"],
        })
    write_json(output_path, shared)
    return len(shared)


def parse_word_log(text):
    synset_pairs = int(re.search(r"Synset content-token pairs in joint vocab\s*:\s*(\d+)", text).group(1))
    values = [float(item) for item in re.findall(r"top-[15] precision\s*:\s*([0-9.]+)", text)]
    row = {
        "synset_pairs": synset_pairs,
        "word_top1": values[0],
        "word_top5": values[1],
    }
    sentence_match = re.search(r"Sentence pairs evaluated\s*:\s*(\d+)", text)
    if sentence_match:
        row["sentence_pairs"] = int(sentence_match.group(1))
        row["sent_top1"] = values[2]
        row["sent_top5"] = values[3]
    else:
        row["sentence_pairs"] = 0
        row["sent_top1"] = None
        row["sent_top5"] = None
    return row


def parse_accuracy_log(text):
    patterns = {
        "dev_sentences": r"Dev sentences:\s*(\d+)",
        "masked_tokens": r"Masked tokens evaluated:\s*(\d+)",
        "mlm_top1": r"top-1 accuracy:\s*([0-9.]+)",
        "mlm_top5": r"top-5 accuracy:\s*([0-9.]+)",
        "mlm_mrr": r"mrr:\s*([0-9.]+)",
    }
    out = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not parse {key} from accuracy output")
        value = match.group(1)
        out[key] = int(value) if key in {"dev_sentences", "masked_tokens"} else float(value)
    return out


def condition_overlap(condition):
    return int(condition.split("_")[1])


def plot_metric(rows, metric_names, title, ylabel, output_path):
    xs = [row["overlap_percent"] for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label in metric_names:
        ys = [row[metric] for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Semantic overlap (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sentence_metric(rows, output_path):
    xs = [row["overlap_percent"] for row in rows if row["sent_top1"] is not None]
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label in [("sent_top1", "Sentence top-1"), ("sent_top5", "Sentence top-5")]:
        ys = [row[metric] for row in rows if row[metric] is not None]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_title("Semantic-overlap sentence retrieval")
    ax.set_xlabel("Semantic overlap (%)")
    ax.set_ylabel("Precision")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    fieldnames = [
        "condition",
        "overlap_percent",
        "word_top1",
        "word_top5",
        "synset_pairs",
        "sent_top1",
        "sent_top5",
        "sentence_pairs",
        "mlm_top1",
        "mlm_top5",
        "mlm_mrr",
        "masked_tokens",
        "dev_sentences",
        "train_perplexity",
        "dev_perplexity",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def format_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(rows, output_path):
    lines = [
        "# Semantic Overlap Seed 42 Evaluation",
        "",
        "Evaluation uses repository scripts:",
        "",
        "- `evaluation/word_trans_sent_retriev.py` for word translation and sentence retrieval.",
        "- `evaluation/accuracy.py` for MLM dev top-k accuracy and MRR.",
        "",
        "Sentence retrieval is computed only over triples shared by L1 and L2. "
        "`overlap_000` has no shared triples, so sentence retrieval is N/A.",
        "",
        "## Alignment Metrics",
        "",
        "| Condition | Overlap | Word top-1 | Word top-5 | Synset pairs | Sentence top-1 | Sentence top-5 | Sentence pairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['condition']}` | {row['overlap_percent']}% | "
            f"{format_value(row['word_top1'])} | {format_value(row['word_top5'])} | "
            f"{row['synset_pairs']} | {format_value(row['sent_top1'])} | "
            f"{format_value(row['sent_top5'])} | {row['sentence_pairs']} |"
        )

    lines.extend([
        "",
        "## MLM Dev Accuracy",
        "",
        "| Condition | Overlap | MLM top-1 | MLM top-5 | MRR | Masked tokens | Dev sentences |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| `{row['condition']}` | {row['overlap_percent']}% | "
            f"{row['mlm_top1']:.4f} | {row['mlm_top5']:.4f} | {row['mlm_mrr']:.4f} | "
            f"{row['masked_tokens']} | {row['dev_sentences']} |"
        )

    lines.extend([
        "",
        "## Training Perplexity Reference",
        "",
        "| Condition | Train perplexity | Dev perplexity |",
        "| --- | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| `{row['condition']}` | {row['train_perplexity']:.4f} | {row['dev_perplexity']:.4f} |"
        )

    lines.extend([
        "",
        "## Visualizations",
        "",
        "- `visualizations/word_translation_precision.png`",
        "- `visualizations/sentence_retrieval_precision.png`",
        "- `visualizations/mlm_dev_accuracy.png`",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, default=SEED_DIR)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        condition_dir = args.seed_dir / condition
        model_dir = condition_dir / "model_multilingual" / "final"
        dev_file = condition_dir / "model_multilingual" / "dev.txt"
        metadata = read_json(condition_dir / "model_multilingual" / "training_metadata.json")

        parallel_path = args.output_dir / "parallel_shared" / f"{condition}_parallel_shared.json"
        sentence_pairs = build_shared_parallel(condition_dir, parallel_path)

        word_log = args.output_dir / f"{condition}_word_retrieval.log"
        if word_log.exists() and not args.force:
            word_text = word_log.read_text(encoding="utf-8")
        else:
            cmd = [
                sys.executable,
                WORD_SCRIPT,
                "--model",
                model_dir,
                "--cjk",
                CJK_DICT,
                "--hiragana",
                HIRAGANA_DICT,
                "--parallel",
                parallel_path,
                "--n_sample",
                args.n_sample,
                "--seed",
                args.seed,
            ]
            if sentence_pairs == 0:
                cmd.extend(["--test", "1"])
            word_text = run(cmd, log_path=word_log)
        row = {
            "condition": condition,
            "overlap_percent": condition_overlap(condition),
            **parse_word_log(word_text),
        }

        acc_log = args.output_dir / f"{condition}_mlm_accuracy.log"
        if acc_log.exists() and not args.force:
            acc_text = acc_log.read_text(encoding="utf-8")
        else:
            acc_text = run(
                [
                    sys.executable,
                    ACCURACY_SCRIPT,
                    "--model_dir",
                    model_dir,
                    "--dev_file",
                    dev_file,
                    "--batch_size",
                    metadata.get("batch_size", 64),
                    "--max_length",
                    metadata.get("max_length", 64),
                    "--mlm_prob",
                    metadata.get("mlm_prob", 0.15),
                    "--seed",
                    args.seed,
                ],
                input_text=f"semantic_{condition}_mlm_accuracy.txt\n",
                log_path=acc_log,
            )
        row.update(parse_accuracy_log(acc_text))
        row["train_perplexity"] = metadata["train_perplexity"]
        row["dev_perplexity"] = metadata["dev_perplexity"]
        rows.append(row)

    rows.sort(key=lambda row: row["overlap_percent"])
    write_json(args.output_dir / "semantic_overlap_seed42_evaluation.json", {"rows": rows})
    write_csv(args.output_dir / "semantic_overlap_seed42_evaluation.csv", rows)
    write_report(rows, args.output_dir / "semantic_overlap_seed42_evaluation.md")

    viz_dir = args.output_dir / "visualizations"
    plot_metric(
        rows,
        [("word_top1", "Word top-1"), ("word_top5", "Word top-5")],
        "Semantic-overlap word translation",
        "Precision",
        viz_dir / "word_translation_precision.png",
    )
    plot_sentence_metric(rows, viz_dir / "sentence_retrieval_precision.png")
    plot_metric(
        rows,
        [("mlm_top1", "MLM top-1"), ("mlm_top5", "MLM top-5"), ("mlm_mrr", "MLM MRR")],
        "Semantic-overlap MLM dev accuracy",
        "Score",
        viz_dir / "mlm_dev_accuracy.png",
    )
    print(f"Wrote {args.output_dir / 'semantic_overlap_seed42_evaluation.md'}")
    print(f"Wrote {args.output_dir / 'visualizations'}")


if __name__ == "__main__":
    main()
