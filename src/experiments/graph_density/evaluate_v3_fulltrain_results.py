#!/usr/bin/env python3
"""Evaluate v3 full-corpus graph-density models and summarize performance."""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "graph_density"
V3_DIR = EXP_DIR / "v3_adj_variants"
EVAL_DIR = V3_DIR / "evaluation_results_fulltrain"
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
    "medium_60_full",
    "high_80_full",
    "medium_60_relation",
    "high_80_relation",
    "full_graph",
]
SEEDS = [42, 43, 44]


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


def parse_word_log(text):
    synset_pairs = int(re.search(r"Synset content-token pairs in joint vocab\s*:\s*(\d+)", text).group(1))
    sentence_pairs = int(re.search(r"Sentence pairs evaluated\s*:\s*(\d+)", text).group(1))
    values = [float(item) for item in re.findall(r"top-[15] precision\s*:\s*([0-9.]+)", text)]
    if len(values) != 4:
        raise ValueError(f"Expected 4 precision values, found {len(values)}")
    return {
        "synset_pairs": synset_pairs,
        "word_top1": values[0],
        "word_top5": values[1],
        "sentence_pairs": sentence_pairs,
        "sent_top1": values[2],
        "sent_top5": values[3],
    }


def parse_accuracy_log(text):
    patterns = {
        "dev_sentences": r"Dev sentences:\s*(\d+)",
        "masked_tokens": r"Masked tokens evaluated:\s*(\d+)",
        "top1": r"top-1 accuracy:\s*([0-9.]+)",
        "top5": r"top-5 accuracy:\s*([0-9.]+)",
        "mrr": r"mrr:\s*([0-9.]+)",
    }
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not parse {key} from accuracy output.")
        value = match.group(1)
        parsed[key] = int(value) if key in {"dev_sentences", "masked_tokens"} else float(value)
    return parsed


def sample_sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows, metrics):
    by_condition = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    summary = []
    for condition in CONDITIONS:
        condition_rows = by_condition[condition]
        out = {"condition": condition, "seeds": [row["seed"] for row in condition_rows]}
        for metric in metrics:
            values = [row[metric] for row in condition_rows]
            out[f"{metric}_mean"] = statistics.mean(values)
            out[f"{metric}_sd"] = sample_sd(values)
        summary.append(out)
    return summary


def fmt(mean_value, sd_value):
    return f"{mean_value:.4f} +/- {sd_value:.4f}"


def write_summary(word_rows, mono_rows):
    word_summary = summarize(word_rows, ["word_top1", "word_top5", "sent_top1", "sent_top5"])
    mono_summary = summarize(mono_rows, ["top1", "top5", "mrr", "masked_tokens"])
    summary_json = {
        "title": "Graph Density V3 Full-Corpus Evaluation Summary",
        "word_retrieval": word_rows,
        "mono_accuracy": mono_rows,
        "word_retrieval_summary": word_summary,
        "mono_accuracy_summary": mono_summary,
    }
    (EVAL_DIR / "v3_fulltrain_evaluation_results.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Graph Density V3 Full-Corpus Evaluation Results",
        "",
        "Full-corpus runs do not use per-epoch downsampling. Each aggregate row is mean +/- sample standard deviation across seeds 42, 43, and 44.",
        "Sentence retrieval uses `--n_sample 500`.",
        "",
        "## Multilingual Alignment",
        "",
        "| Condition | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lookup = {row["condition"]: row for row in word_summary}
    for condition in CONDITIONS:
        row = lookup[condition]
        lines.append(
            f"| `{condition}` | "
            f"{fmt(row['word_top1_mean'], row['word_top1_sd'])} | "
            f"{fmt(row['word_top5_mean'], row['word_top5_sd'])} | "
            f"{fmt(row['sent_top1_mean'], row['sent_top1_sd'])} | "
            f"{fmt(row['sent_top5_mean'], row['sent_top5_sd'])} |"
        )

    lines.extend([
        "",
        "## Monolingual Hiragana MLM Accuracy",
        "",
        "| Condition | MLM top-1 | MLM top-5 | MRR | Masked tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    lookup = {row["condition"]: row for row in mono_summary}
    for condition in CONDITIONS:
        row = lookup[condition]
        lines.append(
            f"| `{condition}` | "
            f"{fmt(row['top1_mean'], row['top1_sd'])} | "
            f"{fmt(row['top5_mean'], row['top5_sd'])} | "
            f"{fmt(row['mrr_mean'], row['mrr_sd'])} | "
            f"{fmt(row['masked_tokens_mean'], row['masked_tokens_sd'])} |"
        )

    lines.extend([
        "",
        "Raw logs and parsed JSON are stored in:",
        "",
        "```text",
        "experiments/graph_density/v3_adj_variants/evaluation_results_fulltrain/",
        "```",
        "",
    ])
    (EVAL_DIR / "v3_fulltrain_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def update_readme():
    readme_path = EXP_DIR / "README.md"
    marker = "## V3 Full-Corpus Performance Results"
    block = (EVAL_DIR / "v3_fulltrain_evaluation_summary.md").read_text(encoding="utf-8")
    block = block.replace("# Graph Density V3 Full-Corpus Evaluation Results", marker, 1)
    readme = readme_path.read_text(encoding="utf-8")
    if marker in readme:
        start = readme.index(marker)
        end = readme.find("\n## ", start + len(marker))
        if end == -1:
            readme = readme[:start].rstrip() + "\n\n" + block + "\n"
        else:
            readme = readme[:start].rstrip() + "\n\n" + block + "\n" + readme[end:]
    else:
        insert_before = "\n## V3 Adjective-Variant Performance Visualizations\n"
        if insert_before in readme:
            readme = readme.replace(insert_before, "\n" + block + "\n" + insert_before, 1)
        else:
            readme = readme.rstrip() + "\n\n" + block + "\n"
    readme_path.write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    word_rows = []
    mono_rows = []

    for condition in CONDITIONS:
        for seed in SEEDS:
            run_dir = V3_DIR / f"{condition}_seed_{seed}"

            word_log = EVAL_DIR / f"{condition}_seed_{seed}_word_retrieval.log"
            if word_log.exists() and not args.force:
                text = word_log.read_text(encoding="utf-8")
            else:
                text = run([
                    args.python_bin,
                    WORD_SCRIPT,
                    "--model",
                    run_dir / "model_multilingual_fulltrain" / "final",
                    "--cjk",
                    CJK_DICT,
                    "--hiragana",
                    HIRAGANA_DICT,
                    "--parallel",
                    run_dir / "parallel_corpus_synset.json",
                    "--n_sample",
                    args.n_sample,
                    "--seed",
                    seed,
                ], log_path=word_log)
            word_rows.append({
                "condition": condition,
                "seed": seed,
                "model": "multilingual_fulltrain",
                "log": str(word_log.relative_to(ROOT)),
                **parse_word_log(text),
            })

            mono_log = EVAL_DIR / f"{condition}_seed_{seed}_mono_accuracy.log"
            mono_result_file = EVAL_DIR / f"{condition}_seed_{seed}_mono_accuracy.txt"
            if mono_log.exists() and not args.force:
                text = mono_log.read_text(encoding="utf-8")
            else:
                filename_for_prompt = "../experiments/graph_density/v3_adj_variants/evaluation_results_fulltrain/" + mono_result_file.name + "\n"
                text = run([
                    args.python_bin,
                    ACCURACY_SCRIPT,
                    "--model_dir",
                    run_dir / "model_mono_hiragana_fulltrain" / "final",
                    "--dev_file",
                    run_dir / "model_mono_hiragana_fulltrain" / "dev.txt",
                    "--seed",
                    seed,
                ], input_text=filename_for_prompt, log_path=mono_log)
            mono_rows.append({
                "condition": condition,
                "seed": seed,
                "model": "mono_hiragana_fulltrain",
                "log": str(mono_log.relative_to(ROOT)),
                "result_file": str(mono_result_file.relative_to(ROOT)),
                **parse_accuracy_log(text),
            })

    write_summary(word_rows, mono_rows)
    update_readme()
    print(EVAL_DIR / "v3_fulltrain_evaluation_summary.md")
    print(EVAL_DIR / "v3_fulltrain_evaluation_results.json")


if __name__ == "__main__":
    main()
