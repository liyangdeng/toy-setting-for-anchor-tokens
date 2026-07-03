#!/usr/bin/env python3
"""Evaluate v3 graph-density models and summarize performance."""

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
EVAL_DIR = V3_DIR / "evaluation_results"
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
    "low_40",
    "medium_60_full",
    "high_80_full",
    "low_40_relation",
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


def mean(values):
    return statistics.mean(values)


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
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_sd"] = sample_sd(values)
        summary.append(out)
    return summary


def fmt(mean_value, sd_value):
    return f"{mean_value:.4f} +/- {sd_value:.4f}"


def write_summary(word_rows, mono_rows):
    word_summary = summarize(word_rows, ["word_top1", "word_top5", "sent_top1", "sent_top5"])
    mono_summary = summarize(mono_rows, ["top1", "top5", "mrr", "masked_tokens"])
    word_lookup = {row["condition"]: row for row in word_summary}
    mono_lookup = {row["condition"]: row for row in mono_summary}

    summary_json = {
        "title": "Graph Density V3 Adjective-Variant Evaluation Summary",
        "word_retrieval": word_rows,
        "mono_accuracy": mono_rows,
        "word_retrieval_summary": word_summary,
        "mono_accuracy_summary": mono_summary,
    }
    (EVAL_DIR / "v3_evaluation_results.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Graph Density V3 Adjective-Variant Evaluation Results",
        "",
        "Evaluation uses the same repository scripts as the earlier graph-density experiments:",
        "",
        "- `evaluation/word_trans_sent_retriev.py` for multilingual word translation and sentence retrieval.",
        "- `evaluation/accuracy.py` for monolingual Hiragana MLM top-k accuracy and MRR.",
        "",
        "All aggregate rows are mean +/- sample standard deviation across seeds 42, 43, and 44.",
        "Sentence retrieval uses `--n_sample 500`.",
        "",
        "## Performance Comparison",
        "",
        "- Best multilingual word top-1 is `full_graph` "
        f"({word_lookup['full_graph']['word_top1_mean']:.4f}), essentially tied with "
        f"`high_80_relation` ({word_lookup['high_80_relation']['word_top1_mean']:.4f}).",
        "- Best multilingual sentence top-1 is `high_80_relation` "
        f"({word_lookup['high_80_relation']['sent_top1_mean']:.4f}), slightly above "
        f"`full_graph` ({word_lookup['full_graph']['sent_top1_mean']:.4f}).",
        "- Relation-controlled pruning is consistently stronger than the corresponding "
        "coverage-first/full-density condition for multilingual alignment: "
        f"+{word_lookup['low_40_relation']['word_top1_mean'] - word_lookup['low_40']['word_top1_mean']:.4f} "
        f"word top-1 at low density, +{word_lookup['medium_60_relation']['word_top1_mean'] - word_lookup['medium_60_full']['word_top1_mean']:.4f} "
        f"at medium density, and +{word_lookup['high_80_relation']['word_top1_mean'] - word_lookup['high_80_full']['word_top1_mean']:.4f} "
        "at high density.",
        "- Monolingual Hiragana MLM accuracy moves in the opposite direction: lower-density "
        "relation-controlled corpora are easiest, with `low_40_relation` giving the best "
        f"MLM top-1 ({mono_lookup['low_40_relation']['top1_mean']:.4f}) and MRR "
        f"({mono_lookup['low_40_relation']['mrr_mean']:.4f}).",
        "- The full graph is a strong multilingual control but not the best monolingual "
        f"MLM setting: compared with `high_80_relation`, it is "
        f"{word_lookup['full_graph']['word_top1_mean'] - word_lookup['high_80_relation']['word_top1_mean']:+.4f} "
        f"on word top-1, {word_lookup['full_graph']['sent_top1_mean'] - word_lookup['high_80_relation']['sent_top1_mean']:+.4f} "
        f"on sentence top-1, and {mono_lookup['full_graph']['top1_mean'] - mono_lookup['high_80_relation']['top1_mean']:+.4f} "
        "on monolingual MLM top-1.",
        "",
        "## Multilingual Alignment",
        "",
        "| Condition | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        row = word_lookup[condition]
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
    for condition in CONDITIONS:
        row = mono_lookup[condition]
        lines.append(
            f"| `{condition}` | "
            f"{fmt(row['top1_mean'], row['top1_sd'])} | "
            f"{fmt(row['top5_mean'], row['top5_sd'])} | "
            f"{fmt(row['mrr_mean'], row['mrr_sd'])} | "
            f"{fmt(row['masked_tokens_mean'], row['masked_tokens_sd'])} |"
        )

    lines.extend([
        "",
        "## Per-Seed Multilingual Results",
        "",
        "| Condition | Seed | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition in CONDITIONS:
        for row in [item for item in word_rows if item["condition"] == condition]:
            lines.append(
                f"| `{condition}` | {row['seed']} | {row['word_top1']:.4f} | "
                f"{row['word_top5']:.4f} | {row['sent_top1']:.4f} | {row['sent_top5']:.4f} |"
            )

    lines.extend([
        "",
        "## Per-Seed Monolingual Results",
        "",
        "| Condition | Seed | Top-1 | Top-5 | MRR | Masked tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition in CONDITIONS:
        for row in [item for item in mono_rows if item["condition"] == condition]:
            lines.append(
                f"| `{condition}` | {row['seed']} | {row['top1']:.4f} | "
                f"{row['top5']:.4f} | {row['mrr']:.4f} | {row['masked_tokens']} |"
            )

    lines.extend([
        "",
        "Raw logs and parsed JSON are stored in:",
        "",
        "```text",
        "experiments/graph_density/v3_adj_variants/evaluation_results/",
        "```",
        "",
    ])
    (EVAL_DIR / "v3_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return word_summary, mono_summary


def update_readme():
    readme_path = EXP_DIR / "README.md"
    marker = "## V3 Adjective-Variant Performance Results"
    block = (EVAL_DIR / "v3_evaluation_summary.md").read_text(encoding="utf-8")
    block = block.replace("# Graph Density V3 Adjective-Variant Evaluation Results", marker, 1)
    readme = readme_path.read_text(encoding="utf-8")
    if marker in readme:
        start = readme.index(marker)
        insert_before = "\n## Evaluation Results\n"
        end = readme.find(insert_before, start + len(marker))
        if end == -1:
            readme = readme[:start].rstrip() + "\n\n" + block + "\n"
        else:
            readme = readme[:start].rstrip() + "\n\n" + block + "\n" + readme[end:]
    else:
        insert_before = "\n## Evaluation Results\n"
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
                    run_dir / "model_multilingual" / "final",
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
            parsed = parse_word_log(text)
            word_rows.append({
                "condition": condition,
                "seed": seed,
                "model": "multilingual",
                "log": str(word_log.relative_to(ROOT)),
                **parsed,
            })

            mono_log = EVAL_DIR / f"{condition}_seed_{seed}_mono_accuracy.log"
            mono_result_file = EVAL_DIR / f"{condition}_seed_{seed}_mono_accuracy.txt"
            if mono_log.exists() and not args.force:
                text = mono_log.read_text(encoding="utf-8")
            else:
                filename_for_prompt = "../experiments/graph_density/v3_adj_variants/evaluation_results/" + mono_result_file.name + "\n"
                text = run([
                    args.python_bin,
                    ACCURACY_SCRIPT,
                    "--model_dir",
                    run_dir / "model_mono_hiragana" / "final",
                    "--dev_file",
                    run_dir / "model_mono_hiragana" / "dev.txt",
                    "--seed",
                    seed,
                ], input_text=filename_for_prompt, log_path=mono_log)
            parsed = parse_accuracy_log(text)
            mono_rows.append({
                "condition": condition,
                "seed": seed,
                "model": "mono_hiragana",
                "log": str(mono_log.relative_to(ROOT)),
                "result_file": str(mono_result_file.relative_to(ROOT)),
                **parsed,
            })

    write_summary(word_rows, mono_rows)
    update_readme()
    print(EVAL_DIR / "v3_evaluation_summary.md")
    print(EVAL_DIR / "v3_evaluation_results.json")


if __name__ == "__main__":
    main()
