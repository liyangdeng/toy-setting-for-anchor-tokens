#!/usr/bin/env python3
"""Evaluate and aggregate semantic-overlap results across all seeds."""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-semantic-overlap-all-seeds")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "semantic_overlap"
SEED_EVAL_SCRIPT = EXP_DIR / "evaluate_seed42_results.py"
DEFAULT_OUT = EXP_DIR / "evaluation_all_seeds"
SEEDS = [42, 43, 44]
CONDITIONS = ["overlap_000", "overlap_025", "overlap_050", "overlap_075", "overlap_100"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
BLUE = {"base": "#A3BEFA", "mid": "#5477C4", "dark": "#2E4780", "xlight": "#EAF1FE"}
ORANGE = {"base": "#F0986E", "mid": "#CC6F47", "dark": "#804126", "xlight": "#FFEDDE"}

METRICS = [
    "word_top1",
    "word_top5",
    "sent_top1",
    "sent_top5",
    "mlm_top1",
    "mlm_top5",
    "mlm_mrr",
    "train_perplexity",
    "dev_perplexity",
]


def run(cmd):
    print("Running:", " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], cwd=ROOT, check=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_seed(seed, output_dir, n_sample, force):
    seed_dir = EXP_DIR / f"v1_seed_{seed}"
    seed_eval_dir = output_dir / f"evaluation_seed{seed}"
    json_path = seed_eval_dir / "semantic_overlap_seed42_evaluation.json"
    if force or not json_path.exists():
        cmd = [
            sys.executable,
            SEED_EVAL_SCRIPT,
            "--seed-dir",
            seed_dir,
            "--output-dir",
            seed_eval_dir,
            "--n-sample",
            n_sample,
            "--seed",
            seed,
        ]
        if force:
            cmd.append("--force")
        run(cmd)
    data = read_json(json_path)
    rows = []
    for row in data["rows"]:
        row = dict(row)
        row["seed"] = seed
        rows.append(row)
    return rows


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def stdev(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def aggregate_rows(rows):
    summary = []
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        out = {
            "condition": condition,
            "overlap_percent": int(condition.split("_")[1]),
            "seed_count": len(condition_rows),
        }
        for metric in METRICS:
            values = [row.get(metric) for row in condition_rows]
            valid = [value for value in values if value is not None]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values)
            out[f"{metric}_n"] = len(valid)
        for count_metric in ["synset_pairs", "sentence_pairs", "masked_tokens", "dev_sentences"]:
            values = [row.get(count_metric) for row in condition_rows if row.get(count_metric) is not None]
            out[f"{count_metric}_mean"] = mean(values)
        summary.append(out)
    return summary


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fmt_pm(row, metric):
    mu = row[f"{metric}_mean"]
    sd = row[f"{metric}_std"]
    if mu is None:
        return "N/A"
    if sd is None:
        return f"{mu:.4f}"
    return f"{mu:.4f} +/- {sd:.4f}"


def write_report(path, summary_rows, seed_rows):
    lines = [
        "# Semantic Overlap All-Seed Evaluation",
        "",
        "Seeds: 42, 43, 44.",
        "",
        "Evaluation uses the same scripts and metric definitions as the seed-42 run. "
        "Sentence retrieval is computed only over triples shared by L1 and L2, so `overlap_000` remains N/A.",
        "",
        "## Alignment Metrics: Mean +/- SD",
        "",
        "| Condition | Overlap | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['condition']}` | {row['overlap_percent']}% | "
            f"{fmt_pm(row, 'word_top1')} | {fmt_pm(row, 'word_top5')} | "
            f"{fmt_pm(row, 'sent_top1')} | {fmt_pm(row, 'sent_top5')} |"
        )

    lines.extend([
        "",
        "## MLM Dev Metrics: Mean +/- SD",
        "",
        "| Condition | Overlap | MLM top-1 | MLM top-5 | MLM MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summary_rows:
        lines.append(
            f"| `{row['condition']}` | {row['overlap_percent']}% | "
            f"{fmt_pm(row, 'mlm_top1')} | {fmt_pm(row, 'mlm_top5')} | {fmt_pm(row, 'mlm_mrr')} |"
        )

    lines.extend([
        "",
        "## Per-Seed Alignment Metrics",
        "",
        "| Seed | Condition | Overlap | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in sorted(seed_rows, key=lambda item: (item["seed"], item["overlap_percent"])):
        lines.append(
            f"| {row['seed']} | `{row['condition']}` | {row['overlap_percent']}% | "
            f"{fmt(row['word_top1'])} | {fmt(row['word_top5'])} | "
            f"{fmt(row['sent_top1'])} | {fmt(row['sent_top5'])} |"
        )

    lines.extend([
        "",
        "## Visualizations",
        "",
        "- `visualizations/paired_alignment_top1_all_seeds.png`",
        "- `visualizations/paired_alignment_top5_all_seeds.png`",
        "- `visualizations/paired_alignment_top1_top5_all_seeds.png`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def setup_axis(ax, ymax):
    ax.set_facecolor(TOKENS["surface"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=1.0, alpha=0.82)
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=1.0, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=12)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlim(-6, 106)
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def finite_error(value):
    if value is None or math.isnan(value):
        return 0.0
    return value


def plot_panel(ax, summary_rows, seed_rows, metric, title, label, marker):
    plot_rows = [row for row in summary_rows if row[f"{metric}_mean"] is not None]
    xs = [row["overlap_percent"] for row in plot_rows]
    ys = [row[f"{metric}_mean"] for row in plot_rows]
    yerr = [finite_error(row[f"{metric}_std"]) for row in plot_rows]
    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        color=BLUE["mid"],
        marker=marker,
        markersize=7,
        linewidth=2.2,
        capsize=4,
        elinewidth=1.5,
        label=label,
    )
    for seed in sorted({row["seed"] for row in seed_rows}):
        seed_plot_rows = [
            row for row in seed_rows if row["seed"] == seed and row.get(metric) is not None
        ]
        seed_plot_rows.sort(key=lambda row: row["overlap_percent"])
        ax.plot(
            [row["overlap_percent"] for row in seed_plot_rows],
            [row[metric] for row in seed_plot_rows],
            color=BLUE["base"],
            linewidth=1.0,
            alpha=0.35,
        )
    ax.set_title(title, fontsize=15, color=TOKENS["ink"], pad=8)
    ax.set_xlabel("Semantic overlap (%)", fontsize=13, color=TOKENS["ink"])


def plot_pair(summary_rows, seed_rows, *, top_k, out_dir):
    if top_k == 1:
        word_metric = "word_top1"
        sent_metric = "sent_top1"
        filename = "paired_alignment_top1_all_seeds.png"
        label = "Mean top-1 precision +/- SD"
        marker = "o"
        subtitle = "Top-1 precision; faint lines show individual seeds, error bars show seed-level standard deviation."
    else:
        word_metric = "word_top5"
        sent_metric = "sent_top5"
        filename = "paired_alignment_top5_all_seeds.png"
        label = "Mean top-5 precision +/- SD"
        marker = "s"
        subtitle = "Top-5 precision; faint lines show individual seeds, error bars show seed-level standard deviation."

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor(TOKENS["surface"])
    ymax = 0.62
    for metric in [word_metric, sent_metric]:
        for row in summary_rows:
            mu = row.get(f"{metric}_mean")
            sd = row.get(f"{metric}_std") or 0.0
            if mu is not None:
                ymax = max(ymax, math.ceil((mu + sd + 0.04) * 10) / 10)
    for ax in axes:
        setup_axis(ax, ymax)

    plot_panel(axes[0], summary_rows, seed_rows, word_metric, "Word translation", label, marker)
    axes[0].set_ylabel("Precision", fontsize=13, color=TOKENS["ink"])
    plot_panel(axes[1], summary_rows, seed_rows, sent_metric, "Sentence retrieval", label, marker)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:1], labels[:1], loc="upper left", bbox_to_anchor=(0.065, 0.78), frameon=False, fontsize=12)
    fig.text(0.07, 0.965, "Semantic overlap effects on cross-lingual alignment", fontsize=22, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.07, 0.915, subtitle, fontsize=13, color=TOKENS["muted"])
    fig.tight_layout(rect=[0, 0, 1, 0.82], w_pad=3.2)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def plot_combined(summary_rows, seed_rows, out_dir):
    series = [
        {
            "label": "Top-1 precision +/- SD",
            "word_metric": "word_top1",
            "sent_metric": "sent_top1",
            "color": BLUE["mid"],
            "context": BLUE["base"],
            "marker": "o",
            "linestyle": "-",
        },
        {
            "label": "Top-5 precision +/- SD",
            "word_metric": "word_top5",
            "sent_metric": "sent_top5",
            "color": ORANGE["mid"],
            "context": ORANGE["base"],
            "marker": "s",
            "linestyle": "--",
        },
    ]

    ymax = 0.62
    for spec in series:
        for metric in [spec["word_metric"], spec["sent_metric"]]:
            for row in summary_rows:
                mu = row.get(f"{metric}_mean")
                sd = row.get(f"{metric}_std") or 0.0
                if mu is not None:
                    ymax = max(ymax, math.ceil((mu + sd + 0.04) * 10) / 10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.patch.set_facecolor(TOKENS["surface"])
    for ax in axes:
        setup_axis(ax, ymax)

    for ax, metric_key, title in [
        (axes[0], "word_metric", "Word translation"),
        (axes[1], "sent_metric", "Sentence retrieval"),
    ]:
        for spec in series:
            metric = spec[metric_key]
            plot_rows = [row for row in summary_rows if row[f"{metric}_mean"] is not None]
            xs = [row["overlap_percent"] for row in plot_rows]
            ys = [row[f"{metric}_mean"] for row in plot_rows]
            yerr = [finite_error(row[f"{metric}_std"]) for row in plot_rows]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                color=spec["color"],
                marker=spec["marker"],
                linestyle=spec["linestyle"],
                markersize=7,
                linewidth=2.2,
                capsize=4,
                elinewidth=1.5,
                label=spec["label"],
            )
            for seed in sorted({row["seed"] for row in seed_rows}):
                seed_plot_rows = [
                    row for row in seed_rows if row["seed"] == seed and row.get(metric) is not None
                ]
                seed_plot_rows.sort(key=lambda row: row["overlap_percent"])
                ax.plot(
                    [row["overlap_percent"] for row in seed_plot_rows],
                    [row[metric] for row in seed_plot_rows],
                    color=spec["context"],
                    linestyle=spec["linestyle"],
                    linewidth=1.0,
                    alpha=0.22,
                )
        ax.set_title(title, fontsize=15, color=TOKENS["ink"], pad=8)
        ax.set_xlabel("Semantic overlap (%)", fontsize=13, color=TOKENS["ink"])

    axes[0].set_ylabel("Precision", fontsize=13, color=TOKENS["ink"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper left",
        bbox_to_anchor=(0.065, 0.805),
        frameon=False,
        fontsize=12,
        ncol=2,
    )
    fig.text(0.07, 0.965, "Semantic overlap effects on cross-lingual alignment", fontsize=22, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.07,
        0.915,
        "Top-1 and top-5 precision; faint lines show individual seeds, error bars show seed-level standard deviation.",
        fontsize=13,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.72], w_pad=3.2)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "paired_alignment_top1_top5_all_seeds.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_seed_rows = []
    for seed in SEEDS:
        all_seed_rows.extend(evaluate_seed(seed, args.output_dir, args.n_sample, args.force))

    all_seed_rows.sort(key=lambda row: (row["seed"], row["overlap_percent"]))
    summary_rows = aggregate_rows(all_seed_rows)

    write_json(args.output_dir / "semantic_overlap_all_seed_evaluation.json", {
        "seeds": SEEDS,
        "rows": all_seed_rows,
        "summary_rows": summary_rows,
    })
    seed_fieldnames = ["seed", "condition", "overlap_percent", "word_top1", "word_top5", "synset_pairs", "sent_top1", "sent_top5", "sentence_pairs", "mlm_top1", "mlm_top5", "mlm_mrr", "masked_tokens", "dev_sentences", "train_perplexity", "dev_perplexity"]
    write_csv(args.output_dir / "semantic_overlap_all_seed_per_seed.csv", all_seed_rows, seed_fieldnames)

    summary_fieldnames = ["condition", "overlap_percent", "seed_count"]
    for metric in METRICS:
        summary_fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
    summary_fieldnames.extend(["synset_pairs_mean", "sentence_pairs_mean", "masked_tokens_mean", "dev_sentences_mean"])
    write_csv(args.output_dir / "semantic_overlap_all_seed_summary.csv", summary_rows, summary_fieldnames)
    write_report(args.output_dir / "semantic_overlap_all_seed_evaluation.md", summary_rows, all_seed_rows)

    viz_dir = args.output_dir / "visualizations"
    print(plot_pair(summary_rows, all_seed_rows, top_k=1, out_dir=viz_dir))
    print(plot_pair(summary_rows, all_seed_rows, top_k=5, out_dir=viz_dir))
    print(plot_combined(summary_rows, all_seed_rows, out_dir=viz_dir))
    print(f"Wrote {args.output_dir / 'semantic_overlap_all_seed_evaluation.md'}")


if __name__ == "__main__":
    main()
