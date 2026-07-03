#!/usr/bin/env python3
"""Visualize v3 graph-density evaluation results.

This mirrors experiments/graph_density/visualize_experiment_results.py, but
reads the v3 adjective-variant evaluation JSON and writes into the v3
experiment directory.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "graph_density"
V3_DIR = EXP_DIR / "v3_adj_variants"
RESULT_JSON = V3_DIR / "evaluation_results" / "v3_evaluation_results.json"
OUT_DIR = V3_DIR / "visualizations"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "coverage": "#5477C4",
    "coverage_light": "#CEDFFE",
    "relation": "#CC6F47",
    "relation_light": "#FFBDA1",
    "full_graph": "#386411",
    "mono": "#B8A037",
    "neutral": "#7A828F",
}

CONDITION_ORDER = [
    "low_40",
    "medium_60_full",
    "high_80_full",
    "low_40_relation",
    "medium_60_relation",
    "high_80_relation",
]

DISPLAY = {
    "low_40": "Low 40",
    "medium_60_full": "Medium 60 full",
    "high_80_full": "High 80 full",
    "low_40_relation": "Low 40 relation",
    "medium_60_relation": "Medium 60 relation",
    "high_80_relation": "High 80 relation",
    "full_graph": "Full graph",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def density_for(condition):
    if condition.startswith("low_40"):
        return 40
    if condition.startswith("medium_60"):
        return 60
    if condition.startswith("high_80"):
        return 80
    if condition == "full_graph":
        return 100
    raise ValueError(condition)


def strategy_for(condition):
    if condition == "full_graph":
        return "full_graph"
    if condition.endswith("_relation"):
        return "relation-first"
    return "coverage-first"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": TOKENS["surface"],
        "axes.facecolor": TOKENS["panel"],
        "axes.edgecolor": TOKENS["axis"],
        "axes.labelcolor": TOKENS["ink"],
        "axes.titlecolor": TOKENS["ink"],
        "xtick.color": TOKENS["muted"],
        "ytick.color": TOKENS["muted"],
        "grid.color": TOKENS["grid"],
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": TOKENS["surface"],
        "savefig.edgecolor": "none",
    })


def add_header(fig, ax, title, subtitle):
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(left, 0.965, title, ha="left", va="top",
             fontsize=14, fontweight="bold", color=TOKENS["ink"])
    fig.text(left, 0.915, subtitle, ha="left", va="top",
             fontsize=9, color=TOKENS["muted"])


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.grid(axis="y")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))


def summary_rows(summary, metric_prefix=None):
    rows = []
    for row in summary:
        condition = row["condition"]
        out = {
            "condition": condition,
            "label": DISPLAY[condition],
            "density": density_for(condition),
            "strategy": strategy_for(condition),
            "n_seeds": len(row["seeds"]),
            "seeds": ",".join(str(seed) for seed in row["seeds"]),
        }
        for key, value in row.items():
            if key not in {"condition", "seeds"}:
                out[key] = value
        rows.append(out)
    order = {condition: i for i, condition in enumerate(CONDITION_ORDER + ["full_graph"])}
    return sorted(rows, key=lambda item: order[item["condition"]])


def plot_density_lines(summary, metrics, title, subtitle, output):
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.8), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    coverage = [row for row in summary if row["strategy"] == "coverage-first"]
    relation = [row for row in summary if row["strategy"] == "relation-first"]
    full = next(row for row in summary if row["condition"] == "full_graph")

    for ax, (metric, label) in zip(axes, metrics):
        for rows, color, name, marker in [
            (coverage, COLORS["coverage"], "Coverage-first mean", "o"),
            (relation, COLORS["relation"], "Relation-first mean", "s"),
        ]:
            rows = sorted(rows, key=lambda item: item["density"])
            xs = [row["density"] for row in rows]
            ys = [row[f"{metric}_mean"] for row in rows]
            yerr = [row[f"{metric}_sd"] for row in rows]
            ax.errorbar(xs, ys, yerr=yerr, color=color, marker=marker,
                        markersize=6, linewidth=1.4, capsize=4, label=name)

        ax.errorbar(
            [100],
            [full[f"{metric}_mean"]],
            yerr=[full[f"{metric}_sd"]],
            color=COLORS["full_graph"],
            marker="D",
            markersize=7,
            linewidth=0,
            capsize=4,
            label="Full graph mean",
            zorder=5,
        )
        ax.set_title(label, fontsize=10, color=TOKENS["ink"])
        ax.set_xlabel("Graph density (%)")
        ax.set_xticks([40, 60, 80, 100])
        clean_axis(ax)

    axes[0].set_ylabel("Precision / accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.08, 0.82),
               ncol=3, frameon=False, fontsize=8)
    add_header(fig, axes[0], title, subtitle)
    fig.subplots_adjust(top=0.74)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_seed42_bars(word_rows, mono_rows, output):
    seed42_word = [row for row in word_rows if row["seed"] == 42]
    seed42_mono = [row for row in mono_rows if row["seed"] == 42]
    order = CONDITION_ORDER + ["full_graph"]
    word_lookup = {row["condition"]: row for row in seed42_word}
    mono_lookup = {row["condition"]: row for row in seed42_mono}

    labels = [DISPLAY[condition] for condition in order]
    multi_values = [float(word_lookup[condition]["sent_top1"]) for condition in order]
    mono_values = [float(mono_lookup[condition]["top1"]) for condition in order]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], multi_values, width=width,
           label="Sentence retrieval top-1", color=COLORS["coverage_light"],
           edgecolor=COLORS["coverage"], linewidth=1.0)
    ax.bar([i + width / 2 for i in x], mono_values, width=width,
           label="Mono Hiragana MLM top-1", color="#FFEA8F",
           edgecolor=COLORS["mono"], linewidth=1.0)
    ax.set_xticks(x, labels, rotation=28, ha="right")
    ax.set_ylabel("Score")
    clean_axis(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02),
              frameon=False, ncol=2, fontsize=8)
    add_header(
        fig,
        ax,
        "Seed 42 shows full graph improves alignment but not monolingual MLM",
        "Bars compare seed 42 across all v3 density conditions and the full-graph control.",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta_from_best(word_summary, mono_summary, output):
    full_word = next(row for row in word_summary if row["condition"] == "full_graph")
    high_rel_word = next(row for row in word_summary if row["condition"] == "high_80_relation")
    full_mono = next(row for row in mono_summary if row["condition"] == "full_graph")
    high_rel_mono = next(row for row in mono_summary if row["condition"] == "high_80_relation")

    rows = [
        ("Word top-1", full_word["word_top1_mean"] - high_rel_word["word_top1_mean"]),
        ("Sentence top-1", full_word["sent_top1_mean"] - high_rel_word["sent_top1_mean"]),
        ("Mono top-1", full_mono["top1_mean"] - high_rel_mono["top1_mean"]),
        ("Mono MRR", full_mono["mrr_mean"] - high_rel_mono["mrr_mean"]),
    ]
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [COLORS["full_graph"] if value >= 0 else COLORS["relation_light"] for value in values]
    edges = ["#1F3B12" if value >= 0 else COLORS["relation"] for value in values]
    bars = ax.barh(labels, values, color=colors, edgecolor=edges, linewidth=1.0)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.set_xlim(-0.055, 0.065)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.set_xlabel("Full graph mean minus high_80_relation mean")
    for bar, value in zip(bars, values):
        x = value + (0.006 if value >= 0 else -0.006)
        ha = "left" if value >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.1%}",
                ha=ha, va="center", fontsize=8, color=TOKENS["ink"])
    add_header(
        fig,
        ax,
        "Full graph is nearly tied with high_80_relation for alignment",
        "Delta uses 3-seed means for both full graph and high_80_relation; monolingual scores move lower.",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_chart_data(word_summary, mono_summary):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "multilingual_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(word_summary[0].keys()))
        writer.writeheader()
        writer.writerows(word_summary)
    with (OUT_DIR / "monolingual_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mono_summary[0].keys()))
        writer.writeheader()
        writer.writerows(mono_summary)


def write_readme(output_paths):
    lines = [
        "# Graph Density V3 Visualizations",
        "",
        "These charts mirror the earlier graph-density visualizations. Density",
        "experiments and the full-graph control are shown as mean +/- sample",
        "standard deviation across seeds 42, 43, and 44.",
        "",
    ]
    for path in output_paths:
        rel = path.relative_to(V3_DIR)
        lines.append(f"- `{rel}`")
    lines.extend([
        "",
        "Chart-ready CSV files:",
        "",
        "- `visualizations/multilingual_summary.csv`",
        "- `visualizations/monolingual_summary.csv`",
        "",
    ])
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def update_main_readme():
    readme_path = EXP_DIR / "README.md"
    marker = "## V3 Adjective-Variant Performance Visualizations"
    lines = [
        marker,
        "",
        "Static charts using the same structure as the earlier graph-density",
        "visualizations are stored in:",
        "",
        "```text",
        "experiments/graph_density/v3_adj_variants/visualizations/",
        "```",
        "",
        "Generated chart files:",
        "",
        "- `multilingual_alignment_top1.png`",
        "- `multilingual_alignment_top5.png`",
        "- `monolingual_hiragana_mlm.png`",
        "- `seed42_condition_comparison.png`",
        "- `full_graph_delta_vs_high80_relation.png`",
        "",
        "Chart-ready CSV summaries:",
        "",
        "```text",
        "experiments/graph_density/v3_adj_variants/visualizations/multilingual_summary.csv",
        "experiments/graph_density/v3_adj_variants/visualizations/monolingual_summary.csv",
        "```",
        "",
    ]
    block = "\n".join(lines)
    readme = readme_path.read_text(encoding="utf-8")
    if marker in readme:
        start = readme.index(marker)
        end = readme.find("\n## Evaluation Results\n", start + len(marker))
        if end == -1:
            readme = readme[:start].rstrip() + "\n\n" + block + "\n"
        else:
            readme = readme[:start].rstrip() + "\n\n" + block + "\n" + readme[end:]
    else:
        insert_before = "\n## Evaluation Results\n"
        if insert_before in readme:
            readme = readme.replace(insert_before, "\n" + block + insert_before, 1)
        else:
            readme = readme.rstrip() + "\n\n" + block + "\n"
    readme_path.write_text(readme, encoding="utf-8")


def main():
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_json(RESULT_JSON)
    word_rows = data["word_retrieval"]
    mono_rows = data["mono_accuracy"]
    word_summary = summary_rows(data["word_retrieval_summary"])
    mono_summary = summary_rows(data["mono_accuracy_summary"])
    write_chart_data(word_summary, mono_summary)

    outputs = [
        OUT_DIR / "multilingual_alignment_top1.png",
        OUT_DIR / "multilingual_alignment_top5.png",
        OUT_DIR / "monolingual_hiragana_mlm.png",
        OUT_DIR / "seed42_condition_comparison.png",
        OUT_DIR / "full_graph_delta_vs_high80_relation.png",
    ]

    plot_density_lines(
        word_summary,
        [("word_top1", "Word translation top-1"), ("sent_top1", "Sentence retrieval top-1")],
        "Relation-preserving graphs dominate multilingual alignment",
        "Mean +/- sample SD for all v3 density experiments; full graph is the 3-seed mean.",
        outputs[0],
    )
    plot_density_lines(
        word_summary,
        [("word_top5", "Word translation top-5"), ("sent_top5", "Sentence retrieval top-5")],
        "Top-5 alignment peaks around high_80_relation and full graph",
        "Mean +/- sample SD for all v3 density experiments; full graph is the 3-seed mean.",
        outputs[1],
    )
    plot_density_lines(
        mono_summary,
        [("top1", "MLM top-1"), ("mrr", "MLM MRR")],
        "Monolingual MLM does not benefit from higher graph density",
        "Mean +/- sample SD for all v3 density experiments; full graph is the 3-seed mean.",
        outputs[2],
    )
    plot_seed42_bars(word_rows, mono_rows, outputs[3])
    plot_delta_from_best(word_summary, mono_summary, outputs[4])
    write_readme(outputs)
    update_main_readme()

    for output in outputs:
        print(output)
    print(OUT_DIR / "multilingual_summary.csv")
    print(OUT_DIR / "monolingual_summary.csv")


if __name__ == "__main__":
    main()
