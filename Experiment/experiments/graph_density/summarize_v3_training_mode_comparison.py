#!/usr/bin/env python3
"""Summarize and visualize downsample vs full-corpus v3 training results."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "graph_density"
V3_DIR = EXP_DIR / "v3_adj_variants"
DOWNSAMPLE_JSON = V3_DIR / "evaluation_results" / "v3_evaluation_results.json"
FULLTRAIN_JSON = V3_DIR / "evaluation_results_fulltrain" / "v3_fulltrain_evaluation_results.json"
OUT_DIR = V3_DIR / "visualizations_training_mode"
SUMMARY_MD = V3_DIR / "v3_training_mode_comparison.md"
SUMMARY_JSON = V3_DIR / "v3_training_mode_comparison.json"

CONDITIONS = [
    "low_40",
    "medium_60_full",
    "high_80_full",
    "low_40_relation",
    "medium_60_relation",
    "high_80_relation",
    "full_graph",
]

FULLTRAIN_CONDITIONS = [
    "medium_60_full",
    "high_80_full",
    "medium_60_relation",
    "high_80_relation",
    "full_graph",
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

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "downsample": "#5477C4",
    "downsample_light": "#CEDFFE",
    "fulltrain": "#CC6F47",
    "fulltrain_light": "#FFBDA1",
    "downsample_relation": "#2E4780",
    "fulltrain_relation": "#804126",
    "full_graph": "#386411",
    "full_graph_light": "#A3D576",
    "positive": "#A3D576",
    "positive_edge": "#386411",
    "negative": "#FFBDA1",
    "negative_edge": "#804126",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def clean_axis(ax, upper=1.0):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.grid(axis="y")
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))


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
        return "relation"
    return "coverage"


def by_condition(rows):
    return {row["condition"]: row for row in rows}


def build_combined_rows():
    downsample = load_json(DOWNSAMPLE_JSON)
    fulltrain = load_json(FULLTRAIN_JSON)
    down_word = by_condition(downsample["word_retrieval_summary"])
    down_mono = by_condition(downsample["mono_accuracy_summary"])
    full_word = by_condition(fulltrain["word_retrieval_summary"])
    full_mono = by_condition(fulltrain["mono_accuracy_summary"])

    rows = []
    for condition in CONDITIONS:
        sources = [
            ("downsample", down_word, down_mono),
        ]
        if condition in FULLTRAIN_CONDITIONS:
            sources.append(("full_corpus", full_word, full_mono))
        for mode, word_lookup, mono_lookup in sources:
            word = word_lookup[condition]
            mono = mono_lookup[condition]
            rows.append({
                "condition": condition,
                "label": DISPLAY[condition],
                "training_mode": mode,
                "word_top1_mean": word["word_top1_mean"],
                "word_top1_sd": word["word_top1_sd"],
                "word_top5_mean": word["word_top5_mean"],
                "word_top5_sd": word["word_top5_sd"],
                "sent_top1_mean": word["sent_top1_mean"],
                "sent_top1_sd": word["sent_top1_sd"],
                "sent_top5_mean": word["sent_top5_mean"],
                "sent_top5_sd": word["sent_top5_sd"],
                "mlm_top1_mean": mono["top1_mean"],
                "mlm_top1_sd": mono["top1_sd"],
                "mlm_top5_mean": mono["top5_mean"],
                "mlm_top5_sd": mono["top5_sd"],
                "mrr_mean": mono["mrr_mean"],
                "mrr_sd": mono["mrr_sd"],
            })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_mode_bars(rows, metrics, title, subtitle, output):
    labels = [DISPLAY[condition] for condition in CONDITIONS]
    lookup = {(row["condition"], row["training_mode"]): row for row in rows}
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.8), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    x = list(range(len(CONDITIONS)))
    width = 0.36

    max_score = 0.0
    for ax, (metric, label) in zip(axes, metrics):
        down_values = [lookup[(condition, "downsample")][f"{metric}_mean"] for condition in CONDITIONS]
        down_err = [lookup[(condition, "downsample")][f"{metric}_sd"] for condition in CONDITIONS]
        full_values = [
            lookup[(condition, "full_corpus")][f"{metric}_mean"] if (condition, "full_corpus") in lookup else float("nan")
            for condition in CONDITIONS
        ]
        full_err = [
            lookup[(condition, "full_corpus")][f"{metric}_sd"] if (condition, "full_corpus") in lookup else 0.0
            for condition in CONDITIONS
        ]
        max_score = max(max_score, max(value + err for value, err in zip(down_values, down_err)))
        full_extents = [value + err for value, err in zip(full_values, full_err) if value == value]
        if full_extents:
            max_score = max(max_score, max(full_extents))
        ax.bar([i - width / 2 for i in x], down_values, width=width,
               label="Downsample", color=COLORS["downsample_light"],
               edgecolor=COLORS["downsample"], linewidth=1.0, yerr=down_err, capsize=3)
        ax.bar([i + width / 2 for i in x], full_values, width=width,
               label="Full corpus", color=COLORS["fulltrain_light"],
               edgecolor=COLORS["fulltrain"], linewidth=1.0, yerr=full_err, capsize=3)
        ax.set_xticks(x, labels, rotation=28, ha="right")
        ax.set_title(label, fontsize=10, color=TOKENS["ink"])
        clean_axis(ax, upper=min(1.0, max(0.2, max_score + 0.08)))
    axes[0].set_ylabel("Score")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.08, 0.82),
               ncol=2, frameon=False, fontsize=8)
    add_header(fig, axes[0], title, subtitle)
    fig.subplots_adjust(top=0.74)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_mode_lines(rows, metrics, title, subtitle, output):
    lookup = {(row["condition"], row["training_mode"]): row for row in rows}
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4.8), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    series = [
        ("downsample", "coverage", "Downsample coverage", COLORS["downsample"], "o", "-"),
        ("downsample", "relation", "Downsample relation", COLORS["downsample_relation"], "s", "-"),
        ("full_corpus", "coverage", "Full corpus coverage", COLORS["fulltrain"], "o", "--"),
        ("full_corpus", "relation", "Full corpus relation", COLORS["fulltrain_relation"], "s", "--"),
    ]
    full_series = [
        ("downsample", "Downsample full graph", COLORS["full_graph_light"], "D"),
        ("full_corpus", "Full corpus full graph", COLORS["full_graph"], "D"),
    ]

    for ax, (metric, label) in zip(axes, metrics):
        max_score = 0.0
        for mode, strategy, name, color, marker, linestyle in series:
            conditions = [
                condition for condition in CONDITIONS
                if condition != "full_graph" and strategy_for(condition) == strategy
                and (condition, mode) in lookup
            ]
            conditions = sorted(conditions, key=density_for)
            xs = [density_for(condition) for condition in conditions]
            ys = [lookup[(condition, mode)][f"{metric}_mean"] for condition in conditions]
            yerr = [lookup[(condition, mode)][f"{metric}_sd"] for condition in conditions]
            if ys:
                max_score = max(max_score, max(value + err for value, err in zip(ys, yerr)))
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=6,
                linewidth=1.4,
                capsize=4,
                label=name,
            )

        for mode, name, color, marker in full_series:
            row = lookup[("full_graph", mode)]
            value = row[f"{metric}_mean"]
            err = row[f"{metric}_sd"]
            max_score = max(max_score, value + err)
            ax.errorbar(
                [100],
                [value],
                yerr=[err],
                color=color,
                marker=marker,
                markersize=7,
                linewidth=0,
                capsize=4,
                label=name,
                zorder=5,
            )

        ax.set_title(label, fontsize=10, color=TOKENS["ink"])
        ax.set_xlabel("Graph density (%)")
        ax.set_xlim(36, 104)
        ax.set_xticks([40, 60, 80, 100])
        clean_axis(ax, upper=min(1.0, max(0.2, max_score + 0.08)))

    axes[0].set_ylabel("Score")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.08, 0.82),
               ncol=3, frameon=False, fontsize=8)
    add_header(fig, axes[0], title, subtitle)
    fig.subplots_adjust(top=0.72)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_delta(rows, metric, title, subtitle, output):
    lookup = {(row["condition"], row["training_mode"]): row for row in rows}
    deltas = []
    for condition in FULLTRAIN_CONDITIONS:
        full_value = lookup[(condition, "full_corpus")][f"{metric}_mean"]
        down_value = lookup[(condition, "downsample")][f"{metric}_mean"]
        deltas.append((DISPLAY[condition], full_value - down_value))
    deltas = sorted(deltas, key=lambda item: item[1])

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    labels = [item[0] for item in deltas]
    values = [item[1] for item in deltas]
    colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in values]
    edges = [COLORS["positive_edge"] if value >= 0 else COLORS["negative_edge"] for value in values]
    bars = ax.barh(labels, values, color=colors, edgecolor=edges, linewidth=1.0)
    ax.axvline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.set_xlabel("Full corpus minus downsample")
    for bar, value in zip(bars, values):
        x = value + (0.006 if value >= 0 else -0.006)
        ha = "left" if value >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{value:+.1%}",
                ha=ha, va="center", fontsize=8, color=TOKENS["ink"])
    add_header(fig, ax, title, subtitle)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt(row, metric):
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_sd']:.4f}"


def write_summary(rows):
    lookup = {(row["condition"], row["training_mode"]): row for row in rows}
    lines = [
        "# V3 Downsample vs Full-Corpus Training Comparison",
        "",
        "This combines the original downsampled v3 runs with the full-corpus follow-up on 60%, 80%, and full-graph conditions.",
        "Low-density rows are included for the downsampled experiment only, because no full-corpus low-density runs were trained.",
        "All rows are mean +/- sample standard deviation across seeds 42, 43, and 44.",
        "",
        "## Multilingual Alignment",
        "",
        "| Condition | Mode | Word top-1 | Word top-5 | Sentence top-1 | Sentence top-5 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        modes = ["downsample"]
        if condition in FULLTRAIN_CONDITIONS:
            modes.append("full_corpus")
        for mode in modes:
            row = lookup[(condition, mode)]
            lines.append(
                f"| `{condition}` | `{mode}` | {fmt(row, 'word_top1')} | {fmt(row, 'word_top5')} | "
                f"{fmt(row, 'sent_top1')} | {fmt(row, 'sent_top5')} |"
            )
    lines.extend([
        "",
        "## Monolingual Hiragana MLM Accuracy",
        "",
        "| Condition | Mode | MLM top-1 | MLM top-5 | MRR |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for condition in CONDITIONS:
        modes = ["downsample"]
        if condition in FULLTRAIN_CONDITIONS:
            modes.append("full_corpus")
        for mode in modes:
            row = lookup[(condition, mode)]
            lines.append(
                f"| `{condition}` | `{mode}` | {fmt(row, 'mlm_top1')} | {fmt(row, 'mlm_top5')} | {fmt(row, 'mrr')} |"
            )
    lines.extend([
        "",
        "## Full-Corpus Delta",
        "",
        "| Condition | Word top-1 | Sentence top-1 | MLM top-1 | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for condition in FULLTRAIN_CONDITIONS:
        down = lookup[(condition, "downsample")]
        full = lookup[(condition, "full_corpus")]
        lines.append(
            f"| `{condition}` | "
            f"{full['word_top1_mean'] - down['word_top1_mean']:+.4f} | "
            f"{full['sent_top1_mean'] - down['sent_top1_mean']:+.4f} | "
            f"{full['mlm_top1_mean'] - down['mlm_top1_mean']:+.4f} | "
            f"{full['mrr_mean'] - down['mrr_mean']:+.4f} |"
        )
    lines.extend([
        "",
        "Chart files are stored in:",
        "",
        "```text",
        "experiments/graph_density/v3_adj_variants/visualizations_training_mode/",
        "```",
        "",
        "Density-style line charts:",
        "",
        "- `training_mode_multilingual_alignment_top1_lines.png`",
        "- `training_mode_multilingual_alignment_top5_lines.png`",
        "- `training_mode_monolingual_lines.png`",
        "",
        "Mode comparison bar and delta charts:",
        "",
        "- `training_mode_multilingual_top1.png`",
        "- `training_mode_multilingual_top5.png`",
        "- `training_mode_monolingual_mlm.png`",
        "- `training_mode_word_top1_delta.png`",
        "- `training_mode_mlm_top1_delta.png`",
        "",
    ])
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_readme():
    readme_path = EXP_DIR / "README.md"
    marker = "## V3 Downsample vs Full-Corpus Training Comparison"
    block = SUMMARY_MD.read_text(encoding="utf-8").replace(
        "# V3 Downsample vs Full-Corpus Training Comparison",
        marker,
        1,
    )
    readme = readme_path.read_text(encoding="utf-8")
    if marker in readme:
        start = readme.index(marker)
        end = readme.find("\n## V3 Adjective-Variant Performance Visualizations\n", start + len(marker))
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
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_combined_rows()
    write_csv(OUT_DIR / "training_mode_summary.csv", rows)
    plot_mode_bars(
        rows,
        [("word_top1", "Word translation top-1"), ("sent_top1", "Sentence retrieval top-1")],
        "Full-corpus training changes multilingual alignment unevenly",
        "Mean +/- sample SD across seeds 42, 43, and 44; bars compare downsample vs full-corpus training.",
        OUT_DIR / "training_mode_multilingual_top1.png",
    )
    plot_mode_bars(
        rows,
        [("word_top5", "Word translation top-5"), ("sent_top5", "Sentence retrieval top-5")],
        "Top-5 alignment comparison across training modes",
        "Mean +/- sample SD across seeds 42, 43, and 44; bars compare downsample vs full-corpus training.",
        OUT_DIR / "training_mode_multilingual_top5.png",
    )
    plot_mode_bars(
        rows,
        [("mlm_top1", "MLM top-1"), ("mrr", "MLM MRR")],
        "Full-corpus training often lowers monolingual MLM accuracy",
        "Mean +/- sample SD across seeds 42, 43, and 44; bars compare downsample vs full-corpus training.",
        OUT_DIR / "training_mode_monolingual_mlm.png",
    )
    plot_training_mode_lines(
        rows,
        [("word_top1", "Word translation top-1"), ("sent_top1", "Sentence retrieval top-1")],
        "Training mode and graph method effects on multilingual top-1",
        "Line chart mirrors the earlier density plots; solid lines are downsample, dashed lines are full-corpus.",
        OUT_DIR / "training_mode_multilingual_alignment_top1_lines.png",
    )
    plot_training_mode_lines(
        rows,
        [("word_top5", "Word translation top-5"), ("sent_top5", "Sentence retrieval top-5")],
        "Training mode and graph method effects on multilingual top-5",
        "Line chart mirrors the earlier density plots; solid lines are downsample, dashed lines are full-corpus.",
        OUT_DIR / "training_mode_multilingual_alignment_top5_lines.png",
    )
    plot_training_mode_lines(
        rows,
        [("mlm_top1", "MLM top-1"), ("mrr", "MLM MRR")],
        "Training mode and graph method effects on monolingual MLM",
        "Line chart mirrors the earlier density plots; solid lines are downsample, dashed lines are full-corpus.",
        OUT_DIR / "training_mode_monolingual_lines.png",
    )
    plot_delta(
        rows,
        "word_top1",
        "Word top-1 delta from full-corpus training",
        "Signed mean difference: full corpus minus downsample.",
        OUT_DIR / "training_mode_word_top1_delta.png",
    )
    plot_delta(
        rows,
        "mlm_top1",
        "MLM top-1 delta from full-corpus training",
        "Signed mean difference: full corpus minus downsample.",
        OUT_DIR / "training_mode_mlm_top1_delta.png",
    )
    write_summary(rows)
    update_readme()
    for path in sorted(OUT_DIR.iterdir()):
        print(path)
    print(SUMMARY_MD)
    print(SUMMARY_JSON)


if __name__ == "__main__":
    main()
