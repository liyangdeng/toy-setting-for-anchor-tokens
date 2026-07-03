#!/usr/bin/env python3
"""Track selected cross-lingual token-pair cosine distances across overlaps."""

import argparse
import csv
import json
import math
import os
import random
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-semantic-overlap-embeddings")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter
import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, PreTrainedTokenizerFast


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "semantic_overlap"
DEFAULT_SEED_DIR = EXP_DIR / "v1_seed_42"
DEFAULT_OUT = EXP_DIR / "embedding_distance_seed42"
GRAPH_EDGES = ROOT / "data" / "semantic_backbones" / "edges_adj.json"
TOKEN_FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
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
CONDITIONS = ["overlap_000", "overlap_025", "overlap_050", "overlap_075", "overlap_100"]
PCA_CONDITIONS = ["overlap_000", "overlap_050", "overlap_100"]
PCA_PAIRS_PER_GROUP = 4

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
OLIVE = {"base": "#A3D576", "mid": "#71B436", "dark": "#386411", "xlight": "#D8ECBD"}
GROUP_STYLES = {
    "close_at_100": {"label": "Close at 100%", "color": BLUE["mid"], "context": BLUE["base"], "marker": "o", "linestyle": "-"},
    "random_true": {"label": "Random graph true pairs", "color": ORANGE["mid"], "context": ORANGE["base"], "marker": "s", "linestyle": "--"},
    "far_at_100": {"label": "Far at 100%", "color": OLIVE["mid"], "context": OLIVE["base"], "marker": "D", "linestyle": ":"},
}
TOKEN_FONT = FontProperties(fname=str(TOKEN_FONT_PATH)) if TOKEN_FONT_PATH.exists() else None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_artificial_pairs(cjk_path, hiragana_path):
    cjk = read_json(cjk_path)
    hiragana = read_json(hiragana_path)
    pairs = []
    for synset in sorted(set(cjk) & set(hiragana)):
        cjk_token = cjk[synset].get("artificial")
        hiragana_token = hiragana[synset].get("artificial")
        if not cjk_token or not hiragana_token:
            continue
        pairs.append({
            "synset": synset,
            "lemma": cjk[synset].get("primary_lemma", synset),
            "cjk_token": cjk_token,
            "hiragana_token": hiragana_token,
        })
    return pairs


def load_graph_nodes(path):
    nodes = set()
    for edge in read_json(path):
        source = edge.get("source")
        target = edge.get("target")
        if source:
            nodes.add(source)
        if target:
            nodes.add(target)
    return nodes


def load_model_embeddings(model_dir):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    model = BertForMaskedLM.from_pretrained(model_dir)
    model.eval()
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight.detach().cpu()
        embeddings = F.normalize(embeddings, dim=1)
    vocab = tokenizer.get_vocab()
    return vocab, embeddings


def pair_distance(pair, vocab, embeddings):
    cjk_id = vocab.get(pair["cjk_token"])
    hira_id = vocab.get(pair["hiragana_token"])
    if cjk_id is None or hira_id is None:
        return None
    similarity = torch.dot(embeddings[cjk_id], embeddings[hira_id]).item()
    return 1.0 - similarity


def valid_in_all_conditions(pair, condition_embeddings):
    for _, (vocab, _) in condition_embeddings.items():
        if pair["cjk_token"] not in vocab or pair["hiragana_token"] not in vocab:
            return False
    return True


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


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def setup_axis(ax):
    ax.set_facecolor(TOKENS["surface"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=1.0, alpha=0.85)
    ax.grid(axis="x", color=TOKENS["grid"], linewidth=1.0, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(axis="both", colors=TOKENS["muted"], labelsize=11)


def plot_distance_trajectory(summary_rows, pair_rows, output_path):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.patch.set_facecolor(TOKENS["surface"])
    setup_axis(ax)
    for group, style in GROUP_STYLES.items():
        group_pair_ids = sorted({row["pair_id"] for row in pair_rows if row["group"] == group})
        for pair_id in group_pair_ids:
            part = [row for row in pair_rows if row["pair_id"] == pair_id]
            part.sort(key=lambda row: row["overlap_percent"])
            ax.plot(
                [row["overlap_percent"] for row in part],
                [row["cosine_distance"] for row in part],
                color=style["context"],
                alpha=0.10,
                linewidth=0.9,
                linestyle=style["linestyle"],
            )

        group_summary = [row for row in summary_rows if row["group"] == group]
        group_summary.sort(key=lambda row: row["overlap_percent"])
        ax.errorbar(
            [row["overlap_percent"] for row in group_summary],
            [row["mean_cosine_distance"] for row in group_summary],
            yerr=[row["std_cosine_distance"] or 0.0 for row in group_summary],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=6.5,
            linewidth=2.1,
            capsize=4,
            elinewidth=1.4,
            label=style["label"],
        )
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Semantic overlap (%)", fontsize=12, color=TOKENS["ink"])
    ax.set_ylabel("Cosine distance", fontsize=12, color=TOKENS["ink"])
    ymax = max(row["cosine_distance"] for row in pair_rows)
    ax.set_ylim(0, max(0.2, math.ceil((ymax + 0.05) * 10) / 10))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), frameon=False, fontsize=10.5, ncol=3)
    fig.text(0.09, 0.965, "Token-pair distances across semantic overlap", fontsize=18, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.09,
        0.915,
        "Only graph endpoint tokens are used; faint lines show individual true CJK-Hiragana synset pairs.",
        fontsize=11,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def pca_2d(matrix):
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2)
    return centered @ v[:, :2]


def choose_representative_pairs(selected_pairs):
    representatives = []
    for group in GROUP_STYLES:
        group_pairs = [pair for pair in selected_pairs if pair["group"] == group]
        if group in {"close_at_100", "far_at_100"}:
            representatives.extend(group_pairs[:PCA_PAIRS_PER_GROUP])
            continue

        distances = sorted(pair["cosine_distance_100"] for pair in group_pairs)
        median_distance = distances[len(distances) // 2]
        representatives.extend(
            sorted(
                group_pairs,
                key=lambda pair: (abs(pair["cosine_distance_100"] - median_distance), pair["rank"]),
            )[:PCA_PAIRS_PER_GROUP]
        )
    return representatives


def annotate_token(ax, x, y, label, color, *, dx=0.012, dy=0.012):
    kwargs = {}
    if TOKEN_FONT is not None:
        kwargs["fontproperties"] = TOKEN_FONT
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=6.5,
        color=color,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": TOKENS["panel"],
            "edgecolor": color,
            "linewidth": 0.35,
            "alpha": 0.82,
        },
        **kwargs,
    )


def plot_pca_panels(selected_pairs, condition_embeddings, output_path):
    representative_pairs = choose_representative_pairs(selected_pairs)
    fig, axes = plt.subplots(1, len(PCA_CONDITIONS), figsize=(18, 6.0))
    fig.patch.set_facecolor(TOKENS["surface"])
    for ax, condition in zip(axes, PCA_CONDITIONS):
        setup_axis(ax)
        vocab, embeddings = condition_embeddings[condition]
        vectors = []
        groups = []
        for pair in representative_pairs:
            vectors.append(embeddings[vocab[pair["cjk_token"]]])
            groups.append(pair["group"])
            vectors.append(embeddings[vocab[pair["hiragana_token"]]])
            groups.append(pair["group"])
        coords = pca_2d(torch.stack(vectors)).numpy()
        for i in range(0, len(coords), 2):
            group = representative_pairs[i // 2]["group"]
            style = GROUP_STYLES[group]
            ax.plot(
                [coords[i, 0], coords[i + 1, 0]],
                [coords[i, 1], coords[i + 1, 1]],
                color=style["context"],
                linewidth=0.8,
                alpha=0.35,
                zorder=1,
            )
        for group, style in GROUP_STYLES.items():
            indices = [idx for idx, pair in enumerate(representative_pairs) if pair["group"] == group]
            cjk_points = coords[[2 * idx for idx in indices]]
            hira_points = coords[[2 * idx + 1 for idx in indices]]
            ax.scatter(
                cjk_points[:, 0],
                cjk_points[:, 1],
                s=46,
                color=style["color"],
                edgecolor=style["color"],
                linewidth=0.6,
                label=f"{style['label']} CJK",
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                hira_points[:, 0],
                hira_points[:, 1],
                s=46,
                marker="s",
                facecolor=TOKENS["panel"],
                edgecolor=style["color"],
                linewidth=0.9,
                label=f"{style['label']} Hiragana",
                alpha=0.9,
                zorder=2,
            )
        for idx, pair in enumerate(representative_pairs):
            style = GROUP_STYLES[pair["group"]]
            cjk_x, cjk_y = coords[2 * idx]
            hira_x, hira_y = coords[2 * idx + 1]
            label_base = pair["lemma"] or pair["synset"]
            annotate_token(ax, cjk_x, cjk_y, f"{label_base} (CJK)", style["color"], dx=0.014, dy=0.014)
            annotate_token(ax, hira_x, hira_y, f"{label_base} (Hira)", style["color"], dx=0.014, dy=-0.014)
        ax.set_title(condition.replace("overlap_", "Overlap ") + "%", fontsize=13, color=TOKENS["ink"])
        ax.set_xlabel("PC1", fontsize=11, color=TOKENS["ink"])
        if ax is axes[0]:
            ax.set_ylabel("PC2", fontsize=11, color=TOKENS["ink"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::2], [GROUP_STYLES[group]["label"] for group in GROUP_STYLES], loc="upper left", bbox_to_anchor=(0.08, 0.83), frameon=False, fontsize=10.5, ncol=3)
    fig.text(0.08, 0.965, "PCA view of representative token-pair embeddings", fontsize=18, fontweight="bold", color=TOKENS["ink"])
    fig.text(
        0.08,
        0.915,
        "Each group contributes four representative graph pairs; labels show the original English synset lemma and language.",
        fontsize=11,
        color=TOKENS["muted"],
    )
    fig.tight_layout(rect=[0, 0, 1, 0.80], w_pad=2.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return representative_pairs


def write_report(path, selected_pairs, summary_rows, selected_rows, valid_pair_count, graph_node_count):
    lines = [
        "# Semantic Overlap Token Embedding Distance Probe",
        "",
        "This probe uses model input token embeddings and cosine distance.",
        "",
        "Pairs are true CJK-Hiragana artificial-token pairs that share the same synset, and the synset must appear as a `source` or `target` endpoint in `data/semantic_backbones/edges_adj.json`.",
        "",
        f"Graph endpoint nodes used for filtering: {graph_node_count}. Valid graph-token pairs present in every model: {valid_pair_count}.",
        "",
        "## Distance Summary",
        "",
        "| Group | Condition | Overlap | Mean cosine distance | SD | Min | Max | Pairs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {GROUP_STYLES[row['group']]['label']} | `{row['condition']}` | {row['overlap_percent']}% | "
            f"{row['mean_cosine_distance']:.4f} | {row['std_cosine_distance']:.4f} | "
            f"{row['min_cosine_distance']:.4f} | {row['max_cosine_distance']:.4f} | {row['pair_count']} |"
        )
    lines.extend([
        "",
        "## Selected Graph Pairs",
        "",
        "| Group | Rank | Synset | Lemma | CJK token | Hiragana token | Distance at 100% |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ])
    for row in selected_rows:
        lines.append(
            f"| {GROUP_STYLES[row['group']]['label']} | {row['rank']} | `{row['synset']}` | {row['lemma']} | "
            f"{row['cjk_token']} | {row['hiragana_token']} | {row['cosine_distance_100']:.4f} |"
        )
    lines.extend([
        "",
        "## Visualizations",
        "",
        "- `visualizations/cosine_distance_trajectory.png`",
        "- `visualizations/pca_selected_pairs.png`",
        "",
        "Note: PCA panels are reduced separately per model, so they are useful for within-condition geometry, not absolute cross-condition axis comparison. The PCA figure labels representative points by original English synset lemma; artificial tokens remain in `pca_representative_pairs.csv`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    condition_embeddings = {}
    for condition in CONDITIONS:
        model_dir = args.seed_dir / condition / "model_multilingual" / "final"
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Missing model directory: {model_dir}")
        condition_embeddings[condition] = load_model_embeddings(model_dir)

    graph_nodes = load_graph_nodes(GRAPH_EDGES)
    all_pairs = load_artificial_pairs(CJK_DICT, HIRAGANA_DICT)
    graph_pairs = [pair for pair in all_pairs if pair["synset"] in graph_nodes]
    valid_pairs = [pair for pair in graph_pairs if valid_in_all_conditions(pair, condition_embeddings)]
    vocab100, emb100 = condition_embeddings["overlap_100"]
    ranked = []
    for pair in valid_pairs:
        distance = pair_distance(pair, vocab100, emb100)
        if distance is not None:
            ranked.append({**pair, "cosine_distance_100": distance})
    ranked.sort(key=lambda row: row["cosine_distance_100"])
    if len(ranked) < args.top_n * 3:
        raise ValueError(f"Need at least {args.top_n * 3} valid pairs, found {len(ranked)}")

    close_pairs = [{**row, "group": "close_at_100", "rank": rank} for rank, row in enumerate(ranked[: args.top_n], start=1)]
    far_pairs = [{**row, "group": "far_at_100", "rank": rank} for rank, row in enumerate(ranked[-args.top_n:][::-1], start=1)]
    excluded = {row["synset"] for row in close_pairs + far_pairs}
    random_pool = [row for row in ranked if row["synset"] not in excluded]
    random_pairs_raw = random.sample(random_pool, args.top_n)
    random_pairs = [{**row, "group": "random_true", "rank": rank} for rank, row in enumerate(random_pairs_raw, start=1)]
    selected_pairs = close_pairs + random_pairs + far_pairs

    pair_rows = []
    for pair_id, pair in enumerate(selected_pairs, start=1):
        for condition in CONDITIONS:
            vocab, embeddings = condition_embeddings[condition]
            distance = pair_distance(pair, vocab, embeddings)
            pair_rows.append({
                "pair_id": pair_id,
                "group": pair["group"],
                "selection_rank": pair["rank"],
                "condition": condition,
                "overlap_percent": int(condition.split("_")[1]),
                "synset": pair["synset"],
                "lemma": pair["lemma"],
                "cjk_token": pair["cjk_token"],
                "hiragana_token": pair["hiragana_token"],
                "cosine_distance": distance,
            })

    summary_rows = []
    for group in GROUP_STYLES:
        for condition in CONDITIONS:
            values = [row["cosine_distance"] for row in pair_rows if row["condition"] == condition and row["group"] == group]
            summary_rows.append({
                "group": group,
                "condition": condition,
                "overlap_percent": int(condition.split("_")[1]),
                "pair_count": len(values),
                "mean_cosine_distance": mean(values),
                "std_cosine_distance": stdev(values) or 0.0,
                "min_cosine_distance": min(values),
                "max_cosine_distance": max(values),
            })

    selected_rows = []
    for pair in selected_pairs:
        selected_rows.append({
            "group": pair["group"],
            "rank": pair["rank"],
            "synset": pair["synset"],
            "lemma": pair["lemma"],
            "cjk_token": pair["cjk_token"],
            "hiragana_token": pair["hiragana_token"],
            "cosine_distance_100": pair["cosine_distance_100"],
        })

    write_csv(
        args.output_dir / "selected_pairs.csv",
        selected_rows,
        ["group", "rank", "synset", "lemma", "cjk_token", "hiragana_token", "cosine_distance_100"],
    )
    write_csv(
        args.output_dir / "pair_distances_by_overlap.csv",
        pair_rows,
        ["pair_id", "group", "selection_rank", "condition", "overlap_percent", "synset", "lemma", "cjk_token", "hiragana_token", "cosine_distance"],
    )
    write_csv(
        args.output_dir / "distance_summary.csv",
        summary_rows,
        ["group", "condition", "overlap_percent", "pair_count", "mean_cosine_distance", "std_cosine_distance", "min_cosine_distance", "max_cosine_distance"],
    )
    write_json(args.output_dir / "embedding_distance_probe.json", {
        "seed_dir": str(args.seed_dir),
        "selection_condition": "overlap_100",
        "top_n": args.top_n,
        "graph_edges": str(GRAPH_EDGES),
        "graph_endpoint_node_count": len(graph_nodes),
        "graph_pair_count": len(graph_pairs),
        "valid_pair_count": len(valid_pairs),
        "selected_pairs": selected_rows,
        "summary_rows": summary_rows,
        "pair_rows": pair_rows,
    })

    viz_dir = args.output_dir / "visualizations"
    plot_distance_trajectory(summary_rows, pair_rows, viz_dir / "cosine_distance_trajectory.png")
    representative_pairs = plot_pca_panels(selected_pairs, condition_embeddings, viz_dir / "pca_selected_pairs.png")
    representative_rows = [
        {
            "group": pair["group"],
            "rank": pair["rank"],
            "synset": pair["synset"],
            "lemma": pair["lemma"],
            "cjk_token": pair["cjk_token"],
            "hiragana_token": pair["hiragana_token"],
            "cosine_distance_100": pair["cosine_distance_100"],
        }
        for pair in representative_pairs
    ]
    write_csv(
        args.output_dir / "pca_representative_pairs.csv",
        representative_rows,
        ["group", "rank", "synset", "lemma", "cjk_token", "hiragana_token", "cosine_distance_100"],
    )
    write_report(args.output_dir / "embedding_distance_probe.md", selected_pairs, summary_rows, selected_rows, len(valid_pairs), len(graph_nodes))

    print(f"Graph endpoint nodes: {len(graph_nodes)}")
    print(f"Graph-filtered true token pairs available in every model: {len(valid_pairs)}")
    print(f"Selected pairs: {len(selected_pairs)}")
    print(args.output_dir / "embedding_distance_probe.md")
    print(viz_dir / "cosine_distance_trajectory.png")
    print(viz_dir / "pca_selected_pairs.png")


if __name__ == "__main__":
    main()
