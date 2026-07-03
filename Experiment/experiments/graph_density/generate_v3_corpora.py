#!/usr/bin/env python3
"""Generate v3 adjective-variant graph-density corpora and a corpus report."""

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "graph_density"
DEFAULT_OUTPUT_ROOT = EXP_DIR / "v3_adj_variants"
EDGES = ROOT / "data" / "semantic_backbones" / "edges_adj.json"
GRAMMAR = ROOT / "data" / "grammar" / "grammar_templates_adj.py"
GENERATOR = ROOT / "data" / "generate_sentences" / "v3_generate_sentences.py"
CORPUS_BUILDER = EXP_DIR / "build_corpus_exact.py"
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
PREPARE_GRAPH = EXP_DIR / "prepare_density_graph.py"

CONDITIONS = [
    ("low_40", 0.40, "coverage"),
    ("medium_60_full", 0.60, "coverage"),
    ("high_80_full", 0.80, "coverage"),
    ("low_40_relation", 0.40, "relation"),
    ("medium_60_relation", 0.60, "relation"),
    ("high_80_relation", 0.80, "relation"),
    ("full_graph", 1.00, "coverage"),
]
SEEDS = [42, 43, 44]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(cmd):
    print("Running:", " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], cwd=ROOT, check=True)


def line_count(path):
    return len([line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()])


def relation_counts(edges):
    return dict(sorted(Counter(edge["relation"] for edge in edges).items()))


def source_type_counts(edges):
    return dict(sorted(Counter(edge.get("source_type", "") for edge in edges).items()))


def has_quality_count(edges):
    return sum(1 for edge in edges if edge.get("relation") == "HasQuality")


def percent(value, total):
    return 100 * value / total if total else 0.0


def distribution_table(counter, total):
    return [
        (key, count, percent(count, total))
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def mean_relation_counts(rows, condition):
    condition_rows = [row for row in rows if row["condition"] == condition]
    keys = sorted({key for row in condition_rows for key in row["relation_counts"]})
    mean_edges = sum(row["edges"] for row in condition_rows) / len(condition_rows)
    return [
        (
            key,
            sum(row["relation_counts"].get(key, 0) for row in condition_rows) / len(condition_rows),
            mean_edges,
        )
        for key in keys
    ]


def mean_source_type_counts(rows, condition):
    condition_rows = [row for row in rows if row["condition"] == condition]
    keys = sorted({key for row in condition_rows for key in row["source_type_counts"]})
    mean_edges = sum(row["edges"] for row in condition_rows) / len(condition_rows)
    return [
        (
            key,
            sum(row["source_type_counts"].get(key, 0) for row in condition_rows) / len(condition_rows),
            mean_edges,
        )
        for key in keys
    ]


def generate_graphs(output_root, force):
    for condition, fraction, strategy in CONDITIONS:
        for seed in SEEDS:
            run_dir = output_root / f"{condition}_seed_{seed}"
            edges_path = run_dir / "edges.json"
            if edges_path.exists() and not force:
                print(f"Skipping existing graph: {edges_path}")
                continue
            run([
                sys.executable,
                PREPARE_GRAPH,
                "--edges",
                EDGES,
                "--nodes-from-edges",
                "--fraction",
                fraction,
                "--sampling-strategy",
                strategy,
                "--condition",
                condition,
                "--seed",
                seed,
                "--output-root",
                output_root,
            ])


def generate_corpora(output_root, args):
    for condition, _fraction, _strategy in CONDITIONS:
        for seed in SEEDS:
            run_dir = output_root / f"{condition}_seed_{seed}"
            sentences_path = run_dir / "generated_sentences.json"
            corpus_metadata = run_dir / "corpus_metadata.json"
            if corpus_metadata.exists() and not args.force:
                print(f"Skipping existing corpus: {run_dir}")
                continue

            run([
                sys.executable,
                GENERATOR,
                "--edges",
                run_dir / "edges.json",
                "--grammar",
                GRAMMAR,
                "--output",
                sentences_path,
                "--s1",
                args.s1,
                "--s2",
                args.s2,
                "--s3",
                args.s3,
                "--adj_when_available",
                args.adj_when_available,
                "--adj_variant_prob",
                args.adj_variant_prob,
                "--max_adj_variants",
                args.max_adj_variants,
                "--n_samples",
                args.n_samples,
                "--seed",
                seed,
            ])
            run([
                sys.executable,
                CORPUS_BUILDER,
                "--sentences",
                sentences_path,
                "--cjk",
                CJK_DICT,
                "--hiragana",
                HIRAGANA_DICT,
                "--out-dir",
                run_dir,
            ])


def build_report(output_root, args):
    full_edges = read_json(EDGES)
    rows = []
    for condition, fraction, strategy in CONDITIONS:
        for seed in SEEDS:
            run_dir = output_root / f"{condition}_seed_{seed}"
            sampled_edges = read_json(run_dir / "edges.json")
            graph_meta = read_json(run_dir / "metadata.json")
            corpus_meta = read_json(run_dir / "corpus_metadata.json")
            generated = read_json(run_dir / "generated_sentences.json")
            generated_sentence_count = sum(len(item["sentences"]) for item in generated["results"])
            rows.append({
                "condition": condition,
                "seed": seed,
                "fraction": fraction,
                "strategy": strategy,
                "edges": len(sampled_edges),
                "active_nodes": graph_meta["sampled_graph"]["active_node_count"],
                "isolated_nodes": graph_meta["sampled_graph"]["isolated_node_count"],
                "has_quality_edges": has_quality_count(sampled_edges),
                "records": corpus_meta["records"],
                "generated_sentences": generated_sentence_count,
                "cjk_sentences": corpus_meta["cjk_sentences"],
                "hiragana_sentences": corpus_meta["hiragana_sentences"],
                "missing_cjk": corpus_meta["missing_cjk_count"],
                "missing_hiragana": corpus_meta["missing_hiragana_count"],
                "wc_cjk_lines": line_count(run_dir / "corpus_cjk.txt"),
                "wc_hiragana_lines": line_count(run_dir / "corpus_hiragana.txt"),
                "relation_counts": relation_counts(sampled_edges),
                "source_type_counts": source_type_counts(sampled_edges),
            })

    report_path = output_root / "v3_adj_variant_corpus_report.md"
    summary_path = output_root / "v3_adj_variant_corpus_summary.json"
    full_relation_counts = relation_counts(full_edges)
    full_source_type_counts = source_type_counts(full_edges)
    full_edge_count = len(full_edges)
    conditions = [condition for condition, _fraction, _strategy in CONDITIONS]
    write_json(summary_path, {
        "title": "Graph Density V3 Adjective-Variant Corpus Summary",
        "output_root": str(output_root),
        "generator": str(GENERATOR),
        "full_graph_edges": full_edge_count,
        "full_graph_nodes": len({edge["source"] for edge in full_edges} | {edge["target"] for edge in full_edges}),
        "full_relation_counts": full_relation_counts,
        "full_source_type_counts": full_source_type_counts,
        "generation_parameters": {
            "s1": args.s1,
            "s2": args.s2,
            "s3": args.s3,
            "adj_when_available": float(args.adj_when_available),
            "adj_variant_prob": float(args.adj_variant_prob),
            "max_adj_variants": int(args.max_adj_variants),
            "n_samples": int(args.n_samples),
        },
        "rows": rows,
    })

    lines = [
        "# Graph Density V3 Adjective-Variant Corpus Report",
        "",
        "This report is for the new v3 sentence generator with adjective minimal-pair variants.",
        "It is intentionally separated from the earlier graph-density corpus report.",
        "",
        "## Generation Parameters",
        "",
        "| Parameter | Value |",
        "| --- | ---: |",
        f"| `s1` | {args.s1} |",
        f"| `s2` | {args.s2} |",
        f"| `s3` | {args.s3} |",
        f"| `adj_when_available` | {args.adj_when_available} |",
        f"| `adj_variant_prob` | {args.adj_variant_prob} |",
        f"| `max_adj_variants` | {args.max_adj_variants} |",
        f"| `n_samples` | {args.n_samples} |",
        "",
        "## Full Graph",
        "",
        "| Field | Value |",
        "| --- | ---: |",
        f"| Edges | {full_edge_count} |",
        f"| Endpoint nodes | {len({edge['source'] for edge in full_edges} | {edge['target'] for edge in full_edges})} |",
        f"| HasQuality edges | {has_quality_count(full_edges)} |",
        "",
        "## Full Graph Source-Type Distribution",
        "",
        "| Source type | Edges | Percent |",
        "| --- | ---: | ---: |",
    ]
    for source_type, count, pct in distribution_table(full_source_type_counts, full_edge_count):
        lines.append(f"| `{source_type}` | {count} | {pct:.2f}% |")

    lines.extend([
        "",
        "## Corpus Summary By Seed",
        "",
        "| Condition | Seed | Strategy | Edges | Active nodes | Isolated nodes | HasQuality | Records | Sentences/lang | Missing CJK | Missing Hiragana |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| `{row['condition']}` | {row['seed']} | {row['strategy']} | "
            f"{row['edges']} | {row['active_nodes']} | {row['isolated_nodes']} | "
            f"{row['has_quality_edges']} | {row['records']} | {row['cjk_sentences']} | "
            f"{row['missing_cjk']} | {row['missing_hiragana']} |"
        )

    lines.extend([
        "",
        "## Experiment Group Summary",
        "",
        "| Condition | Seeds | Sampling | Candidate fraction | Edges | Active nodes | Isolated nodes | HasQuality mean | Avg degree mean | Max degree range | Sentences/lang mean |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ])
    for condition, fraction, strategy in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        graph_metas = [
            read_json(output_root / f"{condition}_seed_{row['seed']}" / "metadata.json")["sampled_graph"]
            for row in condition_rows
        ]
        seeds = ",".join(str(row["seed"]) for row in condition_rows)
        edges_text = ",".join(str(row["edges"]) for row in condition_rows)
        active_text = ",".join(str(row["active_nodes"]) for row in condition_rows)
        isolated_text = ",".join(str(row["isolated_nodes"]) for row in condition_rows)
        has_quality_mean = sum(row["has_quality_edges"] for row in condition_rows) / len(condition_rows)
        avg_degree_mean = sum(meta["average_degree_all_nodes"] for meta in graph_metas) / len(graph_metas)
        max_degrees = [meta["max_degree"] for meta in graph_metas]
        sentence_mean = sum(row["cjk_sentences"] for row in condition_rows) / len(condition_rows)
        lines.append(
            f"| `{condition}` | {seeds} | `{strategy}` | {fraction:.2f} | "
            f"{edges_text} | {active_text} | {isolated_text} | {has_quality_mean:.1f} | "
            f"{avg_degree_mean:.3f} | {min(max_degrees)}-{max(max_degrees)} | {sentence_mean:.1f} |"
        )

    lines.extend([
        "",
        "## Full Graph Relation Distribution",
        "",
        "| Relation | Edges | Percent |",
        "| --- | ---: | ---: |",
    ])
    for relation, count, pct in distribution_table(full_relation_counts, full_edge_count):
        lines.append(f"| `{relation}` | {count} | {pct:.2f}% |")

    lines.extend([
        "",
        "## Relation Distribution by Experiment Group",
        "",
        "Counts are averaged across seeds 42, 43, and 44. Percent is mean count divided by mean edge count for that condition.",
    ])
    full_relation_percents = {
        relation: percent(count, full_edge_count)
        for relation, count in full_relation_counts.items()
    }
    for condition in conditions:
        lines.extend([
            "",
            f"### `{condition}`",
            "",
            "| Relation | Mean edges | Percent | Full graph percent |",
            "| --- | ---: | ---: | ---: |",
        ])
        rows_for_condition = sorted(
            mean_relation_counts(rows, condition),
            key=lambda item: (-item[1], item[0]),
        )
        for relation, mean_count, mean_edges in rows_for_condition:
            lines.append(
                f"| `{relation}` | {mean_count:.1f} | "
                f"{percent(mean_count, mean_edges):.2f}% | "
                f"{full_relation_percents.get(relation, 0.0):.2f}% |"
            )

    lines.extend([
        "",
        "## Source-Type Distribution by Experiment Group",
        "",
        "Counts are averaged across seeds 42, 43, and 44.",
    ])
    full_source_type_percents = {
        source_type: percent(count, full_edge_count)
        for source_type, count in full_source_type_counts.items()
    }
    for condition in conditions:
        lines.extend([
            "",
            f"### `{condition}`",
            "",
            "| Source type | Mean edges | Percent | Full graph percent |",
            "| --- | ---: | ---: | ---: |",
        ])
        rows_for_condition = sorted(
            mean_source_type_counts(rows, condition),
            key=lambda item: (-item[1], item[0]),
        )
        for source_type, mean_count, mean_edges in rows_for_condition:
            lines.append(
                f"| `{source_type}` | {mean_count:.1f} | "
                f"{percent(mean_count, mean_edges):.2f}% | "
                f"{full_source_type_percents.get(source_type, 0.0):.2f}% |"
            )

    lines.extend([
        "",
        "## Mean Sentences Per Language",
        "",
        "| Condition | Mean | Min | Max |",
        "| --- | ---: | ---: | ---: |",
    ])
    for condition, _fraction, _strategy in CONDITIONS:
        vals = [row["cjk_sentences"] for row in rows if row["condition"] == condition]
        lines.append(f"| `{condition}` | {sum(vals) / len(vals):.1f} | {min(vals)} | {max(vals)} |")

    low_rows = [row for row in rows if row["condition"] in {"low_40", "low_40_relation"}]
    low_min = min(row["cjk_sentences"] for row in low_rows)
    low_min_row = min(low_rows, key=lambda row: row["cjk_sentences"])
    lines.extend([
        "",
        "## Recommended Epoch Sample Size",
        "",
        "Use the smallest low-density single-language corpus as the matched",
        "per-epoch sample budget for the next training step.",
        "",
        "| Setting | Value |",
        "| --- | ---: |",
        f"| Source condition | `{low_min_row['condition']}` |",
        f"| Source seed | {low_min_row['seed']} |",
        f"| Monolingual epoch sample size | {low_min} |",
        f"| Multilingual epoch sample size | {low_min * 2} |",
        "",
    ])

    lines.extend([
        "",
        "## Relation Counts",
        "",
        "Full relation distributions for every generated graph are stored in:",
        "",
        f"```text\n{summary_path.relative_to(ROOT)}\n```",
        "",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--s1", default="0")
    parser.add_argument("--s2", default="1")
    parser.add_argument("--s3", default="0")
    parser.add_argument("--adj_when_available", default="0.8")
    parser.add_argument("--adj_variant_prob", default="0.3")
    parser.add_argument("--max_adj_variants", default="3")
    parser.add_argument("--n_samples", default="8")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    generate_graphs(args.output_root, args.force)
    generate_corpora(args.output_root, args)
    build_report(args.output_root, args)


if __name__ == "__main__":
    main()
