"""
Evaluates lexical-overlap models across different experimental conditions, aggregating results and generating plots.
Using seed 42 as the default seed for evaluation.

Adapted from semantic_overlap/evaluate_seed42_results.py for uniform evaluation and visualization.

Metrics computed:
- Word translation precision (top-1 and top-5)
- Sentence retrieval precision (top-1 and top-5)

Needed files:
- synset_pos_artificial_cjk_edges_adj_augmented.json
- synset_pos_artificial_hiragana_edges_adj_augmented.json
- word_trans_sent_retriev.py
- modified parallel corpora (see build_overlapped_corpora.py)

Outputs:
- lexical_overlap_evaluation.json
- lexical_overlap_evaluation.csv
- visualizations/word_translation_precision.png
- visualizations/sentence_retrieval_precision.png

As well as evaluation logs for each condition and the shared parallel corpora used for evaluation.

Usage example:
    python evaluation/evaluate_lexical_overlap.py \
        --checkpoints-dir checkpoints \
        --output-dir evaluation_lexical_overlap \
        --n-sample 500 \
        --seed 42 \
        --force
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt

CHECKPOINTS_DIR = "checkpoints"
EVAL_DIR = "evaluation_lexical_overlap"
WORD_SCRIPT = "word_trans_sent_retriev.py"

CJK_DICT = "synset_pos_artificial_cjk_edges_adj_augmented.json"
HIRAGANA_DICT = "synset_pos_artificial_hiragana_edges_adj_augmented.json"

CONDITIONS = [
    "high_P0", "high_P2", "high_P5", "high_P7", "high_P10",
    "mid_P0", "mid_P2", "mid_P5", "mid_P7", "mid_P10",
    "low_P0", "low_P2", "low_P5", "low_P7", "low_P10",
]

# Auxiliary functions for reading/writing
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

# Function to run a subprocess command and capture its output
def run(cmd, *, log_path=None):
    """
    Runs a subprocess command and captures its output. Optionally logs the output to a file.
    """
    print("Running:", " ".join(str(part) for part in cmd))
    
    result = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    return result.stdout

# Function to build a parallel corpus from two text files
def build_shared_parallel(corpus_a_path, corpus_b_path, output_path):
    """
    Reads two text files line by line and constructs a parallel corpus in JSON format.
    Cannot use parallel_corpus_synset.json, because of modifications of corpora in the Lexical Overlap Experiment.
    """
    lines_a = Path(corpus_a_path).read_text(encoding="utf-8").splitlines()
    lines_b = Path(corpus_b_path).read_text(encoding="utf-8").splitlines()
    shared = []
    for a, b in zip(lines_a, lines_b):
        a_strip = a.strip()
        b_strip = b.strip()
        if not a_strip and not b_strip:
            continue
        shared.append({"lang_a": [a_strip], "lang_b": [b_strip]})
    write_json(output_path, shared)
    return len(shared)

# Function to parse the evaluation log and extract relevant metrics
def parse_word_log(text):
    """
    Parses the evaluation log text to extract relevant metrics.
    """

    # Word translation precision metrics
    synset_pairs = int(re.search(r"Synset content-token pairs in joint vocab\s*:\s*(\d+)", text).group(1))
    values = [float(item) for item in re.findall(r"top-[15] precision\s*:\s*([0-9.]+)", text)]
    row = {
        "synset_pairs": synset_pairs,
        "word_top1": values[0] if len(values) > 0 else 0.0,
        "word_top5": values[1] if len(values) > 1 else 0.0,
    }

    # Sentence retrieval precision metrics
    sentence_match = re.search(r"Sentence pairs evaluated\s*:\s*(\d+)", text)
    if sentence_match:
        row["sentence_pairs"] = int(sentence_match.group(1))
        row["sent_top1"] = values[2] if len(values) > 2 else 0.0
        row["sent_top5"] = values[3] if len(values) > 3 else 0.0
    else:
        row["sentence_pairs"] = 0
        row["sent_top1"] = None
        row["sent_top5"] = None
    return row

# Function to plot metrics and save the plots
def plot_metrics(data_rows, output_dir):
    """
    Plots the evaluation metrics and saves the plots.
    Each strategy is plotted separately.
    x-axis: lexical overlap percentage
    y-axis: precision metrics (word translation and sentence retrieval)
    """
    strategies = ["high", "mid", "low"]
    x_ticks_values = [0.0, 2.5, 5.0, 7.5, 10.0]
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------
    # PLOT 1: Word Translation Precision
    # -------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, strategy in enumerate(strategies):
        ax = axes1[i]
        strat_data = [d for d in data_rows if d["frequency_strategy"] == strategy]
        strat_data.sort(key=lambda x: x["lexical_overlap_percentage"])
        
        if not strat_data:
            continue
            
        xs = [d["lexical_overlap_percentage"] for d in strat_data]
        w1_vals = [d["word_top1"] for d in strat_data]
        w5_vals = [d["word_top5"] for d in strat_data]
        
        # Word Top-1 Plot
        ax.plot(xs, w1_vals, marker="o", linewidth=2.5, markersize=6, color="#1f77b4", label="Top-1")
        
        # Word Top-5 Plot
        ax.plot(xs, w5_vals, marker="s", linewidth=2.5, markersize=6, color="#aec7e8", linestyle="--", label="Top-5")
        
        ax.set_title(f"{strategy.capitalize()} Frequency", fontsize=12, fontweight="normal", pad=12)
        ax.set_xlabel("Lexical overlap", fontsize=11, labelpad=8)
        ax.set_xticks(x_ticks_values)
        ax.set_xticklabels([f"{val}%" for val in x_ticks_values])
        
        if i == 0:
            ax.set_ylabel("Word Translation Precision", fontsize=11, labelpad=8)
            ax.legend(loc="lower left", frameon=True, fontsize=10)
            
        ax.set_ylim(0.3, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig1.tight_layout()
    plot1_path = viz_dir / "word_translation_precision.png"
    fig1.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {plot1_path}")

    # -------------------------------------------------------------
    # PLOT 2: Sentence Retrieval Precision
    # -------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, strategy in enumerate(strategies):
        ax = axes2[i]
        strat_data = [d for d in data_rows if d["frequency_strategy"] == strategy]
        strat_data.sort(key=lambda x: x["lexical_overlap_percentage"])
        
        if not strat_data:
            continue
            
        xs = [d["lexical_overlap_percentage"] for d in strat_data]
        s1_vals = [d["sent_top1"] for d in strat_data if d["sent_top1"] is not None]
        s5_vals = [d["sent_top5"] for d in strat_data if d["sent_top5"] is not None]
        
        # Sentence Top-1 Plot
        if s1_vals:
            ax.plot(xs, s1_vals, marker="o", linewidth=2.5, markersize=6, color="#d62728", label="Top-1")
            
        # Sentence Top-5 Plot
        if s5_vals:
            ax.plot(xs, s5_vals, marker="s", linewidth=2.5, markersize=6, color="#ff9896", linestyle="--", label="Top-5")
            
        ax.set_title(f"{strategy.capitalize()} Frequency", fontsize=12, fontweight="normal", pad=12)
        ax.set_xlabel("Lexical overlap", fontsize=11, labelpad=8)
        ax.set_xticks(x_ticks_values)
        ax.set_xticklabels([f"{val}%" for val in x_ticks_values])
        
        if i == 0:
            ax.set_ylabel("Sentence Retrieval Precision", fontsize=11, labelpad=8)
            ax.legend(loc="lower left", frameon=True, fontsize=10)
            
        ax.set_ylim(0.3, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig2.tight_layout()
    plot2_path = viz_dir / "sentence_retrieval_precision.png"
    fig2.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {plot2_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in CONDITIONS:
        condition_dir = args.checkpoints_dir / condition
        model_dir = condition_dir / "final"
        metadata_file = condition_dir / "training_metadata.json"
        metadata = read_json(metadata_file)
        overlap_percentage = float(metadata.get("lexical_overlap_percentage", 0.0))
        strategy = metadata.get("frequency_strategy", "unknown")

        print(f"Evaluating: {condition} | Strategy: {strategy} | Overlap: {overlap_percentage}%")

        parallel_path = args.output_dir / f"{condition}_shared_parallel.json"

        corpus_a_path = Path(metadata["corpus_a"])
        corpus_b_path = Path(metadata["corpus_b"])
        sentence_pairs = build_shared_parallel(corpus_a_path, corpus_b_path, parallel_path)

        word_log = args.output_dir / f"{condition}_evaluation.log"
        if word_log.exists() and not args.force:
            word_text = word_log.read_text(encoding="utf-8")
        else:
            cmd = [
                sys.executable,
                WORD_SCRIPT,
                "--model", model_dir,
                "--cjk", CJK_DICT,
                "--hiragana", HIRAGANA_DICT,
                "--parallel", parallel_path,
                "--n_sample", args.n_sample,
                "--seed", args.seed
            ]
            if sentence_pairs == 0:
                cmd.extend(["--test", "1"])
            word_text = run(cmd, log_path=word_log)

        parsed_metrics = parse_word_log(word_text)

        row = {
            "condition": condition,
            "frequency_strategy": strategy,
            "lexical_overlap_percentage": overlap_percentage,
            "word_top1": parsed_metrics["word_top1"],
            "word_top5": parsed_metrics["word_top5"],
            "sent_top1": parsed_metrics.get("sent_top1"),
            "sent_top5": parsed_metrics.get("sent_top5"),
            "train_perplexity": metadata.get("train_perplexity", 0.0),
            "dev_perplexity": metadata.get("dev_perplexity", 0.0)
        }
        rows.append(row)

    # Store JSON and CSV results
    rows.sort(key=lambda row: row["lexical_overlap_percentage"])
    write_json(args.output_dir / "lexical_overlap_evaluation.json", {"rows": rows})
    write_csv(args.output_dir / "lexical_overlap_evaluation.csv", rows)

    # Plot final charts
    plot_metrics(rows, args.output_dir)
    print(f"\nEvaluation pipeline finished successfully! Results stored in {args.output_dir}")


if __name__ == "__main__":
    main()
