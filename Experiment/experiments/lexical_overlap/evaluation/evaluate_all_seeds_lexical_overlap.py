#!/usr/bin/env python3
"""Evaluate lexical-overlap models across multiple seeds (Word Translation & Sentence Retrieval) and create aggregate plots."""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-lexical-overlap-eval")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[0] if "__file__" in locals() else Path(".")
CHECKPOINTS_DIR = ROOT / "checkpoints"
EVAL_DIR = ROOT / "evaluation_lexical_multi_seed"
WORD_SCRIPT = ROOT / "word_trans_sent_retriev.py"

CJK_DICT = ROOT / "synset_pos_artificial_cjk.json"
HIRAGANA_DICT = ROOT / "synset_pos_artificial_hiragana.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(cmd, *, log_path=None):
    print("Running:", " ".join(str(part) for part in cmd))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    result = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    
    if result.returncode != 0:
        print("\n" + "="*50)
        print("ERROR OCCURRED IN SUBPROCESS:")
        print(result.stdout)
        print("="*50 + "\n")
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
        
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    return result.stdout


def parse_word_log(text):
    synset_pairs_match = re.search(r"Synset content-token pairs in joint vocab\s*:\s*(\d+)", text)
    synset_pairs = int(synset_pairs_match.group(1)) if synset_pairs_match else 0
    
    values = [float(item) for item in re.findall(r"top-[15] precision\s*:\s*([0-9.]+)", text)]
    
    row = {
        "synset_pairs": synset_pairs,
        "word_top1": values[0] if len(values) > 0 else 0.0,
        "word_top5": values[1] if len(values) > 1 else 0.0,
    }
    
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


def build_parallel_corpus_from_txt(corpus_a_path, corpus_b_path, output_path):
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


def plot_aggregated_metrics(aggregated_data, output_dir):
    strategies = ["high", "mid", "low"]
    x_ticks_values = [0.0, 2.5, 5.0, 7.5, 10.0]
    
    # -------------------------------------------------------------
    # PLOT 1: Word Translation Accuracy
    # -------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, strategy in enumerate(strategies):
        ax = axes1[i]
        strat_data = [d for d in aggregated_data if d["strategy"] == strategy]
        strat_data.sort(key=lambda x: x["overlap_percent"])
        
        if not strat_data:
            continue
            
        xs = [d["overlap_percent"] for d in strat_data]
        
        w1_mean = [d["word_top1_mean"] for d in strat_data]
        w1_std = [d["word_top1_std"] for d in strat_data]
        w5_mean = [d["word_top5_mean"] for d in strat_data]
        w5_std = [d["word_top5_std"] for d in strat_data]
        
        # Word Top-1 Plot with shaded standard deviation region
        ax.plot(xs, w1_mean, marker="o", linewidth=2.5, markersize=6, color="#1f77b4", label="Top-1")
        ax.fill_between(xs, np.array(w1_mean) - np.array(w1_std), np.array(w1_mean) + np.array(w1_std), color="#1f77b4", alpha=0.15)
        
        # Word Top-5 Plot with shaded standard deviation region
        ax.plot(xs, w5_mean, marker="s", linewidth=2.5, markersize=6, color="#aec7e8", linestyle="--", label="Top-5")
        ax.fill_between(xs, np.array(w5_mean) - np.array(w5_std), np.array(w5_mean) + np.array(w5_std), color="#aec7e8", alpha=0.15)
        
        ax.set_title(f"{strategy.capitalize()} Frequency", fontsize=12, fontweight="normal", pad=12)
        ax.set_xlabel("Lexical overlap", fontsize=11, labelpad=8)
        ax.set_xticks(x_ticks_values)
        ax.set_xticklabels([f"{val}%" for val in x_ticks_values])
        
        if i == 0:
            ax.set_ylabel("Word Translation Accuracy", fontsize=11, labelpad=8)
            ax.legend(loc="lower left", frameon=True, fontsize=10)
            
        ax.set_ylim(0.3, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig1.tight_layout()
    plot1_path = output_dir / "word_translation_accuracy.png"
    fig1.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {plot1_path}")

    # -------------------------------------------------------------
    # PLOT 2: Sentence Retrieval Accuracy
    # -------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, strategy in enumerate(strategies):
        ax = axes2[i]
        strat_data = [d for d in aggregated_data if d["strategy"] == strategy]
        strat_data.sort(key=lambda x: x["overlap_percent"])
        
        if not strat_data:
            continue
            
        xs = [d["overlap_percent"] for d in strat_data]
        
        s1_mean = [d["sent_top1_mean"] for d in strat_data if d["sent_top1_mean"] is not None]
        s1_std = [d["sent_top1_std"] for d in strat_data if d["sent_top1_std"] is not None]
        s5_mean = [d["sent_top5_mean"] for d in strat_data if d["sent_top5_mean"] is not None]
        s5_std = [d["sent_top5_std"] for d in strat_data if d["sent_top5_std"] is not None]
        
        # Sentence Top-1 Plot with shaded standard deviation region
        if s1_mean:
            ax.plot(xs, s1_mean, marker="o", linewidth=2.5, markersize=6, color="#d62728", label="Top-1")
            ax.fill_between(xs, np.array(s1_mean) - np.array(s1_std), np.array(s1_mean) + np.array(s1_std), color="#d62728", alpha=0.15)
            
        # Sentence Top-5 Plot with shaded standard deviation region
        if s5_mean:
            ax.plot(xs, s5_mean, marker="s", linewidth=2.5, markersize=6, color="#ff9896", linestyle="--", label="Top-5")
            ax.fill_between(xs, np.array(s5_mean) - np.array(s5_std), np.array(s5_mean) + np.array(s5_std), color="#ff9896", alpha=0.15)
            
        ax.set_title(f"{strategy.capitalize()} Frequency", fontsize=12, fontweight="normal", pad=12)
        ax.set_xlabel("Lexical overlap", fontsize=11, labelpad=8)
        ax.set_xticks(x_ticks_values)
        ax.set_xticklabels([f"{val}%" for val in x_ticks_values])
        
        if i == 0:
            ax.set_ylabel("Sentence Retrieval Accuracy", fontsize=11, labelpad=8)
            ax.legend(loc="lower left", frameon=True, fontsize=10)
            
        ax.set_ylim(0.3, 1.02)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig2.tight_layout()
    plot2_path = output_dir / "sentence_retrieval_accuracy.png"
    fig2.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {plot2_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--n-sample", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.checkpoints_dir.exists():
        print(f"Error: Checkpoints directory '{args.checkpoints_dir}' does not exist!")
        return

    print(f"Scanning checkpoint directories in: {args.checkpoints_dir.resolve()}")
    
    raw_results = {}
    condition_metadata = {}

    for condition_dir in args.checkpoints_dir.iterdir():
        if not condition_dir.is_dir():
            continue
            
        condition = condition_dir.name
        match = re.match(r"^(low|mid|high|none)_P([\d\.]+)$", condition, re.IGNORECASE)
        if not match:
            continue
            
        strategy = match.group(1).lower()
        model_dir = condition_dir / "final"
        metadata_file = condition_dir / "training_metadata.json"

        if not metadata_file.exists():
            print(f"Skipping {condition}: 'training_metadata.json' not found.")
            continue

        metadata = read_json(metadata_file)
        overlap_pct = float(metadata.get("lexical_overlap_percentage", 0.0))
        
        condition_metadata[condition] = {
            "strategy": strategy,
            "overlap_percent": overlap_pct,
            "train_perplexity": metadata.get("train_perplexity", 0.0),
            "dev_perplexity": metadata.get("dev_perplexity", 0.0)
        }
        raw_results[condition] = {}

        for seed in args.seeds:
            print(f"Evaluating: {condition} | Strategy: {strategy} | Overlap: {overlap_pct}% | Seed: {seed}")
            parallel_path = args.output_dir / f"{condition}_parallel.json"
            
            corpus_a_path = ROOT / metadata["corpus_a"]
            corpus_b_path = ROOT / metadata["corpus_b"]
            if not corpus_a_path.exists() or not corpus_b_path.exists():
                corpus_a_path = Path(metadata["corpus_a"])
                corpus_b_path = Path(metadata["corpus_b"])

            if not corpus_a_path.exists():
                sent_eval_possible = False
            else:
                sent_eval_possible = True
                build_parallel_corpus_from_txt(corpus_a_path, corpus_b_path, parallel_path)

            word_log = args.output_dir / f"{condition}_seed{seed}_evaluation.log"
            if word_log.exists() and not args.force:
                word_text = word_log.read_text(encoding="utf-8")
            else:
                cmd = [
                    sys.executable,
                    WORD_SCRIPT,
                    "--model", model_dir,
                    "--cjk", CJK_DICT,
                    "--hiragana", HIRAGANA_DICT,
                    "--parallel", parallel_path if sent_eval_possible else ROOT,
                    "--n_sample", args.n_sample,
                    "--seed", seed,
                ]
                if not sent_eval_possible:
                    cmd += ["--test", "1"]
                word_text = run(cmd, log_path=word_log)

            raw_results[condition][seed] = parse_word_log(word_text)

    # Aggregate raw results to compute means and standard deviations
    aggregated_rows = []
    for condition, seeds_dict in raw_results.items():
        if not seeds_dict:
            continue
        meta = condition_metadata[condition]
        
        w1_vals = [s["word_top1"] for s in seeds_dict.values()]
        w5_vals = [s["word_top5"] for s in seeds_dict.values()]
        s1_vals = [s["sent_top1"] for s in seeds_dict.values() if s.get("sent_top1") is not None]
        s5_vals = [s["sent_top5"] for s in seeds_dict.values() if s.get("sent_top5") is not None]
        
        row = {
            "condition": condition,
            "strategy": meta["strategy"],
            "overlap_percent": meta["overlap_percent"],
            "train_perplexity": meta["train_perplexity"],
            "dev_perplexity": meta["dev_perplexity"],
            "word_top1_mean": np.mean(w1_vals),
            "word_top1_std": np.std(w1_vals),
            "word_top5_mean": np.mean(w5_vals),
            "word_top5_std": np.std(w5_vals),
            "sent_top1_mean": np.mean(s1_vals) if s1_vals else None,
            "sent_top1_std": np.std(s1_vals) if s1_vals else None,
            "sent_top5_mean": np.mean(s5_vals) if s5_vals else None,
            "sent_top5_std": np.std(s5_vals) if s5_vals else None,
        }
        aggregated_rows.append(row)

    if not aggregated_rows:
        print("\nNo valid evaluations were compiled.")
        return

    # Store aggregated JSON structure
    write_json(args.output_dir / "lexical_overlap_evaluation_aggregated.json", {"rows": aggregated_rows})
    
    # Store aggregated CSV rows
    csv_fields = [
        "condition", "strategy", "overlap_percent", 
        "word_top1_mean", "word_top1_std", "word_top5_mean", "word_top5_std",
        "sent_top1_mean", "sent_top1_std", "sent_top5_mean", "sent_top5_std",
        "train_perplexity", "dev_perplexity"
    ]
    csv_path = args.output_dir / "lexical_overlap_evaluation_aggregated.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in aggregated_rows:
            writer.writerow(r)

    # Plot final charts
    plot_aggregated_metrics(aggregated_rows, args.output_dir)
    print(f"\nAggregated evaluation pipeline finished successfully! Results stored in {args.output_dir}")


if __name__ == "__main__":
    main()
