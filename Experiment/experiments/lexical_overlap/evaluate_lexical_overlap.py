#!/usr/bin/env python3
"""Evaluate and plot results for a single seed for Lexical Overlap & Anchoring."""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

STRATEGIES = ["high", "mid", "low"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single seed for Lexical Anchors")
    parser.add_argument("--base_checkpoints_dir", type=str, required=True, help="Path where the seed directory is kept (e.g., ./checkpoints_lexical)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save final charts and tables")
    parser.add_argument("--seed", type=int, default=42, help="The specific seed to evaluate (default: 42)")
    return parser.parse_args()


def generate_plots(rows, seed, output_dir):
    """Generates charts showing the three strategies compared side-by-side for a single seed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["word_top1", "sent_top1", "mlm_top1", "dev_perplexity"]
    titles = {
        "word_top1": f"Word Translation Accuracy (Top-1) - Seed {seed}",
        "sent_top1": f"Sentence Retrieval Accuracy (Top-1) - Seed {seed}",
        "mlm_top1": f"MLM Masked Token Accuracy (Top-1) - Seed {seed}",
        "dev_perplexity": f"Development Set Perplexity - Seed {seed}"
    }
    
    colors = {"high": "#2E4780", "#mid": "#CC6F47", "low": "#2CA02C"}
    
    # Group data by strategy for plotting
    data_by_strategy = {strat: [] for strat in STRATEGIES}
    for row in rows:
        strat = row["strategy"]
        if strat in data_by_strategy:
            data_by_strategy[strat].append(row)

    # Sort each group by overlap percentage so lines render correctly
    for strat in STRATEGIES:
        data_by_strategy[strat].sort(key=lambda x: x["overlap_percent"])

    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        
        for strat in STRATEGIES:
            group = data_by_strategy[strat]
            if not group:
                continue
                
            x = [item["overlap_percent"] for item in group]
            y = [item[metric] for item in group]
            
            plt.plot(x, y, marker='o', linewidth=2, label=f"Strategy: {strat}")
            
        plt.title(titles[metric], fontsize=12, fontweight='bold', pad=15)
        plt.xlabel("Lexical Overlap (%)", fontsize=10)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        
        # Save visualization
        plt.savefig(output_dir / f"lexical_anchors_seed{seed}_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    args = parse_args()
    base_dir = Path(args.base_checkpoints_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Path to the specific single seed results file
    # Expects format: checkpoints_dir/seed_42/lexical_overlap_results.json (or similar metadata format)
    seed_dir = base_dir / f"seed_{args.seed}"
    seed_results_json = seed_dir / "lexical_overlap_results.json"
    
    if not seed_results_json.exists():
        print(f"Error: Could not find results file for seed {args.seed} at {seed_results_json}")
        print("Make sure your training script saves individual results there first.")
        return

    print(f"Loading results for seed {args.seed}...")
    data = json.loads(seed_results_json.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    
    if not rows:
        print("Warning: No evaluation rows found in the JSON file.")
        return

    # Filter/verify rows belong to this seed if necessary
    for row in rows:
        row["seed"] = args.seed

    # Save a filtered copy of the summary table just for this seed
    with open(out_dir / f"lexical_anchors_seed{args.seed}_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
        
    print(f"Generating single-seed plots inside: {out_dir}")
    generate_plots(rows, args.seed, out_dir)
    print("Done! Evaluation completed successfully.")


if __name__ == "__main__":
    main()