"""
This script automates the LM-head evaluation across different lexical overlap conditions
and plots the aggregated top-1 accuracy results.

Needed files:
    - lm_head_lexical_overlap.py
    - probing_run/final_omitted.json
    - probing_run/final_omitted_corpus/parallel_corpus_synset.json
    - synset_pos_artificial_cjk_edges_adj_augmented.json
    - synset_pos_artificial_hiragana_edges_adj_augmented.json
    - checkpoints_probe/{strategy}_P{int(float(percentage))}

Where {PERCENTAGE} is one of 0.0, 2.5, 5.0, 7.5, 10.0 and {STRATEGY} is one of high, mid, low.

"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

import lm_head_lexical_overlap as head

STRATEGIES = ["low", "mid", "high"]
PERCENTAGE_VALUES = ["0.0", "2.5", "5.0", "7.5", "10.0"]

CJK_DICT = "synset_pos_artificial_cjk_edges_adj_augmented.json"
HIRA_DICT = "synset_pos_artificial_hiragana_edges_adj_augmented.json"
FINAL_OMITTED = "probing_run/final_omitted.json"
PARALLEL = "probing_run/final_omitted_corpus/parallel_corpus_synset.json"

for percentage in PERCENTAGE_VALUES:
    for strategy in STRATEGIES:
        if percentage == "0.0" and strategy != "high":
            continue

        pct_label = int(float(percentage))
        condition = f"{strategy}_P{pct_label}"
        
        checkpoint_dir = f"./checkpoints_probe/{condition}/final"
        out_dir = f"./lm_head_results/{condition}"

        sys.argv = [
            "lm_head_eval.py",
            "--model_dir", checkpoint_dir,
            "--final_omitted", FINAL_OMITTED,
            "--parallel", PARALLEL,
            "--cjk_dict", CJK_DICT,
            "--hira_dict", HIRA_DICT,
            "--out_dir", out_dir
        ]
        head.main()

data = []
for percentage in PERCENTAGE_VALUES:
    pct_float = float(percentage)
    pct_label = int(pct_float) if pct_float.is_integer() else pct_float
    
    for strategy in STRATEGIES:
        strat_key = "high" if percentage == "0.0" else strategy
        csv_path = Path(f"./lm_head_results/{strat_key}_P{pct_label}/lm_head_accuracy.csv")
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            row_token = df[df["metric"] == "strict_token"]
            row_concept = df[df["metric"] == "strict_concept"]
            
            if not row_token.empty and not row_concept.empty:
                data.append({
                    "percentage": pct_float,
                    "strategy": strategy,
                    "strict_token_top1": row_token["top1"].values[0],
                    "strict_concept_top1": row_concept["top1"].values[0]
                })

results_df = pd.DataFrame(data)

if not results_df.empty and 0.0 in results_df["percentage"].values:
    zero_val = results_df[results_df["percentage"] == 0.0].iloc[0]
    for strat in ["low", "mid"]:
        if results_df[(results_df["percentage"] == 0.0) & (results_df["strategy"] == strat)].empty:
            new_row = zero_val.to_dict()
            new_row["strategy"] = strat
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, dpi=300)

styles = {
    "low": {"color": "#1f77b4", "marker": "o", "label": "LM Head Low"},
    "mid": {"color": "#ff7f0e", "marker": "s", "label": "LM Head Mid"},
    "high": {"color": "#2ca02c", "marker": "^", "label": "LM Head High"}
}

metrics_info = [
    {"col": "strict_token_top1", "title": "Strict Token (Top-1)", "ylabel": "Accuracy"},
    {"col": "strict_concept_top1", "title": "Strict Concept (Top-1)", "ylabel": "Accuracy"}
]

for idx, info in enumerate(metrics_info):
    ax = axes[idx]
    
    for strat in STRATEGIES:
        sub = results_df[results_df["strategy"] == strat].sort_values("percentage")
        if not sub.empty:
            ax.plot(
                sub["percentage"], sub[info["col"]] * 100,
                color=styles[strat]["color"],
                marker=styles[strat]["marker"],
                linewidth=2,
                markersize=6,
                label=styles[strat]["label"]
            )

    ax.set_title(info["title"], fontsize=13, pad=10)
    ax.set_xlabel("Lexical overlap (%)", fontsize=11)
    ax.set_ylabel(info["ylabel"] if idx == 0 else "", fontsize=11)
    ax.set_xticks([0.0, 2.5, 5.0, 7.5, 10.0])
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.yaxis.set_major_formatter("{x:.0f}%")
    ax.grid(True, linestyle="-", alpha=0.2, color="gray")
    
    if idx == 0:
        ax.legend(frameon=True, loc="upper right", fontsize=9.5)

plt.tight_layout()

output_fig_path = Path("./lm_head_results/lmh_lexical_overlap.png")
output_fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
