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
            print(
                f"Skipping strategy: {strategy} for overlap: {percentage}% "
                "(only one strategy is needed for 0% overlap)"
            )
            continue

        print(f"Evaluating LM head for strategy: {strategy}, overlap: {percentage}%")

        pct_label = int(float(percentage))
        condition = f"{strategy}_P{pct_label}"
        
        checkpoint_dir = f"./checkpoints/{condition}/final"
        out_dir = f"./lm_head_results/{condition}"

        sys.argv = [
            "lm_head_eval.py",
            "--model_dir", checkpoint_dir,
            "--final_omitted", FINAL_OMITTED,
            "--parallel", PARALLEL,
            "--cjk_dict", CJK_DICT,
            "--hira_dict", HIRA_DICT,
            "--out_dir", out_dir,
            "--topk", "3"
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
            row = df[df["metric"] == "strict_token"]
            if not row.empty:
                data.append({
                    "percentage": pct_float,
                    "strategy": strategy,
                    "top1": row["top1"].values[0]
                })

results_df = pd.DataFrame(data)

if not results_df.empty and 0.0 in results_df["percentage"].values:
    zero_val = results_df[results_df["percentage"] == 0.0].iloc[0]
    for strat in ["low", "mid"]:
        if results_df[(results_df["percentage"] == 0.0) & (results_df["strategy"] == strat)].empty:
            new_row = zero_val.to_dict()
            new_row["strategy"] = strat
            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

fig, ax = plt.subplots(figsize=(7, 5))

styles = {
    "low": {"color": "#1f77b4", "marker": "o", "label": "Low"},
    "mid": {"color": "#ff7f0e", "marker": "s", "label": "Mid"},
    "high": {"color": "#2ca02c", "marker": "^", "label": "High"}
}

for strat in STRATEGIES:
    sub = results_df[results_df["strategy"] == strat].sort_values("percentage")
    if not sub.empty:
        ax.plot(
            sub["percentage"], sub["top1"] * 100,
            color=styles[strat]["color"],
            marker=styles[strat]["marker"],
            linewidth=2,
            markersize=7,
            label=f"LM Head {styles[strat]['label']} top-1"
        )

ax.set_xlabel("Lexical overlap (%)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xticks([0.0, 2.5, 5.0, 7.5, 10.0])
ax.set_ylim(0, 20)
ax.set_yticks([0, 5, 10, 15, 20])
ax.yaxis.set_major_formatter("{x:.0f}%")
ax.grid(True, linestyle="-", alpha=0.2, color="gray")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, loc="upper right")

plt.tight_layout()
output_fig_path = Path("./lm_head_results/lm_head_overlap_top1_accuracy.png")
output_fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved summary plot with reordered legend -> {output_fig_path}")