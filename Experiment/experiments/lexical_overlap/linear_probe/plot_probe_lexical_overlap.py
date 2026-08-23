"""
Script for plotting linear probing accuracy across layers for the Lexical Overlap Experiment.
Separated from execution scripts to avoid building probing corpus and training new models again.
Draws three subplots showing each strategy's performance across layers with varying target percentages.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

strategies = ["low", "mid", "high"]
percentages = ["P0", "P2.5", "P5", "P7.5", "P10"]

colors_plot1 = {
    "P0": "#1f77b4",
    "P2.5": "#2ca02c",
    "P5": "#ff7f0e",
    "P7.5": "#d62728",
    "P10": "#9467bd"
}
markers_plot1 = {
    "P0": "o", "P2.5": "s", "P5": "^", "P7.5": "D", "P10": "v"
}
labels_plot1 = {
    "P0": "0% overlap", "P2.5": "2.5% overlap", "P5": "5% overlap", "P7.5": "7.5% overlap", "P10": "10% overlap"
}

base_results_dir = "./probe_results"
output_plots_dir = "./plots"
os.makedirs(output_plots_dir, exist_ok=True)

plot1_raw_data = {} 

for strategy in strategies:
    plot1_raw_data[strategy] = {pct: {} for pct in percentages}
    
    for pct in percentages:
        actual_strategy = "high" if pct == "P0" else strategy
        
        csv_path = os.path.join(base_results_dir, f"{actual_strategy}_{pct}", "layerwise_accuracy.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                sorted_df = df.sort_values("layer")
                plot1_raw_data[strategy][pct] = {
                    "layers": sorted_df["layer"].tolist(),
                    "accuracies": sorted_df["accuracy"].tolist()
                }

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, dpi=300)

for i, strategy in enumerate(strategies):
    ax = axes1[i]
    ax.set_title(f"{strategy.capitalize()} Frequency Strategy", fontsize=12, pad=10)
    ax.set_xlabel("Layer", fontsize=11, labelpad=8)
    
    if i == 0:
        ax.set_ylabel("Probe Accuracy", fontsize=11, labelpad=8)
        
    for pct in percentages:
        pct_data = plot1_raw_data[strategy][pct]
        if pct_data:
            ax.plot(
                pct_data["layers"], 
                pct_data["accuracies"], 
                marker=markers_plot1[pct], 
                linewidth=1.8, 
                markersize=5, 
                color=colors_plot1[pct], 
                label=labels_plot1[pct]
            )
            
    ax.set_xticks(range(5))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    
    if i == 0:
        ax.legend(loc="upper left", frameon=True, fontsize=8.5)

fig1.tight_layout()
plot1_out = os.path.join(output_plots_dir, "lp_lexical_overlap.png")
fig1.savefig(plot1_out, dpi=300, bbox_inches="tight")
plt.close(fig1)

print(f"Plot saved to '{plot1_out}'")
