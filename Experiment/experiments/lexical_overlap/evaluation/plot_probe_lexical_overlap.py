import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Experimental setup definitions
strategies = ["high", "mid", "low"]
percentages = ["P0", "P2.5", "P5", "P7.5", "P10"]
seeds = [42, 43, 44]
x_ticks_values = [0, 2.5, 5, 7.5, 10]

# Formatting configs to match user requirements exactly
colors_plot1 = {
    "P0": "#1f77b4",    # Blue
    "P2.5": "#2ca02c",  # Green
    "P5": "#ff7f0e",    # Orange
    "P7.5": "#d62728",  # Red
    "P10": "#9467bd"    # Purple
}
markers_plot1 = {
    "P0": "o", "P2.5": "s", "P5": "^", "P7.5": "D", "P10": "v"
}
labels_plot1 = {
    "P0": "0% overlap", "P2.5": "2.5% overlap", "P5": "5% overlap", "P7.5": "7.5% overlap", "P10": "10% overlap"
}

colors_plot2 = {"Best": "#1f77b4", "Final": "#d62728", "Embedding": "#2ca02c"}

base_results_dir = "./probe_results"
output_plots_dir = "./plots"
os.makedirs(output_plots_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# STEP 1: Load all csv data across seeds and aggregate metrics
# -----------------------------------------------------------------------------
plot1_raw_data = {} 
plot2_agg_data = {}

for strategy in strategies:
    plot1_raw_data[strategy] = {pct: {} for pct in percentages}
    plot2_agg_data[strategy] = []
    
    for pct in percentages:
        pct_num = float(pct.replace("P", ""))
        
        cell_best_accs = []
        cell_final_accs = []
        cell_embed_accs = []
        
        layer_accs_across_seeds = {}
        
        for seed in seeds:
            csv_path = os.path.join(base_results_dir, f"{strategy}_{pct}", f"seed_{seed}", "layerwise_accuracy.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if not df.empty:
                    embed_acc = df[df["layer"] == 0]["accuracy"].values[0]
                    final_acc = df[df["layer"] == df["layer"].max()]["accuracy"].values[0]
                    best_acc = df["accuracy"].max()
                    
                    cell_embed_accs.append(embed_acc)
                    cell_final_accs.append(final_acc)
                    cell_best_accs.append(best_acc)
                    
                    for _, row in df.iterrows():
                        L = int(row["layer"])
                        acc = float(row["accuracy"])
                        if L not in layer_accs_across_seeds:
                            layer_accs_across_seeds[L] = []
                        layer_accs_across_seeds[L].append(acc)
        
        # Aggregate Layer-wise metrics (Mean AND StDev) for Plot 1 sliding regions
        if layer_accs_across_seeds:
            sorted_layers = sorted(layer_accs_across_seeds.keys())
            mean_accs_per_layer = [np.mean(layer_accs_across_seeds[layer]) for layer in sorted_layers]
            std_accs_per_layer = [np.std(layer_accs_across_seeds[layer]) for layer in sorted_layers]
            print(f"[{strategy} - {pct}] StDev per layer: {std_accs_per_layer}")
            plot1_raw_data[strategy][pct] = {
                "layers": sorted_layers,
                "mean_accuracies": mean_accs_per_layer,
                "std_accuracies": std_accs_per_layer
            }
            
        if cell_best_accs:
            plot2_agg_data[strategy].append({
                "pct_val": pct_num,
                "best_mean": np.mean(cell_best_accs), "best_std": np.std(cell_best_accs),
                "final_mean": np.mean(cell_final_accs), "final_std": np.std(cell_final_accs),
                "embed_mean": np.mean(cell_embed_accs), "embed_std": np.std(cell_embed_accs)
            })

# -----------------------------------------------------------------------------
# STEP 2: RENDER PLOT 1 (Accuracy vs Layers with Shaded StDev Regions)
# -----------------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for i, strategy in enumerate(strategies):
    ax = axes1[i]
    ax.set_title(f"{strategy.capitalize()} Frequency Strategy", fontsize=12, fontweight="normal", pad=12)
    ax.set_xlabel("Layer", fontsize=11, labelpad=8)
    
    if i == 0:
        ax.set_ylabel("Probe Accuracy", fontsize=11, labelpad=8)
        
    has_data = False
    for pct in percentages:
        pct_data = plot1_raw_data[strategy][pct]
        if pct_data:
            has_data = True
            layers = np.array(pct_data["layers"])
            means = np.array(pct_data["mean_accuracies"])
            stds = np.array(pct_data["std_accuracies"])
            
            # Plot the central mean line
            ax.plot(
                layers, 
                means, 
                marker=markers_plot1[pct], 
                linewidth=2.0, 
                markersize=6, 
                color=colors_plot1[pct], 
                label=labels_plot1[pct]
            )
            # Added sliding shaded regions for Plot 1 layer deviations
            ax.fill_between(layers, means - stds, means + stds, color=colors_plot1[pct], alpha=0.30)
            
    if not has_data:
        ax.text(0.5, 0.5, f"No data for {strategy}", ha="center", va="center", transform=ax.transAxes)
        continue
        
    ax.set_xticks(sorted(list(layer_accs_across_seeds.keys())))
    # Y-axis adjusted to run from 0.0 to 0.7 exactly
    ax.set_ylim(0.0, 0.7)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    if i == 0:
        ax.legend(loc="upper left", frameon=True, fontsize=9)

fig1.tight_layout()
plot1_out = os.path.join(output_plots_dir, "lexical_overlap_combined.png")
fig1.savefig(plot1_out, dpi=300, bbox_inches="tight")
plt.close(fig1)
print(f"Saved Plot 1 (Layer-wise profiles) to: {plot1_out}")

# -----------------------------------------------------------------------------
# STEP 3: RENDER PLOT 2 (Accuracy vs Overlap with Shaded Standard Deviation)
# -----------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for i, strategy in enumerate(strategies):
    ax = axes2[i]
    strat_cells = plot2_agg_data[strategy]
    
    strat_cells.sort(key=lambda x: x["pct_val"])
    
    if not strat_cells:
        ax.text(0.5, 0.5, f"No data for {strategy}", ha="center", va="center", transform=ax.transAxes)
        continue
        
    xs = [c["pct_val"] for c in strat_cells]
    
    b_mean = np.array([c["best_mean"] for c in strat_cells])
    f_mean = np.array([c["final_mean"] for c in strat_cells])
    e_mean = np.array([c["embed_mean"] for c in strat_cells])
    
    b_std = np.array([c["best_std"] for c in strat_cells])
    f_std = np.array([c["final_std"] for c in strat_cells])
    e_std = np.array([c["embed_std"] for c in strat_cells])
    
    ax.plot(xs, b_mean, marker="o", linewidth=2.5, markersize=6, color=colors_plot2["Best"], label="Best layer")
    ax.plot(xs, f_mean, marker="o", linewidth=2.5, markersize=6, color=colors_plot2["Final"], label="Final layer")
    ax.plot(xs, e_mean, marker="o", linewidth=2.5, markersize=6, color=colors_plot2["Embedding"], label="Embedding layer")
    
    ax.fill_between(xs, b_mean - b_std, b_mean + b_std, color=colors_plot2["Best"], alpha=0.15)
    ax.fill_between(xs, f_mean - f_std, f_mean + f_std, color=colors_plot2["Final"], alpha=0.15)
    ax.fill_between(xs, e_mean - e_std, e_mean + e_std, color=colors_plot2["Embedding"], alpha=0.15)
    
    ax.set_title(f"{strategy.capitalize()} Frequency Strategy", fontsize=12, fontweight="normal", pad=12)
    ax.set_xlabel("Lexical overlap", fontsize=11, labelpad=8)
    
    if i == 0:
        ax.set_ylabel("Probe Accuracy", fontsize=11, labelpad=8)
        ax.legend(loc="upper left", frameon=True, fontsize=10)
        
    ax.set_xticks(x_ticks_values)
    ax.set_xticklabels([f"{v}%" for v in x_ticks_values])
    # Y-axis adjusted to run from 0.0 to 0.7 exactly
    ax.set_ylim(0.0, 0.7)
    ax.grid(True, linestyle="--", alpha=0.5)

fig2.tight_layout()
plot2_out = os.path.join(output_plots_dir, "lexical_overlap_trends.png")
fig2.savefig(plot2_out, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved Plot 2 (Overlap trendlines with StDev shadows) to: {plot2_out}")