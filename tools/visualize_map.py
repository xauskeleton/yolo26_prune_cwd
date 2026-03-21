"""Visualize mAP50 and mAP50-95 across temperatures (1-10) vs finetune."""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "prune_ckpt/prune"

def find_csv(name):
    """Find results.csv for a given experiment."""
    candidates = [
        os.path.join(BASE, name, "results", "train", "results.csv"),
        os.path.join(BASE, name, "results.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

# Load data
experiments = {f"t{i}": f"T={i}" for i in range(1, 11)}
experiments["finetune"] = "Finetune"

best_map50 = {}
best_map50_95 = {}

for key, label in experiments.items():
    csv_path = find_csv(key)
    if csv_path is None:
        print(f"Warning: CSV not found for {key}")
        continue
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    best_map50[label] = df["metrics/mAP50(B)"].max() * 100
    best_map50_95[label] = df["metrics/mAP50-95(B)"].max() * 100

# Separate temperature and finetune
temp_labels = [f"T={i}" for i in range(1, 11)]
ft_label = "Finetune"

temp_map50 = [best_map50[l] for l in temp_labels]
temp_map50_95 = [best_map50_95[l] for l in temp_labels]
ft_map50 = best_map50[ft_label]
ft_map50_95 = best_map50_95[ft_label]

# --- Plot 1: Bar chart comparing all ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

all_labels = temp_labels + [ft_label]
all_map50 = temp_map50 + [ft_map50]
all_map50_95 = temp_map50_95 + [ft_map50_95]

colors = ["#4C72B0"] * 10 + ["#DD8452"]  # blue for temps, orange for finetune

# mAP50
ax = axes[0]
bars = ax.bar(all_labels, all_map50, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("mAP50 (%)", fontsize=13)
ax.set_title("Best mAP50 (%)", fontsize=14, fontweight="bold")
ax.set_ylim(min(all_map50) - 0.3, max(all_map50) + 0.3)
for bar, val in zip(bars, all_map50):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(y=ft_map50, color="#DD8452", linestyle="--", alpha=0.7, label=f"Finetune = {ft_map50:.2f}%")
ax.legend(fontsize=10)
ax.tick_params(axis="x", rotation=45)

# mAP50-95
ax = axes[1]
bars = ax.bar(all_labels, all_map50_95, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("mAP50-95 (%)", fontsize=13)
ax.set_title("Best mAP50-95 (%)", fontsize=14, fontweight="bold")
ax.set_ylim(min(all_map50_95) - 0.3, max(all_map50_95) + 0.3)
for bar, val in zip(bars, all_map50_95):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(y=ft_map50_95, color="#DD8452", linestyle="--", alpha=0.7, label=f"Finetune = {ft_map50_95:.2f}%")
ax.legend(fontsize=10)
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("prune_ckpt/map_comparison_bar.png", dpi=200, bbox_inches="tight")
print("Saved: prune_ckpt/map_comparison_bar.png")

# --- Plot 2: Line chart (temperature trend) ---
fig, ax1 = plt.subplots(figsize=(10, 6))

temps = list(range(1, 11))
ax1.plot(temps, temp_map50, "o-", color="#4C72B0", linewidth=2, markersize=8, label="mAP50 (%)")
ax1.axhline(y=ft_map50, color="#4C72B0", linestyle="--", alpha=0.5, label=f"Finetune mAP50 = {ft_map50:.2f}%")

ax1.set_xlabel("Temperature", fontsize=13)
ax1.set_ylabel("mAP50 (%)", fontsize=13, color="#4C72B0")
ax1.set_xticks(temps)
ax1.tick_params(axis="y", labelcolor="#4C72B0")

# Annotate mAP50
for t, v in zip(temps, temp_map50):
    ax1.annotate(f"{v:.2f}", (t, v), textcoords="offset points", xytext=(0, 10),
                 ha="center", fontsize=8, color="#4C72B0", fontweight="bold")

ax2 = ax1.twinx()
ax2.plot(temps, temp_map50_95, "s-", color="#C44E52", linewidth=2, markersize=8, label="mAP50-95 (%)")
ax2.axhline(y=ft_map50_95, color="#C44E52", linestyle="--", alpha=0.5, label=f"Finetune mAP50-95 = {ft_map50_95:.2f}%")

ax2.set_ylabel("mAP50-95 (%)", fontsize=13, color="#C44E52")
ax2.tick_params(axis="y", labelcolor="#C44E52")

# Annotate mAP50-95
for t, v in zip(temps, temp_map50_95):
    ax2.annotate(f"{v:.2f}", (t, v), textcoords="offset points", xytext=(0, -15),
                 ha="center", fontsize=8, color="#C44E52", fontweight="bold")

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", fontsize=10, ncol=2)

plt.title("CWD Temperature vs mAP (Best epoch)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("prune_ckpt/map_comparison_line.png", dpi=200, bbox_inches="tight")
print("Saved: prune_ckpt/map_comparison_line.png")

# --- Print summary table ---
print("\n" + "=" * 65)
print(f"{'Experiment':<12} {'mAP50 (%)':<14} {'mAP50-95 (%)':<14} {'vs FT (mAP50-95)'}")
print("=" * 65)
for label in temp_labels + [ft_label]:
    diff = best_map50_95[label] - ft_map50_95
    sign = "+" if diff >= 0 else ""
    marker = " <-- best" if best_map50_95[label] == max(all_map50_95) else ""
    print(f"{label:<12} {best_map50[label]:<14.2f} {best_map50_95[label]:<14.2f} {sign}{diff:.2f}%{marker}")
print("=" * 65)

print("Done!")
