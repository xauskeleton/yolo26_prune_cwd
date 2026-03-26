"""Plot pruning methods benchmark - bar charts + training curves from CSV data."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "prune_bm"
SAVE_DIR = "prune_bm"

# Experiment configs: folder_name -> (display_name, color)
EXPERIMENTS = {
    "baseline": ("Baseline\n(unpruned)", "#607D8B"),
    "l1_norm":  ("L1 Norm",             "#2196F3"),
    "fpgm":     ("FPGM",                "#4CAF50"),
    "taylor":   ("Taylor",              "#FF9800"),
    "gamma+cwd":("BN Gamma\n+ CWD",     "#9C27B0"),
    "gamma":    ("BN Gamma",            "#E91E63"),
    "random":   ("Random",              "#F44336"),
}

# For training curves (shorter names, no newlines)
CURVE_NAMES = {
    "baseline": "Baseline",
    "l1_norm":  "L1 Norm",
    "fpgm":     "FPGM",
    "taylor":   "Taylor",
    "gamma+cwd":"BN Gamma+CWD",
    "gamma":    "BN Gamma",
    "random":   "Random",
}

def load_data():
    """Load all results.csv files."""
    data = {}
    for key in EXPERIMENTS:
        csv_path = os.path.join(BASE, key, "results.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        data[key] = df
    return data


def plot_bar_charts(data):
    """Plot 1: Bar chart comparing best mAP50 and mAP50-95."""
    methods, map50, map50_95, colors = [], [], [], []
    for key, (label, color) in EXPERIMENTS.items():
        if key not in data:
            continue
        df = data[key]
        methods.append(label)
        map50.append(df["metrics/mAP50(B)"].max())
        map50_95.append(df["metrics/mAP50-95(B)"].max())
        colors.append(color)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- mAP50 ---
    bars1 = ax1.bar(methods, map50, color=colors, edgecolor="white", linewidth=0.5)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0003,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_title("mAP50", fontsize=14, fontweight="bold")
    ax1.set_ylim(min(map50) - 0.015, max(map50) + 0.010)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", labelsize=9)

    # --- mAP50-95 ---
    bars2 = ax2.bar(methods, map50_95, color=colors, edgecolor="white", linewidth=0.5)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0003,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.set_title("mAP50-95", fontsize=14, fontweight="bold")
    ax2.set_ylim(min(map50_95) - 0.015, max(map50_95) + 0.010)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=9)

    fig.suptitle("Pruning Methods Benchmark (Best mAP)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "prune_benchmark_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_training_curves(data):
    """Plot 2: mAP50 and mAP50-95 training curves over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for key in EXPERIMENTS:
        if key not in data:
            continue
        df = data[key]
        _, color = EXPERIMENTS[key]
        label = CURVE_NAMES[key]
        epochs = df["epoch"].values

        ax1.plot(epochs, df["metrics/mAP50(B)"].values, color=color, linewidth=1.5,
                 label=f"{label} (best={df['metrics/mAP50(B)'].max():.3f})", alpha=0.85)
        ax2.plot(epochs, df["metrics/mAP50-95(B)"].values, color=color, linewidth=1.5,
                 label=f"{label} (best={df['metrics/mAP50-95(B)'].max():.3f})", alpha=0.85)

    for ax, title in [(ax1, "mAP50"), (ax2, "mAP50-95")]:
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Pruning Methods - Training Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "prune_benchmark_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_loss_curves(data):
    """Plot 3: Training loss curves (box + cls + dfl combined)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for key in EXPERIMENTS:
        if key not in data:
            continue
        df = data[key]
        _, color = EXPERIMENTS[key]
        label = CURVE_NAMES[key]
        epochs = df["epoch"].values

        train_loss = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]
        val_loss = df["val/box_loss"] + df["val/cls_loss"] + df["val/dfl_loss"]

        ax1.plot(epochs, train_loss.values, color=color, linewidth=1.5, label=label, alpha=0.85)
        ax2.plot(epochs, val_loss.values, color=color, linewidth=1.5, label=label, alpha=0.85)

    for ax, title in [(ax1, "Train Loss (box+cls+dfl)"), (ax2, "Val Loss (box+cls+dfl)")]:
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Pruning Methods - Loss Curves", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "prune_benchmark_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_delta_chart(data):
    """Plot 4: mAP drop from baseline (delta chart)."""
    if "baseline" not in data:
        print("Skipping delta chart: no baseline data")
        return

    base_map50 = data["baseline"]["metrics/mAP50(B)"].max()
    base_map50_95 = data["baseline"]["metrics/mAP50-95(B)"].max()

    methods, delta50, delta50_95, colors = [], [], [], []
    for key, (label, color) in EXPERIMENTS.items():
        if key == "baseline" or key not in data:
            continue
        df = data[key]
        methods.append(label)
        delta50.append((df["metrics/mAP50(B)"].max() - base_map50) * 100)
        delta50_95.append((df["metrics/mAP50-95(B)"].max() - base_map50_95) * 100)
        colors.append(color)

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, delta50, width, label="mAP50 drop (%)", color=[c + "AA" for c in colors],
                   edgecolor="white")
    bars2 = ax.bar(x + width / 2, delta50_95, width, label="mAP50-95 drop (%)", color=colors,
                   edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., val - 0.05,
                    f"{val:.2f}%", ha="center", va="top", fontweight="bold", fontsize=9, color="white")

    ax.set_ylabel("mAP Drop from Baseline (%)", fontsize=12)
    ax.set_title("mAP Drop After Pruning (vs Baseline)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "prune_benchmark_delta.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary(data):
    """Print summary table."""
    if "baseline" not in data:
        return

    base_50 = data["baseline"]["metrics/mAP50(B)"].max()
    base_50_95 = data["baseline"]["metrics/mAP50-95(B)"].max()

    print(f"\n{'=' * 75}")
    print(f"{'Method':<18} {'Epochs':>6} {'mAP50':>10} {'mAP50-95':>10} {'Drop50':>10} {'Drop50-95':>10}")
    print(f"{'=' * 75}")
    for key in EXPERIMENTS:
        if key not in data:
            continue
        df = data[key]
        label = CURVE_NAMES[key]
        m50 = df["metrics/mAP50(B)"].max()
        m50_95 = df["metrics/mAP50-95(B)"].max()
        d50 = (m50 - base_50) * 100
        d50_95 = (m50_95 - base_50_95) * 100
        epochs = len(df)
        sign50 = "+" if d50 >= 0 else ""
        sign95 = "+" if d50_95 >= 0 else ""
        print(f"{label:<18} {epochs:>6} {m50:>10.4f} {m50_95:>10.4f} {sign50}{d50:>9.2f}% {sign95}{d50_95:>9.2f}%")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    data = load_data()
    if not data:
        print("No data found!")
        exit(1)

    plot_bar_charts(data)
    plot_training_curves(data)
    plot_loss_curves(data)
    plot_delta_chart(data)
    print_summary(data)
    print("\nDone! All plots saved to prune_bm/")
