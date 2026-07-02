"""Plot DMS per-layer pruning ratios as horizontal bar chart."""

import matplotlib
import yaml

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Layer grouping for visualization
GROUPS = {
    "Stem": ["model.0.bn", "model.1.bn"],
    "Block 2": [
        k
        for k in []
        or [
            "model.2.cv1.bn",
            "model.2.cv2.bn",
            "model.2.m.0.cv2.bn",
            "model.2.m.0.cv3.bn",
            "model.2.m.0.m.0.cv1.bn",
            "model.2.m.0.m.1.cv1.bn",
        ]
    ],
    "Block 3-4": [
        "model.3.bn",
        "model.4.cv1.bn",
        "model.4.cv2.bn",
        "model.4.m.0.cv2.bn",
        "model.4.m.0.cv3.bn",
        "model.4.m.0.m.0.cv1.bn",
        "model.4.m.0.m.1.cv1.bn",
    ],
    "Block 5-6": [
        "model.5.bn",
        "model.6.cv1.bn",
        "model.6.cv2.bn",
        "model.6.m.0.cv2.bn",
        "model.6.m.0.cv3.bn",
        "model.6.m.0.m.0.cv1.bn",
        "model.6.m.0.m.1.cv1.bn",
    ],
    "Block 7-8": [
        "model.7.bn",
        "model.8.cv1.bn",
        "model.8.cv2.bn",
        "model.8.m.0.cv2.bn",
        "model.8.m.0.cv3.bn",
        "model.8.m.0.m.0.cv1.bn",
        "model.8.m.0.m.1.cv1.bn",
    ],
    "SPPF+C2PSA": [
        "model.9.cv1.bn",
        "model.9.cv2.bn",
        "model.10.cv2.bn",
    ],
    "Neck P3 (16)": [
        "model.16.cv1.bn",
        "model.16.cv2.bn",
        "model.16.m.0.cv2.bn",
        "model.16.m.0.cv3.bn",
        "model.16.m.0.m.0.cv1.bn",
        "model.16.m.0.m.1.cv1.bn",
    ],
    "Neck P4 (13,19)": [
        "model.13.cv1.bn",
        "model.13.cv2.bn",
        "model.13.m.0.cv2.bn",
        "model.13.m.0.cv3.bn",
        "model.13.m.0.m.0.cv1.bn",
        "model.13.m.0.m.1.cv1.bn",
        "model.17.bn",
        "model.19.cv1.bn",
        "model.19.cv2.bn",
        "model.19.m.0.cv2.bn",
        "model.19.m.0.cv3.bn",
        "model.19.m.0.m.0.cv1.bn",
        "model.19.m.0.m.1.cv1.bn",
        "model.20.bn",
    ],
    "Neck P5 (22)": [
        "model.22.cv2.bn",
        "model.22.m.0.0.cv1.bn",
    ],
    "Detect cv2": [
        k
        for k in [
            "model.23.cv2.0.0.bn",
            "model.23.cv2.0.1.bn",
            "model.23.cv2.1.0.bn",
            "model.23.cv2.1.1.bn",
            "model.23.cv2.2.0.bn",
            "model.23.cv2.2.1.bn",
        ]
    ],
    "Detect cv3": [
        k
        for k in [
            "model.23.cv3.0.0.0.bn",
            "model.23.cv3.0.0.1.bn",
            "model.23.cv3.0.1.0.bn",
            "model.23.cv3.0.1.1.bn",
            "model.23.cv3.1.0.0.bn",
            "model.23.cv3.1.0.1.bn",
            "model.23.cv3.1.1.0.bn",
            "model.23.cv3.1.1.1.bn",
            "model.23.cv3.2.0.0.bn",
            "model.23.cv3.2.0.1.bn",
            "model.23.cv3.2.1.0.bn",
            "model.23.cv3.2.1.1.bn",
        ]
    ],
}

GROUP_COLORS = {
    "Stem": "#F44336",
    "Block 2": "#E91E63",
    "Block 3-4": "#9C27B0",
    "Block 5-6": "#673AB7",
    "Block 7-8": "#3F51B5",
    "SPPF+C2PSA": "#2196F3",
    "Neck P3 (16)": "#009688",
    "Neck P4 (13,19)": "#4CAF50",
    "Neck P5 (22)": "#8BC34A",
    "Detect cv2": "#FF9800",
    "Detect cv3": "#FF5722",
}


def plot_ratios(ratios, save_path="dms_ratios.png", title="DMS Per-Layer Pruning Ratios"):
    """Plot horizontal bar chart of DMS ratios grouped by region."""
    labels = []
    values = []
    colors = []
    group_positions = []

    pos = 0
    for group_name, layer_names in GROUPS.items():
        group_start = pos
        color = GROUP_COLORS.get(group_name, "#607D8B")
        found = False
        for name in layer_names:
            if name in ratios:
                short = name.replace("model.", "").replace(".bn", "")
                labels.append(short)
                values.append(ratios[name])
                colors.append(color)
                pos += 1
                found = True
        if found:
            group_positions.append((group_name, group_start, pos - 1))
            pos += 0.5  # gap between groups

    # Collect ungrouped layers
    grouped_layers = set()
    for layer_names in GROUPS.values():
        grouped_layers.update(layer_names)
    ungrouped = {k: v for k, v in ratios.items() if k not in grouped_layers}
    if ungrouped:
        group_start = pos
        for name, val in sorted(ungrouped.items()):
            short = name.replace("model.", "").replace(".bn", "")
            labels.append(short)
            values.append(val)
            colors.append("#607D8B")
            pos += 1
        group_positions.append(("Other", group_start, pos - 1))

    fig_height = max(8, len(labels) * 0.28)
    _fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.3, height=0.7)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Pruning Ratio (a)", fontsize=11)
    ax.set_xlim(0, 1.05)
    ax.axvline(
        x=np.mean(list(ratios.values())),
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"avg={np.mean(list(ratios.values())):.3f}",
    )

    # Group labels on right side
    for group_name, start, end in group_positions:
        mid = (start + end) / 2
        color = GROUP_COLORS.get(group_name, "#607D8B")
        ax.text(
            1.02,
            mid,
            group_name,
            va="center",
            fontsize=8,
            fontweight="bold",
            color=color,
            transform=ax.get_yaxis_transform(),
        )

    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot DMS per-layer pruning ratios")
    parser.add_argument("--ratios", type=str, default="dms_ratios.yaml", help="Path to DMS ratios YAML file")
    parser.add_argument("--output", type=str, default="dms_ratios_plot.png", help="Output image path")
    parser.add_argument("--title", type=str, default="DMS Per-Layer Pruning Ratios", help="Plot title")
    args = parser.parse_args()

    with open(args.ratios) as f:
        ratios = yaml.safe_load(f)

    print(f"Loaded {len(ratios)} layers from {args.ratios}")
    print(f"  avg={np.mean(list(ratios.values())):.4f}, min={min(ratios.values()):.4f}, max={max(ratios.values()):.4f}")

    plot_ratios(ratios, save_path=args.output, title=args.title)


if __name__ == "__main__":
    main()
