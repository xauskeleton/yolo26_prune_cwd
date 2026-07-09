import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dms.dms_utils import build_ignore_bn_list
from ultralytics import YOLO


def check_sparsity(model_path, threshold=0.01):
    yolo_model = YOLO(model_path)
    model = yolo_model.model
    model.eval()

    ignore_bn_list = build_ignore_bn_list(model)

    all_gammas = []
    prunable_gammas = []
    layer_info = []

    for k, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            g = m.weight.data.abs().cpu()
            all_gammas.append(g)
            ignored = k in ignore_bn_list
            if not ignored:
                prunable_gammas.append(g)
            sparse = (g < threshold).sum().item()
            layer_info.append((k, len(g), sparse, ignored))

    all_g = torch.cat(all_gammas).numpy()
    prune_g = torch.cat(prunable_gammas).numpy()

    # Stats
    all_sparsity = (all_g < threshold).mean() * 100
    prune_sparsity = (prune_g < threshold).mean() * 100

    print(f"{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"Threshold: {threshold}")
    print(f"{'=' * 60}")
    print(f"{'Tat ca BN':<25} Channels: {len(all_g):>6}  Sparsity: {all_sparsity:.2f}%")
    print(f"{'Loai ignore BN':<25} Channels: {len(prune_g):>6}  Sparsity: {prune_sparsity:.2f}%")
    print(f"Ignore BN layers: {len(ignore_bn_list)}")
    print(f"Gamma mean: {prune_g.mean():.4f}  std: {prune_g.std():.4f}")
    print(f"{'=' * 60}")

    # Per-layer info
    print(f"\n{'Layer':<45} {'Ch':>5} {'Sparse':>6} {'%':>7} {'Ignore'}")
    print("-" * 75)
    for name, total, sparse, ignored in layer_info:
        pct = sparse / total * 100 if total > 0 else 0
        tag = "  [SKIP]" if ignored else ""
        print(f"{name:<45} {total:>5} {sparse:>6} {pct:>6.1f}%{tag}")

    # Plot
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Histogram prunable
    axes[0].hist(prune_g, bins=100, color="crimson", alpha=0.7)
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold}")
    axes[0].set_title(f"Prunable BN Gamma (sparsity={prune_sparsity:.1f}%)")
    axes[0].set_xlabel("|gamma|")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # 2. Sorted gamma
    axes[1].plot(np.sort(prune_g), color="blue", lw=1.5)
    axes[1].axhline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
    axes[1].set_title("Sorted Prunable Gamma")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("|gamma|")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = "weights/yolo26n_sparsed.pt"
    threshold = 0.01
    check_sparsity(path, threshold)
