"""Iterative Pruning: prune N lan, moi lan prune r%.

Tong pruning = 1 - (1-r)^N
Vd: N=5, r=13% => tong = 1 - 0.87^5 = 50%

Usage:
    python scripts/iterative_prune.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pruning"))

from ultralytics import YOLO
from prune_common import load_and_prepare, create_masks, finalize_pruning
from prune_l1norm import compute_l1norm_importance

# ============ CONFIG ============
BASELINE = "weights/yolo26m_baseline.pt"
CFG = "cfg/yolo26m.yaml"
DATA = "VOC.yaml"
MODEL_SIZE = "m"    # n/s/m/l/x

ITERS = 5          # so lan prune
PRUNE_RATIO = 0.13 # prune moi lan (13%)

EPOCHS = 10        # finetune epochs moi iteration
# ================================

if __name__ == "__main__":
    total = 1 - (1 - PRUNE_RATIO) ** ITERS
    print(f"Iterative Pruning: {ITERS} lan x {PRUNE_RATIO*100:.1f}% = ~{total*100:.1f}% tong")

    current = BASELINE
    save_dir = Path("weights/iterative")
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, ITERS + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {i}/{ITERS} — prune {PRUNE_RATIO*100:.1f}% tu {Path(current).name}")
        print(f"{'='*60}")

        # 1. Prune
        model, bn_dict, ignore_bn_list, _, layer_ratio_cfg, pruned_yaml = \
            load_and_prepare(current, CFG, MODEL_SIZE, None)

        importance = compute_l1norm_importance(model, bn_dict, ignore_bn_list)

        maskbndict = create_masks(
            importance, model, ignore_bn_list, layer_ratio_cfg, PRUNE_RATIO, 8
        )

        pruned_path = finalize_pruning(
            model, maskbndict, pruned_yaml, ignore_bn_list,
            current, str(save_dir), 8, PRUNE_RATIO,
            method_name=f"iter{i}"
        )

        # 2. Finetune
        print(f"\nFinetune  ({EPOCHS} epochs)...")
        ft = YOLO(pruned_path)
        ft.train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=640,
            batch=16,
            device=0,
            finetune=True,
            project=str(save_dir / "runs"),
            name=f"iter{i}",
        )

        # Next iteration dung best checkpoint
        best = save_dir / "runs" / f"iter{i}" / "weights" / "best.pt"
        current = str(best if best.exists() else save_dir / "runs" / f"iter{i}" / "weights" / "last.pt")

        done = 1 - (1 - PRUNE_RATIO) ** i
        print(f"Iter {i} done — ~{done*100:.1f}% total pruned — {current}")

    print(f"\n{'='*60}")
    print(f"DONE! Final model: {current}")
    print(f"Total pruning: ~{total*100:.1f}%")
    print(f"{'='*60}")
