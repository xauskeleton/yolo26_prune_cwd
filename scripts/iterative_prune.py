"""Iterative Pruning: prune N lan, moi lan prune r%.

Tong pruning = 1 - (1-r)^N
Vd: N=5, r=13% => tong = 1 - 0.87^5 = 50%

Usage:
    python scripts/iterative_prune.py --weights yolo26m.pt --cfg cfg/yolo26m.yaml --data VOC.yaml \
        --iters 5 --prune-ratio 0.13 --epochs 10
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pruning"))

from ultralytics import YOLO
from prune_common import load_and_prepare, create_masks, finalize_pruning
from prune_l1norm import compute_l1norm_importance


def parse_args():
    p = argparse.ArgumentParser(description="Iterative L1 Pruning")
    p.add_argument("--weights", type=str, required=True, help="Baseline model path")
    p.add_argument("--cfg", type=str, default="cfg/yolo26m.yaml", help="YAML config")
    p.add_argument("--data", type=str, default="VOC.yaml", help="Dataset YAML")
    p.add_argument("--model-size", type=str, default="m", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--iters", type=int, default=5, help="So lan prune")
    p.add_argument("--prune-ratio", type=float, default=0.13, help="Prune ratio moi lan")
    p.add_argument("--divisor", type=int, default=8, choices=[8, 16])
    p.add_argument("--epochs", type=int, default=10, help="Finetune epochs moi iteration")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--save-dir", type=str, default="weights/iterative")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total = 1 - (1 - args.prune_ratio) ** args.iters
    print(f"Iterative Pruning: {args.iters} lan x {args.prune_ratio*100:.1f}% = ~{total*100:.1f}% tong")

    current = args.weights
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, args.iters + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {i}/{args.iters} — prune {args.prune_ratio*100:.1f}% tu {Path(current).name}")
        print(f"{'='*60}")

        # 1. Prune
        model, bn_dict, ignore_bn_list, _, layer_ratio_cfg, pruned_yaml = \
            load_and_prepare(current, args.cfg, args.model_size, None)

        importance = compute_l1norm_importance(model, bn_dict, ignore_bn_list)

        maskbndict = create_masks(
            importance, model, ignore_bn_list, layer_ratio_cfg, args.prune_ratio, args.divisor
        )

        pruned_path = finalize_pruning(
            model, maskbndict, pruned_yaml, ignore_bn_list,
            current, str(save_dir), args.divisor, args.prune_ratio,
            method_name=f"iter{i}"
        )

        # 2. Finetune
        print(f"\nFinetune ({args.epochs} epochs)...")
        ft = YOLO(pruned_path)
        ft.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            finetune=True,
            project=str(save_dir / "runs"),
            name=f"iter{i}",
        )

        # Next iteration dung best checkpoint
        best = save_dir / "runs" / f"iter{i}" / "weights" / "best.pt"
        current = str(best if best.exists() else save_dir / "runs" / f"iter{i}" / "weights" / "last.pt")

        done = 1 - (1 - args.prune_ratio) ** i
        print(f"Iter {i} done — ~{done*100:.1f}% total pruned — {current}")

    print(f"\n{'='*60}")
    print(f"DONE! Final model: {current}")
    print(f"Total pruning: ~{total*100:.1f}%")
    print(f"{'='*60}")