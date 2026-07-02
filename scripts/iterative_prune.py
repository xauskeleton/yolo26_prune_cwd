"""Iterative Pruning: prune N lan, moi lan prune r%.

Tong pruning = 1 - (1-r)^N
Vd: N=5, r=13% => tong = 1 - 0.87^5 = 50%

Usage:
    python scripts/iterative_prune.py --weights weights/yolo26m_baseline.pt --iters 5 --prune-ratio 0.165 --epochs 0
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pruning"))

from prune_common import create_masks, finalize_pruning, load_and_prepare
from prune_l1norm import compute_l1norm_importance

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Iterative L1 Norm Pruning")
    parser.add_argument("--weights", type=str, default="weights/yolo26m_baseline.pt", help="baseline model path")
    parser.add_argument("--cfg", type=str, default="cfg/yolo26m.yaml", help="model YAML config")
    parser.add_argument("--data", type=str, default="VOC.yaml", help="dataset config for finetune")
    parser.add_argument("--model-size", type=str, default="m", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--iters", type=int, default=5, help="number of prune iterations")
    parser.add_argument("--prune-ratio", type=float, default=0.13, help="prune ratio per iteration (0.0-1.0)")
    parser.add_argument("--epochs", type=int, default=10, help="finetune epochs per iteration (0 = skip finetune)")
    parser.add_argument("--batch", type=int, default=16, help="finetune batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="finetune image size")
    parser.add_argument("--device", type=str, default="0", help="device for finetune")
    parser.add_argument("--divisor", type=int, default=8, choices=[8, 16])
    parser.add_argument("--save-dir", type=str, default="weights/iterative", help="output directory")
    parser.add_argument("--kd", action="store_true", help="use CWD distillation during finetune")
    parser.add_argument("--kd-teacher", type=str, default=None, help="teacher model path (default: same as --weights)")
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_args()
    total = 1 - (1 - opt.prune_ratio) ** opt.iters
    print(f"Iterative Pruning: {opt.iters} lan x {opt.prune_ratio * 100:.1f}% = ~{total * 100:.1f}% tong")

    current = opt.weights
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, opt.iters + 1):
        print(f"\n{'=' * 60}")
        print(f"ITERATION {i}/{opt.iters} — prune {opt.prune_ratio * 100:.1f}% tu {Path(current).name}")
        print(f"{'=' * 60}")

        # 1. Prune
        model, bn_dict, ignore_bn_list, _, layer_ratio_cfg, pruned_yaml = load_and_prepare(
            current, opt.cfg, opt.model_size, None
        )

        importance = compute_l1norm_importance(model, bn_dict, ignore_bn_list)

        maskbndict = create_masks(importance, model, ignore_bn_list, layer_ratio_cfg, opt.prune_ratio, opt.divisor)

        pruned_path = finalize_pruning(
            model,
            maskbndict,
            pruned_yaml,
            ignore_bn_list,
            current,
            str(save_dir),
            opt.divisor,
            opt.prune_ratio,
            method_name=f"iter{i}",
        )

        # 2. Finetune
        if opt.epochs > 0:
            print(f"\nFinetune ({opt.epochs} epochs)...")
            ft = YOLO(pruned_path)
            train_args = dict(
                data=opt.data,
                epochs=opt.epochs,
                imgsz=opt.imgsz,
                batch=opt.batch,
                device=opt.device,
                finetune=True,
                project=str(save_dir / "runs"),
                name=f"iter{i}",
            )
            if opt.kd:
                train_args.update(
                    kd=True,
                    kd_teacher=opt.kd_teacher or opt.weights,
                    kd_method="cwd",
                )
            ft.train(**train_args)

            best = save_dir / "runs" / f"iter{i}" / "weights" / "best.pt"
            current = str(best if best.exists() else save_dir / "runs" / f"iter{i}" / "weights" / "last.pt")
        else:
            print("\nSkip finetune (epochs=0)")
            current = pruned_path

        done = 1 - (1 - opt.prune_ratio) ** i
        print(f"Iter {i} done — ~{done * 100:.1f}% total pruned — {current}")

    print(f"\n{'=' * 60}")
    print(f"DONE! Final model: {current}")
    print(f"Total pruning: ~{total * 100:.1f}%")
    print(f"{'=' * 60}")
