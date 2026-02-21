"""
YOLOv26 Pruned Model Validation
================================
So sánh model gốc vs pruned: weight sanity, output consistency, mAP, visual check.

Usage:
    # Basic: weight check + output comparison
    python validate_pruned.py --weights weights/best.pt --pruned weights/pruned_div8.pt

    # Với mAP evaluation
    python validate_pruned.py --weights weights/best.pt --pruned weights/pruned_div8.pt --data data.yaml

    # Với ảnh test
    python validate_pruned.py --weights weights/best.pt --pruned weights/pruned_div8.pt --img test.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ============================================================================
# 1. WEIGHT SANITY CHECK
# ============================================================================

def weight_sanity_check(model, model_name="model"):
    """
    Kiểm tra NaN/Inf và thống kê weight distribution cho mỗi layer.

    Returns:
        bool: True nếu tất cả weights OK (không NaN/Inf)
        list[dict]: Thống kê cho mỗi layer
    """
    print(f"\n{'='*80}")
    print(f" WEIGHT SANITY CHECK: {model_name}")
    print(f"{'='*80}")

    has_issue = False
    stats = []

    for name, param in model.named_parameters():
        nan_count = torch.isnan(param.data).sum().item()
        inf_count = torch.isinf(param.data).sum().item()

        if nan_count > 0 or inf_count > 0:
            has_issue = True
            print(f"  [FAIL] {name}: NaN={nan_count}, Inf={inf_count}")

        p = param.data.float()
        layer_stat = {
            "name": name,
            "shape": list(param.shape),
            "mean": p.mean().item(),
            "std": p.std().item(),
            "min": p.min().item(),
            "max": p.max().item(),
            "nan": nan_count,
            "inf": inf_count,
        }
        stats.append(layer_stat)

    if not has_issue:
        print(f"  [OK] No NaN/Inf found in any parameter")

    # Print top-level summary
    all_params = torch.cat([p.data.float().flatten() for p in model.parameters()])
    print(f"\n  Overall weight stats:")
    print(f"    Total params: {all_params.numel():,}")
    print(f"    Mean: {all_params.mean().item():.6f}")
    print(f"    Std:  {all_params.std().item():.6f}")
    print(f"    Min:  {all_params.min().item():.6f}")
    print(f"    Max:  {all_params.max().item():.6f}")

    return not has_issue, stats


def compare_weight_distributions(stats_orig, stats_pruned):
    """So sánh weight distribution giữa original và pruned model."""
    print(f"\n{'='*80}")
    print(f" WEIGHT DISTRIBUTION COMPARISON")
    print(f"{'='*80}")
    print(f"{'Layer':<45} | {'Orig mean':>10} | {'Prun mean':>10} | {'Orig std':>10} | {'Prun std':>10}")
    print(f"{'-'*45}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    orig_by_name = {s["name"]: s for s in stats_orig}
    pruned_by_name = {s["name"]: s for s in stats_pruned}

    anomaly_count = 0
    for name in pruned_by_name:
        sp = pruned_by_name[name]
        so = orig_by_name.get(name)

        if so is None:
            continue

        # Flag anomaly: pruned std is 5x larger or mean shifted significantly
        mean_shift = abs(sp["mean"] - so["mean"])
        std_ratio = sp["std"] / max(so["std"], 1e-8)
        anomaly = mean_shift > 1.0 or std_ratio > 5.0 or std_ratio < 0.2

        if anomaly:
            anomaly_count += 1
            flag = " <-- ANOMALY"
        else:
            flag = ""

        # Only print Conv/BN layers (skip printing all layers to keep output readable)
        if "conv" in name or "bn" in name or anomaly:
            short_name = name if len(name) <= 44 else "..." + name[-41:]
            print(
                f"{short_name:<45} | {so['mean']:>10.5f} | {sp['mean']:>10.5f} | "
                f"{so['std']:>10.5f} | {sp['std']:>10.5f}{flag}"
            )

    if anomaly_count == 0:
        print(f"\n  [OK] No anomalous weight distributions detected")
    else:
        print(f"\n  [WARN] {anomaly_count} layers with anomalous weight distributions")


# ============================================================================
# 2. OUTPUT CONSISTENCY CHECK
# ============================================================================

def output_consistency_check(model_orig, model_pruned, imgsz=640, device="cuda"):
    """
    So sánh output giữa original và pruned model.
    Dùng random input, so sánh cosine similarity và MSE.

    Returns:
        dict: Metrics (cosine_sim, mse, mae)
    """
    print(f"\n{'='*80}")
    print(f" OUTPUT CONSISTENCY CHECK")
    print(f"{'='*80}")

    model_orig.eval()
    model_pruned.eval()

    # Run multiple random inputs for stable metrics
    n_trials = 3
    cos_sims, mses, maes = [], [], []

    for i in range(n_trials):
        dummy = torch.randn(1, 3, imgsz, imgsz, device=device)

        with torch.no_grad():
            out_orig = model_orig(dummy)
            out_pruned = model_pruned(dummy)

        # Both models return (y, preds) in eval mode; y is the main prediction tensor
        if isinstance(out_orig, (tuple, list)):
            pred_orig = out_orig[0] if isinstance(out_orig[0], torch.Tensor) else out_orig
            pred_pruned = out_pruned[0] if isinstance(out_pruned[0], torch.Tensor) else out_pruned
        else:
            pred_orig = out_orig
            pred_pruned = out_pruned

        flat_orig = pred_orig.float().flatten()
        flat_pruned = pred_pruned.float().flatten()

        cos_sims.append(F.cosine_similarity(flat_orig.unsqueeze(0), flat_pruned.unsqueeze(0)).item())
        mses.append(F.mse_loss(flat_pruned, flat_orig).item())
        maes.append(F.l1_loss(flat_pruned, flat_orig).item())

    cos_sim = sum(cos_sims) / n_trials
    mse = sum(mses) / n_trials
    mae = sum(maes) / n_trials

    print(f"\n  {n_trials} random inputs ({imgsz}x{imgsz}):")
    print(f"    Cosine Similarity: {cos_sim:.6f}  (per-trial: {', '.join(f'{x:.4f}' for x in cos_sims)})")
    print(f"    MSE:               {mse:.6f}")
    print(f"    MAE:               {mae:.6f}")

    # Note: for aggressive pruning (>40%), cosine sim 0.5-0.8 is normal
    # The real quality metric is mAP, not raw output similarity
    if cos_sim > 0.9:
        print(f"    [OK] Cosine similarity > 0.9 (excellent)")
    elif cos_sim > 0.7:
        print(f"    [OK] Cosine similarity > 0.7 (good for pruned model)")
    elif cos_sim > 0.5:
        print(f"    [INFO] Cosine similarity > 0.5 (expected for aggressive pruning)")
        print(f"    --> Run with --data to check actual mAP for real quality metric")
    else:
        print(f"    [WARN] Cosine similarity < 0.5 (significant deviation)")
        print(f"    --> Recommend running mAP evaluation with --data")

    return {"cosine_sim": cos_sim, "mse": mse, "mae": mae}


def feature_map_comparison(model_orig, model_pruned, imgsz=640, device="cuda"):
    """
    So sánh feature maps tại mỗi scale (P3, P4, P5) giữa 2 models.
    Hook vào backbone output layers.
    """
    print(f"\n{'='*80}")
    print(f" FEATURE MAP COMPARISON (per scale)")
    print(f"{'='*80}")

    features_orig = {}
    features_pruned = {}

    def make_hook(store, name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                store[name] = output.detach()
        return hook_fn

    # Hook vào các layer backbone/head output
    # Typical YOLO: P3 (stride 8), P4 (stride 16), P5 (stride 32)
    hooks = []
    for name, module in model_orig.named_modules():
        if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
            continue
        # Hook vào cuối mỗi C3k2/SPPF block
        for block_type in ["C3k2", "SPPF", "C2PSA"]:
            if type(module).__name__.startswith(block_type) or type(module).__name__.startswith(block_type.replace("C3k2", "C3k2Pruned")):
                hooks.append(module.register_forward_hook(make_hook(features_orig, name)))
                break

    for name, module in model_pruned.named_modules():
        for block_type in ["C3k2", "SPPF", "C2PSA", "C3k2Pruned", "SPPFPruned", "C2PSAPruned", "C3k2PrunedAttn"]:
            if type(module).__name__ == block_type:
                hooks.append(module.register_forward_hook(make_hook(features_pruned, name)))
                break

    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        model_orig(dummy)
        model_pruned(dummy)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compare matched features using spatial activation patterns
    # When channels differ (e.g. 512 vs 256), compare spatial patterns:
    #   - Spatial cosine: mean-pool across channels → compare HxW spatial maps
    #   - Activation stats: compare mean activation magnitudes
    print(f"\n  {'Layer':<25} | {'Orig shape':>20} | {'Pruned shape':>20} | {'Spatial cos':>11} | {'Act ratio':>9}")
    print(f"  {'-'*25}-+-{'-'*20}-+-{'-'*20}-+-{'-'*11}-+-{'-'*9}")

    orig_keys = sorted(features_orig.keys())
    pruned_keys = sorted(features_pruned.keys())

    for ko, kp in zip(orig_keys, pruned_keys):
        fo = features_orig[ko]
        fp = features_pruned[kp]

        if fo.shape[2:] == fp.shape[2:]:
            # Spatial cosine: pool channels → [1, 1, H, W] → flatten spatial dims
            fo_spatial = fo.mean(dim=1).flatten()  # [H*W]
            fp_spatial = fp.mean(dim=1).flatten()  # [H*W]
            spatial_cos = F.cosine_similarity(fo_spatial.unsqueeze(0), fp_spatial.unsqueeze(0)).item()

            # Activation magnitude ratio (pruned / original)
            act_orig = fo.abs().mean().item()
            act_pruned = fp.abs().mean().item()
            act_ratio = act_pruned / max(act_orig, 1e-8)
        else:
            spatial_cos = float("nan")
            act_ratio = float("nan")

        short_name = ko if len(ko) <= 24 else "..." + ko[-21:]
        print(
            f"  {short_name:<25} | {str(list(fo.shape)):>20} | {str(list(fp.shape)):>20} | "
            f"{spatial_cos:>11.4f} | {act_ratio:>9.4f}"
        )


# ============================================================================
# 3. mAP EVALUATION
# ============================================================================

def evaluate_map(model_path, data_yaml, imgsz=640, batch=16, device="cuda"):
    """
    Chạy mAP evaluation dùng YOLO.val().

    Args:
        model_path: Path tới model (.pt)
        data_yaml: Path tới data.yaml
        imgsz: Image size
        batch: Batch size
        device: cuda/cpu

    Returns:
        dict: {map50, map50_95}
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device, verbose=False)

    map50 = results.box.map50
    map50_95 = results.box.map

    return {"map50": map50, "map50_95": map50_95}


def compare_map(weights, pruned, data_yaml, imgsz=640, batch=16, device="cuda"):
    """So sánh mAP giữa original và pruned model."""
    print(f"\n{'='*80}")
    print(f" mAP EVALUATION")
    print(f"{'='*80}")

    print(f"\n  Evaluating original model: {weights}")
    metrics_orig = evaluate_map(weights, data_yaml, imgsz, batch, device)

    print(f"  Evaluating pruned model: {pruned}")
    metrics_pruned = evaluate_map(pruned, data_yaml, imgsz, batch, device)

    map50_drop = metrics_orig["map50"] - metrics_pruned["map50"]
    map_drop = metrics_orig["map50_95"] - metrics_pruned["map50_95"]

    print(f"\n  {'Metric':<15} | {'Original':>10} | {'Pruned':>10} | {'Drop':>10}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'mAP@0.5':<15} | {metrics_orig['map50']:>10.4f} | {metrics_pruned['map50']:>10.4f} | {map50_drop:>+10.4f}")
    print(f"  {'mAP@0.5:0.95':<15} | {metrics_orig['map50_95']:>10.4f} | {metrics_pruned['map50_95']:>10.4f} | {map_drop:>+10.4f}")

    if map_drop < 0.05:
        print(f"\n  [OK] mAP drop < 5% ({map_drop:.4f})")
    else:
        print(f"\n  [WARN] mAP drop >= 5% ({map_drop:.4f})")

    return metrics_orig, metrics_pruned


# ============================================================================
# 4. VISUAL CHECK
# ============================================================================

def visual_check(weights, pruned, img_path, imgsz=640, conf=0.25, save_dir="validate_results"):
    """
    Chạy inference trên 1 ảnh, vẽ bounding boxes, save kết quả.
    """
    print(f"\n{'='*80}")
    print(f" VISUAL CHECK: {img_path}")
    print(f"{'='*80}")

    os.makedirs(save_dir, exist_ok=True)
    img_name = Path(img_path).stem

    # Original model
    model_orig = YOLO(weights)
    results_orig = model_orig.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)

    if results_orig and len(results_orig) > 0:
        img_orig = results_orig[0].plot()
        save_path_orig = os.path.join(save_dir, f"{img_name}_original.jpg")
        from PIL import Image
        Image.fromarray(img_orig[..., ::-1]).save(save_path_orig)
        n_orig = len(results_orig[0].boxes) if results_orig[0].boxes is not None else 0
        print(f"  Original: {n_orig} detections -> {save_path_orig}")

    # Pruned model
    model_pruned = YOLO(pruned)
    results_pruned = model_pruned.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)

    if results_pruned and len(results_pruned) > 0:
        img_pruned = results_pruned[0].plot()
        save_path_pruned = os.path.join(save_dir, f"{img_name}_pruned.jpg")
        from PIL import Image
        Image.fromarray(img_pruned[..., ::-1]).save(save_path_pruned)
        n_pruned = len(results_pruned[0].boxes) if results_pruned[0].boxes is not None else 0
        print(f"  Pruned:   {n_pruned} detections -> {save_path_pruned}")

    print(f"\n  Results saved to: {save_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def load_models(weights, pruned, device="cuda"):
    """Load original và pruned model."""
    # Original: dùng AutoBackend (giống prune.py)
    model_orig = AutoBackend(weights, fuse=False)
    model_orig.to(device)
    model_orig.eval()

    # Pruned: torch.load -> lấy model key
    ckpt = torch.load(pruned, weights_only=False, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model_pruned = ckpt["model"]
    else:
        model_pruned = ckpt
    model_pruned.to(device)
    model_pruned.eval()

    return model_orig, model_pruned


def main():
    parser = argparse.ArgumentParser(description="Validate pruned YOLOv26 model")
    parser.add_argument("--weights", type=str, required=True, help="Path to original model (.pt)")
    parser.add_argument("--pruned", type=str, required=True, help="Path to pruned model (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml for mAP evaluation")
    parser.add_argument("--img", type=str, default=None, help="Path to test image for visual check")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for mAP evaluation")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for visual check")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--save-dir", type=str, default="validate_results", help="Directory to save visual results")
    opt = parser.parse_args()

    device = opt.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"\n{'='*80}")
    print(f" YOLOv26 PRUNED MODEL VALIDATION")
    print(f"{'='*80}")
    print(f"  Original: {opt.weights}")
    print(f"  Pruned:   {opt.pruned}")
    print(f"  Device:   {device}")

    # --- Load models ---
    print("\nLoading models...")
    model_orig, model_pruned = load_models(opt.weights, opt.pruned, device)

    # Get the inner nn.Module for analysis
    # AutoBackend wraps the model; .model is the actual DetectionModel
    inner_orig = model_orig.model if hasattr(model_orig, "model") else model_orig
    inner_pruned = model_pruned

    # --- 1. Weight Sanity Check ---
    ok_orig, stats_orig = weight_sanity_check(inner_orig, "Original")
    ok_pruned, stats_pruned = weight_sanity_check(inner_pruned, "Pruned")
    compare_weight_distributions(stats_orig, stats_pruned)

    # --- 2. Output Consistency Check ---
    output_metrics = output_consistency_check(model_orig, model_pruned, opt.imgsz, device)
    feature_map_comparison(inner_orig, inner_pruned, opt.imgsz, device)

    # --- 3. mAP Evaluation (if data provided) ---
    if opt.data:
        compare_map(opt.weights, opt.pruned, opt.data, opt.imgsz, opt.batch, device)
    else:
        print(f"\n  [SKIP] mAP evaluation (no --data provided)")

    # --- 4. Visual Check (if image provided) ---
    if opt.img:
        visual_check(opt.weights, opt.pruned, opt.img, opt.imgsz, opt.conf, opt.save_dir)
    else:
        print(f"\n  [SKIP] Visual check (no --img provided)")

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f" VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"  Weight sanity (original): {'PASS' if ok_orig else 'FAIL'}")
    print(f"  Weight sanity (pruned):   {'PASS' if ok_pruned else 'FAIL'}")
    print(f"  Cosine similarity:        {output_metrics['cosine_sim']:.6f}")
    print(f"  MSE:                      {output_metrics['mse']:.6f}")
    print(f"  MAE:                      {output_metrics['mae']:.6f}")
    if opt.data:
        print(f"  mAP: see above")
    if opt.img:
        print(f"  Visual results: {opt.save_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
