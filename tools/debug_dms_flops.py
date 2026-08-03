"""
Debug script for DMS GFLOPs computation.

Loads a YOLO26 model and prints the full conv→(in_bn, out_bn) mapping
to verify _resolve_internal_in_bn resolves correctly.

Usage:
    python debug_dms_flops.py --weights weights/best.pt --imgsz 640
    python debug_dms_flops.py --weights yolo26m.pt --imgsz 640
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

import torch
from torch import nn

from dms.dms_utils import (
    build_conv_bn_mapping,
    build_ignore_bn_list,
    profile_per_layer_flops,
)
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # Load model — use DetectionModel (not Sequential) for correct naming + forward
    yolo = YOLO(args.weights)
    model = yolo.model  # DetectionModel: names have 'model.' prefix, forward handles skip connections

    device = next(model.parameters()).device
    print(f"Device: {device}")
    print(f"Image size: {args.imgsz}")

    # Build ignore list
    ignore_bn_list = build_ignore_bn_list(model)
    print(f"\n{'=' * 80}")
    print(f"IGNORE BN LIST ({len(ignore_bn_list)} layers)")
    print(f"{'=' * 80}")
    for name in sorted(ignore_bn_list):
        print(f"  {name}")

    # Collect all BN layers
    all_bns = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            all_bns[name] = m.num_features

    prunable_bns = {k: v for k, v in all_bns.items() if k not in ignore_bn_list}
    print(f"\n{'=' * 80}")
    print(f"PRUNABLE BN LAYERS ({len(prunable_bns)} layers)")
    print(f"{'=' * 80}")
    for name in sorted(prunable_bns.keys()):
        print(f"  {name:<50} ch={prunable_bns[name]}")

    # Profile FLOPs
    conv_flops, total_flops = profile_per_layer_flops(model, args.imgsz, device)
    print(f"\n{'=' * 80}")
    print(f"CONV FLOPs PROFILE ({len(conv_flops)} convs, total={total_flops / 1e9:.3f} GFLOPs)")
    print(f"{'=' * 80}")
    for name in sorted(conv_flops.keys()):
        pct = conv_flops[name] / total_flops * 100
        print(f"  {name:<55} {conv_flops[name] / 1e6:>10.2f}M  ({pct:5.2f}%)")

    # Build conv-bn mapping
    conv_bn_map, bn_channels = build_conv_bn_mapping(model, ignore_bn_list)

    print(f"\n{'=' * 80}")
    print(f"CONV → BN MAPPING ({len(conv_bn_map)} convs)")
    print(f"{'=' * 80}")
    print(f"  {'CONV NAME':<55} {'OUT_BN':<40} {'IN_BN':<40} {'DW'}")
    print(f"  {'-' * 55} {'-' * 40} {'-' * 40} {'-' * 3}")

    errors = []
    for name in sorted(conv_bn_map.keys()):
        info = conv_bn_map[name]
        out_bn = info["out_bn"] or "None"
        in_bn = info["in_bn"]
        is_dw = info["is_depthwise"]

        # Format in_bn (could be list for concat)
        if isinstance(in_bn, list):
            in_bn_str = "[" + ", ".join(in_bn) + "]"
        elif in_bn is None:
            in_bn_str = "None"
        else:
            in_bn_str = in_bn

        dw_str = "DW" if is_dw else ""
        print(f"  {name:<55} {out_bn:<40} {in_bn_str:<40} {dw_str}")

        # Validate: check that referenced BNs exist
        if isinstance(in_bn, str) and in_bn not in bn_channels:
            errors.append(f"  MISSING in_bn: {in_bn} (for {name})")
        elif isinstance(in_bn, list):
            for b in in_bn:
                if b not in bn_channels:
                    errors.append(f"  MISSING in_bn: {b} (for {name})")
        if isinstance(out_bn, str) and out_bn != "None" and out_bn not in bn_channels:
            errors.append(f"  MISSING out_bn: {out_bn} (for {name})")

    # Validate: check that prunable BNs appear as out_bn somewhere
    out_bns_used = set()
    for info in conv_bn_map.values():
        if info["out_bn"]:
            out_bns_used.add(info["out_bn"])

    # Simulate resource loss computation
    print(f"\n{'=' * 80}")
    print("SIMULATED RESOURCE LOSS (target=0.3)")
    print(f"{'=' * 80}")

    # Create fake a_params (all at 0.3)
    a_params = {}
    for name in prunable_bns:
        a_params[name] = nn.Parameter(torch.tensor(0.3, device=device))

    effective_total = 0.0
    for conv_name in sorted(conv_flops.keys()):
        flops = conv_flops[conv_name]
        info = conv_bn_map.get(conv_name)

        if info is None:
            effective_total += flops
            print(f"  {conv_name:<55} NO MAPPING → full FLOPs")
            continue

        out_bn = info["out_bn"]
        in_bn = info.get("in_bn")
        is_dw = info["is_depthwise"]

        a_out = a_params.get(out_bn)
        retain_out = (1.0 - a_out.item()) if a_out is not None else 1.0

        if is_dw:
            ratio = retain_out
            r_in_str = "DW"
        elif in_bn is None:
            ratio = retain_out
            r_in_str = "1.000(no_in)"
        elif isinstance(in_bn, list):
            total_ch = 0
            weighted = 0.0
            for b in in_bn:
                ch = bn_channels.get(b, 0)
                a_in = a_params.get(b)
                r = (1.0 - a_in.item()) if a_in is not None else 1.0
                total_ch += ch
                weighted += ch * r
            retain_in = weighted / total_ch if total_ch > 0 else 1.0
            ratio = retain_in * retain_out
            r_in_str = f"{retain_in:.3f}(concat)"
        else:
            a_in = a_params.get(in_bn)
            retain_in = (1.0 - a_in.item()) if a_in is not None else 1.0
            ratio = retain_in * retain_out
            r_in_str = f"{retain_in:.3f}"

        r_out_str = f"{retain_out:.3f}"
        ratio_str = f"{ratio:.3f}"
        effective_total += flops * ratio

        pct = flops / total_flops * 100
        print(f"  {conv_name:<50} r_out={r_out_str:<8} r_in={r_in_str:<18} ratio={ratio_str:<8} ({pct:5.2f}% of total)")

    r_e = effective_total / total_flops
    r_t = 0.5
    print(f"\n  r_e (effective retention) = {r_e:.4f}")
    print(f"  r_t (target retention)    = {r_t:.4f}")
    print(
        f"  loss = {'log(r_e/r_t) = ' + f'{(r_e / r_t):.4f} → {torch.log(torch.tensor(r_e / r_t)).item():.4f}' if r_e > r_t else '0 (under target)'}"
    )

    # Print errors
    if errors:
        print(f"\n{'=' * 80}")
        print(f"ERRORS ({len(errors)})")
        print(f"{'=' * 80}")
        for e in errors:
            print(e)
    else:
        print("\n  All BN references validated OK!")


if __name__ == "__main__":
    main()
