"""
YOLOv8/YOLOv26 Pruning với Divisibility Constraints
===================================================
Pruning script với đảm bảo số kênh chia hết cho 8 để tối ưu GPU

Tính năng:
- Nhiều chiến lược pruning (nearest/up/down/adaptive)
- Tự động điều chỉnh channels để chia hết cho divisor
- Xử lý đặc biệt: chunk (C2f), concat, SPPF, Detect
- Validate divisibility trước khi lưu model
- Layer-wise sparsity control

Cấu trúc file được chỉnh sửa:
- ultralytics/nn/modules/block_pruned.py: BottleneckPruned, C3k2Pruned, SPPFPruned, C2PSAPruned
- ultralytics/nn/modules/head_pruned.py: DetectPruned
- ultralytics/nn/tasks_pruned.py: DetectionModelPruned, parse_model_pruned

Author: YOLOv8 Pruning Expert
Version: 3.0 (with make_divisible)
Date: February 2025
"""

import re
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import argparse

import torch
import torch.nn as nn
from ultralytics.utils import colorstr, LOGGER
from ultralytics.utils.ops import make_divisible
from ultralytics.nn.modules.block import Bottleneck, PSABlock
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules import Conv, Concat

from ultralytics.nn.modules.block_pruned import C3k2Pruned, C3k2PrunedBn, C3k2PrunedAttn, SPPFPruned, C2PSAPruned
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.tasks_pruned import DetectionModelPruned

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from dms_utils import make_divisible_channels, get_layer_ratio, build_pruned_yaml, build_ignore_bn_list


# ============================================================================
# MAIN PRUNING FUNCTION
# ============================================================================

def main(opt):
    """
    Main pruning workflow

    Steps:
    1. Collect BN layers và ignore list (residual connections)
    2. Filter BN layers
    3. Gather gamma values
    4. Sort và tính threshold
    5. Validate threshold hợp lệ
    6. Tạo pruned YAML config
    7. Tạo masks với make_divisible
    8. Validate divisibility
    9. Build pruned model
    10. Copy weights từ original model
    11. Save pruned model
    """

    # Parse options
    weights = opt.weights
    prune_ratio = opt.prune_ratio
    cfg = opt.cfg
    model_size = opt.model_size
    save_dir = opt.save_dir
    divisor = opt.divisor

    # Load layer-wise custom ratios nếu có
    layer_ratio_cfg = {}
    if hasattr(opt, 'layer_ratio') and opt.layer_ratio:
        with open(opt.layer_ratio, encoding='utf-8') as f:
            layer_ratio_cfg = yaml.safe_load(f) or {}
        print(f"  Loaded layer ratio config: {opt.layer_ratio} ({len(layer_ratio_cfg)} rules)")

    print(f"\n{'='*100}")
    print(f"PRUNING CONFIGURATION:")
    print(f"  Model:       {weights}")
    print(f"  Prune ratio: {prune_ratio}")
    print(f"  Divisor:     {divisor}")
    if layer_ratio_cfg:
        print(f"  Layer rules: {layer_ratio_cfg}")
    print(f"{'='*100}\n")

    # Load model
    model = AutoBackend(weights, fuse=False)
    model.eval()

    # Khởi tạo divisor
    print(f"  Divisor: {divisor} (channels sẽ chia hết cho {divisor})")

    # =========================================
    # STEP 1: Thu thập BN layers
    # =========================================
    print("Step 1: Thu thập BatchNorm layers...")

    bn_dict = {}
    chunk_bn_list = []

    # Collect all BN layers and chunk constraints (Bottleneck without residual)
    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_dict[name] = module
        if isinstance(module, Bottleneck) and not module.add:
            chunk_bn = f"{name[:-4]}.cv1.bn"
            chunk_bn_list.append(chunk_bn)

    # Build ignore list using shared utility (Bottleneck residual + PSABlock)
    ignore_bn_list = build_ignore_bn_list(model.model)

    # Validate ignore list - remove non-existent BNs
    missing_bns = [bn for bn in ignore_bn_list if bn not in bn_dict]
    if missing_bns:
        print(f"  WARNING: {len(missing_bns)} BN in ignore list not found in model, removing")
        ignore_bn_list = [bn for bn in ignore_bn_list if bn in bn_dict]

    print(f"  Tổng BN layers: {len(bn_dict)}")
    print(f"  Ignore (residual): {len(ignore_bn_list)}")
    print(f"  Chunk constraint: {len(chunk_bn_list)}")

    # =========================================
    # STEP 2: Filter BN layers
    # =========================================
    print("\nStep 2: Filter BN layers...")

    bn_dict = {k: v for k, v in bn_dict.items() if k not in ignore_bn_list}
    print(f"  BN layers sau filter: {len(bn_dict)}")

    # =========================================
    # STEP 3: Validate prune_ratio hợp lệ
    # =========================================
    # Per-layer structured pruning: mỗi layer prune độc lập theo local threshold
    # → không cần global limit (chỉ cần 0 < ratio < 1.0)
    # Cơ chế divisor đảm bảo mỗi layer giữ ít nhất divisor channels
    print("\nStep 3: Validate prune ratio...")
    print(f"  Prune ratio: {colorstr(f'{prune_ratio:.3f}')}")
    print(f"  Mode: per-layer structured (mỗi layer prune {prune_ratio*100:.0f}% channels riêng)")

    if prune_ratio >= 0.9:
        print(f"  ⚠️ Prune ratio rất cao ({prune_ratio:.2f}), model có thể mất accuracy nghiêm trọng!")
    elif prune_ratio >= 0.7:
        print(f"  ℹ️  Prune ratio cao ({prune_ratio:.2f}), nên fine-tune kỹ sau pruning")

    if layer_ratio_cfg:
        print(f"  Layer-wise custom ratios:")
        for rule_key, rule_ratio in layer_ratio_cfg.items():
            print(f"    {rule_key}: {float(rule_ratio):.3f}")

    # =========================================
    # STEP 6: Tạo pruned YAML
    # =========================================
    print("\nStep 6: Tạo pruned model config...")

    nc = model.model.nc
    pruned_yaml = build_pruned_yaml(cfg, model_size, nc)

    print(f"  nc: {nc}")
    print(f"  scale: {model_size}")
    print(f"  end2end: {pruned_yaml.get('end2end', False)}")
    print(f"  Backbone layers: {len(pruned_yaml['backbone'])}")
    print(f"  Head layers: {len(pruned_yaml['head'])}")
    for idx, (f, n, m, args) in enumerate(pruned_yaml['backbone'] + pruned_yaml['head']):
        print(f"    [{idx:>2}] n={n} {m:<20} args={args}")

    # =========================================
    # STEP 7: Tạo masks với make_divisible
    # =========================================
    print(f"\nStep 7: Tạo pruning masks (divisor={divisor})...")
    print("=" * 110)
    print(f"{'Layer name':<35} | {'Origin':>6} | {'Ratio':>6} | {'Raw':>6} | {'Rounded':>7} | {'Sparsity':>8} | {'Note'}")
    print("=" * 110)

    maskbndict = {}

    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            origin_channels = module.weight.data.size()[0]
            mask = torch.ones(origin_channels)

            if name not in ignore_bn_list:
                # ─────────────────────────────────────
                # Bước 7.1: Lấy ratio cho layer này
                # Ưu tiên: custom config > global threshold
                # ─────────────────────────────────────
                this_ratio = get_layer_ratio(name, layer_ratio_cfg, prune_ratio)

                # ─────────────────────────────────────
                # Bước 7.2: Tính mask theo per-layer ratio
                # Dùng local threshold thay vì global threshold
                # → Chính xác hơn vì mỗi layer có phân phối gamma khác nhau
                # ─────────────────────────────────────
                gamma_abs = module.weight.data.abs().view(-1)
                local_sorted = torch.sort(gamma_abs)[0]
                local_thre = local_sorted[int(len(local_sorted) * this_ratio)]
                mask = gamma_abs.gt(local_thre).float()
                current_channels = mask.sum().int().item()

                # Edge case: nếu tie gây ra quá nhiều/ít channels
                if current_channels == 0:
                    # Giữ ít nhất divisor kênh
                    sorted_idx = torch.argsort(gamma_abs, descending=True)
                    mask = torch.zeros_like(gamma_abs)
                    mask[sorted_idx[:divisor]] = 1.0
                    current_channels = divisor

                # ─────────────────────────────────────
                # Bước 7.3: Làm tròn đến bội số gần nhất của divisor
                # ─────────────────────────────────────
                target_channels = make_divisible_channels(current_channels, origin_channels, divisor)

                # ─────────────────────────────────────
                # Bước 7.4: Tạo lại mask top-k với số kênh chính xác
                # ─────────────────────────────────────
                sorted_idx = torch.argsort(gamma_abs, descending=True)
                mask = torch.zeros_like(gamma_abs)
                mask[sorted_idx[:target_channels]] = 1.0

                # Validate
                assert mask.sum() > 0, f"BN {name} không có kênh nào!"
                assert mask.sum() % divisor == 0, \
                    f"BN {name}: {mask.sum()} channels không chia hết cho {divisor}!"

                # Apply mask
                module.weight.data.mul_(mask)
                module.bias.data.mul_(mask)

                remaining = mask.sum().int().item()
                sparsity = 1 - (remaining / origin_channels)
                note = "⚙ custom" if name in layer_ratio_cfg or any(
                    k in name for k in ['backbone', 'head', 'detect']
                    if k in layer_ratio_cfg
                ) else ""

                print(
                    f"{name:<35} | {origin_channels:>6} | {this_ratio:>6.2f} | "
                    f"{current_channels:>6} | {remaining:>7} | {sparsity:>8.3f} | {note}"
                )
            else:
                print(f"{name:<35} | {origin_channels:>6} | {'  -':>6} | {'  -':>6} | {'  -':>7} | {'  -':>8} | SKIP (residual)")

            maskbndict[name] = mask

    print("=" * 110)

    # =========================================
    # STEP 8: Validate divisibility ⭐
    # =========================================
    print("\nStep 8: Validate divisibility...")

    all_valid = True
    for name, mask in maskbndict.items():
        channels = mask.sum().int().item()
        if channels % divisor != 0:
            print(f"  ❌ {name}: {channels} channels NOT divisible by {divisor}")
            all_valid = False

    if all_valid:
        print(f"  ✅ Tất cả layers đều chia hết cho {divisor}!")
    else:
        raise RuntimeError("Có layers không chia hết cho divisor!")

    # =========================================
    # STEP 9: Build pruned model
    # =========================================
    print("\nStep 9: Build pruned model...")

    pruned_model = DetectionModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    pruned_model.eval()

    # =========================================
    # STEP 10: Copy weights
    # =========================================
    print("\nStep 10: Copy weights từ original model...")

    current_to_prev = pruned_model.current_to_prev

    # Validate current_to_prev
    for xks, xvs in current_to_prev.items():
        xvs = [xvs] if not isinstance(xvs, list) else xvs
        for xk, xv in zip([xks] if not isinstance(xks, list) else xks, xvs):
            assert xk in maskbndict.keys() or 'model.' in xk, f"{xk} from 'current_to_prev' not valid"
            if xv is not None:
                assert xv in maskbndict.keys(), f"{xv} from 'current_to_prev' not in maskbndict"

    changed = []

    # Patterns
    # BUG FIX: C3k[0] có cả cv1 và cv2 đều nhận right_half từ chunk → cần cả 2
    # C3kPruned.forward: y1=cv1(x), y2=cv2(x), x = right_half → cả cv1 và cv2 cần chunk
    # C3k2PrunedAttn: model.X.m.0.0.cv1.bn (Bottleneck.cv1) cũng nhận right_half
    pattern_c3k_first = re.compile(
        r"model\.\d+\.m\.0\.(cv1|cv2)\.bn"       # C3k type: first C3k's cv1/cv2
        r"|model\.\d+\.m\.0\.0\.cv1\.bn"          # Attn type: Bottleneck.cv1 in Sequential
    )
    pattern_detect = re.compile(r"model\.\d+\.(?:cv\d|one2one_cv\d)\.\d\.2")

    # Thu thập SPPF n_param động (tránh hardcode số 4)
    # SPPFPruned.cv2 input = cv1out * (n+1), cần biết n để tạo đúng mask
    sppf_n_params = {}
    for sppf_name, sppf_module in model.model.named_modules():
        if hasattr(sppf_module, 'n') and hasattr(sppf_module, 'cv1') and hasattr(sppf_module, 'cv2') \
                and hasattr(sppf_module, 'm') and isinstance(sppf_module.m, nn.MaxPool2d):
            sppf_n_params[sppf_name] = sppf_module.n
    sppf_cv2_pattern = re.compile(r"model\.(\d+)\.cv2\.conv")

    for (name_org, module_org), (name_pruned, module_pruned) in \
        zip(model.model.named_modules(remove_duplicate=False), pruned_model.named_modules(remove_duplicate=False)):

        assert name_org == name_pruned, f"name mismatch: {name_org} != {name_pruned}"

        # Detect DFL layer - skip (không prune)
        if 'dfl' in name_org:
            continue

        # ─────────────────────────────────────
        # Xử lý Detect head - Conv2d không có BN
        # ─────────────────────────────────────
        if pattern_detect.fullmatch(name_org) is not None:
            current_conv_layer_name = name_org
            prev_bn_layer_name = current_to_prev[current_conv_layer_name]
            in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[:, in_channels_mask, :, :]
            if module_org.bias is not None:
                module_pruned.bias.data = module_org.bias.data
            continue

        # ─────────────────────────────────────
        # Xử lý Conv layers
        # ─────────────────────────────────────
        if isinstance(module_org, nn.Conv2d):
            current_bn_layer_name = name_org[:-4] + 'bn'

            # Skip nếu không có BN (ví dụ: Detect head Conv2d)
            if current_bn_layer_name not in maskbndict:
                continue

            # PSABlock internal layers: BN trong ignore_bn_list, không có current_to_prev
            # → copy weights trực tiếp (PSABlock không bị prune)
            # Đặc biệt quan trọng cho depthwise conv (pe.conv) có weight shape [C,1,k,k]
            if current_bn_layer_name in ignore_bn_list and current_bn_layer_name not in current_to_prev:
                module_pruned.weight.data = module_org.weight.data.clone()
                if module_org.bias is not None:
                    module_pruned.bias.data = module_org.bias.data.clone()
                changed.append(current_bn_layer_name)
                continue

            out_channels_mask = maskbndict[current_bn_layer_name].to(torch.bool)
            prev_bn_layer_name = current_to_prev.get(current_bn_layer_name, None)

            # Xử lý input channels
            if isinstance(prev_bn_layer_name, list):
                # Concat case
                in_channels_masks = [maskbndict[ni] for ni in prev_bn_layer_name]
                in_channels_mask = torch.cat(in_channels_masks, dim=0).to(torch.bool)
            elif prev_bn_layer_name is not None:
                in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)

                # BUG FIX: C3k[0] cv1 VÀ cv2 đều nhận right_half (sau chunk)
                # C3kPruned: cv1(x), cv2(x) với x = right_half của C3k2.cv1
                if pattern_c3k_first.fullmatch(current_bn_layer_name) is not None:
                    # Guard: chỉ chunk nếu KHÔNG phải Bottleneck cv2 (sequential)
                    # C3k cv1/cv2: parallel → cần chunk
                    # Bottleneck cv1: đầu tiên trong chain → cần chunk (nhận right_half)
                    # Bottleneck cv2: sequential → KHÔNG chunk (nhận cv1 output)
                    # Phân biệt bằng cấu trúc: Bottleneck không có cv3, C3k có cv3
                    is_bottleneck_cv2 = False
                    m_cv2 = re.fullmatch(r"model\.\d+\.m\.0\.cv2\.bn", current_bn_layer_name)
                    if m_cv2:
                        parent = current_bn_layer_name.rsplit('.cv2.bn', 1)[0]
                        if f"{parent}.cv3.bn" not in maskbndict:
                            is_bottleneck_cv2 = True
                    if not is_bottleneck_cv2:
                        in_channels_mask = in_channels_mask.chunk(2, 0)[1]

                # BUG FIX: SPPF second conv - dynamic n_param thay vì hardcode 4
                # SPPFPruned.cv2 input = cv1out * (n+1), hardcode 4 chỉ đúng khi n=3
                sppf_match = sppf_cv2_pattern.fullmatch(name_org)
                if sppf_match:
                    sppf_layer = f"model.{sppf_match.group(1)}"
                    if sppf_layer in sppf_n_params:  # chỉ apply nếu thực sự là SPPF
                        n_param = sppf_n_params[sppf_layer]
                        in_channels_mask = torch.cat([in_channels_mask] * (n_param + 1), dim=0)
            else:
                # First layer - no mask
                in_channels_mask = torch.ones(module_org.weight.data.shape[1], dtype=torch.bool)

            # Validate TRƯỚC KHI copy để tránh OOM
            expected_in = in_channels_mask.sum().int().item()
            expected_out = out_channels_mask.sum().int().item()

            if expected_in != module_pruned.in_channels:
                print(f"\n❌ SHAPE MISMATCH DETECTED:")
                print(f"   Layer: {name_org}")
                print(f"   Expected in_channels: {expected_in}")
                print(f"   Actual in_channels:   {module_pruned.in_channels}")
                print(f"   prev_bn: {prev_bn_layer_name}")
                print(f"   current_bn: {current_bn_layer_name}")
                if isinstance(prev_bn_layer_name, list):
                    print(f"   Concat from: {prev_bn_layer_name}")
                    for pbn in prev_bn_layer_name:
                        print(f"     - {pbn}: {maskbndict[pbn].sum().int().item()} ch")
                raise RuntimeError(f"Shape mismatch at {name_org}")

            if expected_out != module_pruned.out_channels:
                raise RuntimeError(f"{name_org} out_channels mismatch: {expected_out} vs {module_pruned.out_channels}")

            # Copy weights
            if module_org.groups > 1 and module_org.groups == module_org.in_channels:
                # Depthwise conv: weight shape [C, 1, k, k], chỉ prune dim 0
                state_dict_org = module_org.weight.data[out_channels_mask, :, :, :]
            else:
                state_dict_org = module_org.weight.data[out_channels_mask, :, :, :]
                state_dict_org = state_dict_org[:, in_channels_mask, :, :]
            module_pruned.weight.data = state_dict_org

            # Copy bias
            if module_org.bias is not None:
                module_pruned.bias.data = module_org.bias.data[out_channels_mask]

            changed.append(current_bn_layer_name)

        # ─────────────────────────────────────
        # Xử lý BatchNorm layers
        # ─────────────────────────────────────
        if isinstance(module_org, nn.BatchNorm2d):
            out_channels_mask = maskbndict[name_org].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[out_channels_mask]
            module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            module_pruned.running_mean = module_org.running_mean[out_channels_mask]
            module_pruned.running_var = module_org.running_var[out_channels_mask]

    # Validate tất cả BN đã được xử lý
    missing = [name for name in maskbndict.keys() if name not in changed and name not in ignore_bn_list]
    assert not missing, f"Missing BN layers: {missing}"

    print("   Copy weights hoàn tất!")

    # =========================================
    # STEP 11: Save model
    # =========================================
    print("\nStep 11: Save pruned model...")

    pruned_model.eval()
    input_name = Path(weights).stem  # vd: "yolo26s" từ "yolo26s.pt"
    save_path = os.path.join(save_dir, f"{input_name}_pruned_div{divisor}.pt")
    torch.save(
        {
            "model": pruned_model,
            "maskbndict": maskbndict,
            "config": {
                "divisor": divisor,
                "prune_ratio": prune_ratio,
            }
        },
        save_path
    )

    print(f"   Model saved: {save_path}")

    # Test forward
    print("\nTesting forward pass...")
    model_test = torch.load(save_path,weights_only=False)["model"].cuda()
    dummies = torch.randn([1, 3, 640, 640], dtype=torch.float32).cuda()
    with torch.no_grad():
        output = model_test(dummies)
    print("   Forward pass successful!")

    # Print summary
    print_summary(maskbndict, divisor, prune_ratio, save_path)

    return maskbndict, pruned_yaml


def print_summary(maskbndict: Dict, divisor: int, prune_ratio: float, save_path: str):
    """In tóm tắt kết quả pruning"""

    total_origin = 0
    total_pruned = 0

    for name, mask in maskbndict.items():
        total_origin += len(mask)
        total_pruned += mask.sum().int().item()

    compression_ratio = total_origin / total_pruned if total_pruned > 0 else 0

    print("\n" + "=" * 100)
    print(" PRUNING SUMMARY")
    print("=" * 100)
    print(f"Divisor:           {divisor}")
    print(f"Prune ratio:       {prune_ratio:.3f}")
    print(f"Total channels:    {total_origin:,} → {total_pruned:,}")
    print(f"Compression:       {compression_ratio:.2f}x")
    print(f"Model saved:       {save_path}")
    print("=" * 100)
    print("\n PRUNING HOÀN TẤT!\n")


def parse_opt():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YOLO26 Pruning với Divisibility Constraints')

    # Basic options
    parser.add_argument('--weights', type=str,
                       default=ROOT / 'weights/best.pt',
                       help='model.pt path')
    parser.add_argument('--cfg', type=str,
                       default=ROOT / 'ultralytics/cfg/models/26/yolo26.yaml',
                       help='model.yaml path')
    parser.add_argument('--model-size', type=str, default='m',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='model size')

    # Pruning options
    parser.add_argument('--prune-ratio', type=float, default=0.5,
                       help='prune ratio toàn cục (0.0-1.0)')
    parser.add_argument('--layer-ratio', type=str, default=None,
                       help='đường dẫn YAML file chứa custom ratio cho từng layer. '
                            'Ví dụ: layer_ratio.yaml với nội dung:\n'
                            '  model.0: 0.1      # layer cụ thể\n'
                            '  backbone: 0.3     # nhóm backbone\n'
                            '  head: 0.5         # nhóm head\n'
                            '  detect: 0.4       # detect head')

    # Divisibility options
    parser.add_argument('--divisor', type=int, default=8,
                       choices=[8, 16],
                       help='divisor cho channels (8 cho GPU thường, 16 cho Tensor Cores')

    # Output options
    parser.add_argument('--save-dir', type=str,
                       default=ROOT / 'weights',
                       help='pruned model save directory')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    """
 Cơ bản nhất
python prune.py \
    --weights runs/train-sparsity/weights/last.pt \
    --cfg ultralytics/cfg/models/v8/yolov8.yaml \
    --prune-ratio 0.3

# Đầy đủ
python prune.py \
    --weights runs/train-sparsity/weights/last.pt \
    --cfg ultralytics/cfg/models/v8/yolov8.yaml \
    --model-size s \
    --prune-ratio 0.4 \
    --divisor 8 \
    --save-dir weights/

# Với custom layer ratio
python prune.py \
    --weights runs/train-sparsity/weights/last.pt \
    --cfg ultralytics/cfg/models/v8/yolov8.yaml \
    --prune-ratio 0.3 \
    --layer-ratio layer_ratio.\
    
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5 --divisor 8
    """