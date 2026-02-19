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

from ultralytics.nn.modules.block_pruned import C3k2Pruned, C3k2PrunedAttn, SPPFPruned, C2PSAPruned
from ultralytics.nn.modules.head_pruned import DetectPruned
from ultralytics.nn.tasks_pruned import DetectionModelPruned

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# ============================================================================
# DIVISIBILITY STRATEGY - Các chiến lược làm tròn channels
# ============================================================================

def make_divisible_channels(channels: int, max_channels: int, divisor: int) -> int:
    """
    Làm tròn channels đến bội số gần nhất của divisor.
    Dùng make_divisible từ Ultralytics.

    Args:
        channels:     Số channels cần làm tròn
        max_channels: Giới hạn trên (không vượt quá origin)
        divisor:      Số chia (8 hoặc 16)

    Returns:
        int: Số channels đã làm tròn, đảm bảo <= max_channels
    """
    return min(make_divisible(channels, divisor), max_channels)


def get_layer_ratio(layer_name: str, layer_ratio_cfg: dict, default_ratio: float) -> float:
    """
    Lấy prune ratio cho một layer cụ thể.

    Matching theo thứ tự ưu tiên (specific → general):
    1. Exact match:   'model.0.bn' → 0.1
    2. Layer index:   'model.0'    → áp dụng cho tất cả BN trong layer 0
    3. Group name:    'backbone'   → model.0 đến model.9
                      'head'       → model.10 trở đi
                      'detect'     → layer Detect
    4. Default:       global prune_ratio

    Args:
        layer_name:      Tên BN layer, ví dụ 'model.2.cv1.bn'
        layer_ratio_cfg: Dict từ YAML, ví dụ {'model.0': 0.1, 'backbone': 0.2}
        default_ratio:   Global prune ratio nếu không match

    Returns:
        float: prune ratio cho layer này

    Example YAML (layer_ratio.yaml):
        # Giữ nhiều kênh ở layer đầu
        model.0: 0.1
        model.1: 0.1
        # Tỉa mạnh ở head
        model.15: 0.6
        model.18: 0.6
        model.21: 0.6
        # Group rules
        backbone: 0.3
        head: 0.5
        detect: 0.4
    """
    if not layer_ratio_cfg:
        return default_ratio

    # 1. Exact match
    if layer_name in layer_ratio_cfg:
        return float(layer_ratio_cfg[layer_name])

    # 2. Layer index match (model.X)
    match = re.match(r'(model\.\d+)', layer_name)
    if match:
        layer_prefix = match.group(1)
        if layer_prefix in layer_ratio_cfg:
            return float(layer_ratio_cfg[layer_prefix])

        # 3. Group match
        layer_idx = int(re.search(r'model\.(\d+)', layer_name).group(1))

        # detect: Detect head layer (thường là 22)
        if 'detect' in layer_ratio_cfg and layer_idx >= 22:
            return float(layer_ratio_cfg['detect'])
        # backbone: model.0 - model.9
        if 'backbone' in layer_ratio_cfg and layer_idx <= 9:
            return float(layer_ratio_cfg['backbone'])
        # head: model.10 - model.21
        if 'head' in layer_ratio_cfg and 10 <= layer_idx <= 21:
            return float(layer_ratio_cfg['head'])

    return default_ratio


# ============================================================================
# DYNAMIC PRUNED YAML GENERATION
# ============================================================================

def build_pruned_yaml(cfg, model_size, nc):
    """
    Build pruned YAML dynamically from original model config.

    Hỗ trợ tất cả model sizes (n, s, m, l, x) bằng cách:
    - Đọc cấu trúc từ original YAML
    - Apply depth_multiple cho repeat counts
    - Map module types sang Pruned versions
    - Giữ nguyên structural parameters (k, stride, shortcut, etc.)

    Args:
        cfg (str): Path to original YAML config
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
        nc (int): Number of classes (từ loaded model)

    Returns:
        dict: Pruned model config ready for DetectionModelPruned
    """
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)

    # Get scale parameters: [depth_multiple, width_multiple, max_channels]
    depth, width, max_ch = model_yamls['scales'][model_size]

    pruned_yaml = {
        'nc': nc,
        'scales': model_yamls['scales'],
        'scale': model_size,
        'end2end': model_yamls.get('end2end', False),
        'reg_max': model_yamls.get('reg_max', 16),
    }

    def map_layer(f, n, m, args):
        """Map a single YAML layer to its pruned equivalent."""
        # Apply depth to repeat count (giống ultralytics parse_model)
        actual_n = max(round(n * depth), 1) if n > 1 else n

        if m == 'C3k2':
            # C3k2 args: [c2, c3k] hoặc [c2, c3k, e] hoặc [c2, c3k, e, attn]
            # attn=True (arg thứ 4) → dùng C3k2PrunedAttn
            has_attn = len(args) >= 4 and args[3] is True
            if has_attn:
                return [f, actual_n, 'C3k2PrunedAttn', [args[0], True]]
            else:
                return [f, actual_n, 'C3k2Pruned', [args[0], True]]
        elif m == 'SPPF':
            # SPPF args: [c2, k, n_pool, shortcut] → giữ nguyên
            return [f, actual_n, 'SPPFPruned', args]
        elif m == 'C2PSA':
            # C2PSA args: [c2] hoặc [c2, e] → giữ nguyên
            return [f, actual_n, 'C2PSAPruned', args]
        elif m == 'Detect':
            return [f, actual_n, 'DetectPruned', [nc]]
        else:
            # Conv, nn.Upsample, Concat → giữ nguyên
            return [f, actual_n, m, args]

    pruned_yaml['backbone'] = [map_layer(*layer) for layer in model_yamls['backbone']]
    pruned_yaml['head'] = [map_layer(*layer) for layer in model_yamls['head']]

    return pruned_yaml


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
    ignore_bn_list = []
    chunk_bn_list = []

    # Debug: print model structure để hiểu architecture
    DEBUG = True  # Set False sau khi fix
    if DEBUG:
        print("\n[DEBUG] Checking Bottleneck modules:")
        bottleneck_count = 0
        for name, module in model.model.named_modules():
            if isinstance(module, Bottleneck):
                bottleneck_count += 1
                print(f"  Found: {name}, add={module.add}")
        print(f"  Total Bottlenecks: {bottleneck_count}")

    for name, module in model.model.named_modules():
        # Bottleneck với residual connection
        if isinstance(module, Bottleneck):
            if module.add:
                # Có residual → không prune cv2
                # C3k standard: model.X.m.j.m.k → parts[-2]='m' → ignore C3k.cv1 + Bottleneck.cv2
                # Attn-type:    model.X.m.j.k   → parts[-2]=digit → chỉ ignore Bottleneck.cv2
                cv2_bn = f"{name}.cv2.bn"
                ignore_bn_list.append(cv2_bn)
                if name.split('.')[-2] == 'm':
                    cv1_bn = f"{name[:-4]}.cv1.bn"
                    ignore_bn_list.append(cv1_bn)
                    if DEBUG:
                        print(f"  Adding to ignore: {cv1_bn}, {cv2_bn}")
                else:
                    if DEBUG:
                        print(f"  Adding to ignore: {cv2_bn}")
            else:
                # Không có residual nhưng có chunk → phải chẵn
                chunk_bn = f"{name[:-4]}.cv1.bn"
                chunk_bn_list.append(chunk_bn)
                if DEBUG:
                    print(f"  Adding to chunk: {chunk_bn}")

        # Thu thập tất cả BN layers
        if isinstance(module, nn.BatchNorm2d):
            bn_dict[name] = module

    # ─────────────────────────────────────
    # Step 1b: PSABlock BNs + cv1.bn của parent layer
    # PSABlock KHÔNG được prune → tất cả BN bên trong phải ignore
    # cv1.bn của layer chứa PSABlock cũng phải ignore (để right_half cố định)
    # ─────────────────────────────────────
    for name, module in model.model.named_modules():
        if isinstance(module, PSABlock):
            # Thêm tất cả BNs bên trong PSABlock vào ignore
            for sub_name, sub_module in module.named_modules():
                if isinstance(sub_module, nn.BatchNorm2d):
                    full_name = f"{name}.{sub_name}"
                    if full_name not in ignore_bn_list:
                        ignore_bn_list.append(full_name)
                        if DEBUG:
                            print(f"  [PSA] ignore: {full_name}")

            # Thêm cv1.bn của layer cha (model.X) vào ignore
            # → đảm bảo right_half không đổi, PSABlock nhận đúng số channels
            parts = name.split('.')
            layer_idx = parts[1]
            cv1_ignore = f"model.{layer_idx}.cv1.bn"
            if cv1_ignore not in ignore_bn_list:
                ignore_bn_list.append(cv1_ignore)
                if DEBUG:
                    print(f"  [PSA parent cv1] ignore: {cv1_ignore}")

    if DEBUG:
        print(f"\n[DEBUG] Total BN layers in model: {len(bn_dict)}")
        print(f"[DEBUG] Expected ignore BNs: {len(ignore_bn_list)}")
        print(f"[DEBUG] Validating ignore_bn_list...")

    # Validate ignore list - SOFT CHECK thay vì hard assert
    missing_bns = []
    for ignore_bn_name in ignore_bn_list:
        if ignore_bn_name not in bn_dict.keys():
            missing_bns.append(ignore_bn_name)

    if missing_bns:
        print(f"\n⚠️  WARNING: {len(missing_bns)} BN không tồn tại trong model:")
        for bn in missing_bns[:10]:  # Hiện tối đa 10
            print(f"    - {bn}")
        print(f"\n  → Loại bỏ các BN không tồn tại khỏi ignore_bn_list")
        ignore_bn_list = [bn for bn in ignore_bn_list if bn in bn_dict.keys()]
        print(f"  → Ignore list sau khi filter: {len(ignore_bn_list)} BNs")

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
    # Tính highest_thre để warn nếu global ratio quá cao
    print("\nStep 3: Validate prune ratio...")

    all_max_gammas = []
    all_gammas = []
    for name, module in bn_dict.items():
        g = module.weight.data.abs().clone().cpu()
        all_max_gammas.append(g.max().item())
        all_gammas.extend(g.tolist())

    highest_thre = min(all_max_gammas)
    sorted_all = torch.sort(torch.tensor(all_gammas))[0]
    percent_limit = (sorted_all == highest_thre).nonzero()[0, 0].item() / len(sorted_all)

    print(f"  Global prune ratio tối đa an toàn: {colorstr(f'{percent_limit:.3f}')}")
    print(f"  Global prune ratio của bạn:         {colorstr(f'{prune_ratio:.3f}')}")

    if prune_ratio > percent_limit:
        prune_ratio = percent_limit
        print(f"  ⚠️ Global ratio giảm xuống {colorstr(f'{prune_ratio:.3f}')} (tránh prune hết 1 layer)")

    # Kiểm tra layer-wise custom ratio
    if layer_ratio_cfg:
        for rule_key, rule_ratio in layer_ratio_cfg.items():
            if float(rule_ratio) > percent_limit:
                print(f"  ⚠️ Layer rule '{rule_key}': ratio={rule_ratio:.3f} > limit={percent_limit:.3f}, có thể nguy hiểm!")
            else:
                print(f"  ✅ Layer rule '{rule_key}': ratio={rule_ratio:.3f} OK")

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
    save_path = os.path.join(save_dir, f"pruned_div{divisor}.pt")
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
    
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3
    """