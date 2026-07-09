"""
YOLOv26 FPGM Pruning (Filter Pruning via Geometric Median).
===========================================================
Based on: He et al., "Filter Pruning via Geometric Median for Deep Convolutional
Neural Networks Acceleration", CVPR 2019

Difference from other pruning methods:
- prune.py:      BN gamma magnitude, same ratio per layer
- prune_lamp.py: BN gamma LAMP scores, adaptive per-layer ratios
- prune_fpgm.py: Conv filter distance to geometric median → prune redundant filters

Y tuong:
- Magnitude-based pruning gia dinh filter nho = khong quan trong → SAI khi filter nho nhung unique
- FPGM: filter gan geometric median = "trung binh", co the bieu dien boi cac filter khac → redundant
- Filter xa geometric median = unique, kho thay the → quan trong
- Geometric median: diem minimize sum khoang cach Euclidean den tat ca filter trong layer

FPGM score cho filter i trong layer l:
    score(i) = ||W_i - GM(W)||_2
    GM(W) = argmin_x sum_i ||W_i - x||_2  (Weiszfeld algorithm)

Filters co score thap (gan geometric median) → bi prune truoc.

Supports 2 modes:
- Per-layer (default): moi layer prune theo cung ratio (nhu prune.py)
- Global:   --global-threshold → pool scores toan cuc, adaptive per-layer (nhu LAMP)

Usage:
    python prune_fpgm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3
    python prune_fpgm.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml --model-size m --prune-ratio 0.5 --divisor 8
    python prune_fpgm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3 --global-threshold
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.block import Bottleneck
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.utils import colorstr

warnings.filterwarnings("ignore")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from dms.dms_utils import build_ignore_bn_list, build_pruned_yaml, get_layer_ratio, make_divisible_channels

# ============================================================================
# FPGM SCORE COMPUTATION
# ============================================================================


def geometric_median(points, max_iter=100, tol=1e-6):
    """Tinh geometric median bang Weiszfeld's algorithm.

    Geometric median = argmin_x sum_i ||x - p_i||_2 Khac voi mean (minimize sum of squared distances), geometric median
    minimize sum of Euclidean distances → robust hon voi outliers.

    Weiszfeld update:
        x_{t+1} = sum(w_i * p_i) / sum(w_i)
        w_i = 1 / ||p_i - x_t||_2

    Args:
        points: tensor shape [N, D] (N filters, D = flattened filter size)
        max_iter: so vong lap toi da
        tol: nguong hoi tu (thay doi tuong doi cua x)

    Returns:
        gm: tensor shape [D], geometric median point
    """
    # Init voi component-wise median (robust init, tot hon mean)
    gm = points.median(dim=0).values.clone()

    for _ in range(max_iter):
        # Khoang cach tu moi point den current estimate
        dists = torch.norm(points - gm.unsqueeze(0), dim=1)  # [N]

        # Tranh chia cho 0 (neu co point trung voi gm)
        dists = dists.clamp(min=1e-8)

        # Weights = 1/distance
        weights = 1.0 / dists  # [N]

        # Weighted average
        gm_new = (weights.unsqueeze(1) * points).sum(dim=0) / weights.sum()

        # Check convergence
        shift = torch.norm(gm_new - gm)
        if shift < tol * (torch.norm(gm) + 1e-8):
            break

        gm = gm_new

    return gm


def compute_fpgm_scores_for_conv(conv_weight):
    """Tinh FPGM score (khoang cach den geometric median) cho 1 Conv layer.

    Score cao = filter xa geometric median = unique = QUAN TRONG → giu lai Score thap = filter gan geometric median =
    redundant = CO THE PRUNE

    Args:
        conv_weight: tensor shape [C_out, C_in, kH, kW]

    Returns:
        scores: tensor shape [C_out], FPGM distance scores
    """
    n_filters = conv_weight.shape[0]

    # Flatten moi filter thanh vector: [C_out, C_in*kH*kW]
    points = conv_weight.reshape(n_filters, -1).detach()

    # Edge case: 1 filter → score = 0 (khong the prune)
    if n_filters <= 1:
        return torch.zeros(n_filters, device=conv_weight.device)

    # Tinh geometric median
    gm = geometric_median(points)

    # Khoang cach Euclidean tu moi filter den geometric median
    scores = torch.norm(points - gm.unsqueeze(0), dim=1)  # [C_out]

    return scores


def compute_all_fpgm_scores(model, bn_dict, ignore_bn_list):
    """Tinh FPGM scores cho tat ca prunable BN layers.

    Map moi BN → Conv tuong ung (bn_name[:-2] + 'conv') → tinh FPGM score tu Conv weights. Ket qua: score per output
    channel (tuong ung BN channel).

    Args:
        model: original model
        bn_dict: {name: nn.BatchNorm2d}
        ignore_bn_list: list BN names khong prune

    Returns:
        fpgm_scores: {bn_name: tensor FPGM scores shape [C_out]}
    """
    # Build mapping: bn_name → Conv module
    conv_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Conv name 'model.X.cv1.conv' → BN name 'model.X.cv1.bn'
            bn_name = name[:-4] + "bn"
            conv_dict[bn_name] = module

    fpgm_scores = {}
    for name in bn_dict:
        if name in ignore_bn_list:
            continue

        conv_module = conv_dict.get(name)
        if conv_module is None:
            # Fallback: dung BN gamma magnitude neu khong tim duoc Conv
            fpgm_scores[name] = bn_dict[name].weight.data.abs()
            continue

        scores = compute_fpgm_scores_for_conv(conv_module.weight.data)
        fpgm_scores[name] = scores

    return fpgm_scores


def compute_fpgm_masks_perlayer(fpgm_scores, bn_dict, prune_ratio, divisor, layer_ratio_cfg=None):
    """Tao masks tu FPGM scores voi per-layer ratio (giong prune.py).

    Moi layer prune theo cung ratio, nhung filter duoc chon dua tren FPGM score thay vi BN gamma.

    Args:
        fpgm_scores: {bn_name: tensor FPGM scores}
        bn_dict: {bn_name: nn.BatchNorm2d}
        prune_ratio: target pruning ratio toan cuc
        divisor: 8 hoac 16
        layer_ratio_cfg: dict custom ratio

    Returns:
        masks: {bn_name: tensor mask shape [C]}
        layer_ratios: {bn_name: float actual ratio}
    """
    masks = {}
    layer_ratios = {}

    for name, scores in fpgm_scores.items():
        origin_channels = bn_dict[name].weight.data.size(0)

        # Lay ratio cho layer nay
        this_ratio = get_layer_ratio(name, layer_ratio_cfg, prune_ratio)

        # So channels giu lai
        n_keep = origin_channels - int(origin_channels * this_ratio)
        n_keep = max(n_keep, divisor)
        n_keep = make_divisible_channels(n_keep, origin_channels, divisor)

        # Top-k theo FPGM score (score cao = xa geometric median = quan trong)
        sorted_idx = torch.argsort(scores, descending=True)
        mask = torch.zeros(origin_channels)
        mask[sorted_idx[:n_keep]] = 1.0

        masks[name] = mask
        layer_ratios[name] = 1.0 - (n_keep / origin_channels)

    return masks, layer_ratios


def compute_fpgm_masks_global(fpgm_scores, bn_dict, prune_ratio, divisor, layer_ratio_cfg=None):
    """Tao masks tu FPGM scores voi global threshold (giong LAMP).

    Pool tat ca FPGM scores (normalized per-layer) thanh 1 list, tim global threshold → adaptive per-layer ratios.

    Normalization quan trong vi FPGM scores co scale khac nhau giua cac layers (Conv 3x3 co filter lon hon Conv 1x1 →
    distances lon hon).

    Args:
        fpgm_scores: {bn_name: tensor FPGM scores}
        bn_dict: {bn_name: nn.BatchNorm2d}
        prune_ratio: target pruning ratio toan cuc
        divisor: 8 hoac 16
        layer_ratio_cfg: dict custom ratio

    Returns:
        masks: {bn_name: tensor mask shape [C]}
        layer_ratios: {bn_name: float actual ratio}
    """
    # Layers co custom ratio → tach rieng
    custom_layers = set()
    if layer_ratio_cfg:
        for name in fpgm_scores:
            custom_ratio = get_layer_ratio(name, layer_ratio_cfg, None)
            if custom_ratio is not None:
                custom_layers.add(name)

    # Normalize FPGM scores per-layer (z-score) de so sanh cross-layer
    # Score cao = quan trong, score thap = redundant → prune
    # Sau normalize: prune channels co normalized score thap nhat globally
    normalized_scores = {}
    for name, scores in fpgm_scores.items():
        if name in custom_layers:
            continue
        mean = scores.mean()
        std = scores.std()
        if std > 1e-8:
            normalized_scores[name] = (scores - mean) / std
        else:
            # All filters gan nhu giong nhau → score = 0 (deu redundant)
            normalized_scores[name] = torch.zeros_like(scores)

    # Pool tat ca normalized scores
    all_scores = []
    for name, scores in normalized_scores.items():
        all_scores.append(scores)

    if all_scores:
        all_scores_cat = torch.cat(all_scores)
        n_total = all_scores_cat.numel()
        n_prune = int(n_total * prune_ratio)
        # Dam bao moi layer con it nhat divisor channels
        n_prune = min(n_prune, n_total - len(normalized_scores) * divisor)
        n_prune = max(n_prune, 0)

        sorted_scores, _ = torch.sort(all_scores_cat)
        if 0 < n_prune < n_total:
            global_threshold = sorted_scores[n_prune].item()
        else:
            global_threshold = float("-inf")

    # Compute per-layer prune counts tu global threshold
    layer_prune_count = {}
    for name, scores in normalized_scores.items():
        n_below = (scores <= global_threshold).sum().item()
        layer_prune_count[name] = int(n_below)

    # Build masks
    masks = {}
    layer_ratios = {}

    for name, raw_scores in fpgm_scores.items():
        origin_channels = bn_dict[name].weight.data.size(0)

        if name in custom_layers:
            # Custom ratio
            this_ratio = get_layer_ratio(name, layer_ratio_cfg, prune_ratio)
            n_keep = origin_channels - int(origin_channels * this_ratio)
        else:
            # FPGM-global determined
            n_prune_layer = layer_prune_count.get(name, 0)
            n_keep = origin_channels - n_prune_layer

        n_keep = max(n_keep, divisor)
        n_keep = make_divisible_channels(n_keep, origin_channels, divisor)

        # Top-k theo raw FPGM score (score cao = giu)
        sorted_idx = torch.argsort(raw_scores, descending=True)
        mask = torch.zeros(origin_channels)
        mask[sorted_idx[:n_keep]] = 1.0

        masks[name] = mask
        layer_ratios[name] = 1.0 - (n_keep / origin_channels)

    return masks, layer_ratios


# ============================================================================
# MAIN PRUNING FUNCTION
# ============================================================================


def main(opt):
    """FPGM pruning workflow.

    Steps:
    1. Collect BN layers va ignore list
    2. Filter BN layers
    3. Validate prune_ratio
    6. Tao pruned YAML config
    7. Tinh FPGM scores va tao masks
    8. Validate divisibility
    9. Build pruned model
    10. Copy weights
    11. Save
    """
    # Parse options
    weights = opt.weights
    prune_ratio = opt.prune_ratio
    cfg = opt.cfg
    model_size = opt.model_size
    save_dir = opt.save_dir
    divisor = opt.divisor
    use_global = opt.global_threshold

    # Load layer-wise custom ratios
    layer_ratio_cfg = {}
    if hasattr(opt, "layer_ratio") and opt.layer_ratio:
        with open(opt.layer_ratio, encoding="utf-8") as f:
            layer_ratio_cfg = yaml.safe_load(f) or {}
        print(f"  Loaded layer ratio config: {opt.layer_ratio} ({len(layer_ratio_cfg)} rules)")

    print(f"\n{'=' * 100}")
    print("FPGM PRUNING CONFIGURATION:")
    print(f"  Model:       {weights}")
    print(f"  Prune ratio: {prune_ratio}")
    print(f"  Divisor:     {divisor}")
    print("  Method:      FPGM (Filter Pruning via Geometric Median)")
    print(f"  Mode:        {'global threshold (adaptive per-layer)' if use_global else 'per-layer (uniform ratio)'}")
    if layer_ratio_cfg:
        print(f"  Layer rules: {layer_ratio_cfg}")
    print(f"{'=' * 100}\n")

    # Load model
    model = AutoBackend(weights, fuse=False)
    model.eval()

    print(f"  Divisor: {divisor} (channels se chia het cho {divisor})")

    # =========================================
    # STEP 1: Thu thap BN layers
    # =========================================
    print("Step 1: Thu thap BatchNorm layers...")

    bn_dict = {}
    chunk_bn_list = []

    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_dict[name] = module
        if isinstance(module, Bottleneck) and not module.add:
            chunk_bn = f"{name[:-4]}.cv1.bn"
            chunk_bn_list.append(chunk_bn)

    ignore_bn_list = build_ignore_bn_list(model.model)

    # Validate ignore list
    missing_bns = [bn for bn in ignore_bn_list if bn not in bn_dict]
    if missing_bns:
        print(f"  WARNING: {len(missing_bns)} BN in ignore list not found in model, removing")
        ignore_bn_list = [bn for bn in ignore_bn_list if bn in bn_dict]

    print(f"  Tong BN layers: {len(bn_dict)}")
    print(f"  Ignore (residual): {len(ignore_bn_list)}")
    print(f"  Chunk constraint: {len(chunk_bn_list)}")

    # =========================================
    # STEP 2: Filter BN layers (for display)
    # =========================================
    print("\nStep 2: Filter BN layers...")

    prunable_bn_dict = {k: v for k, v in bn_dict.items() if k not in ignore_bn_list}
    print(f"  BN layers sau filter: {len(prunable_bn_dict)}")

    # =========================================
    # STEP 3: Validate prune_ratio
    # =========================================
    print("\nStep 3: Validate prune ratio...")
    print(f"  Prune ratio: {colorstr(f'{prune_ratio:.3f}')}")
    mode_str = "FPGM global (adaptive per-layer)" if use_global else "FPGM per-layer (uniform ratio)"
    print(f"  Mode: {mode_str}")

    if prune_ratio >= 0.9:
        print(f"  WARNING: Prune ratio rat cao ({prune_ratio:.2f}), model co the mat accuracy nghiem trong!")
    elif prune_ratio >= 0.7:
        print(f"  INFO: Prune ratio cao ({prune_ratio:.2f}), nen fine-tune ky sau pruning")

    if layer_ratio_cfg:
        print("  Layer-wise custom ratios:")
        for rule_key, rule_ratio in layer_ratio_cfg.items():
            print(f"    {rule_key}: {float(rule_ratio):.3f}")

    # =========================================
    # STEP 6: Tao pruned YAML
    # =========================================
    print("\nStep 6: Tao pruned model config...")

    nc = model.model.nc
    pruned_yaml = build_pruned_yaml(cfg, model_size, nc)

    print(f"  nc: {nc}")
    print(f"  scale: {model_size}")
    print(f"  end2end: {pruned_yaml.get('end2end', False)}")
    print(f"  Backbone layers: {len(pruned_yaml['backbone'])}")
    print(f"  Head layers: {len(pruned_yaml['head'])}")
    for idx, (f, n, m, args) in enumerate(pruned_yaml["backbone"] + pruned_yaml["head"]):
        print(f"    [{idx:>2}] n={n} {m:<20} args={args}")

    # =========================================
    # STEP 7: FPGM scores + masks
    # =========================================
    print(f"\nStep 7: Tinh FPGM scores va tao masks (divisor={divisor})...")

    # 7.1: Compute FPGM scores (distance to geometric median per layer)
    fpgm_scores = compute_all_fpgm_scores(model.model, bn_dict, ignore_bn_list)

    # Print FPGM score statistics
    all_fpgm = torch.cat(list(fpgm_scores.values()))
    print(
        f"  FPGM scores: min={all_fpgm.min():.6f}, max={all_fpgm.max():.6f}, "
        f"mean={all_fpgm.mean():.6f}, median={all_fpgm.median():.6f}"
    )

    # 7.2: Compute masks
    if use_global:
        fpgm_masks, layer_ratios = compute_fpgm_masks_global(
            fpgm_scores, bn_dict, prune_ratio, divisor, layer_ratio_cfg
        )
    else:
        fpgm_masks, layer_ratios = compute_fpgm_masks_perlayer(
            fpgm_scores, bn_dict, prune_ratio, divisor, layer_ratio_cfg
        )

    # 7.3: Build maskbndict (bao gom ca ignored layers voi mask=1)
    print("\n" + "=" * 120)
    print(
        f"{'Layer name':<35} | {'Origin':>6} | {'FPGM ratio':>10} | {'Keep':>6} | {'Rounded':>7} | {'Sparsity':>8} | {'Note'}"
    )
    print("=" * 120)

    maskbndict = {}

    for name, module in model.model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue

        origin_channels = module.weight.data.size(0)

        if name in ignore_bn_list:
            maskbndict[name] = torch.ones(origin_channels)
            print(
                f"{name:<35} | {origin_channels:>6} | {'  -':>10} | {'  -':>6} | {'  -':>7} | {'  -':>8} | SKIP (residual)"
            )
            continue

        mask = fpgm_masks[name]
        actual_ratio = layer_ratios[name]
        remaining = mask.sum().int().item()

        # Validate
        assert mask.sum() > 0, f"BN {name} khong co kenh nao!"
        assert mask.sum() % divisor == 0, f"BN {name}: {mask.sum()} channels khong chia het cho {divisor}!"

        # Apply mask to BN weights (zero out pruned channels)
        module.weight.data.mul_(mask)
        module.bias.data.mul_(mask)

        sparsity = 1 - (remaining / origin_channels)
        note = ""
        if layer_ratio_cfg:
            custom = get_layer_ratio(name, layer_ratio_cfg, None)
            if custom is not None:
                note = "custom"

        print(
            f"{name:<35} | {origin_channels:>6} | {actual_ratio:>10.3f} | "
            f"{origin_channels - int(origin_channels * actual_ratio):>6} | {remaining:>7} | {sparsity:>8.3f} | {note}"
        )

        maskbndict[name] = mask

    print("=" * 120)

    # Print ratio statistics
    ratios = list(layer_ratios.values())
    if ratios:
        print(f"\n  FPGM {'adaptive' if use_global else 'uniform'} ratios:")
        print(f"    Target global:  {prune_ratio:.3f}")
        print(f"    Actual average: {sum(ratios) / len(ratios):.3f}")
        print(f"    Min layer:      {min(ratios):.3f}")
        print(f"    Max layer:      {max(ratios):.3f}")
        print(f"    Std dev:        {torch.tensor(ratios).std():.3f}")

    # =========================================
    # STEP 8: Validate divisibility
    # =========================================
    print("\nStep 8: Validate divisibility...")

    all_valid = True
    for name, mask in maskbndict.items():
        channels = mask.sum().int().item()
        if channels % divisor != 0:
            print(f"  FAIL {name}: {channels} channels NOT divisible by {divisor}")
            all_valid = False

    if all_valid:
        print(f"  OK Tat ca layers deu chia het cho {divisor}!")
    else:
        raise RuntimeError("Co layers khong chia het cho divisor!")

    # =========================================
    # STEP 9: Build pruned model
    # =========================================
    print("\nStep 9: Build pruned model...")

    pruned_model = DetectionModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    pruned_model.eval()

    # =========================================
    # STEP 10: Copy weights
    # =========================================
    print("\nStep 10: Copy weights tu original model...")

    current_to_prev = pruned_model.current_to_prev

    # Validate current_to_prev
    for xks, xvs in current_to_prev.items():
        xvs = [xvs] if not isinstance(xvs, list) else xvs
        for xk, xv in zip([xks] if not isinstance(xks, list) else xks, xvs):
            assert xk in maskbndict.keys() or "model." in xk, f"{xk} from 'current_to_prev' not valid"
            if xv is not None:
                assert xv in maskbndict.keys(), f"{xv} from 'current_to_prev' not in maskbndict"

    changed = []

    # Patterns
    pattern_c3k_first = re.compile(
        r"model\.\d+\.m\.0\.(cv1|cv2)\.bn"
        r"|model\.\d+\.m\.0\.0\.cv1\.bn"
    )
    pattern_detect = re.compile(r"model\.\d+\.(?:cv\d|one2one_cv\d)\.\d\.2")

    # Thu thap SPPF n_param dong
    sppf_n_params = {}
    for sppf_name, sppf_module in model.model.named_modules():
        if (
            hasattr(sppf_module, "n")
            and hasattr(sppf_module, "cv1")
            and hasattr(sppf_module, "cv2")
            and hasattr(sppf_module, "m")
            and isinstance(sppf_module.m, nn.MaxPool2d)
        ):
            sppf_n_params[sppf_name] = sppf_module.n
    sppf_cv2_pattern = re.compile(r"model\.(\d+)\.cv2\.conv")

    for (name_org, module_org), (name_pruned, module_pruned) in zip(
        model.model.named_modules(remove_duplicate=False), pruned_model.named_modules(remove_duplicate=False)
    ):
        assert name_org == name_pruned, f"name mismatch: {name_org} != {name_pruned}"

        if "dfl" in name_org:
            continue

        # ─────────────────────────────────────
        # Detect head - Conv2d khong co BN
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
        # Conv layers
        # ─────────────────────────────────────
        if isinstance(module_org, nn.Conv2d):
            current_bn_layer_name = name_org[:-4] + "bn"

            if current_bn_layer_name not in maskbndict:
                continue

            # PSABlock internal layers
            if current_bn_layer_name in ignore_bn_list and current_bn_layer_name not in current_to_prev:
                module_pruned.weight.data = module_org.weight.data.clone()
                if module_org.bias is not None:
                    module_pruned.bias.data = module_org.bias.data.clone()
                changed.append(current_bn_layer_name)
                continue

            out_channels_mask = maskbndict[current_bn_layer_name].to(torch.bool)
            prev_bn_layer_name = current_to_prev.get(current_bn_layer_name, None)

            # Input channels
            if isinstance(prev_bn_layer_name, list):
                in_channels_masks = [maskbndict[ni] for ni in prev_bn_layer_name]
                in_channels_mask = torch.cat(in_channels_masks, dim=0).to(torch.bool)
            elif prev_bn_layer_name is not None:
                in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)

                if pattern_c3k_first.fullmatch(current_bn_layer_name) is not None:
                    is_bottleneck_cv2 = False
                    m_cv2 = re.fullmatch(r"model\.\d+\.m\.0\.cv2\.bn", current_bn_layer_name)
                    if m_cv2:
                        parent = current_bn_layer_name.rsplit(".cv2.bn", 1)[0]
                        if f"{parent}.cv3.bn" not in maskbndict:
                            is_bottleneck_cv2 = True
                    if not is_bottleneck_cv2:
                        in_channels_mask = in_channels_mask.chunk(2, 0)[1]

                sppf_match = sppf_cv2_pattern.fullmatch(name_org)
                if sppf_match:
                    sppf_layer = f"model.{sppf_match.group(1)}"
                    if sppf_layer in sppf_n_params:
                        n_param = sppf_n_params[sppf_layer]
                        in_channels_mask = torch.cat([in_channels_mask] * (n_param + 1), dim=0)
            else:
                in_channels_mask = torch.ones(module_org.weight.data.shape[1], dtype=torch.bool)

            # Validate
            expected_in = in_channels_mask.sum().int().item()
            expected_out = out_channels_mask.sum().int().item()

            if expected_in != module_pruned.in_channels:
                print("\n  SHAPE MISMATCH DETECTED:")
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
                state_dict_org = module_org.weight.data[out_channels_mask, :, :, :]
            else:
                state_dict_org = module_org.weight.data[out_channels_mask, :, :, :]
                state_dict_org = state_dict_org[:, in_channels_mask, :, :]
            module_pruned.weight.data = state_dict_org

            if module_org.bias is not None:
                module_pruned.bias.data = module_org.bias.data[out_channels_mask]

            changed.append(current_bn_layer_name)

        # ─────────────────────────────────────
        # BatchNorm layers
        # ─────────────────────────────────────
        if isinstance(module_org, nn.BatchNorm2d):
            out_channels_mask = maskbndict[name_org].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[out_channels_mask]
            module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            module_pruned.running_mean = module_org.running_mean[out_channels_mask]
            module_pruned.running_var = module_org.running_var[out_channels_mask]

    # Validate tat ca BN da duoc xu ly
    missing = [name for name in maskbndict.keys() if name not in changed and name not in ignore_bn_list]
    assert not missing, f"Missing BN layers: {missing}"

    print("   Copy weights hoan tat!")

    # =========================================
    # STEP 11: Save model
    # =========================================
    print("\nStep 11: Save pruned model...")

    pruned_model.eval()
    input_name = Path(weights).stem
    suffix = "fpgm_global" if use_global else "fpgm"
    save_path = os.path.join(save_dir, f"{input_name}_{suffix}_pruned_div{divisor}.pt")
    torch.save(
        {
            "model": pruned_model,
            "maskbndict": maskbndict,
            "config": {
                "divisor": divisor,
                "prune_ratio": prune_ratio,
                "method": "fpgm_global" if use_global else "fpgm",
                "layer_ratios": layer_ratios,
            },
        },
        save_path,
    )

    print(f"   Model saved: {save_path}")

    # Test forward
    print("\nTesting forward pass...")
    model_test = torch.load(save_path, weights_only=False)["model"].cuda()
    dummies = torch.randn([1, 3, 640, 640], dtype=torch.float32).cuda()
    with torch.no_grad():
        model_test(dummies)
    print("   Forward pass successful!")

    # Print summary
    print_summary(maskbndict, layer_ratios, divisor, prune_ratio, save_path, use_global)

    return maskbndict, pruned_yaml


def print_summary(
    maskbndict: dict, layer_ratios: dict, divisor: int, prune_ratio: float, save_path: str, use_global: bool
):
    """In tom tat ket qua FPGM pruning."""
    total_origin = 0
    total_pruned = 0

    for name, mask in maskbndict.items():
        total_origin += len(mask)
        total_pruned += mask.sum().int().item()

    compression_ratio = total_origin / total_pruned if total_pruned > 0 else 0

    ratios = list(layer_ratios.values())

    print("\n" + "=" * 100)
    print(" FPGM PRUNING SUMMARY")
    print("=" * 100)
    print("Method:            FPGM (Filter Pruning via Geometric Median)")
    print(f"Mode:              {'global threshold' if use_global else 'per-layer ratio'}")
    print(f"Divisor:           {divisor}")
    print(f"Target ratio:      {prune_ratio:.3f}")
    print(f"Total channels:    {total_origin:,} -> {total_pruned:,}")
    print(f"Actual global:     {1 - total_pruned / total_origin:.3f}")
    print(f"Compression:       {compression_ratio:.2f}x")
    if ratios:
        print(f"Layer ratio range: [{min(ratios):.3f}, {max(ratios):.3f}]")
        print(f"Layer ratio std:   {torch.tensor(ratios).std():.3f}")
    print(f"Model saved:       {save_path}")
    print("=" * 100)
    print("\n FPGM PRUNING HOAN TAT!\n")


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO26 FPGM Pruning (Filter Pruning via Geometric Median)")

    # Basic options
    parser.add_argument("--weights", type=str, default=ROOT / "weights/best.pt", help="model.pt path")
    parser.add_argument(
        "--cfg", type=str, default=ROOT / "ultralytics/cfg/models/26/yolo26.yaml", help="model.yaml path"
    )
    parser.add_argument("--model-size", type=str, default="m", choices=["n", "s", "m", "l", "x"], help="model size")

    # Pruning options
    parser.add_argument("--prune-ratio", type=float, default=0.5, help="target prune ratio toan cuc (0.0-1.0)")
    parser.add_argument("--layer-ratio", type=str, default=None, help="YAML file chua custom ratio cho tongue layer")
    parser.add_argument(
        "--global-threshold",
        action="store_true",
        help="dung global threshold (adaptive per-layer) thay vi per-layer ratio. "
        "Mac dinh: per-layer (moi layer prune cung ratio). "
        "Global: layers co nhieu redundant filters bi prune nhieu hon.",
    )

    # Divisibility options
    parser.add_argument(
        "--divisor",
        type=int,
        default=8,
        choices=[8, 16],
        help="divisor cho channels (8 cho GPU thuong, 16 cho Tensor Cores)",
    )

    # Output options
    parser.add_argument("--save-dir", type=str, default=ROOT / "weights", help="pruned model save directory")

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    """
# Co ban (per-layer ratio, giong prune.py nhung dung FPGM score thay vi BN gamma)
python prune_fpgm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Day du
python prune_fpgm.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml --model-size m --prune-ratio 0.5 --divisor 8

# Global threshold (adaptive per-layer, tuong tu LAMP nhung dung FPGM score)
python prune_fpgm.py --weights weights/yolo26m_baseline.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5 

# Voi custom layer ratio
python prune_fpgm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3 --layer-ratio layer_ratio.yaml

# So sanh 3 phuong phap:
# prune.py:      BN gamma → filter nho bi prune (co the prune filter unique nhung nho)
# prune_lamp.py: LAMP score → normalize theo layer energy → fair hon giua layers
# prune_fpgm.py: FPGM score → filter redundant bi prune (bat ke magnitude)
#
# FPGM tot hon khi:
# - Model chua duoc SR training (BN gamma chua duoc push ve 0)
# - Muon prune dua tren filter redundancy thay vi magnitude
# - Layers co nhieu filters tuong tu nhau (conv 1x1 trong neck)
    """
