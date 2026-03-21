"""
Shared pruning pipeline for all pruning methods.
================================================
Extracts common logic from prune.py so that different pruning criteria
(BN gamma, L1 norm, Taylor, Random) can reuse the same mask-creation,
weight-copying, and model-saving code.

Usage from any prune_*.py:
    from prune_common import load_and_prepare, create_masks, finalize_pruning, add_common_args
"""

import re
import os
import sys
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import torch
import torch.nn as nn
from ultralytics.utils import colorstr, LOGGER
from ultralytics.nn.modules.block import Bottleneck, PSABlock
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from dms_utils import make_divisible_channels, get_layer_ratio, build_pruned_yaml, build_ignore_bn_list

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ============================================================================
# STEP 1-6: Load model and prepare
# ============================================================================

def load_and_prepare(weights, cfg, model_size, layer_ratio_path=None):
    """
    Load model, collect BN layers, build pruned config.

    Returns:
        model:           AutoBackend model (fuse=False)
        bn_dict:         Dict[str, nn.BatchNorm2d] - tất cả BN layers
        ignore_bn_list:  List[str] - BN không được prune (residual, PSABlock)
        chunk_bn_list:   List[str] - BN có chunk constraint
        layer_ratio_cfg: Dict - per-layer custom ratios
        pruned_yaml:     Dict - pruned model config
    """
    # Load layer-wise custom ratios
    layer_ratio_cfg = {}
    if layer_ratio_path:
        with open(layer_ratio_path, encoding='utf-8') as f:
            layer_ratio_cfg = yaml.safe_load(f) or {}
        print(f"  Loaded layer ratio config: {layer_ratio_path} ({len(layer_ratio_cfg)} rules)")

    # Load model
    model = AutoBackend(weights, fuse=False)
    model.eval()

    # Collect BN layers
    print("Step 1: Thu thập BatchNorm layers...")
    bn_dict = {}
    chunk_bn_list = []

    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_dict[name] = module
        if isinstance(module, Bottleneck) and not module.add:
            chunk_bn = f"{name[:-4]}.cv1.bn"
            chunk_bn_list.append(chunk_bn)

    # Build ignore list
    ignore_bn_list = build_ignore_bn_list(model.model)
    missing_bns = [bn for bn in ignore_bn_list if bn not in bn_dict]
    if missing_bns:
        print(f"  WARNING: {len(missing_bns)} BN in ignore list not found, removing")
        ignore_bn_list = [bn for bn in ignore_bn_list if bn in bn_dict]

    print(f"  Tổng BN layers: {len(bn_dict)}")
    print(f"  Ignore (residual): {len(ignore_bn_list)}")
    print(f"  Chunk constraint: {len(chunk_bn_list)}")

    # Filter
    prunable_count = len({k for k in bn_dict if k not in ignore_bn_list})
    print(f"  Prunable BN layers: {prunable_count}")

    # Build pruned YAML
    print("\nStep 6: Tạo pruned model config...")
    nc = model.model.nc
    pruned_yaml = build_pruned_yaml(cfg, model_size, nc)

    print(f"  nc: {nc}, scale: {model_size}")
    print(f"  Backbone layers: {len(pruned_yaml['backbone'])}")
    print(f"  Head layers: {len(pruned_yaml['head'])}")
    for idx, (f, n, m, args) in enumerate(pruned_yaml['backbone'] + pruned_yaml['head']):
        print(f"    [{idx:>2}] n={n} {m:<20} args={args}")

    return model, bn_dict, ignore_bn_list, chunk_bn_list, layer_ratio_cfg, pruned_yaml


# ============================================================================
# STEP 7: Create masks from importance scores
# ============================================================================

def create_masks(importance_scores, model, ignore_bn_list, layer_ratio_cfg,
                 prune_ratio, divisor):
    """
    Create pruning masks from pre-computed importance scores.

    Args:
        importance_scores: Dict[bn_name, Tensor] - importance per channel cho prunable BNs
        model:             AutoBackend model
        ignore_bn_list:    List[str] - BN layers to skip
        layer_ratio_cfg:   Dict - per-layer custom ratios
        prune_ratio:       float - global prune ratio
        divisor:           int - channel divisibility constraint (8 or 16)

    Returns:
        maskbndict: Dict[str, Tensor] - binary mask cho mỗi BN layer
    """
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
                this_ratio = get_layer_ratio(name, layer_ratio_cfg, prune_ratio)

                # Lấy importance scores (fallback: BN gamma nếu không có)
                if name in importance_scores:
                    scores = importance_scores[name].view(-1).float()
                else:
                    scores = module.weight.data.abs().view(-1)

                # Tạo mask từ scores
                local_sorted = torch.sort(scores)[0]
                local_thre = local_sorted[int(len(local_sorted) * this_ratio)]
                mask = scores.gt(local_thre).float()
                current_channels = mask.sum().int().item()

                # Edge case: tất cả channels bị prune
                if current_channels == 0:
                    sorted_idx = torch.argsort(scores, descending=True)
                    mask = torch.zeros_like(scores)
                    mask[sorted_idx[:divisor]] = 1.0
                    current_channels = divisor

                # Làm tròn đến bội số của divisor
                target_channels = make_divisible_channels(current_channels, origin_channels, divisor)

                # Tạo lại mask top-k chính xác
                sorted_idx = torch.argsort(scores, descending=True)
                mask = torch.zeros_like(scores)
                mask[sorted_idx[:target_channels]] = 1.0

                assert mask.sum() > 0, f"BN {name} không có kênh nào!"
                assert mask.sum() % divisor == 0, \
                    f"BN {name}: {mask.sum()} channels không chia hết cho {divisor}!"

                # Apply mask lên BN weights
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
    return maskbndict


# ============================================================================
# STEPS 8-11: Validate, build, copy weights, save
# ============================================================================

def finalize_pruning(model, maskbndict, pruned_yaml, ignore_bn_list,
                     weights, save_dir, divisor, prune_ratio, method_name="pruned"):
    """
    Steps 8-11: validate divisibility, build pruned model, copy weights, save.

    Args:
        model:          AutoBackend model (original)
        maskbndict:     Dict[str, Tensor] - binary masks
        pruned_yaml:    Dict - pruned model config
        ignore_bn_list: List[str] - ignored BN layers
        weights:        str - path to original weights (for naming)
        save_dir:       str - output directory
        divisor:        int - channel divisibility
        prune_ratio:    float - for metadata
        method_name:    str - suffix for saved filename

    Returns:
        save_path: str - path to saved model
    """
    # ─── Step 8: Validate divisibility ───
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

    # ─── Step 9: Build pruned model ───
    print("\nStep 9: Build pruned model...")
    pruned_model = DetectionModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    pruned_model.eval()

    # ─── Step 10: Copy weights ───
    print("\nStep 10: Copy weights từ original model...")
    _copy_weights(model, pruned_model, maskbndict, ignore_bn_list)
    print("   Copy weights hoàn tất!")

    # ─── Step 11: Save ───
    print("\nStep 11: Save pruned model...")
    pruned_model.eval()
    input_name = Path(weights).stem
    save_path = os.path.join(save_dir, f"{input_name}_{method_name}_div{divisor}.pt")
    torch.save(
        {
            "model": pruned_model,
            "maskbndict": maskbndict,
            "config": {
                "divisor": divisor,
                "prune_ratio": prune_ratio,
                "method": method_name,
            }
        },
        save_path
    )
    print(f"   Model saved: {save_path}")

    # Test forward
    print("\nTesting forward pass...")
    model_test = torch.load(save_path, weights_only=False)["model"].cuda()
    dummies = torch.randn([1, 3, 640, 640], dtype=torch.float32).cuda()
    with torch.no_grad():
        output = model_test(dummies)
    print("   Forward pass successful!")

    # Summary
    print_summary(maskbndict, divisor, prune_ratio, save_path, method_name)
    return save_path


def _copy_weights(model, pruned_model, maskbndict, ignore_bn_list):
    """Copy weights from original model to pruned model (Step 10 internals)."""
    current_to_prev = pruned_model.current_to_prev

    # Validate current_to_prev
    for xks, xvs in current_to_prev.items():
        xvs = [xvs] if not isinstance(xvs, list) else xvs
        for xk, xv in zip([xks] if not isinstance(xks, list) else xks, xvs):
            assert xk in maskbndict.keys() or 'model.' in xk, \
                f"{xk} from 'current_to_prev' not valid"
            if xv is not None:
                assert xv in maskbndict.keys(), \
                    f"{xv} from 'current_to_prev' not in maskbndict"

    changed = []

    # Patterns
    pattern_c3k_first = re.compile(
        r"model\.\d+\.m\.0\.(cv1|cv2)\.bn"
        r"|model\.\d+\.m\.0\.0\.cv1\.bn"
    )
    pattern_detect = re.compile(r"model\.\d+\.(?:cv\d|one2one_cv\d)\.\d\.2")

    # Thu thập SPPF n_param động
    sppf_n_params = {}
    for sppf_name, sppf_module in model.model.named_modules():
        if hasattr(sppf_module, 'n') and hasattr(sppf_module, 'cv1') and hasattr(sppf_module, 'cv2') \
                and hasattr(sppf_module, 'm') and isinstance(sppf_module.m, nn.MaxPool2d):
            sppf_n_params[sppf_name] = sppf_module.n
    sppf_cv2_pattern = re.compile(r"model\.(\d+)\.cv2\.conv")

    for (name_org, module_org), (name_pruned, module_pruned) in \
            zip(model.model.named_modules(remove_duplicate=False),
                pruned_model.named_modules(remove_duplicate=False)):

        assert name_org == name_pruned, f"name mismatch: {name_org} != {name_pruned}"

        if 'dfl' in name_org:
            continue

        # Detect head - Conv2d không có BN
        if pattern_detect.fullmatch(name_org) is not None:
            current_conv_layer_name = name_org
            prev_bn_layer_name = current_to_prev[current_conv_layer_name]
            in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[:, in_channels_mask, :, :]
            if module_org.bias is not None:
                module_pruned.bias.data = module_org.bias.data
            continue

        # Conv layers
        if isinstance(module_org, nn.Conv2d):
            current_bn_layer_name = name_org[:-4] + 'bn'

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

            # Xử lý input channels
            if isinstance(prev_bn_layer_name, list):
                in_channels_masks = [maskbndict[ni] for ni in prev_bn_layer_name]
                in_channels_mask = torch.cat(in_channels_masks, dim=0).to(torch.bool)
            elif prev_bn_layer_name is not None:
                in_channels_mask = maskbndict[prev_bn_layer_name].to(torch.bool)

                # C3k[0] cv1/cv2 chunk handling
                if pattern_c3k_first.fullmatch(current_bn_layer_name) is not None:
                    is_bottleneck_cv2 = False
                    m_cv2 = re.fullmatch(r"model\.\d+\.m\.0\.cv2\.bn", current_bn_layer_name)
                    if m_cv2:
                        parent = current_bn_layer_name.rsplit('.cv2.bn', 1)[0]
                        if f"{parent}.cv3.bn" not in maskbndict:
                            is_bottleneck_cv2 = True
                    if not is_bottleneck_cv2:
                        in_channels_mask = in_channels_mask.chunk(2, 0)[1]

                # SPPF cv2 dynamic n_param
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
                print(f"\n❌ SHAPE MISMATCH:")
                print(f"   Layer: {name_org}")
                print(f"   Expected in: {expected_in}, Actual in: {module_pruned.in_channels}")
                print(f"   prev_bn: {prev_bn_layer_name}, current_bn: {current_bn_layer_name}")
                if isinstance(prev_bn_layer_name, list):
                    for pbn in prev_bn_layer_name:
                        print(f"     - {pbn}: {maskbndict[pbn].sum().int().item()} ch")
                raise RuntimeError(f"Shape mismatch at {name_org}")

            if expected_out != module_pruned.out_channels:
                raise RuntimeError(
                    f"{name_org} out_channels mismatch: {expected_out} vs {module_pruned.out_channels}")

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

        # BatchNorm layers
        if isinstance(module_org, nn.BatchNorm2d):
            out_channels_mask = maskbndict[name_org].to(torch.bool)
            module_pruned.weight.data = module_org.weight.data[out_channels_mask]
            module_pruned.bias.data = module_org.bias.data[out_channels_mask]
            module_pruned.running_mean = module_org.running_mean[out_channels_mask]
            module_pruned.running_var = module_org.running_var[out_channels_mask]

    # Validate tất cả BN đã xử lý
    missing = [name for name in maskbndict.keys() if name not in changed and name not in ignore_bn_list]
    assert not missing, f"Missing BN layers: {missing}"


# ============================================================================
# UTILITIES
# ============================================================================

def print_summary(maskbndict, divisor, prune_ratio, save_path, method_name="pruned"):
    """In tóm tắt kết quả pruning."""
    total_origin = 0
    total_pruned = 0
    for name, mask in maskbndict.items():
        total_origin += len(mask)
        total_pruned += mask.sum().int().item()

    compression_ratio = total_origin / total_pruned if total_pruned > 0 else 0

    print("\n" + "=" * 100)
    print(f" PRUNING SUMMARY ({method_name})")
    print("=" * 100)
    print(f"Method:            {method_name}")
    print(f"Divisor:           {divisor}")
    print(f"Prune ratio:       {prune_ratio:.3f}")
    print(f"Total channels:    {total_origin:,} → {total_pruned:,}")
    print(f"Compression:       {compression_ratio:.2f}x")
    print(f"Model saved:       {save_path}")
    print("=" * 100)
    print("\n PRUNING HOÀN TẤT!\n")


def add_common_args(parser):
    """Add common argparse arguments shared by all pruning methods."""
    parser.add_argument('--weights', type=str,
                        default=ROOT / 'weights/best.pt',
                        help='model.pt path')
    parser.add_argument('--cfg', type=str,
                        default=ROOT / 'ultralytics/cfg/models/26/yolo26.yaml',
                        help='model.yaml path')
    parser.add_argument('--model-size', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='model size')
    parser.add_argument('--prune-ratio', type=float, default=0.5,
                        help='prune ratio toàn cục (0.0-1.0)')
    parser.add_argument('--layer-ratio', type=str, default=None,
                        help='YAML file chứa custom ratio cho từng layer')
    parser.add_argument('--divisor', type=int, default=8,
                        choices=[8, 16],
                        help='divisor cho channels (8=GPU, 16=Tensor Cores)')
    parser.add_argument('--save-dir', type=str,
                        default=ROOT / 'weights',
                        help='pruned model save directory')
    return parser