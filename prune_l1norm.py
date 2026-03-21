"""
L1 Norm Pruning
===============
Pruning dựa trên L1 norm của Conv filter weights.
Importance score = ||Conv.weight[i]||_1 (tổng trị tuyệt đối per output filter).

Ưu điểm so với BN gamma:
- Trực tiếp đo lường magnitude của filter, không phụ thuộc vào Sparsity Training (SR)
- Có thể dùng cho model chưa qua SR training
- Reference: "Pruning Filters for Efficient ConvNets" (Li et al., ICLR 2017)

Nhược điểm:
- Không xét đến gradient/activation → filter nhỏ nhưng quan trọng vẫn bị cắt
- Kém hơn Taylor và BN gamma (sau SR) trong thực nghiệm

Usage:
    python prune_l1norm.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3
"""

import argparse
import torch
import torch.nn as nn

from prune_common import (
    ROOT, load_and_prepare, create_masks, finalize_pruning, add_common_args
)


def compute_l1norm_importance(model, bn_dict, ignore_bn_list):
    """
    Tính L1 norm importance cho mỗi BN layer dựa trên Conv filter tương ứng.

    Mapping: BN name `model.X.cv1.bn` → Conv name `model.X.cv1.conv`
    Importance[i] = sum(|Conv.weight[i, :, :, :]|) = L1 norm của filter thứ i

    Returns:
        Dict[str, Tensor] - importance score per channel cho mỗi prunable BN
    """
    importance = {}
    modules_dict = dict(model.model.named_modules())

    for bn_name, bn_module in bn_dict.items():
        if bn_name in ignore_bn_list:
            continue

        # BN name → Conv name: model.X.cv1.bn → model.X.cv1.conv
        conv_name = bn_name[:-2] + 'conv'

        if conv_name not in modules_dict:
            print(f"  WARNING: Conv {conv_name} not found for BN {bn_name}, using BN gamma fallback")
            importance[bn_name] = bn_module.weight.data.abs().view(-1)
            continue

        conv_module = modules_dict[conv_name]

        # L1 norm per output filter: sum(|weight[i, :, :, :]|)
        # weight shape: [out_channels, in_channels/groups, kH, kW]
        scores = conv_module.weight.data.abs().sum(dim=[1, 2, 3])
        importance[bn_name] = scores

    print(f"  L1 Norm importance computed for {len(importance)} layers")
    return importance


def main():
    parser = argparse.ArgumentParser(description='YOLO26 L1 Norm Pruning')
    add_common_args(parser)
    opt = parser.parse_args()

    print(f"\n{'='*100}")
    print(f"L1 NORM PRUNING")
    print(f"  Model:       {opt.weights}")
    print(f"  Prune ratio: {opt.prune_ratio}")
    print(f"  Divisor:     {opt.divisor}")
    print(f"{'='*100}\n")

    # Step 1-6: Load and prepare
    model, bn_dict, ignore_bn_list, chunk_bn_list, layer_ratio_cfg, pruned_yaml = \
        load_and_prepare(opt.weights, opt.cfg, opt.model_size, opt.layer_ratio)

    # Compute L1 norm importance
    print("\nComputing L1 norm importance scores...")
    importance = compute_l1norm_importance(model, bn_dict, ignore_bn_list)

    # Step 7: Create masks
    maskbndict = create_masks(
        importance, model, ignore_bn_list, layer_ratio_cfg, opt.prune_ratio, opt.divisor
    )

    # Steps 8-11: Build, copy, save
    save_path = finalize_pruning(
        model, maskbndict, pruned_yaml, ignore_bn_list,
        opt.weights, opt.save_dir, opt.divisor, opt.prune_ratio,
        method_name="l1norm"
    )

    return maskbndict, pruned_yaml


if __name__ == "__main__":
    main()