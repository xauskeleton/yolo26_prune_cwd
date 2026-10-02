r"""
Random Pruning (Baseline).
=========================
Pruning ngẫu nhiên - chọn channels để giữ lại một cách random.
Dùng làm baseline để so sánh với các phương pháp pruning khác.

Ưu điểm:
- Nhanh nhất, không cần data hay computation
- Baseline công bằng: nếu method X tệ hơn random → method X không có giá trị
- Reproducible với --seed

Kỳ vọng kết quả:
- Kém hơn BN gamma, L1 norm, Taylor ở cùng prune ratio
- Finetune có thể recover phần nào
- Ở prune ratio thấp (<0.2), gap vs structured methods nhỏ

Usage:
    python prune_random.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

    # Reproducible
    python prune_random.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml \\
        --prune-ratio 0.3 --seed 42
"""

import argparse

import torch
from prune_common import add_common_args, create_masks, finalize_pruning, load_and_prepare


def compute_random_importance(model, bn_dict, ignore_bn_list, seed=None):
    """Tạo random importance scores cho mỗi BN layer.

    Mỗi channel nhận 1 score ngẫu nhiên từ uniform(0, 1). Channels có score cao hơn sẽ được giữ lại.

    Args:
        model: AutoBackend model
        bn_dict: Dict[str, BN]
        ignore_bn_list: List[str]
        seed: Optional[int] - random seed cho reproducibility

    Returns:
        Dict[str, Tensor] - random importance per channel
    """
    if seed is not None:
        torch.manual_seed(seed)
        print(f"  Random seed: {seed}")

    importance = {}
    for bn_name, bn_module in bn_dict.items():
        if bn_name in ignore_bn_list:
            continue
        num_channels = bn_module.weight.data.size(0)
        importance[bn_name] = torch.rand(num_channels)

    print(f"  Random importance generated for {len(importance)} layers")
    return importance


def main():
    parser = argparse.ArgumentParser(description="YOLO26 Random Pruning (Baseline)")
    add_common_args(parser)
    parser.add_argument("--seed", type=int, default=None, help="Random seed cho reproducibility")
    opt = parser.parse_args()

    print(f"\n{'=' * 100}")
    print("RANDOM PRUNING (BASELINE)")
    print(f"  Model:       {opt.weights}")
    print(f"  Prune ratio: {opt.prune_ratio}")
    print(f"  Divisor:     {opt.divisor}")
    print(f"  Seed:        {opt.seed or 'None (non-deterministic)'}")
    print(f"{'=' * 100}\n")

    # Step 1-6: Load and prepare
    model, bn_dict, ignore_bn_list, _chunk_bn_list, layer_ratio_cfg, pruned_yaml = load_and_prepare(
        opt.weights, opt.cfg, opt.model_size, opt.layer_ratio
    )

    # Compute random importance
    print("\nGenerating random importance scores...")
    importance = compute_random_importance(model, bn_dict, ignore_bn_list, seed=opt.seed)

    # Step 7: Create masks
    maskbndict = create_masks(importance, model, ignore_bn_list, layer_ratio_cfg, opt.prune_ratio, opt.divisor)

    # Steps 8-11: Build, copy, save
    finalize_pruning(
        model,
        maskbndict,
        pruned_yaml,
        ignore_bn_list,
        opt.weights,
        opt.save_dir,
        opt.divisor,
        opt.prune_ratio,
        method_name="random",
    )

    return maskbndict, pruned_yaml


if __name__ == "__main__":
    main()
