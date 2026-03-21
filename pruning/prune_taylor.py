"""
Taylor Pruning (First-order Taylor Expansion)
=============================================
Pruning dựa trên first-order Taylor approximation.
Importance score = |BN.gamma * grad(BN.gamma)| tích lũy qua calibration data.

Ý tưởng: xấp xỉ ΔLoss khi remove channel i ≈ |θ_i · ∂L/∂θ_i|
→ channel nào mà khi bỏ gây thay đổi loss lớn nhất = quan trọng nhất

Dùng REAL detection loss (box + cls + dfl) với labeled data từ YOLO data.yaml.
Gradient từ detection loss phản ánh đúng importance cho object detection task.

Reference: "Importance Estimation for Neural Network Pruning"
           (Molchanov et al., CVPR 2019)

Ưu điểm:
- Xét cả magnitude VÀ gradient → chính xác hơn L1 norm/BN gamma
- Data-dependent: biết channel nào thực sự contribute cho task
- State-of-the-art trong structured pruning

Nhược điểm:
- Cần labeled data (YOLO data.yaml với val set)
- Chậm hơn do phải chạy forward+backward

Usage:
    # Cơ bản - dùng val set từ data.yaml
    python prune_taylor.py --weights weights/yolo26m_baseline.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5 --data VOC.yaml

    # Custom calibration settings
    python prune_taylor.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml \\
        --prune-ratio 0.3 --data coco.yaml --num-batches 20 --batch-size 8 --imgsz 640
"""

import argparse

import torch
import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.data import build_yolo_dataset, build_dataloader

from prune_common import (
    ROOT, load_and_prepare, create_masks, finalize_pruning, add_common_args
)


def _build_calibration_dataloader(data_yaml, imgsz, batch_size):
    """
    Build YOLO dataloader từ data.yaml cho calibration.
    Dùng val set, mode="val" (không augmentation).

    Returns:
        dataloader: PyTorch DataLoader với labeled data
    """
    # Parse data.yaml
    data = check_det_dataset(data_yaml)

    # Build config tối thiểu cho dataset
    cfg = get_cfg()
    cfg.imgsz = imgsz
    cfg.batch = batch_size

    # Build dataset từ val set, mode="val" để tắt augmentation
    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=data["val"],
        batch=batch_size,
        data=data,
        mode="val",
        rect=False,
        stride=32
    )

    # Build dataloader
    dataloader = build_dataloader(
        dataset=dataset,
        batch=batch_size,
        workers=4,
        shuffle=False,
        rank=-1
    )

    print(f"  Dataset: {data['val']}")
    print(f"  Images: {len(dataset)}")
    print(f"  Batches available: {len(dataloader)}")
    return dataloader


def compute_taylor_importance(model, bn_dict, ignore_bn_list,
                              dataloader, num_batches, device='cuda'):
    """
    Tính Taylor importance = |gamma * grad_gamma| tích lũy qua real detection loss.

    Steps:
    1. Enable grad trên BN gamma
    2. Forward pass → raw predictions (train mode)
    3. Compute real detection loss (box + cls + dfl) với ground truth labels
    4. loss.backward() → gradients phản ánh đúng importance cho detection
    5. Tích lũy |gamma * grad_gamma| cho mỗi BN

    Args:
        model:          AutoBackend model
        bn_dict:        Dict[str, BN] - tất cả BN layers
        ignore_bn_list: List[str] - BN to skip
        dataloader:     YOLO DataLoader với labeled data
        num_batches:    int - số batches calibration
        device:         str - cuda or cpu

    Returns:
        Dict[str, Tensor] - Taylor importance per channel cho mỗi prunable BN
    """
    # Move model lên device và enable gradients trên BN gamma
    model.model.to(device)
    model.model.train()
    for name, module in model.model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.weight.requires_grad_(True)

    # Ensure model.args is namespace (criterion needs .box, .cls, .dfl attribute access)
    if isinstance(model.model.args, dict):
        from ultralytics.cfg import get_cfg
        model.model.args = get_cfg(model.model.args)

    # Initialize detection loss criterion (box + cls + dfl)
    if getattr(model.model, 'criterion', None) is None:
        model.model.criterion = model.model.init_criterion()

    taylor_scores = {}
    actual_batches = min(num_batches, len(dataloader))

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        # Preprocess: move to device, normalize
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        batch["img"] = batch["img"].float() / 255

        # Forward → raw predictions (train mode returns dict)
        preds = model.model(batch["img"])

        # Real detection loss (box + cls + dfl) với ground truth labels
        loss, loss_items = model.model.criterion(preds, batch)
        loss_scalar = loss.sum()

        # Backward
        loss_scalar.backward()

        # Tích lũy |gamma * grad_gamma|
        for name, module in model.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name not in ignore_bn_list:
                if module.weight.grad is not None:
                    score = (module.weight.data * module.weight.grad).abs()
                    if name in taylor_scores:
                        taylor_scores[name] += score
                    else:
                        taylor_scores[name] = score.clone()

        model.model.zero_grad()

        box_loss, cls_loss, dfl_loss = loss_items[0].item(), loss_items[1].item(), loss_items[2].item()
        print(f"  Batch {batch_idx + 1}/{actual_batches} | "
              f"box: {box_loss:.4f} cls: {cls_loss:.4f} dfl: {dfl_loss:.4f}")

    # Average và move về CPU
    for name in taylor_scores:
        taylor_scores[name] = (taylor_scores[name] / actual_batches).cpu()

    # Move model về CPU để đồng bộ với pipeline copy weights
    model.model.cpu()
    model.model.eval()
    print(f"  Taylor importance computed for {len(taylor_scores)} layers "
          f"using {actual_batches} batches with real detection loss")
    return taylor_scores


def main():
    parser = argparse.ArgumentParser(description='YOLO26 Taylor Pruning')
    add_common_args(parser)

    # Taylor-specific args
    parser.add_argument('--data', type=str, required=True,
                        help='YOLO data.yaml (dùng val set cho calibration)')
    parser.add_argument('--num-batches', type=int, default=10,
                        help='Số batches cho calibration (default: 10)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size cho calibration (default: 4)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size cho calibration (default: 640)')
    opt = parser.parse_args()

    print(f"\n{'='*100}")
    print(f"TAYLOR PRUNING (real detection loss)")
    print(f"  Model:       {opt.weights}")
    print(f"  Prune ratio: {opt.prune_ratio}")
    print(f"  Divisor:     {opt.divisor}")
    print(f"  Data:        {opt.data}")
    print(f"  Batches:     {opt.num_batches} x {opt.batch_size}")
    print(f"  Image size:  {opt.imgsz}")
    print(f"{'='*100}\n")

    # Step 1-6: Load and prepare
    model, bn_dict, ignore_bn_list, chunk_bn_list, layer_ratio_cfg, pruned_yaml = \
        load_and_prepare(opt.weights, opt.cfg, opt.model_size, opt.layer_ratio)

    # Build YOLO dataloader với labeled data
    print("\nBuilding calibration dataloader...")
    dataloader = _build_calibration_dataloader(opt.data, opt.imgsz, opt.batch_size)

    # Compute Taylor importance với real detection loss
    print("\nComputing Taylor importance scores (real detection loss)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    importance = compute_taylor_importance(
        model, bn_dict, ignore_bn_list, dataloader,
        num_batches=opt.num_batches, device=device
    )

    # Step 7: Create masks
    maskbndict = create_masks(
        importance, model, ignore_bn_list, layer_ratio_cfg, opt.prune_ratio, opt.divisor
    )

    # Steps 8-11: Build, copy, save
    save_path = finalize_pruning(
        model, maskbndict, pruned_yaml, ignore_bn_list,
        opt.weights, opt.save_dir, opt.divisor, opt.prune_ratio,
        method_name="taylor"
    )

    return maskbndict, pruned_yaml


if __name__ == "__main__":
    main()
