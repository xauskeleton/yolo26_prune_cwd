"""
DepGraph Pruning (Dependency Graph)
===================================
Based on: Fang et al., "DepGraph: Towards Any Structural Pruning", CVPR 2023
Library:  torch-pruning (https://github.com/VainF/Torch-Pruning)

Khac cac phuong phap con lai trong thu muc nay:
- prune_l1norm:  ||W_i||_1 cua RIENG conv sinh ra kenh do
- prune_bn_gamma: |gamma_i| cua RIENG BN do
- prune_fpgm:    khoang cach den trung vi hinh hoc, cung chi trong mot lop
- prune_depgraph: diem cua CA NHOM lop bi rang buoc phai cat cung nhau

Y tuong DepGraph: cat mot kenh o lop A bat buoc phai cat kenh tuong ung o moi
lop phu thuoc vao A (BN di kem, cac conv tieu thu no, nhanh residual...). Vi
the do quan trong phai tinh tren toan nhom chu khong tren mot lop, neu khong
se cat nham mot kenh yeu o lop nay nhung lai dang quan trong o lop khac.

torch-pruning tu dung do thi phu thuoc bang cach do nguoc grad_fn, roi
GroupMagnitudeImportance gop chuan L2 cua moi thanh vien trong nhom.

Hai diem thuc thi can biet:

1. `load_and_prepare` nap model qua AutoBackend nen tat requires_grad. Phai bat
   lai truoc khi dung do thi, khong thi chi trace duoc 1 node.

2. Voi 14 lop co dau ra bi chunk/split (cv1.bn cua C3k2 va m.0.cv3.bn cua C3k),
   torch-pruning 1.6 khong truyen chi so qua nut split nen ham tinh diem bao
   IndexError. Cach xu ly: loc nhom, chi giu thanh vien co chi so hop le, roi
   VAN goi chinh ham cua torch-pruning — khong tu viet lai cong thuc nen khong
   lech so voi 76 lop con lai.

Usage:
    python pruning/prune_depgraph.py --weights weights/yolo26m_baseline.pt \
        --cfg cfg/yolo26m.yaml --prune-ratio 0.5 --divisor 8
"""

import argparse

import torch
import torch.nn as nn

from prune_common import (
    ROOT, load_and_prepare, create_masks, finalize_pruning, add_common_args
)


def _fits(dep, idxs):
    """Chi so co nam trong so chieu cua thanh vien nay khong."""
    m = dep.target.module
    w = getattr(m, "weight", None)
    if w is None or not idxs:
        return False
    # prune_*_in_channels cat chieu 1, con lai cat chieu 0
    dim = 1 if "in_channels" in getattr(dep.handler, "__name__", "") else 0
    if w.dim() <= dim:
        return False
    return max(idxs) < w.shape[dim]


def compute_depgraph_importance(model, bn_dict, ignore_bn_list, p=2):
    """Diem quan trong theo nhom cua DepGraph, tra ve per-channel cho moi BN.

    Returns:
        Dict[str, Tensor] cung dinh dang voi cac phuong phap khac, de dung chung
        create_masks() -> rang buoc chia het cho divisor va duong finetune giu
        nguyen, nen so sanh trong bang chi khac dung mot bien: tieu chi chon kenh.
    """
    try:
        import torch_pruning as tp
    except ImportError:
        raise SystemExit(
            "Thieu torch-pruning. Cai bang:  pip install torch-pruning")

    net = model.model

    # AutoBackend tat requires_grad -> do thi chi trace duoc 1 node.
    was = {}
    for n, prm in net.named_parameters():
        was[n] = prm.requires_grad
        prm.requires_grad_(True)

    example = torch.randn(1, 3, 640, 640).to(next(net.parameters()).device)
    DG = tp.DependencyGraph().build_dependency(net, example_inputs=example)
    print(f"  Do thi phu thuoc: {len(DG.module2node)} node")

    imp_fn = tp.importance.GroupMagnitudeImportance(p=p)
    mods = dict(net.named_modules())

    importance = {}
    n_full = n_partial = n_fallback = 0
    group_sizes = []

    for bn_name, bn in bn_dict.items():
        if bn_name in ignore_bn_list:
            continue

        conv = mods.get(bn_name[:-2] + "conv")
        if conv is None or not isinstance(conv, nn.Conv2d):
            importance[bn_name] = bn.weight.data.abs().view(-1)
            n_fallback += 1
            continue

        idxs = list(range(conv.out_channels))
        group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=idxs)

        try:
            scores = imp_fn(group)
            n_full += 1
            group_sizes.append(len(group))
        except Exception:
            # Loc bo thanh vien co chi so vuot chieu cua no (nut chunk/split)
            kept = tp.dependency.Group()
            kept._DG = DG
            for dep, di in group:
                if _fits(dep, di):
                    kept.add_dep(dep, di)
            try:
                scores = imp_fn(kept)
                n_partial += 1
                group_sizes.append(len(kept))
            except Exception:
                scores = conv.weight.data.abs().sum(dim=[1, 2, 3])
                n_fallback += 1
                importance[bn_name] = scores
                continue

        if scores.numel() != bn.weight.shape[0]:
            scores = conv.weight.data.abs().sum(dim=[1, 2, 3])
            n_fallback += 1
        importance[bn_name] = scores.detach().float().cpu()

    for n, prm in net.named_parameters():
        prm.requires_grad_(was.get(n, False))

    avg = sum(group_sizes) / len(group_sizes) if group_sizes else 0
    print(f"  DepGraph importance: {len(importance)} lop "
          f"({n_full} nhom day du, {n_partial} nhom loc bot, {n_fallback} lui ve L1)")
    print(f"  Trung binh {avg:.1f} lop moi nhom")
    return importance


def main():
    parser = argparse.ArgumentParser(description='YOLO26 DepGraph Pruning')
    add_common_args(parser)
    parser.add_argument('--norm-p', type=int, default=2,
                        help='bac chuan cho diem nhom (mac dinh 2 = L2)')
    opt = parser.parse_args()

    print(f"\n{'='*100}")
    print(f"DEPGRAPH PRUNING  (Fang et al., CVPR 2023)")
    print(f"  Model:       {opt.weights}")
    print(f"  Prune ratio: {opt.prune_ratio}")
    print(f"  Divisor:     {opt.divisor}")
    print(f"  Norm:        L{opt.norm_p} tren toan nhom")
    print(f"{'='*100}\n")

    model, bn_dict, ignore_bn_list, chunk_bn_list, layer_ratio_cfg, pruned_yaml = \
        load_and_prepare(opt.weights, opt.cfg, opt.model_size, opt.layer_ratio)

    print("\nComputing DepGraph group importance...")
    importance = compute_depgraph_importance(
        model, bn_dict, ignore_bn_list, p=opt.norm_p)

    maskbndict = create_masks(
        importance, model, ignore_bn_list, layer_ratio_cfg,
        opt.prune_ratio, opt.divisor
    )

    save_path = finalize_pruning(
        model, maskbndict, pruned_yaml, ignore_bn_list,
        opt.weights, opt.save_dir, opt.divisor, opt.prune_ratio,
        method_name="depgraph"
    )

    return maskbndict, pruned_yaml


if __name__ == "__main__":
    main()
