"""
DepGraph Pruning (Dependency Graph)
===================================
Based on: Fang et al., "DepGraph: Towards Any Structural Pruning", CVPR 2023
Library:  torch-pruning (https://github.com/VainF/Torch-Pruning) — cung la repo
          chinh thuc cua bai, khong co repo rieng.

Khac cac phuong phap con lai trong thu muc nay:
- prune_l1norm:   ||W_i||_1 cua RIENG conv sinh ra kenh do
- prune_bn_gamma: |gamma_i| cua RIENG BN do
- prune_fpgm:     khoang cach den trung vi hinh hoc, cung chi trong mot lop
- prune_depgraph: diem cua CA NHOM lop bi rang buoc phai cat cung nhau

Y tuong DepGraph: cat mot kenh o lop A bat buoc phai cat kenh tuong ung o moi
lop phu thuoc vao A (BN di kem, cac conv tieu thu no, nhanh residual...). Vi
the do quan trong phai tinh tren toan nhom chu khong tren mot lop.

HAI DIEU PHAI LAM, neu khong DepGraph chi giai duoc 76/90 lop:

1. `load_and_prepare` nap model qua AutoBackend nen tat requires_grad, ma
   torch-pruning do nguoc grad_fn de dung do thi -> chi trace duoc 1/506 node.

2. C3k2 lam `y = list(cv1(x).chunk(2, 1)); y.extend(m(y[-1])); cv2(cat(y, 1))`
   — mot manh cua chunk vua chay thang vao concat vua di qua bottleneck.
   torch-pruning khong lan duoc quan he do: suy sai kich thuoc manh va gan nham
   offset, nen 14 lop bao IndexError.

   Cach xu ly lay tu chinh vi du cua tac gia
   (examples/yolov8/yolov8_pruning.py, ham replace_c2f_with_c2f_v2):
   KHONG sua tracer, ma bo chunk di — thay conv bi chunk doi bang HAI conv
   rieng, moi cai sinh mot nua so kenh. Het chunk thi do thi trace sach.

   Do that tren yolo26m, so lop prunable tinh duoc diem nhom:
       khong bo chunk  ->  76/90
       co  bo chunk    ->  90/90

   Viec thay the chi ap len mot BAN SAO dung de dung do thi. Model that giu
   nguyen, nen maskbndict va duong finetune khong doi.

Usage:
    python pruning/prune_depgraph.py --weights weights/yolo26m_baseline.pt \
        --cfg cfg/yolo26m.yaml --prune-ratio 0.5 --divisor 8
"""

import argparse
import copy
import types

import torch
import torch.nn as nn

from prune_common import (
    ROOT, load_and_prepare, create_masks, finalize_pruning, add_common_args
)


def _forward_nochunk(self, x):
    """Forward cua C2f/C3k2 sau khi tach cv1 thanh cv0 + cv1."""
    y = [self.cv0(x), self.cv1(x)]
    y.extend(m(y[-1]) for m in self.m)
    return self.cv2(torch.cat(y, 1))


def drop_chunk(root):
    """Thay conv bi chunk doi bang hai conv rieng, giu nguyen trong so.

    Sau buoc nay BN goc `model.X.cv1.bn` (2c kenh) tro thanh hai BN:
    `model.X.cv0.bn` (c kenh dau) va `model.X.cv1.bn` (c kenh sau) — dung thu
    tu ma chunk(2, 1) cat.
    """
    from ultralytics.nn.modules import Conv
    from ultralytics.nn.modules.block import C2f

    n = 0
    for mod in root.modules():
        if not isinstance(mod, C2f) or hasattr(mod, "cv0"):
            continue
        old = mod.cv1
        cin = old.conv.in_channels
        half = old.conv.out_channels // 2
        dev = old.conv.weight.device
        cv0, cv1 = Conv(cin, half, 1, 1).to(dev), Conv(cin, half, 1, 1).to(dev)
        with torch.no_grad():
            cv0.conv.weight.copy_(old.conv.weight[:half])
            cv1.conv.weight.copy_(old.conv.weight[half:])
            for k in ("weight", "bias", "running_mean", "running_var"):
                src = getattr(old.bn, k)
                getattr(cv0.bn, k).copy_(src[:half])
                getattr(cv1.bn, k).copy_(src[half:])
        mod.cv0, mod.cv1 = cv0, cv1
        mod.forward = types.MethodType(_forward_nochunk, mod)
        n += 1
    return n


def compute_depgraph_importance(model, bn_dict, ignore_bn_list, p=2):
    """Diem quan trong theo nhom cua DepGraph, tra ve per-channel cho moi BN.

    Cung dinh dang voi cac phuong phap khac, de dung chung create_masks() ->
    rang buoc chia het cho divisor va duong finetune giu nguyen, nen so sanh
    trong bang chi khac dung mot bien: tieu chi chon kenh.
    """
    try:
        import torch_pruning as tp
    except ImportError:
        raise SystemExit(
            "Thieu torch-pruning. Cai bang:  pip install torch-pruning")
    print(f"  torch-pruning {tp.__version__}")

    # Ban sao chi de dung do thi. Model that khong bi dong toi.
    net = copy.deepcopy(model.model)
    for prm in net.parameters():
        prm.requires_grad_(True)   # AutoBackend tat di -> do thi chi con 1 node
    n_rep = drop_chunk(net)
    print(f"  Bo chunk o {n_rep} khoi C3k2/C2f")

    example = torch.randn(1, 3, 640, 640).to(next(net.parameters()).device)
    DG = tp.DependencyGraph().build_dependency(net, example_inputs=example)
    print(f"  Do thi phu thuoc: {len(DG.module2node)} node")

    imp_fn = tp.importance.GroupMagnitudeImportance(p=p)
    mods = dict(net.named_modules())
    orig_mods = dict(model.model.named_modules())

    def score_of(bn_name):
        """Diem nhom cua mot BN trong ban sao, hoac None neu khong tinh duoc."""
        conv = mods.get(bn_name[:-2] + "conv")
        if conv is None or not isinstance(conv, nn.Conv2d):
            return None
        try:
            g = DG.get_pruning_group(conv, tp.prune_conv_out_channels,
                                     idxs=list(range(conv.out_channels)))
            s = imp_fn(g)
            return s if s.numel() == conv.out_channels else None
        except Exception:
            return None

    importance = {}
    n_ok = n_fallback = 0

    for bn_name, bn in bn_dict.items():
        if bn_name in ignore_bn_list:
            continue

        # BN bi tach doi -> ghep lai dung thu tu chunk: cv0 truoc, cv1 sau
        cv0_bn = bn_name[:-7] + ".cv0.bn" if bn_name.endswith(".cv1.bn") else None
        if cv0_bn and cv0_bn in mods:
            a, b = score_of(cv0_bn), score_of(bn_name)
            scores = torch.cat([a, b]) if (a is not None and b is not None) else None
        else:
            scores = score_of(bn_name)

        if scores is None or scores.numel() != bn.weight.shape[0]:
            # Lui ve L1 cua rieng conv do, va bao ro — khong vet tam am tham.
            conv = orig_mods.get(bn_name[:-2] + "conv")
            scores = (conv.weight.data.abs().sum(dim=[1, 2, 3]) if conv is not None
                      else bn.weight.data.abs().view(-1))
            n_fallback += 1
        else:
            n_ok += 1

        importance[bn_name] = scores.detach().float().cpu()

    print(f"  DepGraph importance: {n_ok}/{n_ok + n_fallback} lop dung diem nhom"
          + (f", {n_fallback} lui ve L1" if n_fallback else ""))
    del net
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
