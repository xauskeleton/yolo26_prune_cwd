# -*- coding: utf-8 -*-
"""Kiem chung: cong thuc diem nhom tu viet co khop voi torch-pruning khong.

Neu khop tren 76 lop ma torch-pruning xu ly dung, thi dung no cho ca 90 lop
la chinh dang — ke ca 14 lop ma tracer cua torch-pruning tra ve nhom sai.
"""
import contextlib
import io
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "pruning")

import torch
import torch.nn as nn
import torch_pruning as tp
from prune_common import load_and_prepare
from dms.dms_utils import build_conv_bn_mapping

with contextlib.redirect_stdout(io.StringIO()):
    model, bn_dict, ign, _c, _l, _y = load_and_prepare(
        "weights/yolo26m_baseline.pt", "cfg/yolo26m.yaml", "m", None)
net = model.model
conv_bn_map, bn_channels = build_conv_bn_mapping(net, ign)

# BN -> cac conv tieu thu kenh cua no
consumers = {}
for cname, info in conv_bn_map.items():
    ib = info.get("in_bn")
    # in_bn co the la mot ten, hoac mot danh sach (dau vao Concat)
    for b in (ib if isinstance(ib, (list, tuple)) else [ib]):
        if isinstance(b, str) and b:
            consumers.setdefault(b, []).append(cname)

mods = dict(net.named_modules())
P = 2


def norm_mean(x):
    """Chuan hoa nhu normalizer='mean' cua torch-pruning."""
    m = x.mean()
    return x / m if m > 0 else x


def my_score(bn_name):
    """Diem nhom: conv sinh ra kenh + BN di kem + cac conv tieu thu no."""
    parts = []
    conv = mods.get(bn_name[:-2] + "conv")
    if conv is not None:
        w = conv.weight.data.flatten(1)
        parts.append(norm_mean(w.abs().pow(P).sum(1)))
    bn = mods.get(bn_name)
    if bn is not None and bn.weight is not None:
        parts.append(norm_mean(bn.weight.data.abs().pow(P)))
    for cn in consumers.get(bn_name, []):
        c = mods.get(cn)
        if c is None or c.groups != 1:
            continue
        w = c.weight.data.transpose(0, 1).flatten(1)   # theo kenh VAO
        if w.shape[0] != parts[0].numel():
            continue
        parts.append(norm_mean(w.abs().pow(P).sum(1)))
    return torch.stack(parts).mean(0), len(parts)


for p in net.parameters():
    p.requires_grad_(True)
DG = tp.DependencyGraph().build_dependency(net, example_inputs=torch.randn(1, 3, 640, 640))
imp_fn = tp.importance.GroupMagnitudeImportance(p=P)

rows = []
for bn_name in bn_dict:
    if bn_name in ign:
        continue
    conv = mods.get(bn_name[:-2] + "conv")
    if conv is None:
        continue
    try:
        ref = imp_fn(DG.get_pruning_group(
            conv, tp.prune_conv_out_channels, idxs=list(range(conv.out_channels))))
    except Exception:
        rows.append((bn_name, None, None, None))
        continue
    mine, n_parts = my_score(bn_name)
    if ref.numel() != mine.numel():
        rows.append((bn_name, None, None, n_parts))
        continue
    ra = ref.float().argsort().argsort().float()
    rb = mine.float().argsort().argsort().float()
    rho = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    # ti le chon trung khi cat 50%
    k = conv.out_channels // 2
    sa = set(ref.argsort(descending=True)[:k].tolist())
    sb = set(mine.argsort(descending=True)[:k].tolist())
    rows.append((bn_name, rho, len(sa & sb) / k, n_parts))

good = [r for r in rows if r[1] is not None]
bad = [r for r in rows if r[1] is None]
print("Lop torch-pruning xu ly duoc: {}  |  khong duoc: {}".format(len(good), len(bad)))
print()
print("{:<32} {:>8} {:>10} {:>7}".format("layer", "rho", "trung@50%", "n_lop"))
print("-" * 62)
for n, rho, ov, np_ in good[:12]:
    print("{:<32} {:>8.4f} {:>9.1%} {:>7}".format(n, rho, ov, np_))
if good:
    rhos = [r[1] for r in good]
    ovs = [r[2] for r in good]
    print("-" * 62)
    print("{:<32} {:>8.4f} {:>9.1%}".format(
        "TRUNG BINH ({} lop)".format(len(good)),
        sum(rhos) / len(rhos), sum(ovs) / len(ovs)))
    print("rho thap nhat:", round(min(rhos), 4))
print()
print("14 lop torch-pruning bo cuoc, cong thuc tu viet van tinh duoc:")
for n, _, _, np_ in bad[:14]:
    print("   {:<32} {} lop trong nhom".format(n, np_))
