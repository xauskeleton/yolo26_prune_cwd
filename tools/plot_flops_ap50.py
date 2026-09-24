# -*- coding: utf-8 -*-
"""Bieu do Pareto AP@0.5 - GFLOPs: duong quet ti le pruning tren nen cac doi chung.

Huong tot la TREN-TRAI (AP cao, FLOPs thap). Cung bang mau / rcParams voi dcmm.py.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BLUE = "#378ADD"
GREY = "#98A2AE"

# Duong cua ta. Diem dau (74.9, 89.04) chinh la yolo26m baseline chua prune.
OURS = [
    (0, 74.9, 89.04),
    (30, 42.5, 88.09),
    (40, 33.1, 88.05),
    (50, 23.6, 87.96),
    (60, 18.3, 86.78),
    (70, 12.4, 84.98),
]

# Doi chung, 100 epoch cung giao thuc. (ten, GFLOPs, AP@0.5, offset nhan, ha)
RIVALS = [
    ("RT-DETR-L", 110.0, 88.09, (0, 11), "center"),
    ("YOLOv9-C", 103.8, 88.35, (0, 11), "center"),
    ("YOLOv8-M", 79.1, 88.26, (6, 8), "left"),
    ("YOLOv9-M", 77.6, 88.19, (-2, -17), "center"),
    ("YOLO12-M", 71.1, 88.27, (0, 11), "center"),
    ("YOLO11-M", 68.4, 87.41, (7, -4), "left"),
    ("YOLOv5-Mu", 64.4, 87.24, (-7, -4), "right"),
    ("YOLOv10-M", 64.2, 88.28, (-8, -5), "right"),
]

# Nhan ti le tren duong cua ta. 0% trung voi YOLO26-M nen ghi ten model thay vi "0%".
OURS_LABEL = {
    0: ("YOLO26-M", (6, 6), "left"),
    30: ("30%", (0, 12), "center"),
    40: ("40%", (0, 12), "center"),
    50: ("50%", (-4, 12), "center"),
    60: ("60%", (10, 2), "left"),
    70: ("70%", (10, 0), "left"),
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.alpha"] = 0.4

fig, ax = plt.subplots(figsize=(10, 6))

# Doi chung ve truoc de nam duoi duong cua ta.
ax.scatter([r[1] for r in RIVALS], [r[2] for r in RIVALS],
           s=70, color=GREY, zorder=3)
for name, gf, ap, off, ha in RIVALS:
    ax.annotate(name, (gf, ap), textcoords="offset points", xytext=off,
                ha=ha, fontsize=10, color="#6B7480")

x = [d[1] for d in OURS]
y = [d[2] for d in OURS]
ax.plot(x, y, color=BLUE, linewidth=3.0, zorder=4)
# Marker rong, rieng 50% to dac: do la cau hinh de xuat.
for ratio, gf, ap in OURS:
    face = BLUE if ratio == 50 else "white"
    ax.plot([gf], [ap], marker="o", markersize=9, color=BLUE,
            markerfacecolor=face, markeredgewidth=2.5, zorder=5)
    text, off, ha = OURS_LABEL[ratio]
    ax.annotate(text, (gf, ap), textcoords="offset points", xytext=off,
                ha=ha, fontsize=11,
                color="#1F1F1F" if ratio == 50 else "#444444",
                fontweight="bold" if ratio == 50 else "normal")

ax.set_xlabel("GFLOPs")
ax.set_ylabel("AP@0.5")
ax.set_xlim(5, 118)
ax.set_ylim(84.6, 89.5)

sns.despine(left=True, bottom=True)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"results/flops_ap50.{ext}", dpi=250, bbox_inches="tight")
print("results/flops_ap50.png / .pdf")
