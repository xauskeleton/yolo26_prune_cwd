# -*- coding: utf-8 -*-
"""Duong cong AP@0.5 theo GFLOPs cua quet ti le pruning.

Cung bang mau / rcParams voi dcmm.py va yolo_compare.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ratio, GFLOPs, AP@0.5
DATA = [
    (0, 74.9, 89.04),
    (30, 42.5, 88.09),
    (40, 33.1, 88.05),
    (50, 23.6, 87.96),
    (60, 18.3, 86.78),
    (70, 12.4, 84.98),
]
BLUE = "#378ADD"

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.alpha"] = 0.4

x = [d[1] for d in DATA]
y = [d[2] for d in DATA]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(x, y, color=BLUE, linewidth=3.0, marker="o", markersize=8,
        markerfacecolor="white", markeredgewidth=2.5, zorder=5)

# Nhan ti le ngay canh diem: truc x la GFLOPs nen khong doc ra ratio neu thieu.
# 0% va 70% nam o hai dau nen day nhan vao trong de khong tran ra le.
# 60% nam giua doan doc dung nhat: de nhan o tren thi no cham vao duong, nen
# day sang trai (phia duong di len, con nhieu cho trong).
for ratio, gf, ap in DATA:
    if ratio == 0:
        off, ha = (-6, 10), "right"
    elif ratio == 60:
        off, ha = (-12, -2), "right"
    elif ratio == 70:
        off, ha = (10, -4), "left"
    else:
        off, ha = (0, 12), "center"
    ax.annotate(f"{ratio}%", (gf, ap), textcoords="offset points",
                xytext=off, ha=ha, fontsize=11, color="#444444")

ax.set_xlabel("GFLOPs")
ax.set_ylabel("AP@0.5")
ax.invert_xaxis()          # trai -> phai la nen ngay cang manh
ax.set_ylim(84.2, 89.8)

sns.despine(left=True, bottom=True)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"results/flops_ap50.{ext}", dpi=250, bbox_inches="tight")
print("results/flops_ap50.png / .pdf")
