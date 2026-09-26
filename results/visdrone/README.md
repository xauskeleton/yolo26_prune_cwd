# VisDrone2019-DET — ket qua

File theo doi **duy nhat** cho VisDrone. Moi ket qua ghi vao day, khong rai ra
cho khac. (VOC nam o `results/yolo_compare.md`.)

Notebook: `notebooks/share_visdrone/` — 4 cai, moi nguoi mot size.

## Bang chinh

| Model | Params (M) | GFLOPs | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| YOLO26-N | | | | |
| **Ours-N** | | | | |
| YOLO26-S | | | | |
| **Ours-S** | | | | |
| YOLO26-M | | | | |
| **Ours-M** | | | | |
| YOLO26-L | | | | |
| **Ours-L** | | | | |

Ours = L1-norm uniform prune 50% (divisor 8) + finetune 100 epoch voi CWD.

## Trang thai

| Size | Baseline | Ours | Nguoi chay |
|---|---|---|---|
| n | chua | chua | |
| s | chua | chua | |
| m | chua | chua | |
| l | chua | chua | |

## Giao thuc

| | |
|---|---|
| Dataset | VisDrone2019-DET — 6471 train / 548 val, 10 lop |
| Split danh gia | `val` (548 anh) |
| imgsz / batch / seed | 640 / 16 / 0 |
| epochs | 100 |
| optimizer | `auto` -> MuSGD (tu chon lr) |
| cos_lr / patience / warmup | False / 100 / 3.0 |
| Augmentation | mac dinh Ultralytics, khong tinh chinh rieng |
| Phan cung | Kaggle Tesla T4 x2, DDP |
| Prune | L1-norm uniform 50%, divisor 8 |
| CWD | tau=9, kd_lambda=0.5, kd_layers=neck, kd_warmup=5 |
| Teacher | baseline cua **chinh size do** |
| Metric | AP 101-point (Ultralytics) |

**Mot config duy nhat cho ca 4 size**, lay tu cau hinh cua m — do la cau hinh
chinh cua bai. Ly do chi tiet: `notebooks/share_visdrone/README.md`.

## Han che phai ghi trong bai

1. **Chua hoi tu han.** VisDrone chi co 6471 anh train nen 100 epoch = 40500
   buoc, so voi 103500 cua VOC (**39%**). Ca baseline lan Ours deu chua hoi tu,
   nhung ca hai nhan cung ngan sach nen **do chenh** van co nghia — do moi la
   thu can bao cao, khong phai con so tuyet doi.
2. **`max_det=300`** cat bot tren cac anh dong hon 300 vat the. Giu mac dinh de
   con doi chieu duoc voi cac bai khac cung chay tren Ultralytics.
3. **imgsz 640.** Cac bai chuyen ve vat the nho dung 1024-1280 va an hon dang ke
   (640 -> 1280 khoang +25% mAP). 640 la muc chuan cua nhanh **nen model**
   (FDM-YOLO, YOLOv8n-ACW, cac bang YOLOv8n/s) nen so sanh duoc voi ho.

## Doi chung gan nhat

[arXiv 2509.12918](https://arxiv.org/abs/2509.12918) — structured pruning theo
he so BN + CWD tren YOLOv8m/VisDrone cho thiet bi bien:

| | Ho | Ta |
|---|---|---|
| Backbone | YOLOv8m | YOLOv26m |
| Importance | BN gamma | **L1-norm** |
| Distillation | CWD | CWD |
| Ket qua | 25.85M -> 6.85M (-73.5%), mAP50 47.9 (-2.7), 26 -> 68 FPS (TRT) | |

Ba thu ho khong co, phai neu ro trong bai:
- Thuc nghiem cho thay **L1-norm > BN gamma** (ho dung gamma).
- Khao sat do nhay **tau = 1..10**.
- Do that tren **Jetson Nano**.

## Vi sao model pruned cua m va l trong "giong het nhau"

Bang `scales` trong `cfg/yolo26m.yaml`:

| size | depth | width | max_channels |
|---|---:|---:|---:|
| n | 0.5 | 0.25 | 1024 |
| s | 0.5 | 0.50 | 1024 |
| **m** | **0.5** | **1.0** | **512** |
| **l** | **1.0** | **1.0** | **512** |

**m va l co width y het nhau (1.0), chi khac depth.** Ma pruning kenh la thao
tac tren *width*, khong dung den depth. Nen sau khi cat 50%, **124/124 lop dung
chung deu ra dung cung so kenh** — khong lech mot lop nao:

| lop | s | m | l |
|---|---|---|---|
| model.0.bn | 32 -> 16 | 64 -> 32 | 64 -> 32 |
| model.4.cv2.bn | 256 -> 128 | 512 -> 256 | 512 -> 256 |
| model.8.cv2.bn | 512 -> 256 | 512 -> 256 | 512 -> 256 |
| model.22.cv2.bn | 512 -> 256 | 512 -> 256 | 512 -> 256 |

Khac biet giua Ours-M va Ours-L **hoan toan la do depth**: l co 178 lop BN can
prune so voi 124 cua m, tuc la moi khoi C3k2 lap 2 lan thay vi 1 (`n=2` so voi
`n=1` trong log build). Sau prune: 390 lop so voi 278.

Do that (VOC, prune 50% div8):

| size | params goc | pruned | GFLOPs goc | pruned | nen |
|---|---:|---:|---:|---:|---:|
| s | 9.96M | 4.04M | 22.6 | 8.4 | 2.46x |
| m | 21.80M | 7.50M | 74.9 | 23.6 | 2.91x |
| l | 26.21M | 9.70M | 93.3 | 31.9 | 2.70x |

`l` chi to hon `m` 20% ngay tu dau (26.2 so voi 21.8M) cung vi ly do nay — no
sau hon chu khong rong hon.

### He qua khi viet bai

- Doan **m -> l** trong bang theo size la so sanh **chi khac do sau**, khong
  phai "model to hon". Dung noi chung chung la "cac kich thuoc khac nhau".
- Doan **n -> s** thi nguoc lai: cung depth 0.5, chi khac width (0.25 -> 0.50).
- Doan **s -> m** khac ca width (0.5 -> 1.0) lan max_channels (1024 -> 512).

Tuc la moi buoc trong ho model thay doi mot thu khac nhau. Neu bang duoc dung de
noi "phuong phap chay duoc o moi quy mo" thi khong sao; nhung dung dien giai no
nhu mot duong scaling deu.

## Thu muc

```
results/visdrone/
├── README.md
├── ckpt/
│   ├── baseline/{n,s,m,l}/   yolo26<size>_vd_baseline.pt
│   └── pruned/{n,s,m,l}/     yolo26<size>_vd_ours50.pt
└── logs/
    ├── baseline/{n,s,m,l}/   yolo26<size>_vd_baseline.csv
    └── pruned/{n,s,m,l}/     yolo26<size>_vd_ours50.csv
```

Tach theo **loai** (baseline / pruned) roi theo **size** (n/s/m/l). Moi thu muc
cuoi dung mot file, va moi nguoi chi cham vao thu muc size cua minh nen khong
de len nhau khi gop ket qua.

Cell ket qua cua notebook **tu gom** dung bon file nay vao
`/kaggle/working/yolo/results/visdrone/...`, cung layout voi day, nen tai ve la
chep thang vao duoc.

File `.pt` bi `.gitignore:159` chan nen khong len repo — chi nam o may. Muon
day len thi dung Kaggle dataset hoac Drive, dung `git add -f` (checkpoint
nang 5-50 MB moi cai).
