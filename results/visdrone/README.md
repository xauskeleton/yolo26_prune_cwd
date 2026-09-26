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

## Thu muc

```
results/visdrone/
├── README.md                          <- file nay
├── ckpt/
│   └── <size>/                        n, s, m, l
│       ├── yolo26<size>_vd_baseline.pt
│       ├── yolo26<size>_vd_ours50.pt
│       ├── results_baseline.csv
│       └── results_ours.csv
└── logs/                              <- log Kaggle, ghi chu linh tinh
```

Cell ket qua cua notebook **tu gom** dung 4 file tren vao
`/kaggle/working/yolo/results/visdrone/ckpt/<size>/`, cung layout voi day, nen
tai ve la chep thang vao duoc.

File `.pt` bi `.gitignore:159` chan nen khong len repo — chi nam o may. Muon
day len thi dung Kaggle dataset hoac Drive, dung `git add -f` (checkpoint
nang 5-50 MB moi cai).
