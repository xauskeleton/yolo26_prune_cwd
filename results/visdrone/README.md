# VisDrone2019-DET — ket qua

File theo doi **duy nhat** cho VisDrone. Moi ket qua ghi vao day, khong rai ra
cho khac. (VOC nam o `results/yolo_compare.md`.)

Notebook: `notebooks/share_visdrone/` — 4 cai, moi nguoi mot size.

## Bang chinh

| Model | Params (M) | GFLOPs | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| YOLO26-N | 2.38 | | 34.80 | 19.70 |
| **Ours-N** | **1.07** | **2.5** | **28.96** | **16.30** |
| YOLO26-S | 9.47 | | 41.36 | 24.81 |
| **Ours-S** | **4.01** | **8.1** | **35.84** | **20.81** |
| YOLO26-M | 20.36 | 67.9 | 46.90 | 28.70 |
| **Ours-M** | **7.44** | **23.4** | **42.70** | **25.30** |
| YOLO26-L | | | 48.30 | 29.50 |
| **Ours-L** | **9.65** | **31.7** | dang chay 43/100 | |

Params cua Ours la so **sau khi prune, truoc finetune** (kien truc khong doi khi
finetune). GFLOPs cua baseline chua trich ra tu log.

Ours = L1-norm uniform prune 50% (divisor 8) + finetune 100 epoch voi CWD.

## Trang thai

| Size | Baseline | Ours | Thoi gian |
|---|---|---|---|
| n | **xong** | **xong** | 4.14h + 4.51h |
| s | **xong** | **xong** | 4.42h + 5.03h |
| m | **xong** | **xong** | 6.15h + 6.0h |
| l | **xong** | 43/100 epoch | 8.00h + ... |

Nhanh hon uoc tinh ban dau kha nhieu (da du doan n ~2-3h, thuc te 8.7h ca hai;
m du doan 15h, baseline moi het 6.15h).

## Hai dieu rut ra tu n va s (da xong)

### 1. Gia phai tra tren VisDrone cao gap 5 lan so voi VOC

| | AP50 baseline | AP50 Ours | Chenh |
|---|---:|---:|---:|
| VOC (m, prune 50%) | 89.04 | 87.96 | **-1.08** |
| VisDrone (n) | 34.80 | 28.96 | **-5.84** |
| VisDrone (s) | 41.36 | 35.84 | **-5.52** |
| VisDrone (m) | 46.90 | 42.70 | **-4.20** |

Ket luan "cat 50% gan nhu mien phi" rut ra tu VOC **khong chuyen sang VisDrone**.
Hop ly: VisDrone toan vat the nho va dong, ma chinh bang per-class tren VOC da
cho thay nhom vat nho (pottedplant, bottle, chair) chiu thiet nang nhat khi cat
sau. VisDrone la ca dataset toan nhom do.

> Phai ghi thang dieu nay trong bai, dung im lang. No khong pha ket qua — 1.07M
> tham so ma giu duoc 83% AP50 cua ban goc van la mot ti le doi tot — nhung
> dien giai phai khac voi VOC.

Muc giam **giam dan theo kich thuoc model**: n -5.84, s -5.52, m -4.20. Model
cang lon cang chiu prune tot, hop ly vi kenh du thua nhieu hon.

### 2. Ti le nen kenh chi ~1.5x du dat prune ratio 0.5

| size | tong kenh | sau prune | ti le |
|---|---:|---:|---:|
| n | 9,496 | 6,440 | 1.47x |
| s | 18,992 | 12,608 | 1.51x |
| m | 28,096 | 17,488 | 1.61x |
| l | 35,264 | 22,976 | 1.53x |

Vi **34/124 lop BN bi SKIP (residual)** — chung giu nguyen toan bo kenh. Chi
90 lop con lai bi cat 50%. Ti le tham so thi cao hon (2.2x den 2.9x) vi cac lop
bi cat nam o cho nhieu tham so.

Khi viet bai nho phan biet: "prune ratio 50%" la ti le tren **cac lop prune
duoc**, khong phai tren toan model.

## Rui ro can biet: doi chung arXiv 2509.12918

Ho bao **-73.5% params, mAP50 giam 2.7** tren YOLOv8m/VisDrone.
Baseline YOLO26-M cua ta o day la 46.90 — cung hang voi ho (~50.6).

m da xong: **46.90 -> 42.70, giam 4.20**. Do hon n (-5.84) va s (-5.52) dung
nhu du doan model lon chiu prune tot hon, nhung van **gap 1.6 lan** muc giam 2.7
cua ho o cung ti le nen (-65.8% params so voi -73.5%).

Ba huong giai thich, can chon truoc khi viet:
- Ho co **sparsity training** truoc khi prune (day gamma ve 0), ta cat thang.
- Ho train bao nhieu epoch chua ro; ta chi 100 epoch = 39% so buoc cua VOC.
- Ho co the do tren split khac (test-dev thay vi val).

Neu khong khep duoc thi ha ti le prune cho VisDrone xuong 30-40% va bao cao o
muc nen thap hon — van trung thuc va van manh.

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
