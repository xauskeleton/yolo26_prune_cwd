# VisDrone2019-DET — ket qua

File theo doi **duy nhat** cho VisDrone. Moi ket qua ghi vao day, khong rai ra
cho khac. (VOC nam o `results/yolo_compare.md`.)

Notebook: `notebooks/share_visdrone/` — 4 cai, moi nguoi mot size.

## Bang chinh

VisDrone2019-DET **val** (548 anh, 38,759 vat the), imgsz 640, 100 epoch.

| Model      | Params (M) |   GFLOPs |      AP50 |   AP50-95 |       APs |
| ---------- | ---------: | -------: | --------: | --------: | --------: |
| YOLO26-N   |       2.38 |      5.2 |     34.80 |     19.70 |     10.16 |
| **Ours-N** |   **1.07** |  **2.5** | **28.96** | **16.30** |  **7.57** |
| YOLO26-S   |       9.47 |     20.5 |     41.36 |     24.81 |     13.89 |
| **Ours-S** |   **4.01** |  **8.1** | **35.84** | **20.81** | **11.15** |
| YOLO26-M   |      20.36 |     67.9 |     46.90 |     28.70 |     18.11 |
| **Ours-M** |   **7.43** | **23.1** | **42.70** | **25.30** | **15.43** |
| YOLO26-L   |      24.75 |     86.1 |     48.27 |     29.49 |     18.67 |
| **Ours-L** |   **9.63** | **31.2** | **44.70** | **26.92** | **16.29** |

Ours = L1-norm uniform prune 50% (divisor 8) + finetune 100 epoch voi CWD tau=9.
Params va GFLOPs do tren model da fuse.

> `AP50` va `AP50-95` lay tu `model.val()` cua Ultralytics; `APs` tu pycocotools
> (`tools/val_apsmall.py`) vi Ultralytics khong in chi so theo kich thuoc. Hai
> cong cu lech deu 2.24-2.37 diem nen **`APs` khong cung thang do voi hai cot
> kia** — dung de so giua cac dong trong bang, dung tru cheo giua cac cot.

| Size | Giam params | Giam GFLOPs |  Mat AP50 |
| ---- | ----------: | ----------: | --------: |
| n    |      -55.0% |      -51.9% | **-5.84** |
| s    |      -57.7% |      -60.5% | **-5.52** |
| m    |      -63.5% |      -66.0% | **-4.20** |
| l    |      -61.1% |      -63.8% | **-3.57** |

<details><summary>Ban LaTeX</summary>

```latex
\begin{tabular}{lrrrrr}
\toprule
Model & Params (M) & GFLOPs & AP$_{50}$ & AP$_{50:95}$ & AP$_{S}$ \\
\midrule
YOLO26-N & 2.38 & 5.2 & 34.80 & 19.7 & 10.16 \\
\textbf{Ours-N} & \textbf{1.07} & \textbf{2.5} & \textbf{28.96} & \textbf{16.3} & \textbf{7.57} \\
\midrule
YOLO26-S & 9.47 & 20.5 & 41.36 & 24.81 & 13.89 \\
\textbf{Ours-S} & \textbf{4.01} & \textbf{8.1} & \textbf{35.84} & \textbf{20.81} & \textbf{11.15} \\
\midrule
YOLO26-M & 20.36 & 67.9 & 46.90 & 28.70 & 18.11 \\
\textbf{Ours-M} & \textbf{7.43} & \textbf{23.1} & \textbf{42.7} & \textbf{25.3} & \textbf{15.43} \\
\midrule
YOLO26-L & 24.75 & 86.1 & 48.27 & 29.49 & 18.67 \\
\textbf{Ours-L} & \textbf{9.63} & \textbf{31.2} & \textbf{44.7} & \textbf{26.92} & \textbf{16.29} \\
\bottomrule
\end{tabular}
```

</details>

## Ket qua manh nhat: prune thang model to hon la chon model nho

Cau hoi hien nhien cua reviewer: _"Prune yolo26m lam gi, dung thang yolo26s
cho roi?"_ So lieu tra loi duoc:

|                               | Params (M) | GFLOPs |      AP50 |
| ----------------------------- | ---------: | -----: | --------: |
| YOLO26-S (nguyen ban)         |       9.47 |   20.5 |     41.36 |
| **Ours-M** (prune tu yolo26m) |   **7.43** |   23.1 | **42.70** |

**It hon 21.5% than so, AP50 cao hon 1.34 diem.** GFLOPs nhinh hon 12.7%.
Ours-L cung vay: 9.63M dat 44.70, hon YOLO26-S **3.34 diem** o cung muc than so.

Nghia la mo hinh nen tu ban lon **khong phai mot cach xap xi re tien cua ban
nho** — no o mot diem tot hon han tren duong danh doi. Day la lap luan trung
tam nen dung cho bang VisDrone.

## Trang thai

| Size | Baseline | Ours     | Baseline (100ep) | Ours (100ep) |  phut/epoch |
| ---- | -------- | -------- | ---------------: | -----------: | ----------: |
| n    | **xong** | **xong** |            4.14h |        4.51h | 2.48 / 2.71 |
| s    | **xong** | **xong** |            4.42h |        5.03h | 2.65 / 3.02 |
| m    | **xong** | **xong** |            6.15h |       7.23h* | 3.69 / 4.34 |
| l    | **xong** | **xong** |            8.00h |       9.02h* | 4.80 / 5.41 |

*Hai o danh dau: run bi cat lam hai phien nen khong co dong "100 epochs
completed". Suy tu phut/epoch do duoc o phien resume (m: 20 epoch trong 1.45h;
l: 58 epoch trong 5.23h). Truoc day ghi 6.0h va 8.8h — do la uoc luong, khong
phai do.

**CWD dat them khoang 15%**: Ours luon cham hon baseline cung size du model nho
hon 60%, vi teacher forward chay FP32 ngoai autocast moi batch.

**Xong ca 4 size.**

Nhanh hon uoc tinh ban dau kha nhieu (da du doan n ~2-3h, thuc te 8.7h ca hai;
m du doan 15h, baseline moi het 6.15h).

## Hai dieu rut ra tu n va s (da xong)

### 1. Gia phai tra tren VisDrone cao gap 5 lan so voi VOC

|                    | AP50 baseline | AP50 Ours |     Chenh |
| ------------------ | ------------: | --------: | --------: |
| VOC (m, prune 50%) |         89.04 |     87.96 | **-1.08** |
| VisDrone (n)       |         34.80 |     28.96 | **-5.84** |
| VisDrone (s)       |         41.36 |     35.84 | **-5.52** |
| VisDrone (m)       |         46.90 |     42.70 | **-4.20** |
| VisDrone (l)       |         48.27 |     44.70 | **-3.57** |

Ket luan "cat 50% gan nhu mien phi" rut ra tu VOC **khong chuyen sang VisDrone**.
Hop ly: VisDrone toan vat the nho va dong, ma chinh bang per-class tren VOC da
cho thay nhom vat nho (pottedplant, bottle, chair) chiu thiet nang nhat khi cat
sau. VisDrone la ca dataset toan nhom do.

> Phai ghi thang dieu nay trong bai, dung im lang. No khong pha ket qua — 1.07M
> than so ma giu duoc 83% AP50 cua ban goc van la mot ti le doi tot — nhung
> dien giai phai khac voi VOC.

Muc giam **giam deu theo kich thuoc model**: n -5.84, s -5.52, m -4.20,
l -3.57. Bon diem, don dieu, khong co ngoai le — model cang lon cang chiu prune
tot, hop ly vi kenh du thua cang nhieu.

Tinh tuong doi tren nen AP50 thi VisDrone mat **9.0%** (4.20 tren 46.90) con VOC
chi mat **1.2%** (1.08 tren 89.04) — gap 7 lan.

### 2. Ti le nen kenh chi ~1.5x du dat prune ratio 0.5

| size | tong kenh | sau prune | ti le |
| ---- | --------: | --------: | ----: |
| n    |     9,496 |     6,440 | 1.47x |
| s    |    18,992 |    12,608 | 1.51x |
| m    |    28,096 |    17,488 | 1.61x |
| l    |    35,264 |    22,976 | 1.53x |

Vi **34/124 lop BN bi SKIP (residual)** — chung giu nguyen toan bo kenh. Chi
90 lop con lai bi cat 50%. Ti le than so thi cao hon (2.2x den 2.9x) vi cac lop
bi cat nam o cho nhieu than so.

Khi viet bai nho phan biet: "prune ratio 50%" la ti le tren **cac lop prune
duoc**, khong phai tren toan model.

## Rui ro can biet: doi chung arXiv 2509.12918

Ho bao **-73.5% params, mAP50 giam 2.7** tren YOLOv8m/VisDrone.
Baseline YOLO26-M cua ta o day la 46.90 — cung hang voi ho (~50.6).

Diem gan nhat de so la **l**: giam **3.57** o muc nen -61.1% params, so voi ho
giam 2.7 o muc -73.5%. Ta nen it hon ma mat nhieu hon, tuc la **van thua ho**
neu so thang.

Ba huong giai thich, can chon truoc khi viet:

- Ho co **sparsity training** truoc khi prune (day gamma ve 0), ta cat thang.
- Ho train bao nhieu epoch chua ro; ta chi 100 epoch = 39% so buoc cua VOC.
- Ho co the do tren split khac (test-dev thay vi val).

Neu khong khep duoc thi ha ti le prune cho VisDrone xuong 30-40% va bao cao o
muc nen thap hon — van trung thuc va van manh.

## AP theo kich thuoc vat the (pycocotools)

Do bang `tools/val_apsmall.py` tren 8 checkpoint trong `ckpt/`, maxDets=300
cho khop `max_det` cua Ultralytics.

| Size  |                   |      APs |      APm |      APl |
| ----- | ----------------- | -------: | -------: | -------: |
| **n** | YOLO26            |    10.16 |    26.71 |    40.84 |
|       | Ours              |     7.57 |    22.09 |    34.55 |
|       | _mat (diem)_      |  _-2.59_ |  _-4.62_ |  _-6.29_ |
|       | _mat (tuong doi)_ | _-25.5%_ | _-17.3%_ | _-15.4%_ |
| **s** | YOLO26            |    13.89 |    33.13 |    45.99 |
|       | Ours              |    11.15 |    27.85 |    41.47 |
|       | _mat (diem)_      |  _-2.74_ |  _-5.28_ |  _-4.52_ |
|       | _mat (tuong doi)_ | _-19.7%_ | _-15.9%_ |  _-9.8%_ |
| **m** | YOLO26            |    18.11 |    37.26 |    54.13 |
|       | Ours              |    15.43 |    33.22 |    42.24 |
|       | _mat (diem)_      |  _-2.68_ |  _-4.04_ | _-11.89_ |
|       | _mat (tuong doi)_ | _-14.8%_ | _-10.8%_ | _-22.0%_ |
| **l** | YOLO26            |    18.67 |    38.83 |    54.12 |
|       | Ours              |    16.29 |    36.13 |     48.4 |
|       | _mat (diem)_      |  _-2.38_ |  _-2.70_ |  _-5.72_ |
|       | _mat (tuong doi)_ | _-12.7%_ |  _-7.0%_ | _-10.6%_ |

### Doc bang nay cho can than: hai don vi cho hai ket luan nguoc nhau

**Theo diem tuyet doi**, `APs` mat IT hon `APm` o ca 4 size:
n -2.59 so voi -4.62, s -2.74 so voi -5.28, m -2.68 so voi -4.04,
l -2.38 so voi -2.70.

**Theo ti le tuong doi** thi nguoc lai, `APs` mat NHIEU nhat o ca 4 size:
-25.5%, -19.7%, -14.8%, -12.7%.

Ca hai deu dung, chi khac cach chuan hoa. Vi `APs` co nen rat thap (10-19)
nen cung mot so diem mat di se thanh ti le lon hon.

> **Dung vien cau "vat nho chiu thiet nang nhat" ma khong noi ro don vi.**
> Reviewer nhin cot diem tuyet doi se thay dieu nguoc lai. Neu dung khung
> tuong doi thi phai ghi thang la tuong doi, va nen dua ca hai cot.

### Thu thuc su vung

`APs` mat gan nhu **mot hang so 2.4-2.7 diem** bat ke kich thuoc model
(-2.59 / -2.74 / -2.68 / -2.38), trong khi `APm` mat giam dan theo kich thuoc
(-4.62 / -5.28 / -4.04 / -2.70). Model lon hon phuc hoi duoc vat vua nhung
khong phuc hoi duoc vat nho.

Ve cau hoi vi sao VisDrone mat 4.20 con VOC chi mat 1.08: so sanh tuong doi
ro hon nhieu — 4.20 tren nen AP50 46.90 la **9.0%**, con 1.08 tren nen 89.04
chi la **1.2%**. VisDrone kho hon han ngay tu dau, khong chi la chuyen kich
thuoc vat the.

> **Canh bao 1.** So pycocotools thap hon so Ultralytics **deu dan 2.24-2.37
> diem** o ca 8 model (vd YOLO26-M: 44.56 so voi 46.90). Da loai tru nguyen
> nhan `maxDets` — sua tu 100 thanh 300 cho khop, do lech van y nguyen. Day la
> khac biet phuong phap cham diem giua hai cong cu. Do lech deu nen so sanh
> tuong doi khong bi anh huong, nhung **dung tron hai nguon trong mot bang**.

> **Canh bao 2.** Ba cot kich thuoc cua arXiv 2509.12918 **khong doi chieu
> duoc**: bang ho co APmedium 66.2 trong khi AP50 chi 50.2, va APsmall 41.3
> trong khi AP chi 28.3. Voi dinh nghia COCO chuan thi khong the nhu vay.

> **Canh bao 3.** `APl` cua size m mat 11.89 diem trong khi cac size khac chi
> mat 4.5-6.3. VisDrone co rat it vat lon nen cot nay nhieu — dung xay lap luan
> len no.

## Giao thuc

|                            |                                                   |
| -------------------------- | ------------------------------------------------- |
| Dataset                    | VisDrone2019-DET — 6471 train / 548 val, 10 lop   |
| Split danh gia             | `val` (548 anh)                                   |
| imgsz / batch / seed       | 640 / 16 / 0                                      |
| epochs                     | 100                                               |
| optimizer                  | `auto` -> MuSGD (tu chon lr)                      |
| cos_lr / patience / warmup | False / 100 / 3.0                                 |
| Augmentation               | mac dinh Ultralytics, khong tinh chinh rieng      |
| Phan cung                  | Kaggle Tesla T4 x2, DDP                           |
| Prune                      | L1-norm uniform 50%, divisor 8                    |
| CWD                        | tau=9, kd_lambda=0.5, kd_layers=neck, kd_warmup=5 |
| Teacher                    | baseline cua **chinh size do**                    |
| Metric                     | AP 101-point (Ultralytics)                        |

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

|              | Ho                                                              | Ta          |
| ------------ | --------------------------------------------------------------- | ----------- |
| Backbone     | YOLOv8m                                                         | YOLOv26m    |
| Importance   | BN gamma                                                        | **L1-norm** |
| Distillation | CWD                                                             | CWD         |
| Ket qua      | 25.85M -> 6.85M (-73.5%), mAP50 47.9 (-2.7), 26 -> 68 FPS (TRT) |             |

Ba thu ho khong co, phai neu ro trong bai:

- Thuc nghiem cho thay **L1-norm > BN gamma** (ho dung gamma).
- Khao sat do nhay **tau = 1..10**.
- Do that tren **Jetson Nano**.

## Vi sao model pruned cua m va l trong "giong het nhau"

Bang `scales` trong `cfg/yolo26m.yaml`:

| size  |   depth |   width | max_channels |
| ----- | ------: | ------: | -----------: |
| n     |     0.5 |    0.25 |         1024 |
| s     |     0.5 |    0.50 |         1024 |
| **m** | **0.5** | **1.0** |      **512** |
| **l** | **1.0** | **1.0** |      **512** |

**m va l co width y het nhau (1.0), chi khac depth.** Ma pruning kenh la thao
tac tren _width_, khong dung den depth. Nen sau khi cat 50%, **124/124 lop dung
chung deu ra dung cung so kenh** — khong lech mot lop nao:

| lop             | s          | m          | l          |
| --------------- | ---------- | ---------- | ---------- |
| model.0.bn      | 32 -> 16   | 64 -> 32   | 64 -> 32   |
| model.4.cv2.bn  | 256 -> 128 | 512 -> 256 | 512 -> 256 |
| model.8.cv2.bn  | 512 -> 256 | 512 -> 256 | 512 -> 256 |
| model.22.cv2.bn | 512 -> 256 | 512 -> 256 | 512 -> 256 |

Khac biet giua Ours-M va Ours-L **hoan toan la do depth**: l co 178 lop BN can
prune so voi 124 cua m, tuc la moi khoi C3k2 lap 2 lan thay vi 1 (`n=2` so voi
`n=1` trong log build). Sau prune: 390 lop so voi 278.

Do that (VOC, prune 50% div8):

| size | params goc | pruned | GFLOPs goc | pruned |   nen |
| ---- | ---------: | -----: | ---------: | -----: | ----: |
| s    |      9.96M |  4.04M |       22.6 |    8.4 | 2.46x |
| m    |     21.80M |  7.50M |       74.9 |   23.6 | 2.91x |
| l    |     26.21M |  9.70M |       93.3 |   31.9 | 2.70x |

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
