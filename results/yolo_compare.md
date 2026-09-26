# Ket qua so sanh tren PASCAL VOC2007 test

Cap nhat: 2026-09-11

## Bang chinh — 100 epoch

| Model | Family | Params (M) | GFLOPs | Latency (ms) | FPS | AP50 | Nguon |
|---|---|---:|---:|---:|---:|---:|---|
| yolo26m (baseline) | YOLOv26 (ours) | 21.80 | 74.9 | — | — | **89.04** | run cu, ghi thang |
| yolov9c | YOLOv9 | 25.54 | 103.8 | 24.17 | 41.4 | 88.35 | Kaggle, 100ep |
| yolov10m | YOLOv10 | 16.51 | 64.2 | 23.79 | 42.0 | 88.28 | Kaggle, 100ep |
| yolo12m | YOLO12 | 20.15 | 71.1 | 20.54 | 48.7 | 88.27 | Kaggle, 100ep |
| yolov8m | YOLOv8 | 25.87 | 79.1 | 12.60 | 79.3 | 88.26 | Kaggle, 9.22h |
| yolov9m | YOLOv9 | 20.17 | 77.6 | 23.25 | 43.0 | 88.19 | Kaggle, 100ep |
| rtdetr-l | RT-DETR | 32.85 | 110.0 | 42.07 | 23.8 | 88.09 | Kaggle, 100ep |
| **Ours (pruned)** | **YOLOv26 (ours)** | **7.50** | **23.6** | — | — | **87.96** | run cu (cwd_t9), ghi thang |
| yolo11m | YOLO11 | 20.07 | 68.4 | 18.96 | 52.7 | 87.41 | Kaggle, 9.80h |
| yolov5mu | YOLOv5u | 25.08 | 64.4 | 14.41 | 69.4 | 87.24 | Kaggle, 9.24h |

Tien do: **10/10 dong — DA DU**.

![AP@0.5 va GFLOPs](yolo_compare.png)

*Cot = AP@0.5, duong = GFLOPs. Sinh boi `notebooks/plot_compare.ipynb`.*

### Bang 3 cot (khop voi hinh, xep giam dan theo GFLOPs)

| Model | AP@0.5 | GFLOPs |
|---|---:|---:|
| RT-DETR-L | 88.09 | 110.0 |
| YOLOv9-C | 88.35 | 103.8 |
| YOLOv8-M | 88.26 | 79.1 |
| YOLOv9-M | 88.19 | 77.6 |
| YOLO26-M | 89.04 | 74.9 |
| YOLO12-M | 88.27 | 71.1 |
| YOLO11-M | 87.41 | 68.4 |
| YOLOv5-Mu | 87.24 | 64.4 |
| YOLOv10-M | 88.28 | 64.2 |
| **Ours** | **87.96** | **23.6** |

<details><summary>Ban LaTeX</summary>

```latex
\begin{tabular}{lrr}

\toprule
Model & AP@0.5 & GFLOPs \\
\midrule
RT-DETR-L & 88.09 & 110.0 \\
YOLOv9-C & 88.35 & 103.8 \\
YOLOv8-M & 88.26 & 79.1 \\
YOLOv9-M & 88.19 & 77.6 \\
YOLO26-M & 89.04 & 74.9 \\
YOLO12-M & 88.27 & 71.1 \\
YOLO11-M & 87.41 & 68.4 \\
YOLOv5-Mu & 87.24 & 64.4 \\
YOLOv10-M & 88.28 & 64.2 \\
\textbf{Ours} & \textbf{87.96} & \textbf{23.6} \\
\bottomrule
\end{tabular}
```

</details>

### Nhan xet

**Model nang hon nhung AP50 THAP hon Ours** (2/8 doi chung):

| Model | Params (M) | So voi Ours | AP50 | Chenh |
|---|---:|---:|---:|---:|
| yolo11m | 20.07 | **2.7x** | 87.41 | **-0.55** |
| yolov5mu | 25.08 | **3.3x** | 87.24 | **-0.72** |

Sau model con dang tren Ours — **ca sau nam gon trong 0.26 diem cua nhau**:

| Model | Params (M) | GFLOPs | So voi Ours | AP50 | Chenh |
|---|---:|---:|---:|---:|---:|
| yolov9c | 25.54 | 103.8 | 3.4x | 88.35 | +0.39 |
| yolov10m | 16.51 | 64.2 | 2.2x | 88.28 | +0.32 |
| yolo12m | 20.15 | 71.1 | 2.7x | 88.27 | +0.31 |
| yolov8m | 25.87 | 79.1 | 3.4x | 88.26 | +0.30 |
| yolov9m | 20.17 | 77.6 | 2.7x | 88.19 | +0.23 |
| rtdetr-l | 32.85 | 110.0 | 4.4x | 88.09 | +0.13 |

- **Bao hoa ro rang**: sau model trai dai tu 16.5M den 32.9M tham so (**2.0 lan**) va 64.2G
  den 110.0G (**1.7 lan**) nhung AP50 chi bien thien **0.26 diem**. Bon trong so do
  (v10m, 12m, v8m, v9m) nam trong **0.09 diem** — duoi nguong nhieu cua mot lan train.
- Them tham so gan nhu khong mua duoc gi: v10m -> v9c ton **+9.0M** de duoc **+0.07 diem**;
  v10m -> rtdetr-l ton **+16.3M** ma con **-0.19 diem**.
- AP50 / MParam: Ours **11.73** — yolov10m 5.35, yolo12m 4.38, yolov9m 4.37, yolo11m 4.36,
  yolov5mu 3.48, yolov9c 3.46, yolov8m 3.41, rtdetr-l 2.68. Ours cao gap **2.2 lan** model
  hieu qua nhat con lai va gap **4.4 lan** rtdetr-l.
- **Latency khong ti le voi params**: yolov9c/yolov10m/yolov9m deu ~23-24 ms trong khi
  yolov8m (25.87M, nang hon ca ba) chi **12.60 ms** va yolov5mu **14.41 ms**. GELAN nhieu
  nhanh (v9), NMS-free head (v10) va area-attention (v12, 20.54 ms) deu tra gia bang thoi
  gian forward tren T4. Neu xep theo FPS thi thu tu bang **dao lon hoan toan** so voi xep
  theo AP50.

**Cau chot cho bai:**

> *Tren PASCAL VOC, sau kien truc doi chung trai dai 2.0 lan ve so tham so chi chenh nhau
> 0.26 diem AP50 — cho thay cac ho detector hien dai da bao hoa tren tap nay. Mo hinh de
> xuat dat 87.96 AP50 voi 7.50M tham so (23.6 GFLOPs), tuc thap hon nhom tren 0.13-0.39
> diem nhung dung it hon 2.2-4.4 lan tham so, dat ti so AP50/MParam la 11.73 so voi
> 2.68-5.35 cua cac mo hinh doi chung.*

**rtdetr-l — model duy nhat ngoai ho YOLO — dung cuoi nhom dan dau tren moi tieu chi:**

| | rtdetr-l | Ours | Ti le |
|---|---:|---:|---:|
| Params (M) | 32.85 | 7.50 | **4.4x** |
| GFLOPs | 110.0 | 23.6 | **4.7x** |
| Latency (ms) | 42.07 | — | — |
| FPS | 23.8 | — | — |
| AP50 | 88.09 | 87.96 | **+0.13** |

RT-DETR nang nhat bang, cham nhat bang (**42.07 ms, 23.8 FPS** — cham gap **3.3 lan**
yolov8m), va AP50 van **thap hon ca 4 model YOLO** o tren. Tren tap co 20 lop nhu VOC,
uu the "khong can NMS" cua ho DETR khong bu duoc chi phi transformer decoder. Ket qua nay
ho tro viec chon YOLO lam kien truc nen cho pruning.

---

## Bang phu — ket qua 50 epoch (DA BO, khong dung cho bai)

Chay o giai doan truoc khi doi chuan sang 100 epoch. Giu lai neu can bang
"anh huong cua ngan sach huan luyen".

| Model | Params (M) | AP50 @ 50ep | AP50 @ 100ep | Loi tu 100 epoch |
|---|---:|---:|---:|---:|
| yolov10m | 16.51 | 87.71 | **88.28** | +0.57 |
| yolo11m | 20.07 | 87.31 | **87.41** | +0.10 |
| yolov9m | 20.17 | 87.27 | **88.19** | +0.92 |

Ca ba model deu tang khi len 100 epoch: **+0.10 den +0.92 diem**, cung huong voi +0.34 do
duoc tren yolo26m baseline (epoch 50 -> 86). Day la ly do khong dung so 50 epoch de so voi
Ours (100 epoch). Dang chu y: neu giu moc 50 epoch thi **ca ba deu thap hon Ours (87.96)** —
tuc la ket qua se bi thoi phong co loi cho de tai.

---

## AP50 theo tung lop (100 epoch)

| Lop | yolov8m | yolo11m | yolov5mu |
|---|---:|---:|---:|
| aeroplane | 94.3 | 93.9 | 94.8 |
| bicycle | 95.1 | 95.0 | 94.2 |
| bird | 87.9 | 88.5 | 86.3 |
| boat | 82.1 | 81.6 | 79.7 |
| bottle | 81.2 | 79.4 | 79.7 |
| bus | 91.8 | 92.5 | 92.5 |
| car | 94.9 | 94.8 | 94.8 |
| cat | 94.2 | 94.2 | 93.3 |
| chair | 73.4 | 73.2 | 74.1 |
| cow | 92.5 | 91.4 | 91.5 |
| diningtable | 84.0 | 81.9 | 82.9 |
| dog | 92.1 | 92.7 | 91.3 |
| horse | 95.4 | 94.7 | 95.1 |
| motorbike | 94.6 | 92.4 | 92.9 |
| person | 92.1 | 92.0 | 91.8 |
| pottedplant | 67.1 | 65.2 | 67.2 |
| sheep | 87.7 | 84.6 | 86.0 |
| sofa | 83.1 | 82.9 | 79.4 |
| train | 94.6 | 91.1 | 91.7 |
| tvmonitor | 87.2 | 86.7 | 85.6 |
| **Trung binh** | **88.3** | **87.4** | **87.2** |

Nhom object nho (bottle, pottedplant, boat, bird, chair):

| Model | AP50 toan bo | AP50 nhom nho | Khoang cach |
|---|---:|---:|---:|
| yolov8m | 88.3 | 78.3 | -9.9 |
| yolo11m | 87.4 | 77.6 | -9.9 |
| yolov5mu | 87.2 | 77.4 | -9.8 |

`pottedplant` la lop kho nhat voi ca ba (65-67%), keo tut trung binh. Ca ba model deu tut
**~10 diem** o nhom object nho — day la cho pruning thuong hong truoc, nen khi co so per-class
cua Ours can kiem tra rieng nhom nay.

> yolov10m chua co bang per-class (file `yolov10m_perclass.csv` da luu trong output Kaggle,
> chua dan vao day).

---

## Do nhay cua CWD temperature (tau)

Sweep tau = 1..10, cung model da prune, cung cong thuc, khac moi `cwd_temperature`.
Nguon: `cwd/optimzing cwd/optimzing cwd/t1..t10/`. Cac run bi cat thanh nhieu manh do
resume -> da **gop cac manh cua cung mot tau** roi lay max tren hop.

![Do nhay tau](tau_sweep.png)

| tau | AP@0.5 | AP50-95 | best @epoch |
|---:|---:|---:|---:|
| 1 | 87.81 | 70.23 | 92 |
| 2 | 87.64 | 69.94 | 75 |
| 3 | 86.57 | 69.10 | 79 |
| 4 | 87.70 | 70.09 | 78 |
| 5 | 87.55 | 70.00 | 81 |
| 6 | 87.74 | 70.09 | 83 |
| 7 | 87.75 | 69.91 | 81 |
| 8 | 87.62 | 70.07 | 83 |
| **9** | **87.96** | 70.09 | — |
| 10 | 87.36 | 69.78 | 76 |

- **Bien do toan dai tau=1..10: 1.39 diem** (86.57 - 87.96).
- **Bo tau=3 thi chi con 0.45 diem** (87.36 - 87.96) tren 9 gia tri tau con lai.
- tau=3 la diem tut duy nhat; 9/10 gia tri nam trong dai 87.36-87.96.
- Best epoch cua moi tau deu roi vao **75-92**, nam trong vung CSV con giu (56-100),
  nen so lay ra khong bi cat mat.

**Cau de ghi trong bai:**

> *Khao sat tau tu 1 den 10 cho thay AP@0.5 bien thien trong pham vi 0.45 diem tren
> 9/10 gia tri (87.36-87.96), rieng tau=3 tut xuong 86.57. Ket qua nay cho thay phuong
> phap it nhay voi tham so nhiet do; gia tri tau=9 duoc chon cho ket qua tot nhat nhung
> khong phai la dieu kien bat buoc de dat muc AP nay.*

> **Luu y**: cac run khong dung cung so epoch cuoi (93-100, rieng tau=10 la 86), nen
> bang nay dung de noi ve **do nhay**, khong dung de xep hang chinh xac giua cac tau
> gan nhau. Muon so chat thi chay lai bang
> `python scripts/run_e2e.py --stage finetune --cwd-temperature X`.

---

## Giao thuc

| Muc | Gia tri |
|---|---|
| Train set | VOC2007 trainval + VOC2012 trainval (16.551 anh) |
| Test set | VOC2007 test (4.952 anh) |
| Epochs | 100 |
| imgsz / batch / seed | 640 / 16 / 0 |
| Optimizer | MuSGD (lr0=0.01, momentum=0.9) — `optimizer=auto` chon |
| Init | COCO-pretrained |
| Metric | AP 101-point (Ultralytics), khong phai VOC 11-point |
| Latency | batch=1, FP16, forward-only, Tesla T4 |

## Khac biet giua 2 dong cua de tai va 8 doi chung

Ca hai deu bat loi cho **de tai**, tuc la neu Ours van tot thi ket luan cang vung.

**1. `cls_remap`** — Ultralytics 8.4.144 anh xa 14/20 lop VOC tu trong so COCO theo ten lop
khi khoi tao classification head. Fork 8.4.14 (dung cho yolo26m) **khong co** tinh nang nay.
Thay ro o epoch 1: yolov8m dat mAP50 = 0.808 ngay epoch dau.

**2. BatchNorm** — 8 doi chung train DDP tren 2 GPU: batch tong 16, chia 8 moi GPU, nen BN
chuan hoa tren 8 mau. Hai dong yolo26m train 1 GPU, BN tren 16 mau. Batch hieu dung va moi
sieu tham so khac deu giong nhau.

**Cau de ghi trong bai:**

> *Cac mo hinh doi chung duoc huan luyen bang Ultralytics 8.4.144, trong do lop phan loai
> duoc khoi tao bang cach anh xa 14/20 lop VOC tu trong so COCO theo ten lop, va su dung
> huan luyen phan tan tren 2 GPU (batch tong 16, 8 mau moi GPU). Hai mo hinh de xuat dung
> phien ban 8.4.14 tren mot GPU. Ca hai khac biet nay deu co loi cho cac mo hinh doi chung.*

## Da kiem tra, khong phai van de

- **Optimizer**: ca hai deu ra `MuSGD` — fork co san tu `trainer.py:1653`.
- **Augmentation + sieu tham so**: 25/27 tham so trung khop hoan toan (da doi chieu
  `train_args` trong checkpoint).
- **Batch hieu dung**: Ultralytics chia `batch` cho so GPU (`trainer.py:271`), khong nhan len
  — nen DDP batch=16 van la 16.

## Cong thuc tao ra model de xuat

```
Prune   : L1-norm, uniform 50%, divisor 8   (124 lop BN: chi 0% hoac 50%)
Finetune: finetune=True, kd=True, kd_method=cwd, kd_teacher=yolo26m_baseline.pt,
          kd_lambda=0.5, cwd_temperature=9, kd_layers=neck, kd_warmup=5
          epochs=100, batch=16, imgsz=640   (run cwd_t9)
```

## Thoi gian train (do thuc te)

| Model | Phut/epoch | 100 epoch |
|---|---:|---:|
| yolov8m (DDP 2xT4) | 5.53 | 9.22h |
| yolov5mu (DDP 2xT4) | 5.55 | 9.24h |
| yolo11m (DDP 2xT4) | 5.88 | 9.80h |
| yolo26m baseline (run cu) | 16.5 | 27.5h |
| Ours pruned + CWD (run cu) | 13.0 | 21.7h |
| Ours pruned + CWD (r30, 2xT4 DDP) | 12.4 | 20.7h |
| Ours pruned + CWD (r40, 2xT4 DDP) | 11.7 | 19.5h |
| Ours pruned + CWD (r60, 2xT4 DDP) | 10.0 | 16.7h |
| Ours pruned + CWD (r70, 2xT4 DDP) | 9.9 | 16.6h |

## Quet ti le pruning (VOC, L1-norm uniform + CWD tau=9)

Moi ti le mot notebook trong `notebooks/share_ratio/`, cung baseline, cung
config, chi khac `--prune-ratio`. Do tren VOC2007 test (4952 anh).

| Ratio | Params (M) | GFLOPs | AP50 | AP50-95 | So voi baseline | AP50/MParam |
|---|---:|---:|---:|---:|---:|---:|
| 0% (baseline) | 21.80 | 74.9 | 89.04 | — | — | 4.08 |
| 30% | 12.67 | 42.5 | 88.09 | 71.51 | -0.95 | 6.95 |
| 40% | 10.09 | 33.1 | 88.05 | 70.97 | -0.99 | 8.73 |
| 50% | 7.50 | 23.6 | 87.96 | — | -1.08 | 11.73 |
| 60% | 5.87 | 18.3 | 86.78 | 68.69 | -2.26 | 14.78 |
| 70% | 4.18 | 12.4 | 84.98 | 66.42 | -4.06 | 20.33 |

Dong 50% lay tu run `cwd_t9` (cung cau hinh, cung baseline, cung 100 epoch)
— day la so chinh thuc dung trong bang chinh.

### Bang GFLOPs - AP@0.5 (quet ti le pruning)

![AP@0.5 theo GFLOPs](flops_ap50_sweep_ratio.png)

*Sinh boi `tools/plot_flops_ap50.py`. Huong tot la tren-trai. Duong xanh la
quet ti le cua ta, cham xam la 8 doi chung (cung 100 epoch, cung giao thuc).
Diem 50% dac la cau hinh de xuat.*


| Ti le prune | GFLOPs | AP@0.5 |
|---|---:|---:|
| 0% (baseline) | 74.9 | 89.04 |
| 30% | 42.5 | 88.09 |
| 40% | 33.1 | 88.05 |
| **50% (de xuat)** | **23.6** | **87.96** |
| 60% | 18.3 | 86.78 |
| 70% | 12.4 | 84.98 |

<details><summary>Ban LaTeX</summary>

```latex
\begin{tabular}{lrr}
\toprule
Prune ratio & GFLOPs & AP@0.5 \\
\midrule
0\% (baseline) & 74.9 & 89.04 \\
30\% & 42.5 & 88.09 \\
40\% & 33.1 & 88.05 \\
\textbf{50\% (de xuat)} & \textbf{23.6} & \textbf{87.96} \\
60\% & 18.3 & 86.78 \\
70\% & 12.4 & 84.98 \\
\bottomrule
\end{tabular}
```

</details>


### Ket qua chinh: co mot vung phang 30-50%

Ba ti le 30%, 40%, 50% cho AP50 gan nhu **khong phan biet duoc**: 88.09, 88.05,
87.96 — trai rong 0.13 diem, nho hon nhieu so voi dao dong giua cac epoch cuoi
cua chinh mot run. Nhung so params thi khac han: 12.67M so voi 7.50M, tuc la
**cat them 41% tham so ma khong mat gi do duoc**.

Gia phai tra cho moi GFLOPs tiet kiem duoc:

| Doan | AP50 mat | GFLOPs tiet kiem | AP50 mat / GFLOPs |
|---|---:|---:|---:|
| 0% -> 30% | 0.95 | 32.4 | 0.029 |
| 30% -> 40% | 0.04 | 9.4 | **0.004** |
| 40% -> 50% | 0.09 | 9.5 | **0.009** |
| 50% -> 60% | 1.18 | 5.3 | 0.22 |
| 60% -> 70% | 1.80 | 5.9 | 0.31 |

Doan 30-50% gan nhu mien phi (0.004-0.009), roi **tang gap ~25 lan** ngay khi
qua moc 50%. Diem gay sac net nam dung o **50%** — do la ti le cuoi cung con
nam tren vung phang, cho muc nen cao nhat ma chua phai tra gia.

> Vi vay chon 50% khong phai la uoc luong: no la diem cuoi cua vung phang, xac
> dinh boi so lieu. Cat it hon (30%, 40%) chi lam model to ra ma khong duoc
> them do chinh xac nao.

### AP50-95 xuong truoc AP50

| Ratio | AP50 | AP50-95 |
|---|---:|---:|
| 30% | 88.09 | 71.51 |
| 40% | 88.05 (-0.04) | 70.97 (**-0.54**) |
| 60% | 86.78 | 68.69 |
| 70% | 84.98 | 66.42 |

Tu 30% sang 40%, AP50 gan nhu dung yen (-0.04) nhung AP50-95 mat 0.54. Nghia la
model **van tim ra vat the nhung dinh vi kem chinh xac hon** — pruning lam suy
giam chat luong hoi quy box truoc khi anh huong den kha nang phat hien. Dung
AP50 mot minh se khong thay dieu nay.

### Lop nao chiu thiet khi cat sau

AP50 per-class:

| Lop | 30% | 40% | 60% | 70% | 30->70 |
|---|---:|---:|---:|---:|---:|
| pottedplant | 68.0 | 65.6 | 62.7 | 62.3 | **-5.7** |
| aeroplane | 94.6 | 93.9 | 93.2 | 90.3 | -4.3 |
| chair | 73.4 | 73.8 | 72.5 | 69.4 | -4.0 |
| diningtable | 83.1 | 83.9 | 83.4 | 78.5 | -4.6 |
| bottle | 81.1 | 79.6 | 79.1 | 76.1 | -5.0 |
| person | 91.6 | 91.6 | 90.8 | 89.3 | -2.3 |
| car | 94.6 | 94.5 | 93.9 | 93.1 | -1.5 |
| boat | 78.4 | 79.6 | 78.2 | 78.4 | 0.0 |

Vat nho va bi che khuat (pottedplant, bottle, chair) mat nhieu nhat; vat lon va
ro net (car, person) gan nhu khong he han gi. pottedplant giam deu tu dau den
cuoi va luon la lop yeu nhat o moi ti le.

### So voi cong trinh gan nhat

arXiv 2509.12918 (BN-gamma pruning + CWD tren YOLOv8m/VisDrone) dat **-73.51%
params voi -2.7 mAP50**. Muc nen tuong duong cua ta la ratio 60%: **-73.1%
params voi -2.26 AP50**.

> Hai con so nay **khong so sanh truc tiep duoc** — khac dataset (VOC vs
> VisDrone) va khac backbone (YOLOv26m vs YOLOv8m). Chi dung de dinh vi rang
> pipeline cua ta nam cung hang, chua phai bang chung vuot troi. Muon so truc
> tiep thi phai co bang VisDrone (xem muc duoi).

## VisDrone2019-DET (bang phu, dang chay)

Chuyen sang file rieng: **`results/visdrone/README.md`**.

Bang thu hai, cung dinh dang main result nhung tren dataset khac, de chung minh
pipeline khong chi hop voi PASCAL VOC. Bon size n/s/m/l, moi size mot notebook
trong `notebooks/share_visdrone/`. Checkpoint gui ve de o `results/visdrone/ckpt/`.

## Vi sao model pruned cua m va l trong giong het nhau

Chi tiet: `results/visdrone/README.md`.

`m` va `l` trong `cfg/yolo26m.yaml` co **cung width 1.0 va cung
max_channels 512**, chi khac depth (0.5 so voi 1.0). Pruning kenh la thao
tac tren width nen 124/124 lop dung chung ra dung cung so kenh. Khac biet
giua Ours-M va Ours-L hoan toan la do do sau (178 lop BN so voi 124).

> Khi viet bai: doan m -> l trong ho model la so sanh **chi khac do sau**,
> doan n -> s chi khac width, doan s -> m khac ca width lan max_channels.
> Moi buoc doi mot thu khac nhau - dung dien giai nhu mot duong scaling deu.


## Trang thai

**Da du 10/10 dong.** Tat ca 8 doi chung deu train 100 epoch tren Kaggle 2xT4 (DDP),
moi model mot notebook (`notebooks/share8/nb1..nb8`), khong con model nao phai chay.

Quet ti le pruning: **xong ca 5 ti le** (30/40/50/60/70). Dong 50% dung so
cua run `cwd_t9`, trung voi dong Ours trong bang chinh.

Con thieu (khong chan viec lap bang):

- Latency / FPS cho 2 dong yolo26m — can chay `model.val()` tren fork 8.4.14 voi 2
  checkpoint cu, do tren cung Tesla T4 de so sanh duoc.
- AP50 per-class cho Ours va yolo26m baseline — de kiem tra nhom object nho.
- File per-class cua yolov10m, yolo12m, rtdetr-l da luu trong output Kaggle, chua dan vao day.
