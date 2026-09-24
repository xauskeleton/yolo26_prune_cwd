# VisDrone — 4 notebook, moi nguoi mot cai

Muc tieu: **mot bang** tren dataset thu hai, cung dinh dang main result tren VOC,
de chung minh pipeline khong chi hop voi PASCAL VOC.

| Notebook | Model | Phu thuoc | Uoc tinh |
|---|---|---|---|
| `nb1_yolo26m.ipynb` | yolo26m baseline | — | ~7-8h, 1-2 phien |
| `nb2_ours.ipynb` | **Ours** = prune 50% + CWD | **can output cua nb1** | ~8-9h, 2 phien |
| `nb3_yolov8m.ipynb` | yolov8m (doi chung) | — | ~6-7h, 1 phien |
| `nb4_yolo11m.ipynb` | yolo11m (doi chung) | — | ~6-7h, 1 phien |

**nb1, nb3, nb4 chay duoc ngay va song song.** nb2 phai doi nb1 xong, vi baseline
cua nb1 vua la nguon de prune vua la teacher cho CWD.

## Bang can dien

| Model | Params (M) | AP50 | AP50-95 |
|---|---:|---:|---:|
| yolov8m | | | |
| yolo11m | | | |
| yolo26m (baseline) | | | |
| **Ours (prune 50% + CWD)** | | | |

## Cau hinh chung — dung doi

| | |
|---|---|
| Dataset | VisDrone2019-DET, 6471 train / 548 val, 10 lop, 2.3 GB tu dong tai |
| Epoch / batch / imgsz / seed | 100 / 16 / 640 / 0 |
| Phan cung | Kaggle GPU T4 x2 (DDP) |
| Ours | L1-norm uniform 50%, divisor 8, CWD tau=9, kd_layers=neck, kd_warmup=5 |

Bon notebook dung **y het** cac tham so nay. Doi mot cai thoi la bang het so sanh duoc.

`imgsz=640` la muc chuan cua cac bai nen model tren VisDrone (FDM-YOLO,
YOLOv8n-ACW, cac bang YOLOv8n/s) nen so lieu doi chieu duoc voi ho.

## Cach chay

1. Settings -> Accelerator **GPU T4 x2**, **Internet: On**.
   Bat buoc co Internet: VisDrone va trong so COCO deu tai tu mang.
2. **Save & Run All**. Rieng nb2 phai Add Data output cua nb1 truoc.
3. Moi phien tu dung o 10h (`STOP_AFTER_H`) nen output luon duoc luu.
   Cell cuoi bao `CHUA XONG` kem so epoch -> Add Data output cua chinh lan
   chay nay roi Save & Run All lai. Lap den khi bao `XONG`.
4. Xong thi gui lai 3 so o cell cuoi: Params / AP50 / AP50-95.

## Vi sao co cell resume

Kaggle giet phien o 12h, va **phien bi giet thi KHONG luu output** — mat sach
last.pt, mat trang ca phien. Vi the moi notebook tu dung o 10h roi ket thuc
binh thuong. Cell resume chi lam mot viec: tim `runs/<NAME>/weights/last.pt`
trong `/kaggle/input` va chep ve, roi `model.train(resume=True)`.

## Chay tren may khac

Khong dung Kaggle thi co `scripts/run_e2e.sh`:

```bash
DATA=VisDrone.yaml TAG=vd50 ./scripts/run_e2e.sh
```

Chua co `weights/yolo26m_baseline_VisDrone.pt` thi no tu them stage baseline
truoc. Baseline tach theo dataset nen khong de len baseline VOC.
