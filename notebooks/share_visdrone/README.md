# VisDrone — bang ket qua phu

Mot notebook duy nhat: `nb_visdrone.ipynb`.

Khac voi `share_ratio/` (5 nguoi, moi nguoi mot ti le prune tren VOC), o day
chi can **mot bang 2 dong** tren dataset thu hai de chung minh pipeline khong
chi hop voi PASCAL VOC:

| Model | Params (M) | AP50 | AP50-95 |
|---|---:|---:|---:|
| yolo26m (baseline) | | | |
| Ours (prune 50% + CWD) | | | |

## Cau hinh

| | |
|---|---|
| Dataset | VisDrone2019-DET — 6471 train / 548 val, 10 lop, 2.3 GB tu dong tai |
| imgsz | 640 |
| Baseline | yolo26m tu trong so COCO, 100 epoch |
| Ours | L1-norm prune 50% (div 8), finetune 100 epoch, CWD tau=9, kd_layers=neck |
| Phan cung | Kaggle GPU T4 x2 (DDP) |

`imgsz=640` la muc chuan cua cac bai nen model tren VisDrone (FDM-YOLO,
YOLOv8n-ACW, cac bang YOLOv8n/s) nen so lieu so sanh duoc voi ho. Cac bai
chuyen ve vat the nho dung 1024-1280 va an hon dang ke (640 -> 1280 cho
khoang +25% mAP) nhung do la nhanh khac, khong phai nhanh nen model.

## Chay

1. Settings -> Accelerator **GPU T4 x2**, **Internet: On**.
   Bat buoc co Internet: ca VisDrone lan `yolo26m.pt` deu tai tu mang.
2. **Save & Run All**. Lan dau khong can Add Data.
3. Moi phien tu dung o 10h (`STOP_AFTER_H`), output luon duoc luu.
   Cell Ket qua bao con thieu gi -> Add Data output lan nay -> Save & Run All lai.
4. Uoc tinh **2-3 phien**: baseline ~7-8h, prune + CWD ~8-9h.

Xong thi gui lai `results/e2e_manifest.json` trong tab Output.

## Ghi chu ky thuat

- Baseline cua VisDrone **khong dung chung** voi baseline VOC: khac so lop
  (10 vs 20). `run_e2e.py` tach theo dataset — manifest key `baseline_VisDrone`,
  file `weights/yolo26m_baseline_VisDrone.pt`. Tag `vd50` tach khoi `r30..r70`
  nen co the gan chung output VOC va VisDrone ma khong de len nhau.
- DDP an toan: cac tham so `kd`, `kd_teacher`, `finetune`, `cwd_temperature`
  da nam trong `ultralytics/cfg/default.yaml` nen di duoc sang tien trinh con.
  `run_e2e.py` kiem tra lai truoc khi cho chay DDP.
