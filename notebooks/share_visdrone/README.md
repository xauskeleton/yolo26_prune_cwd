# VisDrone — 4 notebook, moi nguoi mot size

Muc tieu: dung lai bang chinh cua VOC nhung tren dataset thu hai, de chung minh
pipeline khong chi hop voi PASCAL VOC.

Moi notebook lam **tron mot size**: baseline -> prune 50% -> finetune + CWD -> val,
cho ra **hai dong** cua bang.

| Notebook | Size | Uoc tinh | Phien |
|---|---|---|---|
| `nb_n.ipynb` | yolo26n | ~2-3h | 1 |
| `nb_s.ipynb` | yolo26s | ~6h | 1 |
| `nb_m.ipynb` | yolo26m | ~15h | 2 |
| `nb_l.ipynb` | yolo26l | ~21h | 2-3 |

Bon notebook **doc lap hoan toan**, chay song song duoc ngay. Trong moi notebook
thi Ours phai doi baseline cua chinh size do xong truoc — notebook tu lo, het gio
giua chung thi cac cell sau tu bo qua va in ra con thieu gi.

## Bang can dien

| Model | Params (M) | AP50 | AP50-95 |
|---|---:|---:|---:|
| YOLO26-N | | | |
| Ours-N | | | |
| YOLO26-S | | | |
| Ours-S | | | |
| YOLO26-M | | | |
| Ours-M | | | |
| YOLO26-L | | | |
| Ours-L | | | |

Dung dinh dang cua `dcmm.py` nhung tren VisDrone thay vi VOC.

## Cau hinh chung — DUNG DOI

| | |
|---|---|
| Dataset | VisDrone2019-DET, 6471 train / 548 val, 10 lop, 2.3 GB tu dong tai |
| Epoch / batch / imgsz / seed | 100 / 16 / 640 / 0 |
| Prune | L1-norm uniform **50%**, divisor 8 |
| CWD | tau=9, kd_lambda=0.5, kd_layers=neck, kd_warmup=5 |
| Teacher | baseline cua **chinh size do** |
| Phan cung | Kaggle GPU T4 x2 (DDP) |

Doi mot tham so thoi la dong cua ban khong con so sanh duoc voi ba dong kia.
Neu buoc phai doi (vi du het VRAM) thi **bao lai**, dung tu sua rooi im lang.

`imgsz=640` la muc chuan cua cac bai nen model tren VisDrone (FDM-YOLO,
YOLOv8n-ACW, cac bang YOLOv8n/s) nen so lieu doi chieu duoc voi ho.

## Cach chay

1. Settings -> Accelerator **GPU T4 x2**, **Internet: On**.
   Bat buoc co Internet: VisDrone va trong so COCO deu tai tu mang.
2. **Save & Run All**. Lan dau khong can Add Data.
3. Moi phien tu dung o 10h (`STOP_AFTER_H`) nen output luon duoc luu.
   Cell cuoi bao con thieu bao nhieu epoch -> Add Data output cua chinh lan
   chay nay roi Save & Run All lai. Lap den khi bao `XONG`.
4. Xong thi gui lai bang 2 dong o cell cuoi.

## Vi sao co cell resume

Kaggle giet phien o 12h, va **phien bi giet thi KHONG luu output** — mat sach
last.pt, mat trang ca phien. Vi the moi notebook tu dung o 10h roi ket thuc
binh thuong. Cell resume tim `runs/<NAME>/weights/last.pt` va file da prune
trong `/kaggle/input`, chep ve, roi `model.train(resume=True)`.

Nguyen tac trong ham `train()`: **co last.pt la resume**. Khong lay so epoch
doc duoc lam dieu kien — doc that bai thi se am tham train lai tu dau, mat ca
chuc gio ma khong bao gi.

## Luu y rieng cho size l

`nb_l` chay pruned-l lam student **cong** yolo26l lam teacher (teacher forward
o FP32, ngoai autocast). Tren T4 15GB o batch 16 DDP thi co the sat tran VRAM.
Neu bao CUDA out of memory: **dung tu ha batch**, bao lai de ca nhom cung ha —
batch khac nhau thi BatchNorm chuan hoa tren so mau khac nhau, bang het so sanh.

## Chay tren may khac

Khong dung Kaggle thi co `scripts/run_e2e.sh`:

```bash
DATA=VisDrone.yaml TAG=vd50 ./scripts/run_e2e.sh
```

Chua co `weights/yolo26m_baseline_VisDrone.pt` thi no tu them stage baseline
truoc. Baseline tach theo dataset nen khong de len baseline VOC.
