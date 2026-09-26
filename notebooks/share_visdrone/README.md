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

## Sieu tham so

Da doc `train_args` tu chinh cac checkpoint VOC. **Cac run VOC theo size KHONG
dong nhat**:

| | yolo26s | yolo26l | yolo26m |
|---|---:|---:|---:|
| batch | 32 | 32 | **16** |
| cos_lr | True | True | **False** |
| patience | 20 | 30 | **100** |
| warmup_epochs | 3.0 | **5.0** | 3.0 |
| device | 0,1 (DDP) | 0,1 (DDP) | **0** |

(`lr0` cung khac nhau — 0.001 / 0.0002 / 0.01 — nhung `optimizer=auto` bo qua
`lr0` va tu chon lay, nen cho nay khong tinh la khac biet that.)

Con lai trung khop het: epochs 100, imgsz 640, seed 0, box/cls/dfl 7.5/0.5/1.5,
nbs 64, mosaic 1.0, close_mosaic 10, mixup/cutmix/copy_paste 0, hsv 0.015/0.7/0.4,
degrees 0, translate 0.1, scale 0.5, shear 0, flipud 0, fliplr 0.5, erasing 0.4,
rect False, multi_scale 0.

> Khong tim thay config cua `yolo26n` — notebook goc khong nam trong repo va may
> nay khong co `yolo26n_baseline.pt`. Ai giu file do thi doc `train_args` ra.

### VisDrone dung mot config duy nhat cho ca 4 size

Lay **config cua m**, vi do la cau hinh chinh cua bai (ca quet ti le, quet tau
va moi so headline deu chay bang no):

```python
EPOCHS = 100        BATCH = 16          IMGSZ = 640         seed = 0
COS_LR = False      PATIENCE = 100      WARMUP = 3.0
optimizer = "auto"  # -> MuSGD, tu chon lr
DEVICE = "0,1"      # DDP cho ca 4 size
```

Ba tham so `cos_lr` / `patience` / `warmup_epochs` duoc **ghi thang trong
notebook** chu khong de mac dinh — de nhin la thay, va de Ultralytics co doi
mac dinh thi bang van khong troi.

Vi sao chon nhu vay:

- **`patience=100`** tat early stop, nen ca 4 size deu la run 100 epoch that.
  Voi patience 20-30 thi co size dung som, co size chay het — bang theo size ma
  moi dong mot do dai thi khong so sanh duoc.
- **`batch=16` chu khong 32.** VisDrone chi co 6471 anh: batch 16 cho 405
  iter/epoch, batch 32 chi con 202 — qua it. Batch 16 con an toan VRAM cho
  `nb_l` (student l + teacher l forward FP32).
- **`DEVICE="0,1"` cho ca 4.** VOC-m chay 1 GPU nen BN chuan hoa tren 16 mau,
  con n/s/l chay DDP nen BN tren 8 mau. Chon DDP cho het de bon dong giong nhau,
  va vi khong dung DDP thi `nb_l` khong kip trong quota.

Phan augmentation giu nguyen mac dinh, khong tinh chinh rieng cho VisDrone:
bang nay la de tra loi "pipeline co chuyen sang dataset khac duoc khong", ma do
lai sieu tham so cho tung dataset thi cau tra loi mat gia tri.

### Hai thu da can nhac va bo

**1. Tang `epochs`.** VisDrone 6471 anh so voi 16551 cua VOC:

| | anh | iter/epoch (batch 16) | 100 epoch |
|---|---:|---:|---:|
| VOC | 16551 | 1035 | 103500 |
| VisDrone | 6471 | 405 | 40500 |

Chi bang **39%** so buoc toi uu; muon bang phai chay ~256 epoch. Van giu 100 vi
cac bai nen model tren VisDrone deu dung 100, vi baseline va Ours nhan cung mot
ngan sach nen do chenh van co nghia, va vi 256 epoch khong du quota.

> Ghi vao phan han che: ca hai dong deu chua hoi tu han o 100 epoch. **Do chenh**
> moi la thu can bao cao, khong phai con so tuyet doi.

**2. Tang `max_det` (300).** VisDrone co anh dong hon 300 vat the nen 300 se cat
bot va ep AP xuong. Nhung cac bai doi chung cung chay mac dinh Ultralytics, tuc
la bi cat y het — doi len se lam so cua ta khong con doi chieu duoc voi ho.
Giu 300, ghi vao phan han che. Dem thu sau khi tai dataset:

```bash
python -c "
import pathlib, collections
d = pathlib.Path('datasets/VisDrone/labels/train')
c = collections.Counter(len([x for x in f.read_text().splitlines() if x.strip()])
                        for f in d.glob('*.txt'))
print('anh >300 vat the:', sum(v for k, v in c.items() if k > 300), '/', sum(c.values()))
print('nhieu nhat:', max(c))
"
```

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

## Weight luu o dau

Trong luc chay, tren Kaggle (`project=runs/`, tat ca nam duoi `/kaggle/working`
nen deu vao tab Output):

| | |
|---|---|
| Baseline | `/kaggle/working/yolo/runs/vd_yolo26<size>/weights/{best,last}.pt` |
| Ours | `/kaggle/working/yolo/runs/vd_ours<size>/weights/{best,last}.pt` |
| Model da prune | `/kaggle/working/yolo/weights/yolo26<size>_vd_pruned50.pt` |

`last.pt` ghi lai moi epoch (de resume), `best.pt` la ban tot nhat.

Chay xong ca hai, cell ket qua **tu gom** lai mot cho cho de tai:

```
/kaggle/working/yolo/results/visdrone/
├── ckpt/baseline/<size>/yolo26<size>_vd_baseline.pt
├── ckpt/pruned/<size>/yolo26<size>_vd_ours50.pt
├── logs/baseline/<size>/yolo26<size>_vd_baseline.csv
└── logs/pruned/<size>/yolo26<size>_vd_ours50.csv
```

Dung layout nhu trong repo, nen tai ve xong la chep thang vao
`results/visdrone/` duoc — moi nguoi chi cham vao thu muc size cua minh nen
khong de len nhau.

## Logic nam trong `vd_common.py`, khong nam trong cell notebook

Cell cua notebook duoc luu trong file `.ipynb` **tren Kaggle**, nen `git clone`
khong va duoc: sua mot loi la phai bat ca 4 nguoi import lai notebook. Da tung
hong vi chuyen nay — goi resume lam cho ban notebook moi, nguoi chay lai dung
ban cu, ket qua la baseline khong duoc nhan ra va train lai tu dau 6 tieng.

Gio toan bo logic o `notebooks/share_visdrone/vd_common.py` trong repo, cell
setup `git pull` moi lan chay. Notebook chi con cau hinh va 6 dong goi ham:

```python
V.restore(BASE_NAME, OURS_NAME, PRUNED, MANUAL_LAST)
n_base = V.train(BASE_NAME, MODEL)
V.prune50(SIZE, BEST_BASE, PRUNED, RATIO)
n_ours = V.train(OURS_NAME, str(PRUNED), finetune=True, kd=True, ...)
V.report(SIZE, BASE_NAME, OURS_NAME)
```

Sua loi trong `vd_common.py` la ca nhom co ngay o lan chay sau.

Kiem tra tu dong:

```bash
python tools/test_nb_resume.py
```

## Dong goi checkpoint de chuyen phien

```bash
python tools/make_resume_zip.py m l        # mac dinh la m va l
```

Sinh `results/visdrone/upload/vd_resume_<size>.zip` voi cau truc:

```
vd_yolo26<size>/weights/best.pt     baseline da xong (nguon prune + teacher CWD)
vd_ours<size>/weights/last.pt       Ours dang train do
```

Upload thanh Kaggle dataset roi **Add Data** — khong phai dien `MANUAL_LAST`,
cell resume tu tim thay vi ten thu muc dung bang `BASE_NAME` / `OURS_NAME`.

Script tu mo lai zip, doc `train_args.name` trong tung file va bao neu lech —
tranh dung lai loi xep nham checkpoint cua size khac.

Baseline chi can `best.pt`: `epochs_of()` thay `epoch = -1` (da strip optimizer
= train xong) nen tra ve 100 va `train()` bo qua, khong goi `resume` tren mot
run da ket thuc.

## Resume thu cong (khi phien bi danh dau **failed**)

Kaggle **khong luu output** cua version bi giet vi qua gio, nen "Add Data ->
Your Work" khong thay no. Van lay lai duoc:

1. Mo version failed -> tab **Output** -> tai `last.pt` ve.
   Trinh duyet doi duoi thanh `.zip` (file `.pt` cua PyTorch von la mot zip).
   **Doi ten lai thanh `.pt`, KHONG giai nen.**
2. Upload thanh Kaggle dataset, roi **Add Data** vao notebook.
3. Neu ban upload ca thu muc va giu nguyen ten (`vd_yolo26m/weights/last.pt`)
   thi **khong can lam gi them** — cell resume tu tim thay o bat ky do sau nao.
   Neu chi upload moi file `last.pt` roi le thi dien duong dan vao cell 2:

```python
MANUAL_LAST = {
    BASE_NAME: "/kaggle/input/vd-resume/last.pt",
    OURS_NAME: "",
}
```

Resume chi can `last.pt`, khong bat buoc co `results.csv` hay `args.yaml`:
so epoch nam ngay trong checkpoint.

Gan nhieu output cung luc thi cell resume lay ban **nhieu epoch nhat**, khong
phai ban dau tien tim thay (thu tu glob khong xac dinh).

Nam truong hop tren duoc kiem tra tu dong:

```bash
python tools/test_nb_resume.py
```

## Chay tren may khac

Khong dung Kaggle thi co `scripts/run_e2e.sh`:

```bash
DATA=VisDrone.yaml TAG=vd50 ./scripts/run_e2e.sh
```

Chua co `weights/yolo26m_baseline_VisDrone.pt` thi no tu them stage baseline
truoc. Baseline tach theo dataset nen khong de len baseline VOC.
