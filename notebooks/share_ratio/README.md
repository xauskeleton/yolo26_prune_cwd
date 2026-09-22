# Chia ratio cho tung nguoi

Moi nguoi mot notebook, chay doc lap tren Kaggle 2xT4. Khong ai phai doi ai.

| Ratio | Notebook | Nguoi chay |
|---|---|---|
| 30% | `nb_r30.ipynb` | |
| 40% | `nb_r40.ipynb` | |
| 50% | `nb_r50.ipynb` | |
| 60% | `nb_r60.ipynb` | |
| 70% | `nb_r70.ipynb` | |

## Truoc khi chia

Trong so baseline da co san tren Kaggle:

```
/kaggle/input/datasets/xauduabo2/weights/yolo26m_baseline.pt
```

Moi nguoi chi can **Add Data -> dataset `xauduabo2`**. Notebook ghim san duong dan
nay; neu Kaggle doi ten mount thi no tu do lai trong `/kaggle/input`.

## Quy uoc bat buoc giong nhau giua moi nguoi

- `EPOCHS=100`, `BATCH=16`, `IMGSZ=640`, seed 0
- `USE_DDP=True` cho CA NHOM (hoac ca nhom tat) - BatchNorm khac nhau neu tron
- CWD: tau=9, kd_layers=neck, kd_warmup=5, kd_lambda=0.5

## Gom ket qua

Moi nguoi gui lai `results/e2e_manifest.json`. Nguoi gom merge cac file do vao
mot manifest chung roi:

```bash
python scripts/sweep_prune.py --ratios 0.3 0.4 0.5 0.6 0.7 --collect-only
```

Lenh do se do lai params/GFLOPs/latency tren cung mot GPU (quan trong: latency
phai do cung mot may moi so duoc), ghi `results/prune_sweep.csv` va ve
`results/prune_sweep.png`.
