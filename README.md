# YOLOv26 Channel Pruning + Knowledge Distillation

Channel pruning cho YOLOv26 object detection, ket hop DMS (Differentiable Model Scaling) va CWD (Channel-Wise Distillation) de giam kich thuoc model va tang toc inference.

## Muc luc

- [Cau truc du an](#cau-truc-du-an)
- [Pipeline tong quan](#pipeline-tong-quan)
- [Cai dat](#cai-dat)
- [1. Sparsity Training](#1-sparsity-training)
- [2. DMS - Differentiable Model Scaling](#2-dms---differentiable-model-scaling)
- [3. Channel Pruning (prune.py)](#3-channel-pruning-prunepy)
- [4. Finetune](#4-finetune)
- [5. CWD - Channel-Wise Distillation](#5-cwd---channel-wise-distillation)
- [6. Do performance (speed.py)](#6-do-performance-speedpy)
- [Chi tiet tung file](#chi-tiet-tung-file)
- [Tham khao nhanh](#tham-khao-nhanh)

---

## Cau truc du an

```
yolo/
├── prune.py                          # Script chinh: cat kenh (channel pruning)
├── finetune.py                       # Finetune model da prune
├── speed.py                          # Do GFLOPs, Params, Latency, FPS
├── dms_utils.py                      # DMS: soft mask, FLOPs profiling, extract ratios
├── cwd_loss.py                       # CWD: KL divergence per channel
│
├── ultralytics/engine/
│   ├── model.py                      # Xu ly custom args (sr, dms, cwd, finetune) → trainer
│   └── trainer.py                    # Training loop voi sr, dms, cwd
│
├── ultralytics/models/yolo/detect/
│   └── train.py                      # get_model(): DetectionModel hoac DetectionModelPruned
│
├── ultralytics/nn/
│   ├── tasks_pruned.py               # Build pruned model tu masks (DetectionModelPruned)
│   └── modules/
│       ├── block_pruned.py           # Pruned modules (C3k2Pruned, SPPFPruned, ...)
│       └── head_pruned.py            # DetectPruned head
│
├── cfg/
│   └── yolo26m.yaml                  # Config goc cua YOLOv26
│
└── weights/
    ├── best.pt                       # Model goc (teacher)
    └── pruned_div8.pt                # Model da prune
```

---

## Pipeline tong quan

```
best.pt (model goc)
  │
  ├─ [1] Sparsity Training (sr)      → ep BN gamma ve 0, chuan bi prune
  ├─ [2] DMS Search (dms)            → tim per-layer prune ratio toi uu
  │
  ├─ [3] Prune (prune.py)            → cat kenh, tao pruned model
  │
  ├─ [4] Finetune                    → recover accuracy
  ├─ [5] CWD Distillation (cwd)      → distill feature tu teacher
  │
  └─ [6] speed.py                    → do va so sanh performance
```

**Pipeline day du:**
```bash
# 1. Sparsity training (optional)
python train_sparsity.py

# 2. DMS search
python train_dms.py
python extract_dms.py

# 3. Prune
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml

# 4. Finetune + CWD
python finetune.py

# 5. Do performance
python speed.py
```

**Pipeline rut gon (khong DMS):**
```bash
python train_sparsity.py
python prune.py --weights runs/.../best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3
python finetune.py
python speed.py
```

---

## Cai dat

Du an dua tren Ultralytics source code. Cac dependency chinh:
```
torch
ultralytics
thop          # do FLOPs (speed.py)
pyyaml
```

---

## 1. Sparsity Training

Them L1 penalty len BN gamma de day kenh khong quan trong ve 0, chuan bi cho pruning.

### Cach dung

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        sr=1e-4,          # sparsity rate
    )
```

### Tham so

| Tham so | Kieu  | Mac dinh | Mo ta |
|---------|-------|----------|-------|
| `sr`    | float | None     | Sparsity rate. Dat > 0 de kich hoat |

### Cach hoat dong

1. Tat AMP
2. Build `ignore_bn_list` (BN layers khong bi ep: residual Bottleneck cv2.bn, PSABlock, parent cv1.bn)
3. Moi step backward:
   ```
   loss.backward()
   sr_actual = sr * (1 - 0.9 * epoch / total_epochs)   # decay tuyen tinh
   BN.weight.grad += sr_actual * sign(BN.weight)        # L1 penalty
   ```
4. Optimizer step truc tiep (khong scaler, khong grad clip)

### Khuyen nghi

| Truong hop | sr | Ghi chu |
|------------|-----|---------|
| Prune nhe (20-30%) | `1e-4` | On dinh, it anh huong accuracy |
| Prune vua (30-50%) | `3e-4` ~ `5e-4` | Can balance voi accuracy |
| Prune manh (>50%) | `5e-4` ~ `1e-3` | Co the mat accuracy, can finetune nhieu |

> `sr` decay tu `sr` → `0.1*sr` theo epoch de tranh ep qua manh cuoi training.

---

## 2. DMS - Differentiable Model Scaling

Tu dong tim ti le prune **toi uu cho tung layer** thong qua gradient descent. Thay vi dat 1 ti le chung, DMS hoc duoc layer nao nen prune nhieu, layer nao nen giu lai.

> Paper: "Differentiable Model Scaling using Differentiable Topk" (ICML 2024)

### Cach dung

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml",
        epochs=50,
        dms=True,
        dms_target=0.3,
    )
```

### Toan bo tham so

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `dms` | bool | `False` | Bat/tat DMS |
| `dms_target` | float | `0.3` | Ti le FLOPs muon giam (0.3 = giam 30%) |
| `dms_lambda` | float | `1.0` | Trong so resource loss |
| `dms_lr` | float | `5e-3` | Learning rate cho a params (Adam rieng) |
| `dms_freeze` | bool | `False` | True = dong bang model, chi train a params |
| `dms_importance` | str | `"gamma"` | `"gamma"` hoac `"taylor"` |

### Chi tiet tham so

**`dms_target`** - Ti le FLOPs muon giam:
| Gia tri | Y nghia |
|---------|---------|
| `0.2` | Giam 20% FLOPs - prune nhe |
| `0.3` | Giam 30% FLOPs - trung binh (khuyen nghi bat dau) |
| `0.5` | Giam 50% FLOPs - prune manh |
| `0.7` | Giam 70% FLOPs - rat manh, co the mat accuracy nhieu |

**`dms_lambda`** - Trong so resource loss:
| Gia tri | Y nghia |
|---------|---------|
| `0.1` | Uu tien accuracy, FLOPs co the khong dat target |
| `1.0` | Can bang (mac dinh) |
| `5.0` | Ep manh ve target, co the hy sinh accuracy |

> Resource loss = `log(r_effective / r_target)` khi effective > target, nguoc lai = 0

**`dms_lr`** - Learning rate cho a params:
| Gia tri | Y nghia |
|---------|---------|
| `1e-3` | Hoc cham, on dinh |
| `5e-3` | Mac dinh, can bang |
| `1e-2` | Nhanh, tot khi `dms_freeze=True` |

> a params dung Adam optimizer RIENG, khong chung scheduler voi model.

**`dms_freeze`**:
- `False`: Train ca model weights + a params (chinh xac hon, lau hon)
- `True`: Dong bang model, chi train a params (nhanh, tot cho search ratio only)

**`dms_importance`**:
| Mode | Cong thuc | Uu/nhuoc |
|------|-----------|----------|
| `"gamma"` | `\|BN.weight\|` | Nhanh, don gian |
| `"taylor"` | `(mask * grad)^2` voi EMA 0.99 | Chinh xac hon, can vai epoch de on dinh |

### Cach hoat dong

Voi moi BN layer co the prune:
```
1. Tao a = Parameter(init=dms_target)
2. Forward hook:
   importance = |gamma|
   c' = rank(importance) / N          # chuan hoa [0,1], detached
   mask = Sigmoid(N * (c' - a))       # grad chay qua a
   output = BN(x) * mask
3. Resource loss = f(effective_FLOPs, target_FLOPs)
4. total_loss = det_loss + lambda * resource_loss
5. Clamp a trong [0.05, 0.95] moi step
```

FLOPs tinh chinh xac per-conv:
- Regular conv: `original * (1-a_in) * (1-a_out)`
- Depthwise: `original * (1-a_out)`
- Concat input: weighted average retention theo so kenh

### Extract ratios sau DMS

```python
from dms_utils import extract_ratios_from_checkpoint

ratios = extract_ratios_from_checkpoint(
    "runs/detect/train/weights/best.pt",
    "dms_ratios.yaml",
)
```

Dung voi prune.py:
```bash
python prune.py --weights runs/.../best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml
```

### Vi du

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # Search nhanh (freeze model)
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=30,
        dms=True, dms_target=0.3, dms_lambda=1.0,
        dms_freeze=True, dms_lr=1e-2,
    )

    # Search + train (chinh xac hon)
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=50,
        dms=True, dms_target=0.3, dms_lambda=1.0,
        dms_lr=5e-3,
    )

    # Prune manh 50%
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=50,
        dms=True, dms_target=0.5, dms_lambda=2.0,
    )

    # Taylor importance
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=80,
        dms=True, dms_target=0.3,
        dms_importance="taylor",
    )
```

### Log

```
[DMS] Epoch 10: avg_a=0.2847, min=0.0500, max=0.6234
```
- `avg_a` tien dan ve `dms_target`
- `min/max` cho thay su phan bo: layer nao giu nhieu (min), layer nao prune nhieu (max)

---

## 3. Channel Pruning (prune.py)

Cat kenh dua tren BN gamma (sau sparsity training) hoac per-layer ratio (tu DMS).

### CLI Arguments

| Argument | Kieu | Mac dinh | Mo ta |
|----------|------|----------|-------|
| `--weights` | str | `weights/best.pt` | Duong dan model .pt (da train hoac sau sparsity) |
| `--cfg` | str | `ultralytics/cfg/models/26/yolo26.yaml` | Duong dan YAML config goc cua model |
| `--model-size` | str | `m` | Kich thuoc model: `n`, `s`, `m`, `l`, `x` |
| `--prune-ratio` | float | `0.5` | Ti le prune chung (0.0 - 1.0). Moi layer prune % kenh nay |
| `--layer-ratio` | str | `None` | Duong dan YAML file chua custom ratio cho tung layer |
| `--divisor` | int | `8` | Channels phai chia het cho gia tri nay (`8` hoac `16`) |
| `--save-dir` | str | `weights` | Thu muc luu model da prune |

### Chi tiet tham so

**`--weights`** - Model input:
- Model goc (best.pt) sau khi train hoac sau sparsity training
- Model phai co BN layers de prune

**`--cfg`** - YAML config goc:
- Dung de build cau truc pruned model
- Phai match voi model architecture (yolo26.yaml cho YOLOv26)

**`--model-size`** - Xac dinh scale parameters:
| Size | Depth | Width | Max channels |
|------|-------|-------|-------------|
| `n` | 0.50 | 0.25 | 1024 |
| `s` | 0.50 | 0.50 | 1024 |
| `m` | 0.50 | 1.00 | 512 |
| `l` | 1.00 | 1.00 | 512 |
| `x` | 1.00 | 1.50 | 512 |

> Scale values lay tu YAML file, bang tren la vi du cho yolo26.

**`--prune-ratio`** - Ti le cat kenh:
| Gia tri | Y nghia |
|---------|---------|
| `0.2` | Nhe - giu 80% kenh |
| `0.3` | Vua phai - giu 70% kenh |
| `0.5` | Manh - giu 50% kenh |
| `0.7` | Rat manh - giu 30% kenh, can finetune ky |

> Moi layer duoc prune doc lap theo local threshold (per-layer structured pruning).
> Divisor dam bao moi layer giu it nhat `divisor` channels.

**`--layer-ratio`** - Custom ratio per-layer:
File YAML voi matching theo thu tu uu tien:
```yaml
# 1. Exact match (cao nhat)
model.0.bn: 0.1
model.2.cv1.bn: 0.2

# 2. Layer index
model.0: 0.1        # tat ca BN trong layer 0
model.15: 0.6       # layer 15

# 3. Group (thap nhat)
backbone: 0.3       # model.0 - model.9
head: 0.5           # model.10 - model.21
detect: 0.4         # model.22+
```

**`--divisor`** - Divisibility constraint:
| Gia tri | Su dung |
|---------|---------|
| `8` | GPU thong thuong (mac dinh) |
| `16` | GPU voi Tensor Cores (A100, RTX 30xx/40xx) |

### Cach dung

```bash
# Co ban - ti le prune chung
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Day du - tat ca options
python prune.py \
    --weights weights/best.pt \
    --cfg cfg/yolo26m.yaml \
    --model-size m \
    --prune-ratio 0.4 \
    --divisor 8 \
    --save-dir weights/

# Per-layer ratio tu DMS
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml

# Voi custom layer ratio
python prune.py \
    --weights weights/best.pt \
    --cfg cfg/yolo26m.yaml \
    --prune-ratio 0.3 \
    --layer-ratio layer_ratio.yaml
```

### Workflow noi bo

```
1. Thu thap BN layers
2. Build ignore list (Bottleneck residual + PSABlock)
3. Validate prune ratio
4. Tao pruned YAML (map C3k2 → C3k2Pruned, Detect → DetectPruned, ...)
5. Tao masks:
   - Tinh per-layer local threshold
   - Lam tron channels → boi so cua divisor
   - Tao top-k mask
6. Validate divisibility (tat ca layers chia het cho divisor)
7. Build DetectionModelPruned tu masks
8. Copy weights tu original → pruned (xu ly: Conv, BN, Detect, SPPF, chunk, concat)
9. Test forward pass
10. Save checkpoint (model + maskbndict + config)
```

### Output

File `.pt` chua:
```python
{
    "model": pruned_model,        # DetectionModelPruned
    "maskbndict": maskbndict,     # Dict {bn_name: mask_tensor}
    "config": {
        "divisor": 8,
        "prune_ratio": 0.3,
    }
}
```

---

## 4. Finetune

Recover accuracy cho model da prune. **Bat buoc** truyen `finetune=True` de trainer rebuild model dung cach qua `DetectionModelPruned`.

### Cach hoat dong

```
model.train(finetune=True)
    → model.py: pop finetune, lay maskbndict tu checkpoint
    → detect/train.py get_model(maskbndict=...):
        finetune=True → DetectionModelPruned(maskbndict, cfg, nc)
            → parse_model_pruned: dung maskbndict tinh so channel thuc moi layer
            → Build: C3k2Pruned, SPPFPruned, DetectPruned, ...
            → Tinh lai stride dung cach
        → model.load(weights) qua intersect_dicts
```

### Tham so

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `finetune` | bool | `False` | **BAT BUOC** khi train pruned model |

> Khi `finetune=True`, trainer doc `maskbndict` tu checkpoint de build dung `DetectionModelPruned`.

### Finetune don gian

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")
    model.train(data="VOC.yaml", epochs=50, finetune=True)
```

> Luu y: Tren Windows **bat buoc** co `if __name__ == '__main__':` de tranh loi multiprocessing spawn.

Dung AMP + scaler + grad clip binh thuong cua Ultralytics.

### Finetune + CWD

Xem [muc 5](#5-cwd---channel-wise-distillation) ben duoi.

---

## 5. CWD - Channel-Wise Distillation

Student (pruned) hoc lai feature tu teacher (original) qua knowledge distillation. CWD distill feature maps tai tung layer, hieu qua hon logit distillation.

> Paper: "Channel-wise Knowledge Distillation for Dense Prediction" (arXiv:2011.13256)

### Cach dung

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")      # student (pruned)
    model.train(
        data="VOC.yaml",
        epochs=50,
        finetune=True,                            # bat buoc cho pruned model
        cwd=True,
        cwd_teacher="weights/best.pt",            # teacher (original)
    )
```

### Toan bo tham so

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `cwd` | bool | `False` | Bat/tat CWD |
| `cwd_teacher` | str | `None` | Path teacher model (**BAT BUOC**) |
| `cwd_lambda` | float | `0.5` | Trong so CWD loss |
| `cwd_temperature` | float/str | `6.0` | Temperature, hoac `"dynamic"` |
| `tau_max` | float | `10.0` | Tau cao nhat khi dynamic (dau + cuoi training) |
| `tau_min` | float | `1.0` | Tau thap nhat khi dynamic (giua training) |
| `cwd_layers` | str | `"neck"` | `"neck"` hoac `"all"` |
| `cwd_layer_weights` | dict/None | `None` | Trong so rieng tung layer |

### Chi tiet tham so

**`cwd_teacher`** (bat buoc):
- Path den checkpoint model goc (chua prune)
- Teacher duoc load → freeze → eval mode
- Forward moi batch (no grad) de lay feature maps

**`cwd_lambda`** - Trong so CWD loss:

| Gia tri | Y nghia |
|---------|---------|
| `0.1` | CWD nhe, uu tien detection loss |
| `0.5` | Can bang (mac dinh) |
| `1.0` | CWD manh, student giong teacher hon |
| `2.0` | Rat manh, co the lam slow convergence |

> `total_loss = detection_loss + cwd_lambda * cwd_loss`

**`cwd_temperature`** - Temperature cho spatial softmax:

*Fixed:*
| Gia tri | Y nghia |
|---------|---------|
| `1.0` | Sharp - phai match chinh xac, kho hoi tu |
| `4.0` | Vua phai |
| `6.0` | Mac dinh - soft, de hoc |
| `10.0` | Rat soft - de hoc nhung mat chi tiet |

*Dynamic:* truyen string `"dynamic"`
```
tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + cos(pi * epoch / total_epochs))
```
- Mac dinh: `tau_max=10.0`, `tau_min=1.0` → cosine: 10 → 1 → 10
- Tuy chinh bang `tau_max` va `tau_min`
- Khuyen nghi cho training dai (>50 epochs)

**`tau_max`** va **`tau_min`** - Gioi han dynamic temperature:

Chi co tac dung khi `cwd_temperature="dynamic"`.
| Gia tri | Y nghia |
|---------|---------|
| `tau_max=10.0` | Mac dinh. Tau cao nhat (dau + cuoi training, soft) |
| `tau_min=1.0` | Mac dinh. Tau thap nhat (giua training, sharp) |
| `tau_max=8.0, tau_min=2.0` | Range hep hon, it bien dong |
| `tau_max=15.0, tau_min=0.5` | Range rong, bien dong nhieu |

**`cwd_layers`** - Layers de distill:

| Mode | Layer indices | Mo ta |
|------|---------------|-------|
| `"neck"` | [13, 16, 19, 22] | Neck + detect input (mac dinh, nhanh) |
| `"all"` | [2, 4, 6, 8, 13, 16, 19, 22] | Backbone + neck (chinh xac hon, cham hon) |

**`cwd_layer_weights`** - Trong so rieng tung layer:

Mac dinh tat ca = 1.0. Truyen dict `{layer_index: weight}`:
```python
cwd_layer_weights = {
    13: 0.5,   # P3 - low-level
    16: 1.0,   # P4
    19: 1.0,   # P5
    22: 1.5,   # detect input - quan trong nhat
}
```

### Cach hoat dong

```
Moi training step:
  1. Forward student (co grad)
  2. Forward teacher (no grad) → feature maps
  3. Voi moi hooked layer:
     - Align channels qua maskbndict (khong can adapter 1x1)
     - Spatial softmax per channel: phi(y_c) = softmax(y_c / tau)
     - CWD loss = (tau^2 / C) * sum KL(teacher || student)
  4. total_loss = det_loss + cwd_lambda * avg(cwd_losses)
```

Channel alignment: `maskbndict` (tu prune checkpoint) chi ra teacher kenh nao duoc giu → `teacher_feat[:, mask, :, :]`.

### Vi du

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # CWD co ban
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=50, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt", cwd_lambda=0.5,
    )

    # Dynamic temperature (mac dinh tau_max=10, tau_min=1)
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt", cwd_lambda=0.5,
        cwd_temperature="dynamic",
    )

    # Dynamic temperature voi custom range
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt", cwd_lambda=0.5,
        cwd_temperature="dynamic",
        tau_max=8.0, tau_min=2.0,
    )

    # Distill tat ca layers
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=50, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt", cwd_lambda=0.5,
        cwd_layers="all",
    )

    # Custom layer weights
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=50, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt", cwd_lambda=0.5,
        cwd_layer_weights={13: 0.5, 16: 1.0, 19: 1.0, 22: 1.5},
    )

    # Full config (all layers + dynamic + custom weights + custom tau range)
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt",
        cwd_lambda=0.5,
        cwd_temperature="dynamic",
        tau_max=10.0, tau_min=1.0,
        cwd_layers="all",
        cwd_layer_weights={
            2: 0.3, 4: 0.3, 6: 0.5, 8: 0.5,
            13: 1.0, 16: 1.0, 19: 1.0, 22: 1.5,
        },
    )
```

### Log

```
[CWD] Epoch 10: tau=6.00       # fixed
[CWD] Epoch 10: tau=8.45       # dynamic
```

---

## 6. Do performance (speed.py)

So sanh GFLOPs, Params, Latency, FPS giua model goc va model da prune.

### Cach dung

```bash
python speed.py
```

Sua `orig_path` va `pruned_path` trong `speed.py`:
```python
orig_path = "weights/best.pt"
pruned_path = "weights/pruned_div8.pt"
```

### Output

```
============================================================
Metric          | Original (Best) | Pruned (Div8)   | Giam (%)
------------------------------------------------------------
GFLOPs          |           38.80 |           25.16 |    35.1%
Params (M)      |           20.07 |           12.45 |    38.0%
Latency (ms)    |            8.23 |            5.67 |    31.1%
FPS             |           121.5 |           176.4 |    45.2%
============================================================
```

### Cach hoat dong

1. Load model qua `YOLO(path)` → chuyen sang CUDA
2. Tinh GFLOPs va Params bang `thop.profile()` (chinh xac voi pruned model)
3. Warm-up 30 lan
4. Do latency 100 lan voi `torch.cuda.Event` → tinh trung binh
5. FPS = 1000 / latency_ms

### Yeu cau

- GPU CUDA (chay tren CPU se khong chinh xac)
- Thu vien `thop` (`pip install thop`)

---

## Chi tiet tung file

### prune.py

**Muc dich:** Script CLI chinh de thuc hien channel pruning.

**Workflow:**
1. Load model tu `--weights`
2. Thu thap BN layers, build ignore list (residual Bottleneck, PSABlock)
3. Voi moi BN layer:
   - Lay ratio (tu `--layer-ratio` hoac `--prune-ratio`)
   - Sap xep BN gamma theo do lon
   - Tinh local threshold, tao mask
   - Lam tron channels → boi so cua `--divisor`
4. Build pruned YAML config (map C3k2 → C3k2Pruned, etc.)
5. Build `DetectionModelPruned`, copy weights, validate, save

**Input:** Model .pt + YAML config
**Output:** Pruned model .pt (model + maskbndict + config)

---

### finetune.py

**Muc dich:** Script de finetune model da prune (co hoac khong co CWD).

**Noi dung hien tai:**
```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="coco8.yaml", epochs=5,
        finetune=True,
        cwd=False, cwd_teacher="yolo26m.pt",
        cwd_lambda=0.5,
        cwd_temperature="dynamic",
        tau_max=10.0, tau_min=1.0,
        cwd_layers="all",
        cwd_layer_weights={
            2: 0.3, 4: 0.3, 6: 0.5, 8: 0.5,
            13: 1.0, 16: 1.0, 19: 1.0, 22: 1.5,
        },
    )
```

**Luu y:**
- **Tren Windows:** `if __name__ == '__main__':` la bat buoc (tranh loi multiprocessing spawn)
- `finetune=True` bat buoc de model rebuild dung `DetectionModelPruned`
- `data` phai co `nc` khop voi so class cua model da prune
- Co the bat/tat CWD bang `cwd=True/False`

---

### dms_utils.py

**Muc dich:** Cac ham utility cho DMS va pruning, duoc dung boi ca `prune.py` va `trainer.py`.

**Cac ham chinh:**

| Ham | Muc dich | Dung boi |
|-----|----------|----------|
| `make_divisible_channels(ch, max_ch, div)` | Lam tron channels → boi so cua divisor | `prune.py` |
| `get_layer_ratio(name, cfg, default)` | Lay prune ratio cho layer (exact → index → group → default) | `prune.py` |
| `build_pruned_yaml(cfg, size, nc)` | Tao pruned YAML config tu YAML goc | `prune.py` |
| `build_ignore_bn_list(model)` | List BN layers khong bi prune (residual, PSABlock) | `prune.py`, `trainer.py` |
| `make_soft_mask_hook(bn, a, imp, taylor)` | Tao DMS forward hook voi differentiable soft mask | `trainer.py` (DMS) |
| `profile_per_layer_flops(model, imgsz, dev)` | Do FLOPs tung Conv2d layer | `trainer.py` (DMS) |
| `build_conv_bn_mapping(model, ignore)` | Map Conv → BN de tinh exact FLOPs | `trainer.py` (DMS) |
| `compute_resource_loss(a, flops, ...)` | GFLOPs constraint loss (differentiable) | `trainer.py` (DMS) |
| `compute_l1_loss(model, ignore)` | L1 penalty tren BN gamma | `trainer.py` (SR) |
| `extract_ratios_from_checkpoint(ckpt, out)` | Extract DMS a params → YAML file | User script |

**Chi tiet cac ham quan trong:**

`build_ignore_bn_list(model)`:
- Bottleneck co `add=True` (residual): ignore `cv2.bn` + parent `cv1.bn`
- PSABlock: ignore tat ca BN ben trong + parent layer `cv1.bn`

`make_soft_mask_hook(bn_name, a_params, importance, taylor_buffers)`:
- Tao forward hook cho DMS
- Gamma mode: importance = |BN.weight|
- Taylor mode: importance = (mask * grad)^2 voi EMA 0.99
- Soft mask: `Sigmoid(N * (rank(importance)/N - a))`
- Gradient chi chay qua `a` (rank la detached)

`compute_resource_loss(a_params, conv_flops, conv_bn_map, bn_channels, total_flops, target_ratio)`:
- Tinh exact FLOPs sau pruning cho moi Conv2d
- Regular: `flops * (1-a_in) * (1-a_out)`
- Depthwise: `flops * (1-a_out)`
- Concat input: weighted average retention
- Loss = `log(r_e / r_t)` neu `r_e > r_t`, nguoc lai 0

`extract_ratios_from_checkpoint(ckpt_path, save_path)`:
- Doc `dms_a_params` tu checkpoint
- Clamp a trong [0.01, 0.95]
- Luu ra YAML file de dung voi `prune.py --layer-ratio`

---

### cwd_loss.py

**Muc dich:** Implement CWD (Channel-Wise Distillation) loss va cac utility hook.

**Cac class/ham:**

| Class/Ham | Muc dich |
|-----------|----------|
| `CWDLoss` | Tinh KL divergence per channel (paper Eq 4-5) |
| `FeatureHook` | Forward hook de bat feature maps |
| `setup_hooks(model, layer_names)` | Dang ky hooks tren cac layers |
| `build_cwd_channel_masks(maskbndict, indices)` | Tao boolean masks cho channel alignment |
| `compute_cwd_loss(s_hooks, t_hooks, ...)` | Tinh weighted CWD loss qua tat ca layers |

**CWDLoss.forward(student, teacher, temperature):**
```
Input: student [B,C,H,W], teacher [B,C,H,W], tau
1. Flatten spatial: [B, C, H*W]
2. Spatial softmax per channel: softmax(x / tau, dim=spatial)
3. KL divergence: KL(teacher_soft || student_soft)
4. Scale: (tau^2 / C) * mean(KL)
```

**build_cwd_channel_masks(maskbndict, layer_indices):**
- Dung `maskbndict` de tim mask cua `model.{i}.cv2.bn`
- Chuyen sang boolean mask
- Dung de chon dung teacher channels tuong ung voi student

**compute_cwd_loss(student_hooks, teacher_hooks, criterion, channel_masks, layer_weights, temperature):**
- Voi moi layer: align channels (mask) → interpolate spatial size → tinh CWD loss
- Weighted average theo `layer_weights`

---

### speed.py

**Muc dich:** Do va so sanh performance giua model goc va model da prune.

**Cach hoat dong:**
1. Load model → CUDA
2. `thop.profile()` → GFLOPs, Params
3. Warm-up 30 lan → do latency 100 lan → trung binh
4. In bang so sanh

**Sua doi:**
- Thay `orig_path` va `pruned_path` bang duong dan model thuc te

---

### ultralytics/engine/model.py (da chinh sua)

**Chinh sua:** Xu ly custom args truoc khi truyen vao Ultralytics trainer.

**Cac args duoc pop:**
```python
sr = args.pop("sr", None)
dms = args.pop("dms", False)
dms_target = args.pop("dms_target", 0.3)
dms_lambda = args.pop("dms_lambda", 1.0)
dms_lr = args.pop("dms_lr", 5e-3)
dms_freeze = args.pop("dms_freeze", False)
dms_importance = args.pop("dms_importance", "gamma")
finetune = args.pop("finetune", False)
cwd = args.pop("cwd", False)
cwd_teacher = args.pop("cwd_teacher", None)
cwd_lambda = args.pop("cwd_lambda", 0.5)
cwd_temperature = args.pop("cwd_temperature", 6.0)
tau_max = args.pop("tau_max", 10.0)
tau_min = args.pop("tau_min", 1.0)
cwd_layers = args.pop("cwd_layers", "neck")
cwd_layer_weights = args.pop("cwd_layer_weights", None)
```

**Ly do pop:** Ultralytics se bao loi neu gap tham so la. Pop truoc, gan vao trainer sau.

**Flow:**
1. Pop tat ca custom args
2. Goi Ultralytics setup binh thuong
3. Gan custom args vao `self.trainer.*`
4. Lay `maskbndict` tu checkpoint (neu co)
5. Truyen `maskbndict` vao `get_model()` (neu `finetune=True`)

---

### ultralytics/engine/trainer.py (da chinh sua)

**Chinh sua:** Training loop ho tro SR, DMS, CWD.

**SR (Sparsity Training):**
- Tat AMP khi `sr > 0`
- Build `ignore_bn_list`
- Sau `loss.backward()`: them L1 gradient len BN gamma (voi decay)
- Bo qua scaler va grad clip

**DMS (Differentiable Model Scaling):**
- Tat AMP
- Tao `a_params` (nn.Parameter) cho moi prunable BN
- Dang ky forward hooks (soft mask)
- Adam optimizer rieng cho a_params
- Them `resource_loss` vao total loss
- Clamp a trong [0.05, 0.95]

**CWD (Channel-Wise Distillation):**
- Load teacher model → freeze → eval
- Dang ky hooks tren student va teacher
- Build channel masks tu maskbndict
- Moi step: forward teacher (no grad) → tinh CWD loss → cong vao total loss
- Tat AMP va grad clip

---

### ultralytics/models/yolo/detect/train.py (da chinh sua)

**Chinh sua:** `get_model()` nhan them `maskbndict` arg.

**Logic:**
- `maskbndict is None` → build `DetectionModel` binh thuong
- `maskbndict is not None` → build `DetectionModelPruned(maskbndict, cfg, nc)`

---

### ultralytics/nn/tasks_pruned.py

**Muc dich:** `DetectionModelPruned` - build pruned model tu `maskbndict`.

**`parse_model_pruned(maskbndict, cfg)`:**
- Doc cau truc tu pruned YAML
- Dung `maskbndict` de tinh so channels thuc cho moi layer
- Map module types: C3k2Pruned, SPPFPruned, C2PSAPruned, DetectPruned
- Tao `current_to_prev` mapping (de copy weights dung)

---

### ultralytics/nn/modules/block_pruned.py

**Muc dich:** Cac module pruned tuong ung voi block.py goc.

**Modules:**
| Module | Goc | Mo ta |
|--------|-----|-------|
| `BottleneckPruned` | `Bottleneck` | Bottleneck voi so channels tuy bien |
| `C3kPruned` | `C3k` | cv1/cv2 parallel + cv3 concat |
| `C3k2Pruned` | `C3k2` | Chunk + N bottlenecks + concat + cv2 |
| `C3k2PrunedAttn` | `C3k2` (attn=True) | Sequential(Bottleneck, PSABlock) |
| `SPPFPruned` | `SPPF` | MaxPool pyramid, dynamic n param |
| `C2PSAPruned` | `C2PSA` | C2PSA voi pruned channels |

---

### ultralytics/nn/modules/head_pruned.py

**Muc dich:** `DetectPruned` - detection head cho pruned model.

**Khac biet voi Detect goc:**
- Nhan so channels tu `maskbndict` (khong tu tinh)
- Khong dung `max_det` hardcode
- `nc` duoc lay tu model goc luc prune

---

## Tham khao nhanh

### Bang tham so

| Nhom | Tham so | Mac dinh | Range khuyen nghi |
|------|---------|----------|-------------------|
| **Finetune** | `finetune` | False | `True` khi train pruned model |
| **SR** | `sr` | None | `1e-4` ~ `5e-4` |
| **DMS** | `dms_target` | 0.3 | 0.2 ~ 0.5 |
| | `dms_lambda` | 1.0 | 0.5 ~ 5.0 |
| | `dms_lr` | 5e-3 | 1e-3 ~ 1e-2 |
| | `dms_freeze` | False | True (nhanh) / False (chinh xac) |
| | `dms_importance` | "gamma" | "gamma" (nhanh) / "taylor" (chinh xac) |
| **CWD** | `cwd_lambda` | 0.5 | 0.3 ~ 1.0 |
| | `cwd_temperature` | 6.0 | 4.0 ~ 10.0 hoac "dynamic" |
| | `tau_max` | 10.0 | 5.0 ~ 15.0 (chi khi dynamic) |
| | `tau_min` | 1.0 | 0.5 ~ 3.0 (chi khi dynamic) |
| | `cwd_layers` | "neck" | "neck" (nhanh) / "all" (chinh xac) |
| | `cwd_layer_weights` | None | dict {layer_idx: weight} |
| **Prune** | `--prune-ratio` | 0.5 | 0.2 ~ 0.7 |
| | `--divisor` | 8 | 8 (GPU) / 16 (Tensor Cores) |
| | `--model-size` | m | n / s / m / l / x |

### Luu y quan trong

1. **`finetune=True` bat buoc** khi train pruned model (rebuild qua `DetectionModelPruned`).
2. **Khong ket hop** sr, dms, cwd cung luc. Chi dung 1 che do moi lan train.
3. **AMP tu dong tat** khi dung sr, dms, hoac cwd.
4. **Gradient clip tu dong tat** khi dung sr, dms, hoac cwd.
5. **DMS can extract ratios** truoc khi prune (`extract_ratios_from_checkpoint`).
6. **CWD can teacher model** chua prune (original best.pt).
7. **CWD tu dong align channels** qua `maskbndict` - khong can chinh tay.
8. **Tat ca che do ho tro resume** training tu checkpoint.
9. **Tren Windows** phai co `if __name__ == '__main__':` trong script de tranh loi multiprocessing.
10. **`data` YAML phai co `nc` khop** voi so class cua model (pruned model giu nguyen `nc` tu model goc).

### Checkpoint luu them

| Che do | Du lieu luu |
|--------|-------------|
| Prune | `model`, `maskbndict`, `config` (divisor, prune_ratio) |
| DMS | `dms_a_params`, `dms_optimizer` |
| CWD | `cwd_state` (teacher, lambda, temp, layers), `maskbndict` |
