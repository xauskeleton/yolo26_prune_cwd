# YOLOv26 Channel Pruning + Knowledge Distillation

Channel pruning cho YOLOv26 object detection, ket hop DMS (Differentiable Model Scaling) va CWD (Channel-Wise Distillation) de giam kich thuoc model va tang toc inference.

## Muc luc

**Phan I — Cach dung**
- [Pipeline tong quan](#pipeline-tong-quan)
- [Cai dat](#cai-dat)
- [1. Sparsity Training](#1-sparsity-training)
- [2. DMS - Differentiable Model Scaling](#2-dms---differentiable-model-scaling)
- [3. Channel Pruning](#3-channel-pruning)
- [4. Finetune](#4-finetune)
- [5. CWD - Channel-Wise Distillation](#5-cwd---channel-wise-distillation)
- [6. Do performance](#6-do-performance)
- [Bang tham khao nhanh](#bang-tham-khao-nhanh)

**Phan II — Chi tiet ky thuat**
- [Cau truc du an](#cau-truc-du-an)
- [Sparsity Training — Ben trong](#sparsity-training--ben-trong)
- [DMS — Ben trong](#dms--ben-trong)
- [Pruning — Ben trong](#pruning--ben-trong)
- [Finetune — Ben trong](#finetune--ben-trong)
- [CWD — Ben trong](#cwd--ben-trong)
- [Chi tiet tung file](#chi-tiet-tung-file)

---

# Phan I — Cach dung

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
# 1. Sparsity training
python train_sparsity.py

# 2. DMS search → extract ratios
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

**Pipeline L1 norm (don gian nhat, khong can SR):**
```bash
# Prune truc tiep tu pretrained model
python prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5

# Finetune (co the kem CWD learnable tau)
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

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml",
        epochs=100,
        batch=16,
        imgsz=640,
        sr=1e-4,
    )
```

### Tham so

| Tham so | Kieu  | Mac dinh | Mo ta |
|---------|-------|----------|-------|
| `sr`    | float | None     | Sparsity rate. Dat > 0 de kich hoat |

### Khuyen nghi

| Muc tieu prune | sr | Ghi chu |
|----------------|-----|---------|
| Nhe (20-30%) | `1e-4` | On dinh, it anh huong accuracy |
| Vua (30-50%) | `3e-4` ~ `5e-4` | Can balance voi accuracy |
| Manh (>50%) | `5e-4` ~ `1e-3` | Co the mat accuracy, can finetune nhieu |

> `sr` decay tu dong: `sr * (1 - 0.9 * epoch/epochs)`, tranh ep qua manh cuoi training.

---

## 2. DMS - Differentiable Model Scaling

Tu dong tim ti le prune toi uu **cho tung layer** thong qua gradient descent. Thay vi dat 1 ti le chung, DMS hoc duoc layer nao nen prune nhieu, layer nao nen giu lai.

> Paper: "Differentiable Model Scaling using Differentiable Topk" (ICML 2024)

### Cach dung co ban

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

### Tham so

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `dms` | bool | `False` | Bat/tat DMS |
| `dms_target` | float | `0.3` | Ti le FLOPs muon giam (0.3 = giam 30%) |
| `dms_lambda` | float | `1.0` | Trong so resource loss |
| `dms_lr` | float | `5e-3` | Learning rate cho a params (Adam rieng) |
| `dms_freeze` | bool | `False` | True = dong bang model, chi train a params |
| `dms_importance` | str | `"gamma"` | `"gamma"`, `"taylor"`, hoac `"l1"` |
| `dms_warmup` | int | `0` | So epoch warmup truoc khi DMS bat dau |

**`dms_target`** — Ti le FLOPs muon giam:

| Gia tri | Y nghia |
|---------|---------|
| `0.2` | Prune nhe |
| `0.3` | Trung binh (khuyen nghi bat dau) |
| `0.5` | Prune manh |
| `0.7` | Rat manh, co the mat accuracy nhieu |

**`dms_lambda`** — Trong so resource loss:

| Gia tri | Y nghia |
|---------|---------|
| `0.1` | Uu tien accuracy, FLOPs co the khong dat target |
| `1.0` | Can bang (mac dinh) |
| `5.0` | Ep manh ve target, co the hy sinh accuracy |

**`dms_freeze`** — Dong bang model:
- `False`: Train ca model weights + a params (chinh xac hon, lau hon)
- `True`: Dong bang model, chi train a params (nhanh, tot cho search ratio only)

**`dms_importance`** — Cach tinh do quan trong cua kenh:

| Mode | Uu/nhuoc |
|------|----------|
| `"gamma"` | Nhanh, don gian. Dung `\|BN.weight\|` |
| `"taylor"` | Chinh xac hon, can vai epoch de on dinh. Dung `(mask * grad)^2` voi EMA |
| `"l1"` | L1 norm cua Conv filter. Consistent voi L1 norm pruning, khong can SR training |

**`dms_warmup`** — So epoch warmup:
- Trong warmup: model train binh thuong, `a` params dong bang, khong co resource loss
- Sau warmup: resource loss ramp up linearly trong 5 epochs tiep theo
- Khuyen nghi: `2` ~ `5` epoch de model on dinh truoc khi DMS bat dau

### Extract ratios sau DMS

```python
from dms_utils import extract_ratios_from_checkpoint

ratios = extract_ratios_from_checkpoint(
    "runs/detect/train/weights/best.pt",
    "weights/dms_ratios.yaml",
)
```

Dung voi prune.py:
```bash
python prune.py --weights runs/.../best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml
```

### Vi du nang cao

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
        dms=True, dms_target=0.3, dms_lr=5e-3,
    )

    # Taylor importance
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=80,
        dms=True, dms_target=0.3, dms_importance="taylor",
    )

    # L1 norm importance + warmup (consistent voi L1 norm pruning)
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=50,
        dms=True, dms_target=0.3,
        dms_importance="l1", dms_warmup=3,
    )
```

### Log

```
[DMS] Epoch 0: WARMUP (1/3), a params frozen
[DMS] Epoch 1: WARMUP (2/3), a params frozen
[DMS] Epoch 2: WARMUP (3/3), a params frozen
[DMS] Epoch 3: avg_a=0.0034, min=0.0000, max=0.0100, ramp=0.20
[DMS] Epoch 10: avg_a=0.2847, min=0.0500, max=0.6234
```
- Warmup: a params dong bang, model train binh thuong
- `avg_a` tien dan ve `dms_target`
- `min/max` cho thay su phan bo: layer nao giu nhieu (min), layer nao prune nhieu (max)
- `ramp`: resource loss ramp up dan tu 0 → 1 trong 5 epoch sau warmup

---

## 3. Channel Pruning

Cat kenh dua tren BN gamma (`prune.py`) hoac L1 norm cua Conv filter (`prune_l1norm.py`).
L1 norm khong can Sparsity Training, co the dung truc tiep tren pretrained model.

### CLI Arguments

| Argument | Kieu | Mac dinh | Mo ta |
|----------|------|----------|-------|
| `--weights` | str | `weights/best.pt` | Duong dan model .pt |
| `--cfg` | str | `ultralytics/cfg/models/26/yolo26.yaml` | Duong dan YAML config |
| `--model-size` | str | `m` | Kich thuoc: `n`, `s`, `m`, `l`, `x` |
| `--prune-ratio` | float | `0.5` | Ti le prune chung (0.0 - 1.0) |
| `--layer-ratio` | str | `None` | YAML file chua custom ratio per-layer |
| `--divisor` | int | `8` | Channels chia het cho gia tri nay |
| `--save-dir` | str | `weights` | Thu muc luu output |

### Cach dung

```bash
# Co ban - ti le prune chung
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Day du
python prune.py \
    --weights weights/best.pt \
    --cfg cfg/yolo26m.yaml \
    --model-size m \
    --prune-ratio 0.4 \
    --divisor 8 \
    --save-dir weights/

# Per-layer ratio tu DMS
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml

# L1 norm pruning (khong can Sparsity Training)
python prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# L1 norm + per-layer ratio tu DMS
python prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml
```

**`--prune-ratio`**:

| Gia tri | Y nghia |
|---------|---------|
| `0.2` | Nhe - giu 80% kenh |
| `0.3` | Vua phai - giu 70% kenh |
| `0.5` | Manh - giu 50% kenh |
| `0.7` | Rat manh - giu 30% kenh |

**`--layer-ratio`** — File YAML voi matching theo thu tu uu tien:
```yaml
# 1. Exact match (cao nhat)
model.0.bn: 0.1

# 2. Layer index
model.0: 0.1

# 3. Group (thap nhat)
backbone: 0.3       # model.0 - model.9
head: 0.5           # model.10 - model.21
detect: 0.4         # model.22+
```

**`--divisor`**:

| Gia tri | Su dung |
|---------|---------|
| `8` | GPU thong thuong (mac dinh) |
| `16` | GPU voi Tensor Cores (A100, RTX 30xx/40xx) |

**`--model-size`**:

| Size | Depth | Width | Max channels |
|------|-------|-------|-------------|
| `n` | 0.50 | 0.25 | 1024 |
| `s` | 0.50 | 0.50 | 1024 |
| `m` | 0.50 | 1.00 | 512 |
| `l` | 1.00 | 1.00 | 512 |
| `x` | 1.00 | 1.50 | 512 |

### Output

File `.pt` chua: `model` (DetectionModelPruned), `maskbndict` (dict mask per BN layer), `config` (divisor, prune_ratio).

---

## 4. Finetune

Recover accuracy cho model da prune. **Bat buoc** truyen `finetune=True`.

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")
    model.train(data="VOC.yaml", epochs=50, finetune=True)
```

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `finetune` | bool | `False` | **BAT BUOC** khi train pruned model |

> `finetune=True` de trainer doc `maskbndict` tu checkpoint va build `DetectionModelPruned`.
> Dung AMP + scaler + grad clip binh thuong cua Ultralytics.

---

## 5. CWD - Channel-Wise Distillation

Student (pruned) hoc lai feature tu teacher (original) qua knowledge distillation per channel.

> Paper: "Channel-wise Knowledge Distillation for Dense Prediction" (arXiv:2011.13256)

### Cach dung co ban

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml",
        epochs=50,
        finetune=True,
        cwd=True,
        cwd_teacher="weights/best.pt",
    )
```

### Tham so

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `cwd` | bool | `False` | Bat/tat CWD |
| `cwd_teacher` | str | `None` | Path teacher model (**BAT BUOC**) |
| `cwd_lambda` | float | `0.5` | Trong so CWD loss |
| `cwd_temperature` | float/str | `6.0` | Temperature, hoac `"dynamic"` |
| `tau_max` | float | `10.0` | Tau max khi dynamic |
| `tau_min` | float | `1.0` | Tau min khi dynamic |
| `cwd_layers` | str | `"neck"` | `"neck"` hoac `"all"` |
| `cwd_layer_weights` | dict/None | `None` | Trong so rieng tung layer |
| `cwd_warmup` | int | `3` | So epoch warmup truoc khi bat CWD |
| `cwd_learnable_tau` | bool | `False` | Tu dong hoc temperature bang gradient |
| `cwd_learnable_tau_lr` | float | `1e-3` | Learning rate cho tau (Adam rieng) |
| `cwd_learnable_tau_init` | float | `6.0` | Tau khoi tao |

**`cwd_lambda`**:

| Gia tri | Y nghia |
|---------|---------|
| `0.1` | CWD nhe, uu tien detection loss |
| `0.5` | Can bang (mac dinh) |
| `1.0` | CWD manh |
| `2.0` | Rat manh, co the slow convergence |

**`cwd_temperature`**:

*Fixed:*

| Gia tri | Y nghia |
|---------|---------|
| `1.0` | Sharp - kho hoi tu |
| `4.0` | Vua phai |
| `6.0` | Mac dinh - soft, de hoc |
| `10.0` | Rat soft - de hoc nhung mat chi tiet |

*Dynamic:* truyen string `"dynamic"` → cosine schedule: `tau_max → tau_min → tau_max`

**`cwd_layers`**:

| Mode | Layer indices | Mo ta |
|------|---------------|-------|
| `"neck"` | [13, 16, 19, 22] | Neck + detect (mac dinh, nhanh) |
| `"all"` | [2, 4, 6, 8, 13, 16, 19, 22] | Backbone + neck (chinh xac hon) |

**`cwd_layer_weights`** — Trong so rieng tung layer (mac dinh tat ca = 1.0):
```python
cwd_layer_weights = {
    13: 0.5,   # P3
    16: 1.0,   # P4
    19: 1.0,   # P5
    22: 1.5,   # detect input - quan trong nhat
}
```

### Vi du nang cao

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # Dynamic temperature
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt",
        cwd_temperature="dynamic",
    )

    # All layers + custom weights + custom tau range
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt",
        cwd_lambda=0.5,
        cwd_temperature="dynamic", tau_max=10.0, tau_min=1.0,
        cwd_layers="all",
        cwd_layer_weights={
            2: 0.3, 4: 0.3, 6: 0.5, 8: 0.5,
            13: 1.0, 16: 1.0, 19: 1.0, 22: 1.5,
        },
    )

    # Learnable temperature (tu dong hoc tau, khong can grid search)
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        cwd=True, cwd_teacher="weights/best.pt",
        cwd_learnable_tau=True,           # bat learnable T
        cwd_learnable_tau_lr=1e-3,        # lr cho tau
        cwd_learnable_tau_init=6.0,       # tau khoi tao
        cwd_lambda=0.5, cwd_warmup=3,
    )
```

---

## 6. Do performance

So sanh GFLOPs, Params, Latency, FPS giua model goc va model da prune.

```bash
python speed.py
```

Sua `orig_path` va `pruned_path` trong `speed.py`. Yeu cau GPU CUDA va thu vien `thop`.

Output:
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

---

## Bang tham khao nhanh

| Nhom | Tham so | Mac dinh | Range khuyen nghi |
|------|---------|----------|-------------------|
| **Finetune** | `finetune` | False | `True` khi train pruned model |
| **SR** | `sr` | None | `1e-4` ~ `5e-4` |
| **DMS** | `dms_target` | 0.3 | 0.2 ~ 0.5 |
| | `dms_lambda` | 1.0 | 0.5 ~ 5.0 |
| | `dms_lr` | 5e-3 | 1e-3 ~ 1e-2 |
| | `dms_freeze` | False | True (nhanh) / False (chinh xac) |
| | `dms_importance` | "gamma" | "gamma" / "taylor" / "l1" |
| | `dms_warmup` | 0 | 0 ~ 5 (khuyen nghi 2-3) |
| **CWD** | `cwd_lambda` | 0.5 | 0.3 ~ 1.0 |
| | `cwd_temperature` | 6.0 | 4.0 ~ 10.0 hoac "dynamic" |
| | `tau_max` | 10.0 | 5.0 ~ 15.0 (chi khi dynamic) |
| | `tau_min` | 1.0 | 0.5 ~ 3.0 (chi khi dynamic) |
| | `cwd_layers` | "neck" | "neck" (nhanh) / "all" (chinh xac) |
| | `cwd_warmup` | 3 | 1 ~ 5 |
| **Prune** | `--prune-ratio` | 0.5 | 0.2 ~ 0.7 |
| | `--divisor` | 8 | 8 (GPU) / 16 (Tensor Cores) |
| | `--model-size` | m | n / s / m / l / x |

### Luu y quan trong

1. **`finetune=True` bat buoc** khi train pruned model.
2. **Khong ket hop** sr, dms, cwd cung luc. Chi dung 1 che do moi lan train.
3. **AMP tu dong tat** khi dung sr hoac dms. CWD co the dung AMP nhung can warmup.
4. **DMS can extract ratios** truoc khi prune (`extract_ratios_from_checkpoint`).
5. **CWD can teacher model** chua prune (original best.pt).
6. **Tren Windows** phai co `if __name__ == '__main__':` trong script.

---

# Phan II — Chi tiet ky thuat

## Cau truc du an

```
yolo/
├── prune.py                          # Script chinh: channel pruning
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

## Sparsity Training — Ben trong

### Y tuong

L1 regularization len BN gamma (`weight`) de day cac kenh khong quan trong ve 0. Sau do prune.py dung magnitude cua gamma de quyet dinh kenh nao bi cat.

### Training loop

```
1. Tat AMP (L1 penalty khong tuong thich voi mixed precision)
2. Build ignore_bn_list:
   - Bottleneck co add=True (residual): ignore cv2.bn + parent cv1.bn
   - PSABlock: ignore tat ca BN ben trong + parent cv1.bn
   → Nhung BN nay khong duoc ep vi cat se pha vo residual connection

3. Moi training step:
   loss.backward()
   sr_actual = sr * (1 - 0.9 * epoch / total_epochs)    # decay tuyen tinh
   for moi BN khong nam trong ignore_list:
       BN.weight.grad += sr_actual * sign(BN.weight)     # L1 penalty
   optimizer.step()    # khong scaler, khong grad clip
```

### Tai sao decay sr?

`sr` giam dan tu `sr` → `0.1*sr` theo epoch. Dau training ep manh de tao sparsity, cuoi training nhe bot de model hoi tu. Tranh truong hop ep qua manh lam model khong hoc duoc.

### Tai sao tat AMP?

AMP dung fp16 cho forward/backward. L1 gradient rat nho (`sr * sign(w)` ~ 1e-4), de bi round ve 0 trong fp16. Tat AMP dam bao L1 penalty thuc su co tac dung.

### ignore_bn_list

Residual connections (`x + bottleneck(x)`) yeu cau input va output cung so kenh. Neu prune BN trong residual path → so kenh lech → crash. PSABlock tuong tu — attention mechanism yeu cau channels khop.

---

## DMS — Ben trong

### Y tuong cot loi

Thay vi dat 1 prune ratio chung cho tat ca layer (vd 30%), DMS hoc **per-layer ratio** bang gradient descent. Moi layer co 1 learnable parameter `a` dai dien cho ti le kenh bi cat.

### Soft mask mechanism

Voi moi BN layer co the prune:

```
1. Tao a = nn.Parameter(init=dms_target)     # vd a=0.3

2. Forward hook sau BN:
   importance = |BN.weight|                    # gamma mode
   rank = argsort(importance) / N              # chuan hoa [0,1], DETACHED (khong co grad)
   mask = Sigmoid(N * (rank - a))              # differentiable soft mask
   output = BN_output * mask                   # ap mask len feature

3. Gradient chi chay qua a (vi rank bi detach)
   → a hoc duoc: tang a = prune nhieu hon, giam a = giu nhieu hon
```

Tai sao dung Sigmoid thay vi hard threshold? Sigmoid lam mask lien tuc → gradient co the flow qua `a` → hoc duoc bang backprop. Hard threshold (0/1) khong co gradient.

### Resource loss

DMS dung FLOPs constraint de dam bao tong FLOPs sau prune gan target:

```
Tinh effective FLOPs cho moi Conv2d:
- Regular conv:  flops_original * (1 - a_input) * (1 - a_output)
- Depthwise:     flops_original * (1 - a_output)
- Concat input:  weighted average retention theo so kenh

effective_ratio = sum(effective_flops) / total_flops
resource_loss = max(0, log(effective_ratio / target_ratio))

total_loss = detection_loss + dms_lambda * resource_loss
```

`log` thay vi hieu tuyen tinh de penalty tang nhanh khi vuot target nhieu, nhung nhe khi gan target.

### Optimizer rieng

`a_params` dung Adam optimizer rieng, **khong chung** scheduler voi model. Ly do: model optimizer co lr decay phuc tap (warmup, cosine, ...), nhung a params chi can lr co dinh de hoi tu nhanh ve target.

### Clamp

Moi step: `a = clamp(a, 0.05, 0.95)`. Tranh a = 0 (khong prune gi) hoac a = 1 (cat het kenh).

### Taylor importance (optional)

Thay vi `|gamma|`, dung first-order Taylor expansion:
```
importance = (mask * gradient)^2
```
Cap nhat bang EMA voi momentum 0.99 de on dinh. Chinh xac hon gamma vi tinh den ca gradient (anh huong thuc te cua kenh len loss), nhung can vai epoch de EMA on dinh.

---

## Pruning — Ben trong

### Workflow cua prune.py

```
 1. Load model tu --weights
 2. Thu thap tat ca BN layers
 3. Build ignore_bn_list (Bottleneck residual, PSABlock)
 4. Voi moi BN layer (khong nam trong ignore list):
    a. Lay prune ratio: --layer-ratio (exact → index → group) hoac --prune-ratio
    b. Sap xep |BN.gamma| tang dan
    c. Tinh local threshold: gamma[int(n_channels * ratio)]
    d. Tao binary mask: gamma >= threshold → 1, nguoi lai → 0
    e. Lam tron so kenh con lai → boi so cua --divisor
    f. Dam bao giu it nhat divisor channels
 5. Build pruned YAML config:
    - Map: C3k2 → C3k2Pruned, SPPF → SPPFPruned, Detect → DetectPruned, ...
 6. Build DetectionModelPruned tu masks
 7. Copy weights tu original → pruned:
    - Conv: copy subset in/out channels theo mask
    - BN: copy subset channels theo mask
    - Detect: xu ly rieng (cv2, cv3 cua tung scale)
    - SPPF: xu1 ly concat
 8. Test forward pass voi dummy input
 9. Save checkpoint: model + maskbndict + config
```

### maskbndict

Dict `{bn_name: mask_tensor}` voi `mask_tensor` la binary tensor (0/1) co shape `[n_channels]`. Day la output chinh cua prune.py, duoc dung boi:
- `finetune`: de build DetectionModelPruned dung so kenh
- `CWD`: de align channels giua teacher va student

### Divisor constraint

GPU thuc hien phep tinh nhanh hon khi so channels la boi so cua 8 (hoac 16 cho Tensor Cores). Divisor dam bao dieu nay sau khi prune.

---

## Finetune — Ben trong

### Flow khi finetune=True

```
model.train(finetune=True)
  → model.py:
      pop finetune=True
      doc maskbndict tu checkpoint["maskbndict"]
  → detect/train.py get_model(maskbndict=...):
      maskbndict != None → DetectionModelPruned(maskbndict, cfg, nc)
  → tasks_pruned.py parse_model_pruned(maskbndict, cfg):
      Doc so kenh thuc tu maskbndict cho moi layer
      Build: C3k2Pruned, SPPFPruned, DetectPruned, ...
      Tinh lai stride
  → model.load(weights) qua intersect_dicts
      → chi copy weights co shape khop
```

### DetectionModelPruned vs DetectionModel

DetectionModel build model tu YAML config voi so kenh theo scale (width multiplier). DetectionModelPruned dung `maskbndict` de tinh so kenh **thuc te** cho moi layer — co the khac nhau giua cac layer (vi moi layer co mask rieng).

### Cac module pruned

| Module | Goc | Khac biet |
|--------|-----|-----------|
| `C3k2Pruned` | `C3k2` | So kenh cv1, cv2, bottleneck lay tu maskbndict |
| `SPPFPruned` | `SPPF` | Dong nhat kenh in/out theo mask |
| `C2PSAPruned` | `C2PSA` | Attention channels theo mask |
| `DetectPruned` | `Detect` | cv2/cv3 channels lay tu maskbndict, khong hardcode |

---

## CWD — Ben trong

### Y tuong

CWD distill feature maps (khong phai logits) tu teacher sang student. Voi moi layer duoc chon, student hoc phan bo spatial cua tung channel sao cho giong teacher.

### Loss formulation

```
Voi moi hooked layer i:
  1. Align channels: teacher_feat = teacher_feat[:, mask_i, :, :]
     (chon subset channels cua teacher khop voi student)
  2. Interpolate neu spatial size khac nhau
  3. Flatten spatial: [B, C, H*W]
  4. Spatial softmax per channel:
     phi(y_c) = softmax(y_c / tau, dim=spatial)
  5. KL divergence:
     kl_c = KL(phi(teacher_c) || phi(student_c))
  6. CWD loss cua layer i:
     L_i = (tau^2 / C) * mean(kl_c)

total_loss = det_loss + cwd_lambda * weighted_avg(L_i)
```

### Channel alignment

Khi prune, moi layer chi giu 1 subset kenh. `maskbndict` ghi lai kenh nao duoc giu. Khi CWD forward teacher, dung mask de chon dung nhung kenh tuong ung:
```python
teacher_feat_aligned = teacher_feat[:, mask_bool, :, :]
# shape: [B, n_pruned_channels, H, W] — khop voi student
```

Khong can projection layer 1x1 (nhu mot so paper khac) vi ta biet chinh xac kenh nao cua teacher map sang kenh nao cua student.

### Warmup + ramp up

- **Van de**: CWD loss qua lon o epoch dau + AMP scaler → overflow → NaN → model hong vinh vien.
- **Fix**:
  1. `cwd_warmup=3`: 3 epoch dau chi train detection loss (khong CWD)
  2. Ramp up: sau warmup, `cwd_lambda` tang dan tu 0 → full trong 5 epochs tiep
  3. Scaler on dinh truoc khi CWD vao

### Dynamic temperature

```
tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + cos(pi * progress))
progress = (epoch - warmup) / (total_epochs - warmup)
```
- Dau training: tau cao (soft) → student de hoc pattern tong quat
- Giua training: tau thap (sharp) → student hoc chi tiet
- Cuoi training: tau cao lai → on dinh hoi tu

### Tai sao `tau^2 / C`?

`tau^2` compensate cho viec softmax(x/tau) lam giam gradient khi tau lon. Chia cho C (so channels) de loss khong phu thuoc vao so kenh cua layer.

---

## Chi tiet tung file

### model.py (ultralytics/engine/)

Xu ly custom args truoc khi truyen vao Ultralytics trainer.

**Flow:**
1. Pop tat ca custom args (`sr`, `dms*`, `cwd*`, `finetune`) khoi `args` — Ultralytics bao loi neu gap tham so la
2. Goi Ultralytics setup binh thuong
3. Gan custom args vao `self.trainer.*`
4. Lay `maskbndict` tu checkpoint (neu co)
5. Truyen `maskbndict` vao `get_model()` (neu `finetune=True`)

### trainer.py (ultralytics/engine/)

Training loop ho tro 3 che do:

**SR:** Tat AMP → build ignore list → sau `loss.backward()` them L1 gradient len BN gamma → optimizer step khong scaler

**DMS:** Tat AMP → tao a_params + Adam rieng → dang ky forward hooks (soft mask) → them resource_loss → clamp a moi step

**CWD:** Load teacher → freeze → dang ky hooks student + teacher → build channel masks → moi step forward teacher (no grad) → tinh CWD loss → cong vao total loss. Warmup + ramp up trong vai epoch dau.

### detect/train.py (ultralytics/models/yolo/detect/)

`get_model(maskbndict=...)`:
- `maskbndict is None` → `DetectionModel` binh thuong
- `maskbndict is not None` → `DetectionModelPruned(maskbndict, cfg, nc)`

### tasks_pruned.py (ultralytics/nn/)

`parse_model_pruned(maskbndict, cfg)`: doc cau truc tu pruned YAML, dung maskbndict tinh so kenh thuc moi layer, map module types sang phien ban Pruned.

### block_pruned.py (ultralytics/nn/modules/)

Cac module pruned nhan so kenh tuy bien tu maskbndict thay vi tu scale config:

| Module | Goc |
|--------|-----|
| `BottleneckPruned` | `Bottleneck` |
| `C3k2Pruned` | `C3k2` |
| `C3k2PrunedAttn` | `C3k2` (attn=True) |
| `SPPFPruned` | `SPPF` |
| `C2PSAPruned` | `C2PSA` |

### head_pruned.py (ultralytics/nn/modules/)

`DetectPruned`: detection head nhan so kenh tu maskbndict, khong hardcode. `nc` lay tu model goc luc prune.

### dms_utils.py

| Ham | Muc dich |
|-----|----------|
| `make_divisible_channels()` | Lam tron channels → boi so cua divisor |
| `get_layer_ratio()` | Lay prune ratio (exact → index → group → default) |
| `build_pruned_yaml()` | Tao pruned YAML tu config goc |
| `build_ignore_bn_list()` | List BN khong bi prune |
| `make_soft_mask_hook()` | DMS forward hook voi soft mask |
| `profile_per_layer_flops()` | Do FLOPs per Conv2d |
| `build_conv_bn_mapping()` | Map Conv → BN |
| `compute_resource_loss()` | FLOPs constraint loss |
| `compute_l1_loss()` | L1 penalty tren BN gamma |
| `extract_ratios_from_checkpoint()` | Extract a params → YAML |

### cwd_loss.py

| Class/Ham | Muc dich |
|-----------|----------|
| `CWDLoss` | KL divergence per channel (spatial softmax) |
| `FeatureHook` | Forward hook bat feature maps |
| `setup_hooks()` | Dang ky hooks tren cac layers |
| `build_cwd_channel_masks()` | Boolean masks cho channel alignment |
| `compute_cwd_loss()` | Weighted CWD loss qua tat ca layers |

### speed.py

Load model → CUDA → `thop.profile()` (GFLOPs, Params) → warm-up 30 lan → do latency 100 lan voi `torch.cuda.Event` → FPS = 1000 / latency_ms.
