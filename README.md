# YOLOv26 Channel Pruning + Knowledge Distillation

Channel pruning cho YOLOv26 object detection, ket hop DMS (Differentiable Model Scaling) va nhieu phuong phap Knowledge Distillation de giam kich thuoc model va tang toc inference.

## Muc luc

**Phan I — Cach dung**
- [Pipeline tong quan](#pipeline-tong-quan)
- [Cai dat](#cai-dat)
- [1. Sparsity Training](#1-sparsity-training)
- [2. DMS - Differentiable Model Scaling](#2-dms---differentiable-model-scaling)
- [3. Channel Pruning (6 methods)](#3-channel-pruning-6-methods)
- [4. Finetune](#4-finetune)
- [5. Knowledge Distillation (4 methods)](#5-knowledge-distillation-4-methods)
- [6. Do performance](#6-do-performance)
- [Bang tham khao nhanh](#bang-tham-khao-nhanh)

**Phan II — Chi tiet ky thuat**
- [Cau truc du an](#cau-truc-du-an)
- [Sparsity Training — Ben trong](#sparsity-training--ben-trong)
- [DMS — Ben trong](#dms--ben-trong)
- [Pruning — Ben trong](#pruning--ben-trong)
- [Finetune — Ben trong](#finetune--ben-trong)
- [CWD — Ben trong](#cwd--ben-trong)

---

# Phan I — Cach dung

## Pipeline tong quan

```
best.pt (model goc)
  │
  ├─ [1] Sparsity Training (sr)      → ep BN gamma ve 0, chuan bi prune
  ├─ [2] DMS Search (dms)            → tim per-layer prune ratio toi uu
  │
  ├─ [3] Prune (6 methods)           → cat kenh, tao pruned model
  │
  ├─ [4] Finetune                    → recover accuracy
  ├─ [5] KD (4 methods)              → distill tu teacher
  │
  └─ [6] speed.py                    → do va so sanh performance
```

**Pipeline day du:**
```bash
# 1. Sparsity training
python scripts/train_sparsity.py

# 2. DMS search → extract ratios
python scripts/train_dms.py
python dms/extract_ratios.py --ckpt runs/.../best.pt --output dms_ratios.yaml



# 3. Prune (chon 1 trong 6 methods)
python pruning/prune_bn_gamma.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml

# 4. Finetune + KD
python scripts/finetune_cwd.py

# 5. Do performance
python speed.py
```

**Pipeline rut gon (khong DMS):**
```bash
python scripts/train_sparsity.py
python pruning/prune_bn_gamma.py --weights runs/.../best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3
python scripts/finetune.py
```

**Pipeline L1 norm (don gian nhat, khong can SR):**
```bash
# Prune truc tiep tu pretrained model
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5

# Finetune
python scripts/finetune.py
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

```bash
python dms/extract_ratios.py --ckpt runs/.../best.pt --output dms_ratios.yaml
```

Dung voi bat ky pruning method nao:
```bash
python pruning/prune_bn_gamma.py --weights runs/.../best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml
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

    # L1 norm importance + warmup
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=50,
        dms=True, dms_target=0.3,
        dms_importance="l1", dms_warmup=3,
    )
```

---

## 3. Channel Pruning (6 methods)

6 phuong phap pruning, tat ca dung chung pipeline tu `pruning/prune_common.py`.

### Cac phuong phap

| Method | File | Can SR? | Mo ta |
|--------|------|---------|-------|
| **BN Gamma** | `prune_bn_gamma.py` | Co | Cat kenh dua tren BN gamma magnitude |
| **L1 Norm** | `prune_l1norm.py` | Khong | Cat kenh dua tren L1 norm cua Conv filter |
| **Taylor** | `prune_taylor.py` | Khong | Gradient-based importance (can forward+backward) |
| **LAMP** | `prune_lamp.py` | Khong | Layer-Adaptive Magnitude Pruning |
| **FPGM** | `prune_fpgm.py` | Khong | Filter Pruning via Geometric Median |
| **Random** | `prune_random.py` | Khong | Random pruning (baseline) |

### Cach dung

```bash
# BN gamma (default, can SR training truoc)
python pruning/prune_bn_gamma.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# L1 norm (khong can SR, dung truc tiep tren pretrained)
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Taylor importance (can --data de forward+backward)
python pruning/prune_taylor.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3 --data VOC.yaml

# LAMP (adaptive per-layer)
python pruning/prune_lamp.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# FPGM (geometric median)
python pruning/prune_fpgm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Random (baseline)
python pruning/prune_random.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Bat ky method nao + DMS per-layer ratio
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml \
    --prune-ratio 0.3 --layer-ratio dms_ratios.yaml
```

### CLI Arguments

| Argument | Kieu | Mac dinh | Mo ta |
|----------|------|----------|-------|
| `--weights` | str | `weights/best.pt` | Duong dan model .pt |
| `--cfg` | str | - | Duong dan YAML config |
| `--model-size` | str | `m` | Kich thuoc: `n`, `s`, `m`, `l`, `x` |
| `--prune-ratio` | float | `0.5` | Ti le prune chung (0.0 - 1.0) |
| `--layer-ratio` | str | `None` | YAML file chua custom ratio per-layer |
| `--divisor` | int | `8` | Channels chia het cho gia tri nay |
| `--save-dir` | str | `weights` | Thu muc luu output |
| `--data` | str | `None` | Dataset (chi can cho Taylor) |

### Benchmark (VOC, prune ratio=0.3)

| Method | mAP50 | mAP50-95 |
|--------|-------|----------|
| Baseline (unpruned) | 0.890 | 0.725 |
| **L1 Norm** | **0.878** | **0.701** |
| FPGM | 0.873 | 0.698 |
| Taylor | 0.872 | 0.695 |
| BN Gamma | 0.865 | 0.689 |
| Random | 0.865 | 0.689 |

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

---

## 5. Knowledge Distillation (4 methods)

Student (pruned) hoc lai tu teacher (original) qua knowledge distillation. 4 phuong phap, chon qua `kd_method`.

### Cac phuong phap

| Method | `kd_method` | Loai | Mo ta |
|--------|-------------|------|-------|
| **CWD** | `"cwd"` | Feature-based | Channel-wise KL divergence (default) |
| **Response KD** | `"response"` | Logit-based | Distill output logits |
| **FitNets** | `"fitnets"` | Feature-based | Intermediate feature mimicking |
| **MGD** | `"mgd"` | Feature-based | Masked Generative Distillation |

### Cach dung co ban

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml",
        epochs=50,
        finetune=True,
        kd=True,
        kd_teacher="weights/best.pt",
    )
```

### Chon KD method

```python
# CWD (default)
model.train(
    data="VOC.yaml", epochs=50, finetune=True,
    kd=True, kd_teacher="weights/best.pt",
    kd_method="cwd", kd_lambda=0.5,
)

# Response KD (logit distillation)
model.train(
    data="VOC.yaml", epochs=50, finetune=True,
    kd=True, kd_teacher="weights/best.pt",
    kd_method="response", kd_lambda=0.5,
)

# FitNets (feature mimicking)
model.train(
    data="VOC.yaml", epochs=50, finetune=True,
    kd=True, kd_teacher="weights/best.pt",
    kd_method="fitnets", kd_lambda=0.5,
)

# MGD (masked generative distillation)
model.train(
    data="VOC.yaml", epochs=50, finetune=True,
    kd=True, kd_teacher="weights/best.pt",
    kd_method="mgd", kd_lambda=0.5,
)
```

### KD tham so

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `kd` | bool | `False` | Bat/tat KD pipeline |
| `kd_teacher` | str | `None` | Path teacher model (**BAT BUOC**) |
| `kd_method` | str | `"cwd"` | `"cwd"`, `"response"`, `"fitnets"`, `"mgd"` |
| `kd_lambda` | float | `0.5` | Trong so KD loss |
| `kd_layers` | str | `"neck"` | `"neck"` hoac `"all"` |
| `kd_warmup` | int | `5` | So epoch warmup truoc khi bat KD |

### CWD tham so (chi ap dung cho kd_method="cwd")

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `cwd_temperature` | float/str | `9.0` | Fixed temperature, hoac `"learnable"` |
| `cwd_learnable_tau_lr` | float | `1e-3` | Learning rate cho learnable tau |
| `cwd_learnable_tau_init` | float | `9.0` | Tau khoi tao khi learnable |

### Vi du CWD nang cao

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # CWD + fixed temperature
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        kd=True, kd_teacher="weights/best.pt",
        kd_lambda=0.5, kd_layers="all",
        cwd_temperature=9.0,
    )

    # CWD + learnable temperature (tu dong, khong can grid search)
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        kd=True, kd_teacher="weights/best.pt",
        kd_lambda=0.5, kd_layers="all", kd_warmup=3,
        cwd_temperature="learnable",
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
| **KD** | `kd_method` | "cwd" | "cwd" / "response" / "fitnets" / "mgd" |
| | `kd_lambda` | 0.5 | 0.3 ~ 1.0 |
| | `kd_layers` | "neck" | "neck" (nhanh) / "all" (chinh xac) |
| | `kd_warmup` | 5 | 1 ~ 5 |
| | `cwd_temperature` | 9.0 | 4.0 ~ 15.0 hoac "learnable" |
| **Prune** | `--prune-ratio` | 0.5 | 0.2 ~ 0.7 |
| | `--divisor` | 8 | 8 (GPU) / 16 (Tensor Cores) |
| | `--model-size` | m | n / s / m / l / x |

### Luu y quan trong

1. **`finetune=True` bat buoc** khi train pruned model.
2. **Khong ket hop** sr, dms, kd cung luc. Chi dung 1 che do moi lan train.
3. **AMP tu dong tat** khi dung sr. DMS va KD tuong thich AMP.
4. **DMS can extract ratios** truoc khi prune (`python dms/extract_ratios.py`).
5. **KD can teacher model** chua prune (original best.pt).
6. **Tren Windows** phai co `if __name__ == '__main__':` trong script.

---

# Phan II — Chi tiet ky thuat

## Cau truc du an

```
yolo/
├── pruning/                            # 6 pruning methods
│   ├── prune_common.py                 # Shared pipeline: load_and_prepare, create_masks, finalize_pruning
│   ├── prune_bn_gamma.py              # BN gamma magnitude pruning (default)
│   ├── prune_l1norm.py                # L1 norm pruning
│   ├── prune_taylor.py                # Taylor importance pruning
│   ├── prune_lamp.py                  # LAMP adaptive per-layer pruning
│   ├── prune_fpgm.py                  # FPGM geometric median pruning
│   └── prune_random.py                # Random pruning (baseline)
│
├── distillation/                       # 4 KD methods
│   ├── cwd_loss.py                    # CWD distillation loss
│   └── kd_losses.py                   # Response KD, FitNets, MGD losses
│
├── dms/                                # Differentiable Model Scaling
│   ├── dms_utils.py                   # Soft mask, resource loss, FLOPs profiling
│   └── extract_ratios.py             # Extract DMS ratios from checkpoint → YAML
│
├── scripts/                            # Training/finetune scripts
│   ├── train_sparsity.py              # Sparsity training (SR)
│   ├── train_dms.py                   # DMS search training
│   ├── finetune.py                    # Finetune pruned model
│   ├── finetune_cwd.py                # Finetune with CWD distillation
│   └── finetune_kd.py                 # Finetune with KD (Response/FitNets/MGD)
│
├── tools/                              # Debug & validation tools
│   ├── check_sparsity.py             # Check BN sparsity of model
│   ├── debug_dms_flops.py            # Debug DMS FLOPs computation
│   ├── validate_pruned.py            # Validate pruned model structure
│   └── visualize_map.py              # Visualize mAP results
│
├── cfg/
│   └── yolo26m.yaml                   # YAML config cua yolo26
│
├── ultralytics/
│   ├── engine/
│   │   ├── model.py                   # YOLO.train() - nhan tat ca custom args
│   │   └── trainer.py                 # BaseTrainer - xu ly SR/DMS/KD/finetune
│   └── nn/
│       ├── tasks_pruned.py            # Build pruned model tu masks
│       └── modules/
│           ├── block_pruned.py        # C3k2Pruned, SPPFPruned, C2PSAPruned, ...
│           └── head_pruned.py         # DetectPruned
│
├── speed.py                           # Do GFLOPs, Params, Latency, FPS
└── weights/                           # Model weights
```

---

## Sparsity Training — Ben trong

### Y tuong

L1 regularization len BN gamma (`weight`) de day cac kenh khong quan trong ve 0. Sau do pruning dung magnitude cua gamma de quyet dinh kenh nao bi cat.

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

### Tai sao tat AMP?

AMP dung fp16 cho forward/backward. L1 gradient rat nho (`sr * sign(w)` ~ 1e-4), de bi round ve 0 trong fp16. Tat AMP dam bao L1 penalty thuc su co tac dung.

---

## DMS — Ben trong

### Y tuong cot loi

Thay vi dat 1 prune ratio chung cho tat ca layer (vd 30%), DMS hoc **per-layer ratio** bang gradient descent. Moi layer co 1 learnable parameter `a` dai dien cho ti le kenh bi cat.

### Soft mask mechanism

```
1. Tao a = nn.Parameter(init=0.0)

2. Forward hook sau BN:
   importance = |BN.weight|              # gamma mode (hoac taylor, l1)
   rank = argsort(importance) / N        # chuan hoa [0,1], DETACHED
   mask = Sigmoid(N * (rank - a))        # differentiable soft mask
   output = BN_output * mask             # ap mask len feature

3. Gradient chi chay qua a (vi rank bi detach)
   → a hoc duoc: tang a = prune nhieu hon, giam a = giu nhieu hon
```

### Resource loss

```
Tinh effective FLOPs cho moi Conv2d:
- Regular conv:  flops_original * (1 - a_input) * (1 - a_output)
- Depthwise:     flops_original * (1 - a_output)
- Concat input:  weighted average retention theo so kenh

effective_ratio = sum(effective_flops) / total_flops
resource_loss = max(0, log(effective_ratio / target_ratio))

total_loss = detection_loss + dms_lambda * resource_loss
```

---

## Pruning — Ben trong

### Shared pipeline (prune_common.py)

Tat ca 6 methods dung chung 3 ham:

1. **`load_and_prepare(args)`**: Load model, build ignore_bn_list, collect prunable BNs
2. **`create_masks(bn_weights, args)`**: Tao binary mask tu importance scores (method-specific)
3. **`finalize_pruning(model, masks, args)`**: Build pruned YAML, DetectionModelPruned, copy weights, save

Moi method chi can implement cach tinh importance scores, con lai dung chung pipeline.

### maskbndict

Dict `{bn_name: mask_tensor}` voi `mask_tensor` la binary tensor (0/1) co shape `[n_channels]`. Duoc dung boi:
- `finetune`: de build DetectionModelPruned dung so kenh
- `KD`: de align channels giua teacher va student

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
  → model.load(weights) qua intersect_dicts
```

---

## CWD — Ben trong

### Loss formulation

```
Voi moi hooked layer i:
  1. Align channels: teacher_feat = teacher_feat[:, mask_i, :, :]
  2. Flatten spatial: [B, C, H*W]
  3. Spatial softmax per channel:
     phi(y_c) = softmax(y_c / tau, dim=spatial)
  4. KL divergence:
     kl_c = KL(phi(teacher_c) || phi(student_c))
  5. CWD loss:
     L_i = (tau^2 / C) * mean(kl_c)

total_loss = det_loss + kd_lambda * avg(L_i)
```

### Channel alignment

Khi prune, moi layer chi giu 1 subset kenh. `maskbndict` ghi lai kenh nao duoc giu:
```python
teacher_feat_aligned = teacher_feat[:, mask_bool, :, :]
# shape: [B, n_pruned_channels, H, W] — khop voi student
```

Khong can projection layer 1x1 vi ta biet chinh xac kenh nao cua teacher map sang kenh nao cua student (index alignment).

### Warmup + ramp up

- **Van de**: KD loss qua lon o epoch dau + AMP scaler → overflow → NaN → model hong.
- **Fix**:
  1. `kd_warmup=5`: epoch dau chi train detection loss (khong KD)
  2. Ramp up: sau warmup, `kd_lambda` tang dan tu 0 → full trong 5 epochs tiep
  3. Scaler on dinh truoc khi KD vao

### Learnable temperature

- `cwd_temperature="learnable"`: tu dong hoc tau bang gradient descent
- `log_tau = nn.Parameter(log(tau_init))` → tau = exp(log_tau).clamp(0.5, 20)
- Optimizer rieng: Adam, tach khoi main optimizer
- Khong can grid search temperature