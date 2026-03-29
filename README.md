# YOLOv26 Channel Pruning + Knowledge Distillation

Channel pruning cho YOLOv26 object detection, ket hop DMS (Differentiable Model Scaling) va nhieu phuong phap Knowledge Distillation de giam kich thuoc model va tang toc inference.

## Muc luc

**Phan I — Cach dung**
- [Pipeline tong quan](#pipeline-tong-quan)
- [Cai dat](#cai-dat)
- [1. DMS - Differentiable Model Scaling](#1-dms---differentiable-model-scaling)
- [2. Channel Pruning (6 methods)](#2-channel-pruning-6-methods)
- [3. Finetune](#3-finetune)
- [4. Knowledge Distillation (4 methods)](#4-knowledge-distillation-4-methods)
- [5. Do performance](#5-do-performance)
- [Bang tham khao nhanh](#bang-tham-khao-nhanh)

**Phan II — Chi tiet ky thuat**
- [Cau truc du an](#cau-truc-du-an)
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
  ├─ [1] DMS Search (dms)            → tim per-layer prune ratio toi uu
  │
  ├─ [2] Prune (6 methods)           → cat kenh, tao pruned model
  │
  ├─ [3] Finetune                    → recover accuracy
  ├─ [4] KD (4 methods)              → distill tu teacher
  │
  └─ [5] tools/speed.py              → do va so sanh performance
```

**Pipeline chinh (L1 norm, don gian nhat):**
```bash
# Prune truc tiep tu pretrained model (khong can SR)
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5

# Finetune + CWD distillation
python scripts/finetune_cwd.py
```

**Pipeline DMS (per-layer ratio):**
```bash
# 1. DMS search → extract ratios
python scripts/train_dms.py
python dms/extract_ratios.py --ckpt runs/.../best.pt --output dms_ratios.yaml

# 2. Prune voi per-layer ratio
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml \
    --prune-ratio 0.5 --layer-ratio dms_ratios.yaml

# 3. Finetune + CWD
python scripts/finetune_cwd.py
```

### Ket qua thuc nghiem

| Pipeline | mAP50-95 | Ghi chu |
|----------|----------|---------|
| L1 uniform prune + CWD | >= baseline | Regularization effect, method chinh |
| DMS L1 per-layer + prune | kem uniform | Magnitude ≠ importance, tao bottleneck |
| DMS Taylor per-layer + prune | dang optimize | Gradient-based, do dung contribution |

> **Ket luan**: L1 norm pruning > BN gamma pruning. Sparsity Regularization khong can thiet cho L1.

---

## Cai dat

Du an dua tren Ultralytics source code. Cac dependency chinh:
```
torch
ultralytics
thop          # do FLOPs (tools/speed.py)
pyyaml
```

---

## 1. DMS - Differentiable Model Scaling

Tu dong tim ti le prune toi uu **cho tung layer** thong qua gradient descent. Thay vi dat 1 ti le chung, DMS hoc duoc layer nao nen prune nhieu, layer nao nen giu lai.

> Paper: "Differentiable Model Scaling using Differentiable Topk" (ICML 2024)

### Cach dung co ban

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml",
        epochs=10,
        dms=True,
        dms_target=0.3,
    )
```

### Tham so

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `dms` | bool | `False` | Bat/tat DMS |
| `dms_target` | float | `0.3` | Ti le FLOPs muon giam (0.3 = giam 30%) |
| `dms_lambda` | float | `100.0` | Trong so resource loss (repo goc: 100~1000) |
| `dms_lr` | float | `4e-4` | Learning rate cho a params (constant, khong decay) |
| `dms_taylor_type` | str | `"taylor"` | `"taylor"`, `"snip"`, hoac `"fisher"` |
| `dms_decay_ratio` | float | `0.8` | Ti le epochs cho progressive target phase |
| `dms_grad_scale` | float | `-1.0` | Gradient normalization scale (-1=OFF, >=0=ON) |

**`dms_target`** — Ti le FLOPs muon giam:

| Gia tri | Y nghia |
|---------|---------|
| `0.2` | Prune nhe |
| `0.3` | Trung binh (khuyen nghi bat dau) |
| `0.5` | Prune manh |
| `0.7` | Rat manh, co the mat accuracy nhieu |

**`dms_taylor_type`** — Cach tinh do quan trong cua kenh:

| Mode | Cong thuc | Mo ta |
|------|-----------|-------|
| `"taylor"` | `(mask * grad)^2` | First-order Taylor expansion (default) |
| `"snip"` | `\|mask * grad\|` | SNIP criterion, sensitive voi outlier |
| `"fisher"` | `grad^2` | Fisher information (Hessian diagonal approx) |

**`dms_grad_scale`** — Gradient normalization:
- `-1.0` (default): OFF — dung `dms_lambda` de can bang thu cong
- `>= 0`: ON — auto normalize flop gradient to match task gradient magnitude (EMA)
- Khi ON: `dms_lambda` van dung cho loss logging nhung khong anh huong gradient balance

**`dms_decay_ratio`** — 2-phase scheduler:
- Phase 1 [0%, decay_ratio): Progressive target tang dan tu 0 → final_target
- Phase 2 [decay_ratio, 100%]: Fixed target, resource loss van ON (stabilize)

### Extract ratios sau DMS

```bash
python dms/extract_ratios.py --ckpt runs/.../best.pt --output dms_ratios.yaml
```

Dung voi bat ky pruning method nao:
```bash
python pruning/prune_l1norm.py --weights runs/.../best.pt --cfg cfg/yolo26m.yaml --layer-ratio dms_ratios.yaml
```

### Vi du nang cao

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # Taylor importance + gradient normalization
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=10,
        dms=True, dms_target=0.3,
        dms_lambda=100.0, dms_lr=4e-4,
        dms_taylor_type="taylor",
        dms_grad_scale=1.0,       # auto balance task/flop gradient
    )

    # Fisher importance + manual lambda
    model = YOLO("weights/best.pt")
    model.train(
        data="VOC.yaml", epochs=10,
        dms=True, dms_target=0.3,
        dms_lambda=500.0, dms_lr=4e-4,
        dms_taylor_type="fisher",
        dms_grad_scale=-1.0,      # OFF, dung lambda thu cong
    )
```

---

## 2. Channel Pruning (6 methods)

6 phuong phap pruning, tat ca dung chung pipeline tu `pruning/prune_common.py`.
**L1 norm la method chinh**, cho ket qua tot nhat (mAP50-95 >= baseline sau finetune).

### Cac phuong phap

| Method | File | Mo ta |
|--------|------|-------|
| **L1 Norm** | `prune_l1norm.py` | **Method chinh.** Cat kenh dua tren L1 norm cua Conv filter |
| **BN Gamma** | `prune_bn_gamma.py` | Cat kenh dua tren BN gamma magnitude |
| **Taylor** | `prune_taylor.py` | Gradient-based importance (can forward+backward) |
| **LAMP** | `prune_lamp.py` | Layer-Adaptive Magnitude Pruning |
| **FPGM** | `prune_fpgm.py` | Filter Pruning via Geometric Median |
| **Random** | `prune_random.py` | Random pruning (baseline) |

### Cach dung

```bash
# L1 norm (method chinh, khong can SR)
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5

# BN gamma
python pruning/prune_bn_gamma.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5

# Taylor importance (can --data de forward+backward)
python pruning/prune_taylor.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5 --data VOC.yaml

# LAMP (adaptive per-layer)
python pruning/prune_lamp.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5

# FPGM (geometric median)
python pruning/prune_fpgm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5

# Random (baseline)
python pruning/prune_random.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.5

# Bat ky method nao + DMS per-layer ratio
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml \
    --prune-ratio 0.5 --layer-ratio dms_ratios.yaml
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

### Output

File `.pt` chua: `model` (DetectionModelPruned), `maskbndict` (dict mask per BN layer), `config` (divisor, prune_ratio).

---

## 3. Finetune

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

## 4. Knowledge Distillation (4 methods)

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
| `kd_layers` | str | `"neck"` | `"neck"`, `"all"`, `"backbone"`, hoac list indices |
| `kd_warmup` | int | `5` | So epoch warmup truoc khi bat KD |

### CWD tham so (chi ap dung cho kd_method="cwd")

| Tham so | Kieu | Mac dinh | Mo ta |
|---------|------|----------|-------|
| `cwd_temperature` | float | `9.0` | Temperature cho softmax. Grid search: 4.0 ~ 15.0 |

### Vi du CWD nang cao

```python
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="VOC.yaml", epochs=100, finetune=True,
        kd=True, kd_teacher="weights/best.pt",
        kd_lambda=0.5, kd_layers="all",
        cwd_temperature=9.0,
    )
```

---

## 5. Do performance

So sanh GFLOPs, Params, Latency, FPS giua model goc va model da prune.

```bash
# GPU benchmark
python tools/speed.py --orig weights/yolo26m_baseline.pt --pruned weights/l1_best.pt --device 0

# CPU benchmark
python tools/speed.py --orig weights/yolo26m_baseline.pt --pruned weights/l1_best.pt --device cpu --threads 8

# Custom batch sizes
python tools/speed.py --orig weights/yolo26m_baseline.pt --pruned weights/l1_best.pt --batch 1 4 8 16 32 64
```

| Argument | Kieu | Mac dinh | Mo ta |
|----------|------|----------|-------|
| `--orig` | str | - | Path model goc |
| `--pruned` | str | - | Path model pruned |
| `--device` | str | `cuda` | `cuda` hoac `cpu` |
| `--batch` | int[] | `1 2 4` | Batch sizes |
| `--warmup` | int | auto | Warmup iterations |
| `--reps` | int | auto | Benchmark repetitions |
| `--threads` | int | `1` | CPU threads |

Output:
```
=================================================================
Metric          | Original          | Pruned            | Giam (%)
-----------------------------------------------------------------
GFLOPs          |             38.80 |             23.30 |    39.9%
Params (M)      |             18.90 |              7.50 |    60.3%
Latency (ms)    |          8.2±0.3  |          5.1±0.2  |    37.8%
FPS (bs=1)      |             121.5 |             196.1 |    61.4%
=================================================================
```

---

## Bang tham khao nhanh

| Nhom | Tham so | Mac dinh | Range khuyen nghi |
|------|---------|----------|-------------------|
| **Finetune** | `finetune` | False | `True` khi train pruned model |
| **DMS** | `dms_target` | 0.3 | 0.2 ~ 0.5 |
| | `dms_lambda` | 100.0 | 100 ~ 1000 |
| | `dms_lr` | 4e-4 | constant, khong decay |
| | `dms_taylor_type` | "taylor" | "taylor" / "snip" / "fisher" |
| | `dms_decay_ratio` | 0.8 | 0.7 ~ 0.9 |
| | `dms_grad_scale` | -1.0 | -1 (OFF) hoac 1.0 (auto) |
| **KD** | `kd_method` | "cwd" | "cwd" / "response" / "fitnets" / "mgd" |
| | `kd_lambda` | 0.5 | 0.3 ~ 1.0 |
| | `kd_layers` | "neck" | "neck" (nhanh) / "all" (chinh xac) |
| | `kd_warmup` | 5 | 1 ~ 5 |
| | `cwd_temperature` | 9.0 | 4.0 ~ 15.0 (grid search) |
| **Prune** | `--prune-ratio` | 0.5 | 0.2 ~ 0.7 |
| | `--divisor` | 8 | 8 (GPU) / 16 (Tensor Cores) |
| | `--model-size` | m | n / s / m / l / x |

### Luu y quan trong

1. **`finetune=True` bat buoc** khi train pruned model.
2. **Khong ket hop** dms va kd cung luc. Chi dung 1 che do moi lan train.
3. **DMS tuong thich AMP**, KD co warmup de tranh NaN.
4. **DMS can extract ratios** truoc khi prune (`python dms/extract_ratios.py`).
5. **KD can teacher model** chua prune (original best.pt).
6. **Tren Windows** phai co `if __name__ == '__main__':` trong script.
7. **L1 norm la method chinh** — khong can Sparsity Training, prune truc tiep tu pretrained.

---

# Phan II — Chi tiet ky thuat

## Cau truc du an

```
yolo/
├── pruning/                            # 6 pruning methods
│   ├── prune_common.py                 # Shared pipeline: load_and_prepare, create_masks, finalize_pruning
│   ├── prune_bn_gamma.py              # BN gamma magnitude pruning
│   ├── prune_l1norm.py                # L1 norm pruning (method chinh)
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
│   ├── train_dms.py                   # DMS search training
│   ├── finetune.py                    # Finetune pruned model
│   ├── finetune_cwd.py                # Finetune with CWD distillation
│   └── finetune_kd.py                 # Finetune with KD (Response/FitNets/MGD)
│
├── tools/                              # Debug & benchmark tools
│   ├── speed.py                       # Benchmark: GFLOPs, Params, Latency, FPS
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
│   │   └── trainer.py                 # BaseTrainer - xu ly DMS/KD/finetune
│   └── nn/
│       ├── tasks_pruned.py            # Build pruned model tu masks
│       └── modules/
│           ├── block_pruned.py        # C3k2Pruned, SPPFPruned, C2PSAPruned, ...
│           └── head_pruned.py         # DetectPruned
│
└── weights/                           # Model weights
```

---

## DMS — Ben trong

### Y tuong cot loi

Thay vi dat 1 prune ratio chung cho tat ca layer (vd 50%), DMS hoc **per-layer ratio** bang gradient descent. Moi layer co 1 learnable parameter `a` dai dien cho ti le kenh duoc giu lai.

### Soft Mask Pipeline

Moi layer co N channels, tham so `a ∈ [16/N, 1-8/N]`, Taylor buffer T ∈ R^N.

**Buoc 1 — Taylor Importance**: Tinh dong gop cua tung channel toi loss.
```
T_new = (mask × grad)²                    # first-order Taylor expansion
T ← 0.99 × T + 0.01 × T_new              # EMA on dinh qua nhieu batch
```
- `grad` phai chia cho AMP scale truoc khi tinh (vi AMP nhan gradient ~65536x)
- Variants: taylor `(m·g)²`, snip `|m·g|`, fisher `g²`

**Buoc 2 — STE Differentiable Ranking**: Sap xep channels ma van co gradient.
```
vm = T_i - T_j                             # pairwise difference [N×N]
c = (vm >= 0).float() - vm.detach() + vm   # STE trick
c_ranked = mean(c, dim=-1)                 # rank ∈ [0,1]
c_prime = 1 - c_ranked                     # dao: quan trong → gia tri thap
```
STE (Straight-Through Estimator): forward dung hard comparison (chinh xac),
backward truyen gradient nhu linear (co gradient).

**Buoc 3 — Sigmoid Mask**:
```
mask = sigmoid(-(c_prime - a) × N)
```
- `a` la nguong cat (learnable): a lon → giu nhieu, a nho → cat nhieu
- `N` lam mask gan binary: channel duoc giu (≈1) hoac cat (≈0)

### Ap dung mask

Mask nhan vao **input cua Conv2d** (forward pre-hook):
```
Conv.input ← mask × Conv.input
```

### Loss

```
L = L_detect + λ × L_resource
L_resource = log(FLOPs_hien_tai / FLOPs_target)   khi > target, else 0
```
FLOPs_hien_tai tinh qua **mask** (sigmoid), KHONG qua `(1-a)` truc tiep.
`soft_mask_sum` dung STE: forward = hard count, backward = soft sum (qua sigmoid).

### 2-Phase Scheduler

- Phase 1 [0%, 80%): Progressive target tang dan `1-(1-final)^ratio`. Cho Taylor importance thoi gian tich luy.
- Phase 2 [decay_ratio, 100%]: Fixed target, resource loss van ON (stabilize).

### Gradient normalization (dms_grad_scale)

Matching `DMSMutator.norm_gradient()` tu ICML 2024 repo.
- **Van de**: detection gradient >> resource gradient tren `a` → can lambda lon (100~1000)
- **Giai phap**: tu dong normalize flop grad L2 norm = task grad L2 norm (EMA tracking)
- **Khi ON** (`dms_grad_scale >= 0`): auto balance, lambda khong anh huong gradient
- **Khi OFF** (`dms_grad_scale=-1`): can lambda thu cong de can bang

### Clamp

- `a_min = 16/N`: moi layer giu toi thieu 16 channels (tranh bottleneck)
- `a_max = 1 - 8/N`: dam bao ket qua chia het cho 8 (GPU alignment)

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

### ignore_bn_list

BN layers khong duoc prune (de bao ve residual connections):
- Bottleneck co `add=True`: ignore cv2.bn + parent cv1.bn
- PSABlock: ignore tat ca BN ben trong + parent cv1.bn

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
