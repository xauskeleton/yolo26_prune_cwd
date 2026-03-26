# YOLOv26 Pruning + Distillation Project

## Cau truc project
```
yolo/
├── pruning/                            # Pruning scripts
│   ├── prune_common.py                 # Shared pipeline: load_and_prepare, create_masks, finalize_pruning
│   ├── prune_bn_gamma.py              # BN gamma magnitude pruning (default)
│   ├── prune_l1norm.py                # L1 norm pruning
│   ├── prune_taylor.py                # Taylor importance pruning
│   ├── prune_lamp.py                  # LAMP adaptive per-layer pruning
│   ├── prune_fpgm.py                  # FPGM geometric median pruning
│   └── prune_random.py                # Random pruning (baseline)
├── distillation/                       # Knowledge distillation
│   ├── cwd_loss.py                    # CWD distillation loss
│   └── kd_losses.py                   # Response KD, FitNets, MGD losses
├── dms/                                # Differentiable Model Scaling
│   ├── dms_utils.py                   # Soft mask, resource loss, FLOPs profiling
│   └── extract_ratios.py             # Extract DMS ratios from checkpoint → YAML
├── scripts/                            # Training/finetune scripts
│   ├── train_sparsity.py              # Sparsity training (SR)
│   ├── train_dms.py                   # DMS search training
│   ├── finetune.py                    # Finetune pruned model
│   ├── finetune_cwd.py                # Finetune with CWD distillation
│   └── finetune_kd.py                 # Finetune with KD (Response/FitNets/MGD)
├── tools/                              # Debug & validation tools
│   ├── check_sparsity.py             # Check BN sparsity of model
│   ├── debug_dms_flops.py            # Debug DMS FLOPs computation
│   ├── validate_pruned.py            # Validate pruned model structure
│   └── visualize_map.py              # Visualize mAP results
├── cfg/
│   └── yolo26m.yaml                   # YAML config cua yolo26
├── ultralytics/
│   ├── engine/
│   │   ├── model.py                   # YOLO.train() - nhan tat ca custom args
│   │   └── trainer.py                 # BaseTrainer - xu ly SR/DMS/CWD/finetune
│   └── nn/
│       ├── tasks_pruned.py            # Build pruned model tu masks
│       └── modules/
│           ├── block_pruned.py        # C3k2Pruned, C3k2PrunedBn, C3k2PrunedAttn, SPPFPruned, C2PSAPruned
│           └── head_pruned.py         # DetectPruned
```

## Pipeline tong quat
```
[Chinh] Baseline → L1 norm prune (uniform ratio) → Finetune + CWD
[Thu nghiem] Baseline → DMS Taylor → L1 prune (per-layer ratio) → Finetune + CWD
```

### Ket qua thuc nghiem
- L1 uniform prune: mAP50-95 >= baseline (regularization effect), mAP50 giam ~5%
- DMS L1 per-layer ratio: kem hon uniform L1 → magnitude ≠ importance, per-layer ratio tao bottleneck
- DMS Taylor: dang optimize, gradient-based nen do dung contribution to loss

### Ket luan da rut ra
- **L1 norm pruning > BN gamma pruning** (thuc nghiem)
- **Magnitude-based importance (L1, gamma) sai cho DMS** → ratio "nhin hop ly" nhung tao information bottleneck (1 so layer bi cat 90%+)
- **SR (Sparsity Regularization) deprecated**: chi phuc vu gamma pruning, L1 norm khong can SR
- **DMS Taylor la huong duy nhat con kha thi** cho per-layer ratio optimization

## Cac che do training

### 1. Sparsity Training (SR) — DEPRECATED
> Khong dung nua. SR chi phuc vu gamma pruning, L1 norm pruning tot hon gamma nen SR khong can thiet.

L1 penalty len BN gamma de day cac kenh khong quan trong ve 0.
```python
model = YOLO("yolo26m.pt")
model.train(data="coco.yaml", epochs=100, sr=1e-3)
```
- `sr` (float): He so L1 penalty. 0 = tat. Mac dinh 0.
- Decay: sr giam dan theo epoch: sr_tmp = sr * (1 - 0.9 * epoch/epochs)
- Tu dong tat AMP khi sr > 0
- BN trong ignore_bn_list (residual, PSABlock) khong bi phat

### 2. DMS (Differentiable Model Scaling)
Tim pruning ratio toi uu cho tung layer bang gradient.
```python
model = YOLO("yolo26m.pt")
model.train(
    data="coco.yaml", epochs=10,
    dms=True,
    dms_target=0.3,          # target pruning ratio (0.3 = cat 30%)
    dms_lambda=1.0,           # trong so resource loss
    dms_lr=2e-5,              # learning rate cho a params (constant, khong decay)
    dms_taylor_type="taylor", # "taylor" (default), "snip", "fisher"
    dms_decay_ratio=0.8,      # 80% epochs: progressive target (default=0.8)
    dms_refine_ratio=0.2,     # 20% epochs: tat resource loss, refine detection (default=0.2)
)
```
- Importance: Taylor (Conv input pre-hook + STE ranking + AMP unscale)
- `dms_taylor_type`:
  - `"taylor"` (default): (mask * grad)^2 — first-order Taylor expansion
  - `"snip"`: |mask * grad| — SNIP criterion, it sensitive voi outlier
  - `"fisher"`: grad^2 — Fisher information (Hessian diagonal approximation)
- `dms_decay_ratio` (float): ti le epochs cho progressive target phase (default=0.8)
- `dms_refine_ratio` (float): ti le epochs cho refine phase, tat resource loss (default=0.2)
- **2-phase scheduler** (matching ICML 2024 paper):
  - Phase 1 [0%, 80%): Progressive target tang dan tu 0 → final_target. Formula: `target = 1-(1-final)^ratio`
  - Phase 2 [80%, 100%]: Tat resource loss hoan toan, chi train detection (refine accuracy)
- Output: checkpoint chua `dms_a_params` → extract bang `python dms/extract_ratios.py --ckpt <path>`
- Ket qua: file YAML chua per-layer ratio → dung voi `pruning/prune_*.py --layer-ratio`

### 3. Pruning (6 methods)
6 pruning methods, tat ca dung chung pipeline tu `pruning/prune_common.py`.
**L1 norm la method chinh**, cho ket qua tot nhat (mAP50-95 >= baseline sau finetune).
```bash
# L1 norm (default, method chinh)
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# BN gamma
python pruning/prune_bn_gamma.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Taylor importance
python pruning/prune_taylor.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3 --data VOC.yaml

# LAMP (adaptive per-layer)
python pruning/prune_lamp.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# FPGM (geometric median)
python pruning/prune_fpgm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Random (baseline)
python pruning/prune_random.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Voi DMS per-layer ratio (bat ky method nao)
python pruning/prune_bn_gamma.py --weights weights/best.pt --cfg cfg/yolo26m.yaml \
    --prune-ratio 0.3 --layer-ratio dms_ratios.yaml
```
Args:
- `--weights`: duong dan model.pt
- `--cfg`: duong dan YAML config
- `--model-size`: n/s/m/l/x (mac dinh m)
- `--prune-ratio`: ti le prune toan cuc 0.0-1.0
- `--divisor`: 8 hoac 16 (GPU alignment)
- `--layer-ratio`: YAML file chua custom ratio cho tung layer
- `--save-dir`: thu muc luu output

### 4. Finetune
Train lai pruned model de phuc hoi accuracy.
```python
model = YOLO("weights/pruned_div8.pt")
model.train(data="coco.yaml", epochs=100, finetune=True)
```
- `finetune=True`: load maskbndict tu checkpoint, build DetectionModelPruned

### 5. Knowledge Distillation (KD)
Knowledge distillation tu teacher model (full) sang student (pruned).
4 methods: CWD (default), Response KD, FitNets, MGD.

General KD args dung `kd_*`, CWD-specific args giu `cwd_*`.

```python
model = YOLO("weights/pruned_div8.pt")
model.train(
    data="coco.yaml", epochs=100,
    finetune=True,
    kd=True,                            # bat KD pipeline
    kd_teacher="yolo26m.pt",            # teacher model path
    kd_lambda=0.5,                      # trong so KD loss
    kd_method="cwd",                    # "cwd", "response", "fitnets", "mgd"
    kd_layers="neck",                   # "all", "neck", "backbone", hoac list indices
    kd_warmup=5,                        # so epoch warmup truoc khi bat KD (default=5)
    cwd_temperature=9.0,                # float=fixed tau, "learnable"=auto (CWD only)
)
```

#### KD Args
- `kd=True`: bat KD pipeline
- `kd_teacher`: duong dan teacher model
- `kd_lambda` (float): trong so KD loss, default 0.5
- `kd_method`: "cwd" (default), "response", "fitnets", "mgd"
- `kd_layers`: "neck" (default), "all", "backbone", hoac list indices
- `kd_warmup` (int): so epoch warmup, default 5

#### CWD-specific Args
- `cwd_temperature`: float (fixed tau, default 9.0) hoac "learnable" (auto)
- `cwd_learnable_tau_lr` (float): lr cho learnable tau, default 1e-3
- `cwd_learnable_tau_init` (float): tau khoi tao khi learnable, default 9.0

#### CWD Learnable Temperature
Tu dong hoc temperature (tau) bang gradient descent thay vi grid search.
```python
model.train(
    ...,
    kd=True, kd_teacher="yolo26m.pt",
    kd_method="cwd",
    cwd_temperature="learnable",        # bat learnable mode
    cwd_learnable_tau_lr=1e-3,          # lr rieng cho tau (Adam)
    cwd_learnable_tau_init=9.0,         # tau khoi tao
)
```
- `log_tau = nn.Parameter(log(tau_init))` → tau = exp(log_tau).clamp(0.5, 20)
- Optimizer rieng: Adam, tach khoi main optimizer
- Gradient unscale thu cong (giong DMS) vi scaler chi biet main optimizer
- Warmup + ramp up van ap dung nhu fixed CWD
- Save/resume: luu log_tau + optimizer state vao checkpoint

#### KD Methods
- `kd_method="cwd"`: Channel-Wise Distillation (default) - spatial softmax per channel → KL div
- `kd_method="response"`: Response-based KD (Hinton 2015) - channel softmax per spatial → KL div
- `kd_method="fitnets"`: FitNets (Romero 2015) - MSE giua normalized feature maps
- `kd_method="mgd"`: Masked Generative Distillation (Yang 2022) - mask channels + generator reconstruct

#### KD warmup + ramp up (fix NaN epoch 1)
- **Van de**: KD + AMP scaler gay NaN o epoch dau → mAP=0 vinh vien
- **Fix**:
  1. `kd_warmup=5`: 5 epoch dau chi train detection loss (khong KD), scaler on dinh
  2. Ramp up: sau warmup, kd_lambda tang dan tu 0 → full trong 5 epochs tiep theo
- Channel alignment dung mask tu maskbndict (prune.py tao) → chon subset channels teacher khop student, khong can projection layer

#### DMS validation fix: AMP dtype mismatch
- **Van de**: DMS hooks tao float32 output, AMP cast model sang float16 → RuntimeError khi validation
- **Fix**: Remove DMS hooks truoc validation, re-register sau validation trong `validate()` method

## dms/dms_utils.py - Cac ham co san

### Pruning helpers
- `make_divisible_channels(channels, max_channels, divisor)` → int: Lam tron channels den boi so cua divisor (8/16)
- `get_layer_ratio(layer_name, layer_ratio_cfg, default_ratio)` → float: Lay prune ratio cho 1 layer (exact match > layer index > group name > default)
- `build_pruned_yaml(cfg, model_size, nc)` → dict: Build pruned YAML tu original config, map module sang Pruned versions
- `build_ignore_bn_list(model)` → list: List BN layers khong duoc prune (residual Bottleneck, PSABlock)

### DMS core
- `DMSMaskManager(a_params, taylor_buffers, bn_modules, bn_channels, ...)`: Paper-matching mask manager
  - Conv input pre-hook (thay vi BN output hook): gradient truc tiep hon, khong qua activation
  - STE differentiable ranking: `(vm>=0).float() - vm.detach() + vm` — gradient flow qua ranking
  - AMP grad unscale: `grad / scaler` truoc khi tinh taylor — fix bug importance bi corrupt boi AMP scale
  - Taylor types: taylor `(m*g)^2`, snip `|m*g|`, fisher `g^2`
- `make_soft_mask_hook(...)` → DEPRECATED: legacy BN hook cho gamma/l1 (khong dung cho taylor nua)
- `profile_per_layer_flops(model, imgsz, device)` → (dict, float): Profile FLOPs/MACs per Conv2d layer
- `build_conv_bn_mapping(model, ignore_bn_list)` → (conv_bn_map, bn_channels): Mapping Conv2d → output BN + input BN
- `compute_resource_loss(a_params, conv_flops, conv_bn_map, bn_channels, total_flops, target_ratio)` → tensor: GFLOPs resource constraint loss
- `compute_l1_loss(model, ignore_bn_list)` → tensor: L1 penalty tren BN gamma (Σ|γ|)

### DMS extract
- `extract_ratios_from_checkpoint(ckpt_path, save_path, divisor)` → dict: Extract a params tu checkpoint → YAML file dung voi pruning/prune_*.py --layer-ratio

### Internal helpers (khong can goi truc tiep)
- `_resolve_internal_in_bn(conv_name, layer_idx, bn_channels)`: Resolve in_bn cho conv trong C3k2 blocks
- `_resolve_detect_in_bn(conv_name, layer_idx, scale_inputs)`: Resolve in_bn cho Detect head convs

## So sanh DMS importance criteria: L1 (magnitude) vs Taylor (gradient)

> **Ket luan: L1 importance da bi loai.** Magnitude-based (L1, gamma) tao bottleneck, kem uniform.
> **Taylor dang duoc optimize** — gradient-based, do dung contribution to loss.

### Setup
- Cung baseline: yolo26m, cung DMS target ~0.3, cung config
- Chi khac `dms_importance`: "l1" vs "taylor"
- Ratio files: `weights/dms_ratios_l1norm.yaml` (L1), `weights/dms_ratios_taylor_last.yaml` (Taylor)
- Ratio = ti le prune (0=giu het, 1=cat het)

### Ket qua ratio
- Mean ratio gan giong: L1=41.5%, Taylor=41.4% → cung GFLOPs (~11.4G), cung Params (~9.2M)
- 18/90 layers co diff > 20% (20% layers khac nhau dang ke)

### Phan bo per-layer (ratio = % cat)
| Component              | L1    | Taylor | Delta  |
|------------------------|-------|--------|--------|
| Backbone early (0-3)   | 47.2% | 46.3%  | -0.9%  |
| Backbone P3-P4 (4-6)   | 48.6% | 37.6%  | -11%   |
| Backbone P5 (7-8)      | 27.3% | 38.7%  | +11%   |
| SPPF + C2PSA (9-10)    | 21.6% | 31.1%  | +10%   |
| Neck fusion (13-20)    | 52.1% | 48.7%  | -3.4%  |
| Neck P5 + Attn (22)    | 7.7%  | 35.7%  | +28%   |
| Detect box head        | 29.8% | 29.3%  | -0.5%  |
| Detect cls head        | 45.0% | 46.2%  | +1.2%  |

### Tai sao L1/gamma importance that bai
- **Magnitude ≠ importance**: weight lon khong co nghia la quan trong cho detection
- **Bottleneck**: 1 so neck layers bi cat 90%+ (vd model.19.m.0.cv2: 93.75%) → nghen thong tin
- **Uniform ratio khong co van de nay**: moi layer mat it, information flow giu nguyen
- Taylor dua tren gradient nen do dung contribution, phan bo deu hon, tranh bottleneck

## Files chinh da chinh sua
- `ultralytics/engine/model.py`: them args SR/DMS/KD/finetune vao train()
- `ultralytics/engine/trainer.py`: xu ly setup + training loop cho tat ca modes (KD general args: kd_*, CWD-specific: cwd_*)
- `pruning/prune_common.py`: shared pruning pipeline
- `pruning/prune_*.py`: 6 pruning methods
- `dms/dms_utils.py`: soft mask hooks, resource loss, FLOPs profiling
- `distillation/cwd_loss.py`: CWD loss, feature hooks, channel alignment
- `distillation/kd_losses.py`: Response KD, FitNets, MGD losses
- `ultralytics/nn/tasks_pruned.py`: parse_model_pruned, DetectionModelPruned
- `ultralytics/nn/modules/block_pruned.py`: cac module pruned
- `ultralytics/nn/modules/head_pruned.py`: DetectPruned

## Test commands (5 sizes)
```bash
python pruning/prune_l1norm.py --weights weights/yolo26n.pt --cfg cfg/yolo26m.yaml --model-size n --prune-ratio 0.5 --divisor 8
python pruning/prune_l1norm.py --weights weights/yolo26s.pt --cfg cfg/yolo26m.yaml --model-size s --prune-ratio 0.5 --divisor 8
python pruning/prune_l1norm.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml --model-size m --prune-ratio 0.5 --divisor 8
python pruning/prune_l1norm.py --weights weights/yolo26l.pt --cfg cfg/yolo26m.yaml --model-size l --prune-ratio 0.5 --divisor 8
python pruning/prune_l1norm.py --weights weights/yolo26x.pt --cfg cfg/yolo26m.yaml --model-size x --prune-ratio 0.5 --divisor 8
```
