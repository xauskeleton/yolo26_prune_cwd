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
[1] Train baseline → [2] Sparsity training (SR) → [3] DMS search (optional)
→ [4] Prune (prune.py) → [5] Finetune → [6] CWD distillation (optional)
```

## Cac che do training

### 1. Sparsity Training (SR)
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
    data="coco.yaml", epochs=30,
    dms=True,
    dms_target=0.3,      # target pruning ratio (0.3 = cat 30%)
    dms_lambda=1.0,       # trong so resource loss
    dms_lr=5e-3,          # learning rate cho a params
    dms_freeze=False,     # True = dong bang model, chi train a
    dms_importance="gamma", # 'gamma', 'taylor', hoac 'l1'
    dms_warmup=3,         # so epoch warmup truoc khi DMS bat dau (default=0)
)
```
- `dms_importance`: criterion de sort channels trong soft mask
  - `"gamma"`: |BN.weight| (mac dinh, giong paper goc)
  - `"taylor"`: gradient-based (mask * grad)^2 voi EMA, ly thuyet tot nhat nhung cham hon
  - `"l1"`: L1 norm cua Conv filter weights, consistent voi L1 norm pruning
- `dms_warmup`: so epoch warmup truoc khi DMS bat dau (default=0)
  - Trong warmup: model train binh thuong, a params dong bang, khong co resource loss
  - Sau warmup: resource loss ramp up linearly trong 5 epochs
- Output: checkpoint chua `dms_a_params` → extract bang `python dms/extract_ratios.py --ckpt <path>`
- Ket qua: file YAML chua per-layer ratio → dung voi `pruning/prune_*.py --layer-ratio`

### 3. Pruning (6 methods)
6 pruning methods, tat ca dung chung pipeline tu `pruning/prune_common.py`.
```bash
# BN gamma (default)
python pruning/prune_bn_gamma.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# L1 norm
python pruning/prune_l1norm.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

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

### 5. CWD (Channel-Wise Distillation)
Knowledge distillation tu teacher model (full) sang student (pruned).
```python
model = YOLO("weights/pruned_div8.pt")
model.train(
    data="coco.yaml", epochs=100,
    finetune=True,
    cwd=True,
    cwd_teacher="yolo26m.pt",      # teacher model path
    cwd_lambda=0.5,                 # trong so CWD loss
    cwd_temperature="dynamic",      # "dynamic" hoac float (vd: 6.0)
    tau_max=10.0,                   # nhiet do max (dynamic mode)
    tau_min=1.0,                    # nhiet do min (dynamic mode)
    cwd_layers="all",               # "all", "neck", "backbone", hoac list indices
    cwd_warmup=3,                   # so epoch warmup truoc khi bat CWD (default=3)
    cwd_layer_weights={             # trong so rieng cho tung layer (optional)
        2: 0.3, 4: 0.3, 6: 0.5, 8: 0.5,
        13: 1.0, 16: 1.0, 19: 1.0, 22: 1.5,
    },
)
```

### 6. CWD Learnable Temperature
Tu dong hoc temperature (tau) cua CWD bang gradient descent thay vi grid search.
Chi hoc tau — lambda va layer_weights giu fixed (learnable lambda/layer_weights khong on dinh).

```python
model = YOLO("weights/pruned_div8.pt")
model.train(
    data="coco.yaml", epochs=100,
    finetune=True,
    cwd=True,
    cwd_teacher="yolo26m.pt",
    cwd_learnable_tau=True,           # bat learnable temperature
    cwd_learnable_tau_lr=1e-3,        # lr rieng cho tau (Adam)
    cwd_learnable_tau_init=6.0,       # tau khoi tao
    cwd_lambda=0.5,                   # lambda van fixed
    cwd_warmup=3,                     # warmup van ap dung
    cwd_layers="all",                 # chon layers distill
    cwd_layer_weights={               # layer weights van fixed (optional)
        2: 0.3, 4: 0.3, 6: 0.5, 8: 0.5,
        13: 1.0, 16: 1.0, 19: 1.0, 22: 1.5,
    },
)
```

#### Y tuong
- Grid search tau can nhieu lan train → tot thoi gian
- Tau co loss landscape smooth → gradient tim duoc gia tri tot
- Lambda va layer_weights KHONG learnable (lambda bat on, layer_weights collapse ve 1 layer)
- Chi 1 learnable parameter: `log_tau` (tau = exp(log_tau), luon duong)

#### Implementation details
- `log_tau = nn.Parameter(log(tau_init))` → tau = exp(log_tau).clamp(0.5, 20)
- Optimizer rieng: Adam lr=1e-3, tach khoi main optimizer
- Gradient unscale thu cong (giong DMS) vi scaler chi biet main optimizer
- Clamp: log_tau ∈ [log(0.5), log(20)] tranh degenerate
- CWDLoss.forward KHONG can sua — PyTorch tu handle tensor tau, gradient flow tu dong
- Warmup + ramp up van ap dung nhu fixed CWD
- Save/resume: luu log_tau + optimizer state vao checkpoint
- Epoch logging: in tau hien tai va log_tau

#### Files da sua
- `model.py`: them 3 args (cwd_learnable_tau, cwd_learnable_tau_lr, cwd_learnable_tau_init)
- `trainer.py`: setup log_tau param, optimizer rieng, optimizer step, logging, save/resume

#### CWD fix: warmup + ramp up (fix NaN epoch 1)
- **Van de**: CWD + AMP scaler gay NaN o epoch dau → mAP=0 vinh vien
- **Nguyen nhan**: CWD loss qua lon khi chua warmup, scaler scale len → overflow → NaN gradients → scaler skip optimizer step → model hong
- **Fix hien tai**:
  1. `cwd_warmup=3`: 3 epoch dau chi train detection loss (khong CWD), scaler on dinh
  2. Ramp up: sau warmup, cwd_lambda tang dan tu 0 → full trong 5 epochs tiep theo
  3. Dynamic temp progress tinh tu epoch warmup thay vi epoch 0
- **Scaler**: chua chac chan scaler lam model tot hon. Dang thu nghiem. Bo scaler thi khong NaN nhung mAP thap hon finetune thong thuong (co the do nguyen nhan khac). Can test them de ket luan.
- Channel alignment dung mask tu maskbndict (prune.py tao) → chon subset channels teacher khop student, khong can projection layer

### 7. KD Methods (Response KD, FitNets, MGD)
Ngoai CWD, co them 3 phuong phap KD khac, chon qua `kd_method` arg.
```python
model = YOLO("weights/pruned_div8.pt")
model.train(
    data="coco.yaml", epochs=100,
    finetune=True,
    cwd=True,                          # bat KD pipeline
    cwd_teacher="yolo26m.pt",          # teacher model
    kd_method="response",              # "cwd", "response", "fitnets", "mgd"
    cwd_lambda=0.5,                    # trong so KD loss
)
```
- `kd_method="cwd"`: Channel-Wise Distillation (default, feature-based)
- `kd_method="response"`: Response-based KD (logit distillation)
- `kd_method="fitnets"`: FitNets (intermediate feature mimicking)
- `kd_method="mgd"`: Masked Generative Distillation

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
- `make_soft_mask_hook(bn_name, a_params, importance, taylor_buffers)` → hook: Forward hook ap soft mask sau BN (mask = Sigmoid(N*(c'-a)))
- `profile_per_layer_flops(model, imgsz, device)` → (dict, float): Profile FLOPs/MACs per Conv2d layer
- `build_conv_bn_mapping(model, ignore_bn_list)` → (conv_bn_map, bn_channels): Mapping Conv2d → output BN + input BN
- `compute_resource_loss(a_params, conv_flops, conv_bn_map, bn_channels, total_flops, target_ratio)` → tensor: GFLOPs resource constraint loss
- `compute_l1_loss(model, ignore_bn_list)` → tensor: L1 penalty tren BN gamma (Σ|γ|)

### DMS extract
- `extract_ratios_from_checkpoint(ckpt_path, save_path, divisor)` → dict: Extract a params tu checkpoint → YAML file dung voi pruning/prune_*.py --layer-ratio

### Internal helpers (khong can goi truc tiep)
- `_resolve_internal_in_bn(conv_name, layer_idx, bn_channels)`: Resolve in_bn cho conv trong C3k2 blocks
- `_resolve_detect_in_bn(conv_name, layer_idx, scale_inputs)`: Resolve in_bn cho Detect head convs

## Files chinh da chinh sua
- `ultralytics/engine/model.py`: them args SR/DMS/CWD/KD/finetune vao train()
- `ultralytics/engine/trainer.py`: xu ly setup + training loop cho tat ca modes
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
python pruning/prune_bn_gamma.py --weights weights/yolo26n.pt --cfg cfg/yolo26m.yaml --model-size n --prune-ratio 0.5 --divisor 8
python pruning/prune_bn_gamma.py --weights weights/yolo26s.pt --cfg cfg/yolo26m.yaml --model-size s --prune-ratio 0.5 --divisor 8
python pruning/prune_bn_gamma.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml --model-size m --prune-ratio 0.5 --divisor 8
python pruning/prune_bn_gamma.py --weights weights/yolo26l.pt --cfg cfg/yolo26m.yaml --model-size l --prune-ratio 0.5 --divisor 8
python pruning/prune_bn_gamma.py --weights weights/yolo26x.pt --cfg cfg/yolo26m.yaml --model-size x --prune-ratio 0.5 --divisor 8
```
