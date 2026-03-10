# YOLOv26 Pruning + Distillation Project

## Cau truc project
```
yolo/
├── prune.py                            # Pruning script chinh
├── finetune.py                         # Finetune pruned model
├── dms_utils.py                        # DMS utilities (soft mask, resource loss, etc.)
├── cwd_loss.py                         # CWD distillation loss
├── cfg/
│   └── yolo26m.yaml                    # YAML config cua yolo26
├── ultralytics/
│   ├── engine/
│   │   ├── model.py                    # YOLO.train() - nhan tat ca custom args
│   │   └── trainer.py                  # BaseTrainer - xu ly SR/DMS/CWD/finetune
│   └── nn/
│       ├── tasks_pruned.py             # Build pruned model tu masks
│       └── modules/
│           ├── block_pruned.py         # C3k2Pruned, C3k2PrunedBn, C3k2PrunedAttn, SPPFPruned, C2PSAPruned
│           └── head_pruned.py          # DetectPruned
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
    dms_importance="gamma" # 'gamma' hoac 'taylor'
)
```
- Output: checkpoint chua `dms_a_params` → extract bang `dms_utils.extract_ratios_from_checkpoint()`
- Ket qua: file YAML chua per-layer ratio → dung voi `prune.py --layer-ratio`

### 3. Pruning (prune.py)
Cat kenh dua tren BN gamma magnitude.
```bash
# Co ban
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

# Day du
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml \
    --model-size m --prune-ratio 0.5 --divisor 8 --save-dir weights/

# Voi DMS per-layer ratio
python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml \
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

### 6. CWD Learnable (Auto-tune Hyperparameters)
Tu dong hoc cac hyperparameters cua CWD bang gradient descent thay vi grid search.
Dua tren Kendall et al. (2018) "Multi-Task Learning Using Uncertainty to Weigh Losses".

```python
model = YOLO("weights/pruned_div8.pt")
model.train(
    data="coco.yaml", epochs=100,
    finetune=True,
    cwd=True,
    cwd_teacher="yolo26m.pt",
    cwd_learnable=True,              # bat auto-tune mode
    cwd_learnable_lr=1e-3,           # lr rieng cho learnable params
    cwd_learnable_tau_init=6.0,      # nhiet do khoi tao
    cwd_warmup=3,                    # warmup van ap dung
    cwd_layers="all",                # chon layers distill
    # cwd_lambda, cwd_temperature, cwd_layer_weights bi BO QUA khi cwd_learnable=True
)
```

#### Y tuong
- Thay vi grid search N lan train de tim tau, lambda, layer_weights toi uu
  → model tu hoc trong 1 lan train bang gradient-based optimization
- 4 learnable parameters (nn.Parameter):
  1. `log_sigma_det`: uncertainty weight cho detection loss
  2. `log_sigma_cwd`: uncertainty weight cho CWD loss
  3. `log_tau`: temperature (tau = exp(log_tau), luon duong)
  4. `cwd_log_layer_weights`: trong so per-layer (softmax → sum=1)

#### Loss formulation (Kendall et al. 2018)
```
L = exp(-log_σ_det) × L_det + log_σ_det
  + ramp × (exp(-log_σ_cwd) × L_cwd + log_σ_cwd)
```
- Term `log_σ` la regularization, ngan σ→∞ (model bo loss)
- `exp(-log_σ)` la precision = 1/(2σ²), tu dong can bang 2 loss
- Tau la tensor tren computation graph → gradient flow qua softmax/KL div
- Layer weights dung softmax → bounded, sum=1

#### Implementation details
- Optimizer rieng: Adam lr=1e-3, tach khoi main optimizer (giong DMS)
- Clamp: tau ∈ [0.5, 20], log_sigma ∈ [-5, 5] tranh degenerate
- CWDLoss.forward KHONG can sua — PyTorch tu handle tensor tau
- Gradient cua learnable params can unscale manually (giong DMS) vi scaler chi biet main optimizer
- Warmup + ramp up van ap dung nhu fixed CWD
- Save/resume: luu learnable params + optimizer state vao checkpoint

#### Files can sua
- `model.py`: them 3 args (cwd_learnable, cwd_learnable_lr, cwd_learnable_tau_init)
- `trainer.py`: setup learnable params, optimizer rieng, loss computation, optimizer step, logging, save/resume
- `cwd_loss.py`: them `compute_cwd_loss_learnable()` nhan tensor tau va layer weights

#### Ky vong ket qua
- 60-70%: bang fixed CWD (van la contribution: 1 lan train thay vi 10+ lan grid search)
- 20%: tot hon fixed CWD 0.1-0.3 AP50
- 10%: kem hon do instability

#### CWD fix: warmup + ramp up (fix NaN epoch 1)
- **Van de**: CWD + AMP scaler gay NaN o epoch dau → mAP=0 vinh vien
- **Nguyen nhan**: CWD loss qua lon khi chua warmup, scaler scale len → overflow → NaN gradients → scaler skip optimizer step → model hong
- **Fix hien tai**:
  1. `cwd_warmup=3`: 3 epoch dau chi train detection loss (khong CWD), scaler on dinh
  2. Ramp up: sau warmup, cwd_lambda tang dan tu 0 → full trong 5 epochs tiep theo
  3. Dynamic temp progress tinh tu epoch warmup thay vi epoch 0
- **Scaler**: chua chac chan scaler lam model tot hon. Dang thu nghiem. Bo scaler thi khong NaN nhung mAP thap hon finetune thong thuong (co the do nguyen nhan khac). Can test them de ket luan.
- Channel alignment dung mask tu maskbndict (prune.py tao) → chon subset channels teacher khop student, khong can projection layer

## dms_utils.py - Cac ham co san

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
- `extract_ratios_from_checkpoint(ckpt_path, save_path, divisor)` → dict: Extract a params tu checkpoint → YAML file dung voi prune.py --layer-ratio

### Internal helpers (khong can goi truc tiep)
- `_resolve_internal_in_bn(conv_name, layer_idx, bn_channels)`: Resolve in_bn cho conv trong C3k2 blocks
- `_resolve_detect_in_bn(conv_name, layer_idx, scale_inputs)`: Resolve in_bn cho Detect head convs

## Files chinh da chinh sua
- `ultralytics/engine/model.py`: them args SR/DMS/CWD/finetune vao train()
- `ultralytics/engine/trainer.py`: xu ly setup + training loop cho tat ca modes
- `prune.py`: script pruning chinh
- `dms_utils.py`: soft mask hooks, resource loss, FLOPs profiling
- `cwd_loss.py`: CWD loss, feature hooks, channel alignment
- `ultralytics/nn/tasks_pruned.py`: parse_model_pruned, DetectionModelPruned
- `ultralytics/nn/modules/block_pruned.py`: cac module pruned
- `ultralytics/nn/modules/head_pruned.py`: DetectPruned

## Test commands (5 sizes)
```bash
python prune.py --weights weights/yolo26n.pt --cfg cfg/yolo26m.yaml --model-size n --prune-ratio 0.5 --divisor 8
python prune.py --weights weights/yolo26s.pt --cfg cfg/yolo26m.yaml --model-size s --prune-ratio 0.5 --divisor 8
python prune.py --weights weights/yolo26m.pt --cfg cfg/yolo26m.yaml --model-size m --prune-ratio 0.5 --divisor 8
python prune.py --weights weights/yolo26l.pt --cfg cfg/yolo26m.yaml --model-size l --prune-ratio 0.5 --divisor 8
python prune.py --weights weights/yolo26x.pt --cfg cfg/yolo26m.yaml --model-size x --prune-ratio 0.5 --divisor 8
```
