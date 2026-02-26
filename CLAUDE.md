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
    cwd_layer_weights={             # trong so rieng cho tung layer (optional)
        2: 0.3, 4: 0.3, 6: 0.5, 8: 0.5,
        13: 1.0, 16: 1.0, 19: 1.0, 22: 1.5,
    },
)
```

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
