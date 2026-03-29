# Pipeline Chi Tiet — YOLOv26 Pruning + Distillation

## 1. Tong quan

De tai: **Structured Channel Pruning ket hop Knowledge Distillation cho YOLOv26 object detection**, huong toi giam kich thuoc model va tang toc inference tren edge devices ma giu accuracy.

### Pipeline chinh

```
Baseline (YOLOv26m, pretrained)
  │
  ├─── Pipeline A (Uniform ratio — method chinh)
  │      │
  │      ├─ L1 Norm Pruning (uniform 30%)
  │      └─ Finetune + CWD ──→ Final Model A
  │
  └─── Pipeline B (Per-layer ratio — thu nghiem)
         │
         ├─ DMS Taylor Search (10 epochs) ──→ per-layer ratios YAML
         ├─ L1 Norm Pruning (per-layer ratio)
         └─ Finetune + CWD ──→ Final Model B
```

### So sanh 2 pipeline

| | Pipeline A (Uniform) | Pipeline B (DMS per-layer) |
|---|---|---|
| Pruning ratio | Cung 30% moi layer | Moi layer ratio rieng (DMS tim) |
| Ket qua | mAP50-95 >= baseline | Kem uniform (bottleneck) |
| Do phuc tap | Don gian, 2 buoc | Phuc tap, 3 buoc |
| Ket luan | **Method chinh** | Dang optimize (Taylor) |

---

## 2. Chi tiet tung buoc

### Buoc 0: Baseline

Train YOLOv26m tren VOC dataset tu pretrained COCO weights.

```python
from ultralytics import YOLO
model = YOLO("yolo26m.pt")  # pretrained COCO
model.train(data="VOC.yaml", epochs=100, imgsz=640)
```

- Output: `weights/yolo26m_baseline.pt`
- Params: 21.80M | GFLOPs: 67.9

### Buoc 1 (Optional): DMS Search — Tim per-layer pruning ratio

> Chi can cho Pipeline B. Pipeline A bo qua buoc nay.

**Muc tieu**: Thay vi cat cung 30% moi layer, DMS dung gradient descent de hoc layer nao nen cat nhieu, layer nao nen giu lai.

```python
model = YOLO("weights/yolo26m_baseline.pt")
model.train(
    data="VOC.yaml", epochs=10,
    dms=True,
    dms_target=0.3,           # muon giam 30% FLOPs
    dms_lambda=100.0,         # trong so resource loss
    dms_lr=4e-4,              # lr cho learnable params
    dms_taylor_type="taylor", # Taylor first-order importance
    dms_decay_ratio=0.8,      # 80% epochs progressive target
)
```

Extract ratios:
```bash
python dms/extract_ratios.py --ckpt runs/.../best.pt --output dms_ratios.yaml
```

**Cach DMS hoat dong**:

1. **Moi layer** co tham so `a` (learnable) dieu khien ti le channels giu lai
2. **Taylor importance**: tinh dong gop cua tung channel toi loss = `(mask * grad)^2`
3. **Soft mask**: `mask = sigmoid(-(rank - a) * N)` — differentiable, gradient chay qua duoc
4. **2 loss**: `L = L_detect + lambda * L_resource`
   - `L_detect`: detection loss binh thuong
   - `L_resource = log(FLOPs_hien_tai / FLOPs_target)`: phat khi FLOPs vuot target
5. **2-phase scheduler**:
   - Phase 1 [0%, 80%): target tang dan (cho importance tich luy)
   - Phase 2 [80%, 100%]: target co dinh (stabilize)
6. Sau training, extract `a` → per-layer pruning ratio

**Ket qua DMS**:
- DMS + L1 importance: that bai (magnitude ≠ importance, tao bottleneck o 1 so layer 90%+)
- DMS + Taylor importance: dang optimize, co trien vong hon vi gradient-based

### Buoc 2: Structured Channel Pruning

6 phuong phap pruning. **L1 Norm la method chinh** (ket qua tot nhat).

```bash
# Pipeline A: Uniform ratio
python pruning/prune_l1norm.py \
    --weights weights/yolo26m_baseline.pt \
    --cfg cfg/yolo26m.yaml \
    --model-size m \
    --prune-ratio 0.3 \
    --divisor 8

# Pipeline B: DMS per-layer ratio
python pruning/prune_l1norm.py \
    --weights weights/yolo26m_baseline.pt \
    --cfg cfg/yolo26m.yaml \
    --prune-ratio 0.3 \
    --layer-ratio dms_ratios.yaml \
    --divisor 8
```

**6 pruning methods**:

| Method | Mo ta | Basis |
|--------|-------|-------|
| **L1 Norm** | L1 norm cua Conv filter weights | Magnitude |
| BN Gamma | BN scaling factor magnitude | Magnitude |
| Taylor | Gradient * activation (can data) | Gradient |
| LAMP | Layer-adaptive, can bang sensitivity | Magnitude + adaptive |
| FPGM | Geometric median distance | Geometry |
| Random | Random selection (baseline) | Random |

**Quy trinh pruning** (chung cho ca 6):
1. Load model, xay dung danh sach BN khong duoc prune (residual, PSABlock)
2. Tinh importance score cho tung channel (method-specific)
3. Tao binary mask dua tren importance + prune ratio
4. Build `DetectionModelPruned` voi so channels moi
5. Copy weights theo mask, luu maskbndict

Output: `.pt` file chua model pruned + `maskbndict`

### Buoc 3: Finetune + CWD (Channel-Wise Distillation)

Finetune pruned model de recover accuracy. Dung CWD de student (pruned) hoc tu teacher (baseline).

```python
model = YOLO("weights/yolo26m_baseline_l1norm_div8.pt")
model.train(
    data="VOC.yaml",
    epochs=100,
    imgsz=640,
    finetune=True,              # BAT BUOC cho pruned model
    kd=True,                    # bat KD pipeline
    kd_teacher="weights/yolo26m_baseline.pt",
    kd_method="cwd",            # Channel-Wise Distillation
    kd_lambda=0.5,              # trong so KD loss
    kd_layers="neck",           # distill o neck layers
    kd_warmup=5,                # 5 epoch warmup tranh NaN
    cwd_temperature=9.0,        # temperature cho softmax
)
```

Output: `weights/prune_kd_best.pt`

**CWD (Channel-Wise Distillation)**:
- Spatial softmax per channel: `phi(y_c) = softmax(y_c / tau, dim=spatial)`
- KL divergence giua teacher va student per channel
- Channel alignment: dung `maskbndict` de chon subset channels teacher khop student (khong can projection layer)
- Warmup: 5 epoch dau chi train detection loss, sau do ramp up KD loss tu 0 → full trong 5 epochs tiep (tranh NaN do AMP scaler)

**4 KD methods ho tro**:

| Method | Loai | Mo ta |
|--------|------|-------|
| **CWD** (default) | Feature-based | Channel-wise KL divergence |
| Response KD | Logit-based | Distill output predictions (Hinton 2015) |
| FitNets | Feature-based | MSE giua normalized feature maps |
| MGD | Feature-based | Mask channels + generator reconstruct |

### Buoc 4: Benchmark Speed

```bash
# Local
python tools/speed.py \
    --orig weights/yolo26m_baseline.pt \
    --pruned weights/prune_kd_best.pt

# Kaggle (T4 GPU) — GPU FP32 + TensorRT FP16
python tools/speed_kaggle.py \
    --weights weights/yolo26m_baseline.pt weights/prune_kd_best.pt \
    --mode gpu_fp32 tensorrt
```

---

## 3. Ket qua thuc nghiem

### Model compression

| Metric | Baseline (YOLOv26m) | Pruned + CWD | Giam |
|--------|---------------------|--------------|------|
| Parameters | 21.80M | 7.50M | 65.6% |
| GFLOPs | 67.9 | 23.3 | 65.7% |
| Model size | 44 MB (stripped) | 15 MB | 66% |

### Accuracy (VOC)

| Model | mAP50 | mAP50-95 |
|-------|-------|----------|
| Baseline (YOLOv26m) | - | - |
| Pruned + CWD | - | - |

> Dien ket qua khi co.

### Inference Speed (Tesla T4, batch=1)

| Model | GPU FP32 (ms) | FPS | TensorRT FP16 (ms) | FPS |
|-------|---------------|-----|---------------------|-----|
| Baseline | 26.7 | 37.4 | 6.4 | 155.5 |
| Pruned + CWD | 16.4 | 61.1 | 3.6 | 280.6 |
| **Speedup** | | **1.63x** | | **1.80x** |

### Tai sao TensorRT nhanh hon nhieu (~4x so voi PyTorch FP32)

- FP32 → FP16: ~2x (T4 tensor cores)
- PyTorch → TensorRT: ~2x (kernel fusion, Conv+BN+ReLU gop thanh 1 kernel, loai bo op thua)
- 2x * 2x ≈ 4x

---

## 4. Cac van de da gap va cach giai quyet

### DMS: L1/Gamma importance that bai
- **Van de**: DMS + L1 importance cho per-layer ratio "nhin hop ly" nhung 1 so layer bi cat 90%+ → information bottleneck
- **Nguyen nhan**: Magnitude (L1, gamma) ≠ importance. Weight lon khong co nghia la quan trong cho detection
- **Giai phap**: Chuyen sang Taylor importance (gradient-based, do dung contribution to loss)

### KD + AMP: NaN epoch dau
- **Van de**: KD loss qua lon o epoch 1 + AMP scaler → overflow → NaN → mAP=0 vinh vien
- **Giai phap**: Warmup 5 epochs (chi detection loss), sau do ramp up KD loss tu 0 → full

### DMS + AMP: Importance bi corrupt
- **Van de**: AMP nhan gradient len ~65536x → Taylor importance sai
- **Giai phap**: Chia gradient cho AMP scale truoc khi tinh importance

### DMS validation: dtype mismatch
- **Van de**: DMS hooks tao float32 output, AMP cast model sang float16 → RuntimeError
- **Giai phap**: Remove DMS hooks truoc validation, re-register sau

### Export pruned model: Tracer error
- **Van de**: `DetectPruned` ke thua `nn.Module` (khong phai `Detect`), exporter khong set `export=True` → forward tra ve tuple → torch.jit.trace fail
- **Giai phap**: Them `DetectPruned` vao isinstance check trong exporter

---

## 5. Cong cu ho tro

| Tool | Chuc nang |
|------|-----------|
| `tools/speed.py` | Benchmark local: GFLOPs, Params, Latency, FPS |
| `tools/speed_kaggle.py` | Benchmark Kaggle: GPU FP32, FP16, TensorRT, ONNX, CPU |
| `tools/check_sparsity.py` | Kiem tra BN sparsity cua model |
| `tools/debug_dms_flops.py` | Debug DMS FLOPs computation |
| `tools/validate_pruned.py` | Validate cau truc pruned model |
| `tools/visualize_map.py` | Visualize mAP results |
| `dms/extract_ratios.py` | Extract DMS ratios tu checkpoint → YAML |

---

## 6. References

- Liu et al., "Differentiable Model Scaling using Differentiable Topk", ICML 2024
- Shu et al., "Channel-Wise Knowledge Distillation for Dense Prediction", ICCV 2021
- Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
- Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015
- Yang et al., "Masked Generative Distillation", ECCV 2022
