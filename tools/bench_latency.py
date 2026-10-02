"""Benchmark latency/FPS: PyTorch FP32 vs TensorRT FP16 for all models in ./benchmark/.

Steps:
  1. PyTorch FP32 forward pass (CUDA events timing, sigma clipping)
  2. Export to TensorRT FP16 engine
  3. TensorRT FP16 inference (AutoBackend, CUDA events timing)
  4. Print combined table + CSV

Usage:
    python tools/bench_latency.py
    python tools/bench_latency.py --warmup 200 --reps 100
    python tools/bench_latency.py --skip-trt
    python tools/bench_latency.py --skip-pytorch
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend


def sigma_clip(data, sigma=2, max_iters=3):
    """Iterative sigma clipping — loai outlier."""
    data = np.array(data)
    for _ in range(max_iters):
        mean, std = np.mean(data), np.std(data)
        if std == 0:
            break
        clipped = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
        if len(clipped) == len(data):
            break
        data = clipped
    return data


def bench_pytorch(model_path, imgsz, warmup, reps):
    """Benchmark PyTorch FP32 forward pass."""
    model = YOLO(str(model_path))
    model.to("cuda")
    nn = model.model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup):
            nn(dummy)
    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            nn(dummy)
            e.record()
            torch.cuda.synchronize()
            timings.append(s.elapsed_time(e))

    clean = sigma_clip(timings)
    lat = float(np.median(clean))
    del model, nn, dummy
    torch.cuda.empty_cache()
    return lat


def bench_tensorrt(model_path, imgsz, warmup, reps):
    """Export to TensorRT FP16 and benchmark."""
    model = YOLO(str(model_path))
    exported = model.export(format="engine", imgsz=imgsz, half=True, device=0)

    backend = AutoBackend(exported, device=torch.device("cuda:0"), fp16=True)
    backend.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz, device="cuda", dtype=torch.float16)

    with torch.no_grad():
        for _ in range(warmup):
            backend(dummy)
    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            backend(dummy)
            e.record()
            torch.cuda.synchronize()
            timings.append(s.elapsed_time(e))

    clean = sigma_clip(timings)
    lat = float(np.median(clean))
    engine_size = Path(exported).stat().st_size / (1024 * 1024)
    del model, backend, dummy
    torch.cuda.empty_cache()
    return lat, engine_size


# Fixed mAP values from validation results
FIXED_DATA = {
    ("YOLO26n", "baseline"): {"params": 2.51, "size_mb": 5.2, "gflops": 5.8, "ap50": 84.76, "ap50_95": 65.82},
    ("YOLO26n", "pruned"): {"params": 1.8, "size_mb": 2.5, "gflops": 2.6, "ap50": 79.32, "ap50_95": 59.56},
    ("YOLO26s", "baseline"): {"params": 9.96, "size_mb": 19.4, "gflops": 22.6, "ap50": 87.06, "ap50_95": 69.73},
    ("YOLO26s", "pruned"): {"params": 4.4, "size_mb": 8.2, "gflops": 8.4, "ap50": 84.55, "ap50_95": 66.20},
    ("YOLO26m", "baseline"): {"params": 21.80, "size_mb": 42.1, "gflops": 74.9, "ap50": 89.04, "ap50_95": 72.49},
    ("YOLO26m", "pruned"): {"params": 7.50, "size_mb": 14.9, "gflops": 23.6, "ap50": 87.96, "ap50_95": 70.17},
    ("YOLO26l", "baseline"): {"params": 26.21, "size_mb": 50.6, "gflops": 93.3, "ap50": 89.59, "ap50_95": 73.70},
    ("YOLO26l", "pruned"): {"params": 9.70, "size_mb": 19.3, "gflops": 31.9, "ap50": 88.50, "ap50_95": 71.5},
}

MODELS = [
    ("YOLO26n", "baseline", "benchmark/n/baseline/yolo26n_baseline.pt"),
    ("YOLO26n", "pruned", "benchmark/n/bm/yolo26n_pruned_best.pt"),
    ("YOLO26s", "baseline", "benchmark/s/baseline/yolo26s_baseline.pt"),
    ("YOLO26s", "pruned", "benchmark/s/bm/yolo26s_pruned_best.pt"),
    ("YOLO26m", "baseline", "benchmark/m/baseline/yolo26m_baseline.pt"),
    ("YOLO26m", "pruned", "benchmark/m/bm/yolo26m_pruned_best.pt"),
    ("YOLO26l", "baseline", "benchmark/l/baseline/yolo26l_baseline.pt"),
    ("YOLO26l", "pruned", "benchmark/l/bm/yolo26l_pruned_best.pt"),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--skip-trt", action="store_true", help="Skip TensorRT benchmark")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch benchmark")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    torch.backends.cudnn.benchmark = True

    # TF32 only on Ampere+, harmless no-op on older GPUs (Maxwell/Pascal)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Image size: {args.imgsz} | Warmup: {args.warmup} | Reps: {args.reps}")
    print(f"Modes: {'PyTorch FP32' if not args.skip_pytorch else ''} {'TensorRT FP16' if not args.skip_trt else ''}")
    print()

    results = []

    for model_name, variant, model_path in MODELS:
        p = Path(model_path)
        if not p.exists():
            print(f"SKIP {model_name} {variant}: {model_path} not found")
            continue

        fixed = FIXED_DATA[(model_name, variant)]
        entry = {
            "model": model_name,
            "variant": variant,
            "params": fixed["params"],
            "size_mb": fixed["size_mb"],
            "gflops": fixed["gflops"],
            "ap50": fixed["ap50"],
            "ap50_95": fixed["ap50_95"],
            "pt_latency": None,
            "pt_fps": None,
            "trt_latency": None,
            "trt_fps": None,
            "trt_size_mb": None,
        }

        # PyTorch benchmark
        if not args.skip_pytorch:
            print(f"[PyTorch]   {model_name} {variant}...", end=" ", flush=True)
            try:
                lat = bench_pytorch(p, args.imgsz, args.warmup, args.reps)
                entry["pt_latency"] = lat
                entry["pt_fps"] = 1000.0 / lat
                print(f"{lat:.2f} ms  ({entry['pt_fps']:.1f} FPS)")
            except torch.cuda.OutOfMemoryError:
                print("OOM — skip (not enough GPU memory)")
                torch.cuda.empty_cache()
            except Exception as ex:
                print(f"ERROR: {ex}")

        # TensorRT benchmark
        if not args.skip_trt:
            print(f"[TensorRT]  {model_name} {variant}...", end=" ", flush=True)
            try:
                lat, eng_size = bench_tensorrt(p, args.imgsz, args.warmup, args.reps)
                entry["trt_latency"] = lat
                entry["trt_fps"] = 1000.0 / lat
                entry["trt_size_mb"] = eng_size
                print(f"{lat:.2f} ms  ({entry['trt_fps']:.1f} FPS)  engine={eng_size:.1f}MB")
            except torch.cuda.OutOfMemoryError:
                print("OOM — skip (not enough GPU memory)")
                torch.cuda.empty_cache()
            except Exception as ex:
                print(f"ERROR: {ex}")

        results.append(entry)
        print()

    # ===== COMBINED TABLE =====
    print(f"\n{'=' * 140}")
    hdr = (
        f"{'Model':<10} | {'Variant':<8} | {'Params':>7} | {'GFLOPs':>6} "
        f"| {'PT ms':>7} | {'PT FPS':>7} "
        f"| {'TRT ms':>7} | {'TRT FPS':>8} "
        f"| {'Speedup':>7} "
        f"| {'AP50':>6} | {'AP50-95':>7}"
    )
    print(hdr)
    print(f"{'-' * 140}")

    for r in results:
        pt_lat = f"{r['pt_latency']:.2f}" if r["pt_latency"] else "N/A"
        pt_fps = f"{r['pt_fps']:.1f}" if r["pt_fps"] else "N/A"
        trt_lat = f"{r['trt_latency']:.2f}" if r["trt_latency"] else "N/A"
        trt_fps = f"{r['trt_fps']:.1f}" if r["trt_fps"] else "N/A"
        if r["pt_latency"] and r["trt_latency"]:
            speedup = f"{r['pt_latency'] / r['trt_latency']:.2f}x"
        else:
            speedup = "N/A"
        print(
            f"{r['model']:<10} | {r['variant']:<8} | {r['params']:>6.2f}M | {r['gflops']:>6.1f} "
            f"| {pt_lat:>7} | {pt_fps:>7} "
            f"| {trt_lat:>7} | {trt_fps:>8} "
            f"| {speedup:>7} "
            f"| {r['ap50']:>6.2f} | {r['ap50_95']:>7.2f}"
        )
    print(f"{'=' * 140}")

    # ===== PRUNING SPEEDUP (baseline vs pruned, per size) =====
    print(f"\n{'=' * 100}")
    print("PRUNING SPEEDUP (baseline vs pruned)")
    print(f"{'-' * 100}")
    print(
        f"{'Size':<10} | {'PT base':>8} | {'PT prune':>9} | {'PT speedup':>10} "
        f"| {'TRT base':>9} | {'TRT prune':>10} | {'TRT speedup':>11}"
    )
    print(f"{'-' * 100}")

    sizes = ["YOLO26n", "YOLO26s", "YOLO26m", "YOLO26l"]
    for size in sizes:
        base = next((r for r in results if r["model"] == size and r["variant"] == "baseline"), None)
        prun = next((r for r in results if r["model"] == size and r["variant"] == "pruned"), None)
        if not base or not prun:
            continue

        pt_b = f"{base['pt_latency']:.1f}ms" if base["pt_latency"] else "N/A"
        pt_p = f"{prun['pt_latency']:.1f}ms" if prun["pt_latency"] else "N/A"
        if base["pt_latency"] and prun["pt_latency"]:
            pt_sp = f"{base['pt_latency'] / prun['pt_latency']:.2f}x"
        else:
            pt_sp = "N/A"

        trt_b = f"{base['trt_latency']:.1f}ms" if base["trt_latency"] else "N/A"
        trt_p = f"{prun['trt_latency']:.1f}ms" if prun["trt_latency"] else "N/A"
        if base["trt_latency"] and prun["trt_latency"]:
            trt_sp = f"{base['trt_latency'] / prun['trt_latency']:.2f}x"
        else:
            trt_sp = "N/A"

        print(f"{size:<10} | {pt_b:>8} | {pt_p:>9} | {pt_sp:>10} | {trt_b:>9} | {trt_p:>10} | {trt_sp:>11}")
    print(f"{'=' * 100}")

    # ===== CSV =====
    print("\n--- CSV ---")
    print(
        "Model,Variant,Params(M),Size(MB),GFLOPs,PT_Latency(ms),PT_FPS,TRT_Latency(ms),TRT_FPS,TRT_Speedup,AP50,AP50-95"
    )
    for r in results:
        pt_l = f"{r['pt_latency']:.2f}" if r["pt_latency"] else ""
        pt_f = f"{r['pt_fps']:.1f}" if r["pt_fps"] else ""
        trt_l = f"{r['trt_latency']:.2f}" if r["trt_latency"] else ""
        trt_f = f"{r['trt_fps']:.1f}" if r["trt_fps"] else ""
        if r["pt_latency"] and r["trt_latency"]:
            sp = f"{r['pt_latency'] / r['trt_latency']:.2f}"
        else:
            sp = ""
        print(
            f"{r['model']},{r['variant']},{r['params']},{r['size_mb']},{r['gflops']},"
            f"{pt_l},{pt_f},{trt_l},{trt_f},{sp},{r['ap50']},{r['ap50_95']}"
        )
