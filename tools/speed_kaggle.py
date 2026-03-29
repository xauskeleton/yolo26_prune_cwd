"""Benchmark inference speed on Kaggle (T4/P100) - simulate edge deployment.

Modes:
  1. GPU FP32/FP16  — raw forward, CUDA events timing
  2. TensorRT FP16  — export + benchmark (realistic edge deployment)
  3. ONNX CPU       — simulate CPU-only edge device
  4. CPU limited    — simulate weak edge CPU (2-4 threads)

Usage (Kaggle notebook cell):
    !python tools/speed_kaggle.py --weights weights/baseline.pt weights/pruned.pt --mode all
    !python tools/speed_kaggle.py --weights weights/baseline.pt weights/pruned.pt --mode gpu_fp16 tensorrt
    !python tools/speed_kaggle.py --weights weights/baseline.pt weights/pruned.pt --mode cpu --threads 2
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sigma_clip(data, sigma=2, max_iters=3):
    """Iterative sigma clipping (Ultralytics method)."""
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


def bench_pytorch(model_nn, imgsz, device, batch=1, half=False, warmup=50, reps=200):
    """Benchmark raw model.forward() with CUDA events or perf_counter."""
    model_nn.eval().to(device)
    dtype = torch.float16 if half else torch.float32
    if half:
        model_nn.half()
    dummy = torch.randn(batch, 3, imgsz, imgsz, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(warmup):
            model_nn(dummy)
        if device.type == torch.device("cuda").type:
            torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(reps):
            if device.type == torch.device("cuda").type:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                model_nn(dummy)
                e.record()
                torch.cuda.synchronize()
                timings.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                model_nn(dummy)
                timings.append((time.perf_counter() - t0) * 1000)

    clean = sigma_clip(timings)
    return {
        "median_ms": float(np.median(clean)),
        "mean_ms": float(np.mean(clean)),
        "std_ms": float(np.std(clean)),
        "fps": batch / (float(np.median(clean)) / 1000),
    }


def bench_export(model_path, fmt, imgsz, half=False, device="cpu"):
    """Export model and benchmark with AutoBackend directly."""
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import AutoBackend
    model = YOLO(model_path)

    # Export
    export_args = dict(format=fmt, imgsz=imgsz, half=half)
    if fmt == "engine":
        export_args["device"] = 0
    exported = model.export(**export_args)

    # Load exported model via AutoBackend
    dev = torch.device("cuda:0" if fmt == "engine" else device)
    backend = AutoBackend(exported, device=dev, fp16=half)
    backend.eval()

    dtype = torch.float16 if half else torch.float32
    dummy = torch.randn(1, 3, imgsz, imgsz, device=dev, dtype=dtype)

    # Warmup
    warmup = 30
    reps = 100
    with torch.no_grad():
        for _ in range(warmup):
            backend(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(reps):
            if dev.type == "cuda":
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                backend(dummy)
                e.record()
                torch.cuda.synchronize()
                timings.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                backend(dummy)
                timings.append((time.perf_counter() - t0) * 1000)

    clean = sigma_clip(timings)
    return {
        "median_ms": float(np.median(clean)),
        "mean_ms": float(np.mean(clean)),
        "std_ms": float(np.std(clean)),
        "fps": 1.0 / (float(np.median(clean)) / 1000),
        "exported_path": exported,
    }


def get_model_info(model_path):
    """Get params and FLOPs."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    params = sum(p.numel() for p in model.model.parameters()) / 1e6

    try:
        from thop import profile as thop_profile
        dummy = torch.randn(1, 3, 640, 640)
        m = deepcopy(model.model).cpu()
        if hasattr(m, 'model') and hasattr(m.model[-1], 'export'):
            m.model[-1].export = True
        flops, _ = thop_profile(m, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
    except Exception:
        gflops = 0.0

    return params, gflops, model


def print_header(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", required=True,
                        help="Model paths (first=baseline, rest=pruned)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Display names for each model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, nargs="+", default=[1])
    parser.add_argument("--mode", nargs="+", default=["gpu_fp32", "gpu_fp16"],
                        choices=["gpu_fp32", "gpu_fp16", "tensorrt", "onnx_cpu", "cpu", "all"],
                        help="Benchmark modes")
    parser.add_argument("--threads", type=int, default=2, help="CPU threads for cpu/onnx_cpu mode")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    if "all" in args.mode:
        args.mode = ["gpu_fp32", "gpu_fp16", "tensorrt", "onnx_cpu", "cpu"]

    if args.names is None:
        args.names = []
        for i, w in enumerate(args.weights):
            args.names.append("Baseline" if i == 0 else f"Pruned_{i}")

    has_cuda = torch.cuda.is_available()

    # ---- System info ----
    print_header("System Info")
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  GPU: None (CPU only)")
    import platform
    print(f"  CPU: {platform.processor() or 'unknown'}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Image size: {args.imgsz}")

    # ---- Model info ----
    print_header("Model Info")
    models_info = OrderedDict()
    for name, wpath in zip(args.names, args.weights):
        params, gflops, model_obj = get_model_info(wpath)
        models_info[name] = {"path": wpath, "params": params, "gflops": gflops, "model": model_obj}
        print(f"  {name:<20s}  Params={params:.2f}M  GFLOPs={gflops * 2:.1f}  ({wpath})")

    # ---- Benchmarks ----
    all_results = OrderedDict()  # mode -> {name -> result}

    for mode in args.mode:
        # --- GPU FP32 ---
        if mode == "gpu_fp32" and has_cuda:
            print_header(f"GPU FP32 (batch={args.batch})")
            torch.backends.cudnn.benchmark = True
            results = OrderedDict()
            for name, info in models_info.items():
                results[name] = {}
                for bs in args.batch:
                    r = bench_pytorch(deepcopy(info["model"].model), args.imgsz,
                                      torch.device("cuda"), bs, half=False,
                                      warmup=args.warmup, reps=args.reps)
                    results[name][bs] = r
                    print(f"  {name:<20s} bs={bs}  {r['median_ms']:.1f}ms  {r['fps']:.1f} FPS")
            all_results["GPU FP32"] = results

        # --- GPU FP16 ---
        elif mode == "gpu_fp16" and has_cuda:
            print_header(f"GPU FP16 (batch={args.batch})")
            torch.backends.cudnn.benchmark = True
            results = OrderedDict()
            for name, info in models_info.items():
                results[name] = {}
                for bs in args.batch:
                    r = bench_pytorch(deepcopy(info["model"].model), args.imgsz,
                                      torch.device("cuda"), bs, half=True,
                                      warmup=args.warmup, reps=args.reps)
                    results[name][bs] = r
                    print(f"  {name:<20s} bs={bs}  {r['median_ms']:.1f}ms  {r['fps']:.1f} FPS")
            all_results["GPU FP16"] = results

        # --- TensorRT FP16 ---
        elif mode == "tensorrt" and has_cuda:
            print_header("TensorRT FP16 (export + benchmark)")
            results = OrderedDict()
            for name, info in models_info.items():
                try:
                    r = bench_export(info["path"], "engine", args.imgsz, half=True, device="cuda")
                    if r:
                        results[name] = {1: r}
                        print(f"  {name:<20s}  {r['median_ms']:.1f}ms  {r['fps']:.1f} FPS")
                    else:
                        print(f"  {name:<20s}  FAILED")
                except Exception as ex:
                    print(f"  {name:<20s}  ERROR: {ex}")
            all_results["TensorRT FP16"] = results

        # --- ONNX CPU ---
        elif mode == "onnx_cpu":
            print_header(f"ONNX CPU (threads={args.threads})")
            torch.set_num_threads(args.threads)
            results = OrderedDict()
            for name, info in models_info.items():
                try:
                    r = bench_export(info["path"], "onnx", args.imgsz, half=False, device="cpu")
                    if r:
                        results[name] = {1: r}
                        print(f"  {name:<20s}  {r['median_ms']:.1f}ms  {r['fps']:.1f} FPS")
                    else:
                        print(f"  {name:<20s}  FAILED")
                except Exception as ex:
                    print(f"  {name:<20s}  ERROR: {ex}")
            all_results["ONNX CPU"] = results

        # --- CPU raw ---
        elif mode == "cpu":
            print_header(f"CPU FP32 (threads={args.threads}, batch={args.batch})")
            torch.set_num_threads(args.threads)
            results = OrderedDict()
            for name, info in models_info.items():
                results[name] = {}
                for bs in args.batch:
                    r = bench_pytorch(deepcopy(info["model"].model), args.imgsz,
                                      torch.device("cpu"), bs, half=False,
                                      warmup=min(args.warmup, 10), reps=min(args.reps, 30))
                    results[name][bs] = r
                    print(f"  {name:<20s} bs={bs}  {r['median_ms']:.1f}ms  {r['fps']:.1f} FPS")
            all_results[f"CPU FP32 ({args.threads}T)"] = results

    # ---- Summary table ----
    print_header("SUMMARY TABLE")
    base_name = args.names[0]

    # Header
    modes_with_results = [m for m in all_results if all_results[m]]
    hdr = f"{'Model':<20s} | {'Params':>8s} | {'GFLOPs':>7s}"
    for m in modes_with_results:
        hdr += f" | {m + ' (ms)':>18s} | {'FPS':>7s}"
    print(hdr)
    print("-" * len(hdr))

    # Rows
    for name, info in models_info.items():
        row = f"{name:<20s} | {info['params']:>7.2f}M | {info['gflops'] * 2:>7.1f}"
        for m in modes_with_results:
            mr = all_results[m]
            if name in mr and 1 in mr[name]:
                r = mr[name][1]
                row += f" | {r['median_ms']:>13.1f}±{r['std_ms']:<3.1f} | {r['fps']:>7.1f}"
            else:
                row += f" | {'N/A':>18s} | {'N/A':>7s}"
        print(row)

    # Speedup row
    if len(args.names) >= 2:
        print("-" * len(hdr))
        for pname in args.names[1:]:
            row = f"{'Speedup (' + pname + ')':<20s} | {'':>8s} | {'':>7s}"
            for m in modes_with_results:
                mr = all_results[m]
                if base_name in mr and pname in mr and 1 in mr[base_name] and 1 in mr[pname]:
                    speedup = mr[pname][1]["fps"] / mr[base_name][1]["fps"]
                    row += f" | {'':>18s} | {speedup:>6.2f}x"
                else:
                    row += f" | {'':>18s} | {'N/A':>7s}"
            print(row)

    print(f"{'='*80}")
    print("Note: Latency = median after 2-sigma clipping. FPS = 1000/latency (bs=1).")
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()


if __name__ == "__main__":
    main()
