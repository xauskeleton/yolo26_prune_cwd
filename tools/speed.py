"""Benchmark speed: Original vs Pruned model.

Best practices from PyTorch docs, Ultralytics, MMDetection, NVIDIA:
- Raw model.forward() only (no predict/preprocess/NMS overhead)
- cudnn.benchmark=True for optimal kernel selection
- torch.set_num_threads for reproducible CPU results
- Iterative sigma clipping (Ultralytics method) for outlier removal
- Report median + mean + std

Usage:
    python tools/speed.py --orig weights/yolo26m_baseline.pt --pruned weights/yolo26m_baseline_l1norm_div8_dayroi.pt
    python tools/speed.py --orig weights/yolo26m_baseline.pt --pruned weights/yolo26m_baseline_l1norm_div8_dayroi.pt --device cpu
    python tools/speed.py --orig weights/yolo26m_baseline.pt --pruned weights/yolo26m_baseline_l1norm_div8_dayroi.pt --batch 1 4 8 16 32 64
    python tools/speed.py --orig weights/yolo26m_baseline.pt --pruned weights/yolo26m_baseline_l1norm_div8_dayroi.pt --device cpu --threads 4
"""

import argparse
import time
import numpy as np
import torch
from ultralytics import YOLO
from thop import profile
from copy import deepcopy


def iterative_sigma_clipping(data, sigma=2, max_iters=3):
    """Ultralytics-style outlier removal: loai bo > 2 std, lap lai 3 lan."""
    data = np.array(data)
    for _ in range(max_iters):
        mean, std = np.mean(data), np.std(data)
        clipped = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
        if len(clipped) == len(data):
            break
        data = clipped
    return data


def time_forward(model_nn, dummy, device, warmup=50, repetitions=200):
    """Benchmark raw forward pass with proper warmup and sigma clipping."""
    model_nn.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model_nn(dummy)
        if device == 'cuda':
            torch.cuda.synchronize()

    # Benchmark
    if device == 'cuda':
        timings = []
        with torch.no_grad():
            for _ in range(repetitions):
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                model_nn(dummy)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
    else:
        timings = []
        with torch.no_grad():
            for _ in range(repetitions):
                t0 = time.perf_counter()
                model_nn(dummy)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)

    # Iterative sigma clipping (Ultralytics method)
    timings_clean = iterative_sigma_clipping(timings, sigma=2, max_iters=3)

    median_ms = float(np.median(timings_clean).item())
    mean_ms = float(np.mean(timings_clean).item())
    std_ms = float(np.std(timings_clean).item())
    return median_ms, mean_ms, std_ms


def load_model(model_path, device):
    """Load model and move to device."""
    model = YOLO(model_path)
    model.to(device)
    model.model.eval()
    return model


def get_flops_params(model_nn, img_size, device):
    """Get GFLOPs and params."""
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    flops, params = profile(deepcopy(model_nn), inputs=(dummy,), verbose=False)
    return flops / 1e9, params / 1e6


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Original vs Pruned")
    parser.add_argument("--orig", type=str, required=True, help="Path to original model")
    parser.add_argument("--pruned", type=str, required=True, help="Path to pruned model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch", type=int, nargs="+", default=None)
    parser.add_argument("--warmup", type=int, default=None, help="Warmup iterations (default: 20 GPU, 10 CPU)")
    parser.add_argument("--reps", type=int, default=None, help="Benchmark repetitions (default: 50 GPU, 30 CPU)")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (default: 1 for reproducibility)")
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA khong kha dung, chuyen sang CPU")
        args.device = 'cpu'

    if args.batch is None:
        args.batch = [1, 2, 4] if args.device == 'cpu' else [1, 2, 4]

    warmup = args.warmup or (100 if args.device == 'cuda' else 10)
    reps = args.reps or (50 if args.device == 'cuda' else 30)

    # === Setup device ===
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print("NOTE: For best results, lock GPU clocks: nvidia-smi -lgc <base_clock>,<base_clock>")
    else:
        torch.set_num_threads(args.threads)
        print(f"CPU threads: {args.threads}")

    print(f"Device: {args.device.upper()}")
    print(f"Image size: {args.imgsz}")
    print(f"Warmup: {warmup} | Repetitions: {reps}")
    print(f"Timing: raw model.forward() | Outlier: iterative sigma clipping (2σ)")
    print(f"Metric: median (robust to outliers)")

    # Load models
    print(f"\nLoading Original: {args.orig}")
    m_orig = load_model(args.orig, args.device)
    print(f"Loading Pruned: {args.pruned}")
    m_prun = load_model(args.pruned, args.device)

    # FLOPs & Params
    g_orig, p_orig = get_flops_params(m_orig.model, args.imgsz, args.device)
    g_prun, p_prun = get_flops_params(m_prun.model, args.imgsz, args.device)

    # === BENCHMARK ALL BATCH SIZES ===
    print(f"\n{'='*90}")
    print(f"BENCHMARK ({args.device.upper()}) - raw forward, {reps} reps, sigma clipped")
    print(f"{'='*90}")

    res_orig = {}
    res_prun = {}

    for bs in args.batch:
        dummy = torch.randn(bs, 3, args.imgsz, args.imgsz).to(args.device)

        # Original
        try:
            med_o, avg_o, std_o = time_forward(m_orig.model, dummy, args.device, warmup, reps)
            fps_o = bs / (med_o / 1000)
            res_orig[bs] = (med_o, avg_o, std_o, fps_o)
            print(f"  Orig  bs={bs:<3d} | median {med_o:.1f}ms  mean {avg_o:.1f}±{std_o:.1f}ms | {fps_o:.1f} FPS")
        except (torch.cuda.OutOfMemoryError if args.device == 'cuda' else MemoryError):
            res_orig[bs] = None
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            print(f"  Orig  bs={bs:<3d} | OOM")

        # Pruned
        try:
            med_p, avg_p, std_p = time_forward(m_prun.model, dummy, args.device, warmup, reps)
            fps_p = bs / (med_p / 1000)
            res_prun[bs] = (med_p, avg_p, std_p, fps_p)
            print(f"  Prune bs={bs:<3d} | median {med_p:.1f}ms  mean {avg_p:.1f}±{std_p:.1f}ms | {fps_p:.1f} FPS")
        except (torch.cuda.OutOfMemoryError if args.device == 'cuda' else MemoryError):
            res_prun[bs] = None
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            print(f"  Prune bs={bs:<3d} | OOM")

        del dummy
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        print()

    # === SUMMARY TABLE ===
    print(f"{'='*65}")
    print(f"{'Metric':<15} | {'Original':<17} | {'Pruned':<17} | {'Giam (%)'}")
    print(f"{'-'*65}")
    print(f"{'GFLOPs':<15} | {g_orig:>17.2f} | {g_prun:>17.2f} | {((g_orig - g_prun) / g_orig) * 100:>8.1f}%")
    print(f"{'Params (M)':<15} | {p_orig:>17.2f} | {p_prun:>17.2f} | {((p_orig - p_prun) / p_orig) * 100:>8.1f}%")

    # Latency & FPS tu batch=1 (dung median)
    o1 = res_orig.get(1)
    p1 = res_prun.get(1)
    if o1 and p1:
        print(f"{'Latency (ms)':<15} | {o1[0]:>12.2f}±{o1[2]:<3.1f} | {p1[0]:>12.2f}±{p1[2]:<3.1f} | {((o1[0] - p1[0]) / o1[0]) * 100:>8.1f}%")
        fps_orig = 1000 / o1[0]
        fps_prun = 1000 / p1[0]
        print(f"{'FPS (bs=1)':<15} | {fps_orig:>17.1f} | {fps_prun:>17.1f} | {((fps_prun - fps_orig) / fps_orig) * 100:>8.1f}%")
    print(f"{'='*65}")

    # === BATCH SIZE COMPARISON TABLE ===
    print(f"\n{'='*95}")
    print(f"{'Batch':<6} | {'Orig median':<12} | {'Orig FPS':<10} | {'Prune median':<13} | {'Prune FPS':<10} | {'Speedup'}")
    print(f"{'-'*95}")
    for bs in args.batch:
        o = res_orig.get(bs)
        p = res_prun.get(bs)
        if o is None and p is None:
            print(f"{bs:<6} | {'OOM':<12} | {'OOM':<10} | {'OOM':<13} | {'OOM':<10} |")
        elif o is None:
            print(f"{bs:<6} | {'OOM':<12} | {'OOM':<10} | {p[0]:>8.1f}ms   | {p[3]:<10.1f} | orig OOM")
        elif p is None:
            print(f"{bs:<6} | {o[0]:>8.1f}ms   | {o[3]:<10.1f} | {'OOM':<13} | {'OOM':<10} |")
        else:
            speedup = p[3] / o[3]
            print(f"{bs:<6} | {o[0]:>8.1f}ms   | {o[3]:<10.1f} | {p[0]:>8.1f}ms    | {p[3]:<10.1f} | {speedup:.2f}x")
    print(f"{'='*95}")