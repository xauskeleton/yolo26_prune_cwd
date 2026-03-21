import torch
import time
from ultralytics import YOLO
from thop import profile
from copy import deepcopy


def get_stats(model_path, img_size=640):
    print(f"\n đang kiểm tra: {model_path}...")

    # 1. Load model
    model = YOLO(model_path)
    model.to('cuda')
    model.model.eval()

    # 2. Tính GFLOPs & Params (Dùng thop cho chính xác với mô hình đã prune)
    input_dummy = torch.randn(1, 3, img_size, img_size).cuda()
    flops, params = profile(deepcopy(model.model), inputs=(input_dummy,), verbose=False)

    gflops = flops / 1e9
    mparams = params / 1e6

    # 3. Đo Latency batch=1
    for _ in range(30):
        _ = model.predict(input_dummy, verbose=False)

    repetitions = 100
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(repetitions):
            starter.record()
            _ = model.predict(input_dummy, verbose=False)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))

    avg_latency = sum(timings) / repetitions
    return gflops, mparams, avg_latency, model


def benchmark_batch(model, batch_sizes, img_size=640, repetitions=50):
    """Đo throughput (FPS) với các batch size khác nhau."""
    results = {}
    model.model.eval()

    for bs in batch_sizes:
        dummy = torch.randn(bs, 3, img_size, img_size).cuda()

        # Warmup
        try:
            with torch.no_grad():
                for _ in range(10):
                    model.model(dummy)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results[bs] = None  # OOM
            print(f"  Batch {bs}: OOM")
            continue

        # Đo
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = []

        with torch.no_grad():
            for _ in range(repetitions):
                starter.record()
                model.model(dummy)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

        avg_ms = sum(timings) / repetitions
        fps = bs / (avg_ms / 1000)
        results[bs] = (avg_ms, fps)
        print(f"  Batch {bs:>3d} | {avg_ms:.1f}ms | {fps:.1f} FPS")

        del dummy
        torch.cuda.empty_cache()

    return results


# --- Thực thi ---
if __name__ == "__main__":
    orig_path = "weights/yolo26m_sparsed.pt"
    pruned_path = "weights/yolo26m_pruned_div8.pt"

    # Đo thông số batch=1
    g_orig, p_orig, l_orig, m_orig = get_stats(orig_path)
    g_prun, p_prun, l_prun, m_prun = get_stats(pruned_path)

    # Bảng so sánh cơ bản
    print("\n" + "=" * 60)
    print(f"{'Metric':<15} | {'Original (Best)':<15} | {'Pruned (Div8)':<15} | {'Giảm (%)'}")
    print("-" * 60)
    print(f"{'GFLOPs':<15} | {g_orig:>15.2f} | {g_prun:>15.2f} | {((g_orig - g_prun) / g_orig) * 100:>8.1f}%")
    print(f"{'Params (M)':<15} | {p_orig:>15.2f} | {p_prun:>15.2f} | {((p_orig - p_prun) / p_orig) * 100:>8.1f}%")
    print(f"{'Latency (ms)':<15} | {l_orig:>15.2f} | {l_prun:>15.2f} | {((l_orig - l_prun) / l_orig) * 100:>8.1f}%")
    print(f"{'FPS (bs=1)':<15} | {1000 / l_orig:>15.1f} | {1000 / l_prun:>15.1f} | {((1000 / l_prun - 1000 / l_orig) / (1000 / l_orig)) * 100:>8.1f}%")
    print("=" * 60)

    # Benchmark batch lớn
    batch_sizes = [1, 4, 8, 16, 32, 64]

    print(f"\n{'='*60}")
    print("BENCHMARK BATCH SIZE")
    print(f"{'='*60}")

    print(f"\n--- Original: {orig_path} ---")
    res_orig = benchmark_batch(m_orig, batch_sizes)

    print(f"\n--- Pruned: {pruned_path} ---")
    res_prun = benchmark_batch(m_prun, batch_sizes)

    # Bảng so sánh batch
    print(f"\n{'='*70}")
    print(f"{'Batch':<8} | {'Orig (ms)':<12} | {'Orig FPS':<12} | {'Prune (ms)':<12} | {'Prune FPS':<12} | {'Speedup'}")
    print(f"{'-'*70}")
    for bs in batch_sizes:
        o = res_orig.get(bs)
        p = res_prun.get(bs)
        if o is None and p is None:
            print(f"{bs:<8} | {'OOM':<12} | {'OOM':<12} | {'OOM':<12} | {'OOM':<12} |")
        elif o is None:
            print(f"{bs:<8} | {'OOM':<12} | {'OOM':<12} | {p[0]:<12.1f} | {p[1]:<12.1f} | pruned OK")
        elif p is None:
            print(f"{bs:<8} | {o[0]:<12.1f} | {o[1]:<12.1f} | {'OOM':<12} | {'OOM':<12} |")
        else:
            speedup = p[1] / o[1]
            print(f"{bs:<8} | {o[0]:<12.1f} | {o[1]:<12.1f} | {p[0]:<12.1f} | {p[1]:<12.1f} | {speedup:.2f}x")
    print(f"{'='*70}")