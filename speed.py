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
    # Truy cập vào model.model vì thop cần nn.Module, không phải class YOLO
    flops, params = profile(deepcopy(model.model), inputs=(input_dummy,), verbose=False)

    gflops = flops / 1e9
    mparams = params / 1e6

    # 3. Đo Latency
    # Warm-up
    for _ in range(30):
        _ = model.predict(input_dummy, verbose=False)

    # Đo thực tế
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
    return gflops, mparams, avg_latency


# --- Thực thi ---
if __name__ == "__main__":
    orig_path = "weights/best.pt"
    pruned_path = "weights/pruned_div8.pt"

    # Đo thông số
    g_orig, p_orig, l_orig = get_stats(orig_path)
    g_prun, p_prun, l_prun = get_stats(pruned_path)

    # In bảng so sánh
    print("\n" + "=" * 60)
    print(f"{'Metric':<15} | {'Original (Best)':<15} | {'Pruned (Div8)':<15} | {'Giảm (%)'}")
    print("-" * 60)
    print(f"{'GFLOPs':<15} | {g_orig:>15.2f} | {g_prun:>15.2f} | {((g_orig - g_prun) / g_orig) * 100:>8.1f}%")
    print(f"{'Params (M)':<15} | {p_orig:>15.2f} | {p_prun:>15.2f} | {((p_orig - p_prun) / p_orig) * 100:>8.1f}%")
    print(f"{'Latency (ms)':<15} | {l_orig:>15.2f} | {l_prun:>15.2f} | {((l_orig - l_prun) / l_orig) * 100:>8.1f}%")
    print(
        f"{'FPS':<15} | {1000 / l_orig:>15.1f} | {1000 / l_prun:>15.1f} | {((1000 / l_prun - 1000 / l_orig) / (1000 / l_orig)) * 100:>8.1f}%")
    print("=" * 60)