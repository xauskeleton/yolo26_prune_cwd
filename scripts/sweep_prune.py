"""Quet ti le pruning va ve duong accuracy - FLOPs - latency.

Moi ratio la mot lan chay run_e2e.py (prune -> finetune -> val) voi tag rieng
(r30, r40, ...) nen khong ghi de len nhau. Sau do do params/GFLOPs/latency cho
tung checkpoint cuoi, gom vao results/prune_sweep.csv va ve results/prune_sweep.png.

Usage:
    # quet day du (moi ratio ~10h tren 1 GPU - nen chay qua nhieu phien voi --resume)
    python scripts/sweep_prune.py --ratios 0.3 0.4 0.5 0.6 0.7

    # da train roi, chi do lai + ve lai
    python scripts/sweep_prune.py --ratios 0.3 0.4 0.5 0.6 0.7 --collect-only

    # xem ke hoach
    python scripts/sweep_prune.py --ratios 0.3 0.5 0.7 --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "results" / "e2e_manifest.json"
CSV_OUT = ROOT / "results" / "prune_sweep.csv"
PNG_OUT = ROOT / "results" / "prune_sweep.png"

WARMUP, REPS = 50, 200


def tag_of(r):
    return f"r{int(round(r * 100))}"


def banner(t):
    print(f"\n{'='*100}\n  {t}\n{'='*100}\n", flush=True)


# ───────────────────────────── do dac ─────────────────────────────

def sigma_clip(data, sigma=2, max_iters=3):
    """Giong cach Ultralytics loc outlier khi do toc do."""
    import numpy as np
    data = np.array(data)
    for _ in range(max_iters):
        mean, std = np.mean(data), np.std(data)
        if std == 0:
            break
        keep = data[(data > mean - sigma * std) & (data < mean + sigma * std)]
        if len(keep) == len(data):
            break
        data = keep
    return data


def measure(weights, imgsz, device, half=True):
    """(params_M, gflops, latency_ms, fps). Tung phan co try rieng: do dac hong
    khong duoc lam mat nhung so con lai."""
    import numpy as np
    import torch
    from ultralytics import YOLO

    nan = float("nan")
    params = gflops = lat = fps = nan
    m = YOLO(str(weights))

    try:
        params = sum(p.numel() for p in m.model.parameters()) / 1e6
    except Exception as e:
        print(f"    (params fail: {e})")

    try:
        info = m.model.info(detailed=False, verbose=False)
        gflops = float(info[3]) if info and len(info) > 3 else nan
    except Exception as e:
        print(f"    (GFLOPs fail: {e})")

    if not torch.cuda.is_available():
        print("    (khong co CUDA -> bo qua latency)")
        return params, gflops, lat, fps

    try:
        dev = torch.device(f"cuda:{device}" if str(device).isdigit() else "cuda")
        net = m.model.to(dev).eval()
        dtype = torch.float16 if half else torch.float32
        if half:
            net.half()
        dummy = torch.randn(1, 3, imgsz, imgsz, device=dev, dtype=dtype)
        with torch.no_grad():
            for _ in range(WARMUP):
                net(dummy)
            torch.cuda.synchronize()
            ts = []
            for _ in range(REPS):
                s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                s.record()
                net(dummy)
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
        lat = float(np.median(sigma_clip(ts)))
        fps = 1000.0 / lat
    except Exception as e:
        print(f"    (latency fail: {e})")
    finally:
        try:
            import gc
            del m
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    return params, gflops, lat, fps


# ───────────────────────────── ve ─────────────────────────────

def plot(rows):
    """x = GFLOPs, truc trai = AP@0.5, truc phai = latency. Mau theo dcmm.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    rows = sorted([r for r in rows if r["GFLOPs"] == r["GFLOPs"]], key=lambda r: r["GFLOPs"])
    if len(rows) < 2:
        print("!! Can it nhat 2 diem de ve duong.")
        return

    g = [r["GFLOPs"] for r in rows]
    ap = [r["AP50"] for r in rows]
    lat = [r["Latency (ms)"] for r in rows]
    lab = [r["label"] for r in rows]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.4
    C_AP, C_LAT, C_TXT = "#378ADD", "#D85A30", "#4A5568"

    fig, ax1 = plt.subplots(figsize=(10, 5.4))
    ax1.plot(g, ap, color=C_AP, marker="o", markersize=8, linewidth=2.6,
             markerfacecolor="white", markeredgewidth=2.2, zorder=4)
    ax1.set_xlabel("GFLOPs", fontsize=13)
    ax1.set_ylabel("AP@0.5 (%)", fontsize=13, color=C_AP)
    ax1.tick_params(axis="y", labelcolor=C_AP)

    # Chua le cho nhan: diem dau/cuoi nam sat bien nen nhan de bi cat.
    gx = max(g) - min(g) or 1.0
    ay = max(ap) - min(ap) or 1.0
    ax1.set_xlim(min(g) - 0.07 * gx, max(g) + 0.07 * gx)
    ax1.set_ylim(min(ap) - 0.12 * ay, max(ap) + 0.22 * ay)

    for xi, yi, t in zip(g, ap, lab):
        # nhan o hai dau lech vao trong de khong tran ra ngoai khung
        dx = 12 if xi == min(g) else (-12 if xi == max(g) else 0)
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        ax1.annotate(t, (xi, yi), textcoords="offset points", xytext=(dx, 11),
                     ha=ha, fontsize=9.5, fontweight="bold", color=C_TXT)

    if any(v == v for v in lat):
        ax2 = ax1.twinx()
        ax2.plot(g, lat, color=C_LAT, marker="s", markersize=7, linewidth=2.4,
                 markerfacecolor="white", markeredgewidth=2, linestyle="--", zorder=3)
        ax2.set_ylabel("Latency (ms)", fontsize=13, color=C_LAT)
        ax2.tick_params(axis="y", labelcolor=C_LAT)
        ax2.grid(False)

    ax1.legend(handles=[
        Line2D([0], [0], color=C_AP, marker="o", markersize=8, markerfacecolor="white",
               markeredgewidth=2, label="AP@0.5"),
        Line2D([0], [0], color=C_LAT, marker="s", markersize=7, markerfacecolor="white",
               markeredgewidth=2, linestyle="--", label="Latency"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False, fontsize=11.5)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(PNG_OUT).replace(".png", f".{ext}"), dpi=250, bbox_inches="tight")
    print(f"  -> {PNG_OUT} / .pdf")


# ───────────────────────────── main ─────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Quet ti le pruning, ve duong accuracy - FLOPs - latency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ratios", nargs="+", type=float,
                   default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--collect-only", action="store_true",
                   help="bo qua train, chi do lai tu checkpoint da co va ve")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stage", nargs="+", default=["prune", "finetune", "val"],
                   help="stage truyen xuong run_e2e.py cho moi ratio")
    p.add_argument("--no-baseline", action="store_true",
                   help="khong ve diem baseline (ratio 0)")
    # chuyen tiep xuong run_e2e.py
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=0)
    p.add_argument("--data", default="VOC.yaml")
    return p.parse_args()


def run_one(a, ratio):
    cmd = [sys.executable, str(ROOT / "scripts" / "run_e2e.py"),
           "--stage", *a.stage,
           "--prune-ratio", str(ratio), "--tag", tag_of(ratio),
           "--epochs", str(a.epochs), "--imgsz", str(a.imgsz),
           "--batch", str(a.batch), "--device", str(a.device),
           "--data", a.data, "--resume"]
    print("  " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main():
    a = parse_args()
    ratios = sorted(a.ratios)

    banner("KE HOACH QUET")
    for r in ratios:
        print(f"  ratio {r:.0%}  -> tag {tag_of(r)}")
    print(f"\n  stage moi ratio: {' '.join(a.stage)}")
    print(f"  {'CHI GOM + VE (khong train)' if a.collect_only else 'train day du'}")
    print(f"  manifest: {MANIFEST}")
    if a.dry_run:
        print("\n  --dry-run: dung o day.")
        return

    # ── train tung ratio ──
    if not a.collect_only:
        for r in ratios:
            banner(f"RATIO {r:.0%}  (tag {tag_of(r)})")
            t0 = time.time()
            rc = run_one(a, r)
            print(f"\n  ratio {r:.0%}: {'OK' if rc == 0 else f'that bai (rc={rc})'} "
                  f"sau {(time.time()-t0)/3600:.2f}h")
            if rc != 0:
                print("  -> bo qua ratio nay, chay tiep cai sau.")

    # ── gom ──
    banner("DO DAC + GOM KET QUA")
    if not MANIFEST.exists():
        raise SystemExit(f"!! Chua co {MANIFEST}. Chay quet truoc.")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))

    rows = []
    targets = []
    if not a.no_baseline and man.get("baseline", {}).get("weights"):
        targets.append(("baseline", 0.0, man["baseline"]["weights"]))
    for r in ratios:
        w = man.get(f"finetune@{tag_of(r)}", {}).get("weights")
        if w and Path(w).exists():
            targets.append((tag_of(r), r, w))
        else:
            print(f"  ratio {r:.0%}: chua co checkpoint -> bo qua")

    for label, r, w in targets:
        print(f"\n  [{label}] {w}")
        params, gflops, lat, fps = measure(w, a.imgsz, a.device)
        # AP50 uu tien lay tu manifest (da val bang dung giao thuc)
        ap = float("nan")
        key = "baseline" if label == "baseline" else f"val@{label}"
        ent = man.get(key, {})
        for sub in ("pruned", "baseline"):
            if isinstance(ent.get(sub), dict) and "AP50" in ent[sub]:
                ap = ent[sub]["AP50"]
                if (sub == "pruned") == (label != "baseline"):
                    break
        rows.append({"label": label if label == "baseline" else f"{r:.0%}",
                     "ratio": r, "weights": str(w),
                     "Params (M)": round(params, 2) if params == params else None,
                     "GFLOPs": round(gflops, 1) if gflops == gflops else float("nan"),
                     "Latency (ms)": round(lat, 2) if lat == lat else float("nan"),
                     "FPS": round(fps, 1) if fps == fps else float("nan"),
                     "AP50": ap})
        print(f"    AP50={ap}  {params:.2f}M  {gflops:.1f}G  {lat:.2f}ms")

    if not rows:
        raise SystemExit("!! Khong gom duoc dong nao.")

    import pandas as pd
    df = pd.DataFrame(rows)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"\n  -> {CSV_OUT}")
    print(df[["label", "Params (M)", "GFLOPs", "Latency (ms)", "FPS", "AP50"]].to_string(index=False))

    banner("VE")
    plot(rows)


if __name__ == "__main__":
    main()
