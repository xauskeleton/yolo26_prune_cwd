"""Chay toan bo pipeline end-to-end trong mot lenh.

    baseline -> prune -> finetune+CWD -> val -> export -> bench

Moi stage ghi ket qua vao results/e2e_manifest.json, stage sau doc duong dan tu
manifest nen khong phai tu noi checkpoint bang tay. Chay lai voi --resume thi
stage nao da co output se bi bo qua.

Usage:
    python scripts/run_e2e.py --stage all
    python scripts/run_e2e.py --stage all --resume
    python scripts/run_e2e.py --stage prune finetune
    python scripts/run_e2e.py --stage all --dry-run

Mac dinh dung dung cong thuc da tao ra ket qua trong bai (run cwd_t9):
    L1-norm uniform 50%, divisor 8, finetune 100 epoch, CWD tau=9, kd_layers=neck,
    kd_warmup=5, batch=16, imgsz=640.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pruning"))

# Fork nay KHONG duoc pip install, chi import duoc qua sys.path. DDP cua Ultralytics
# sinh tien trinh con chay mot file tam trong /root/.config/Ultralytics/DDP/ nen
# sys.path[0] cua no la thu muc do -> "No module named 'ultralytics'".
# PYTHONPATH di theo os.environ xuong moi tien trinh con, ke ca torch.distributed.run.
os.environ["PYTHONPATH"] = os.pathsep.join(
    x for x in (str(ROOT), os.environ.get("PYTHONPATH", "")) if x
)

MANIFEST = ROOT / "results" / "e2e_manifest.json"
STAGES = ["baseline", "prune", "finetune", "val", "export", "bench"]


# ───────────────────────────── manifest ─────────────────────────────

def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(man):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


def record(man, stage, **kv):
    man[stage] = {**man.get(stage, {}), **kv, "at": datetime.now().isoformat(timespec="seconds")}
    save_manifest(man)


def need(man, stage, key):
    """Lay output cua stage truoc; bao loi ro rang thay vi KeyError kho hieu."""
    val = man.get(stage, {}).get(key)
    if not val:
        raise SystemExit(
            f"!! Thieu '{key}' cua stage '{stage}'.\n"
            f"   Chay stage '{stage}' truoc, hoac truyen tay duong dan qua cac co --*-weights."
        )
    if not Path(val).exists():
        raise SystemExit(f"!! '{val}' (tu stage '{stage}') khong ton tai nua.")
    return val


def skey(a, stage):
    """Key trong manifest. baseline dung chung cho moi ratio; cac stage con lai
    phai tach theo tag, neu khong quet nhieu ratio se ghi de len nhau."""
    return stage if stage == "baseline" else f"{stage}@{a.tag}"


def banner(title):
    print(f"\n{'='*100}\n  {title}\n{'='*100}\n", flush=True)


def free_gpu():
    """Tra VRAM lai sau moi stage train - neu khong TRT export rat de OOM."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def done_epochs(run_dir):
    """So epoch DA train xong, doc tu results.csv.

    KHONG duoc dung su ton tai cua best.pt de ket luan "da xong": mot phien bi
    cat ngang van de lai best.pt cua epoch dang do, va no se bi ghi nhan nham
    thanh ket qua cuoi.
    """
    csv = Path(run_dir) / "results.csv"
    if not csv.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()
        return int(df["epoch"].max()) if len(df) and "epoch" in df else 0
    except Exception:
        return 0


def train_or_resume(spec_weights, run_dir, kw, epochs, build=None):
    """Train moi, hoac train TIEP tu last.pt neu phien truoc bi cat ngang.

    Kaggle gioi han 12h/phien con 100 epoch mat ~10h, nen truong hop bi cat la
    binh thuong. Khong co nhanh resume nay thi moi phien deu bat dau lai tu 0.
    """
    from ultralytics import YOLO

    run_dir = Path(run_dir)
    last = run_dir / "weights" / "last.pt"
    done = done_epochs(run_dir)

    if last.exists() and (run_dir / "args.yaml").exists() and 0 < done < epochs:
        print(f"  -> RESUME tu {last}  (da co {done}/{epochs} epoch)")
        m = (build or YOLO)(str(last))
        try:
            m.train(resume=True)
        except Exception as exc:
            # Ultralytics tu choi resume mot run da ket thuc (vd early stop do patience).
            # Khi do coi nhu xong o so epoch dang co, dung best.pt, KHONG train lai tu dau.
            if "finished" in str(exc).lower() or "nothing to resume" in str(exc).lower():
                print(f"  (run da ket thuc som o epoch {done}: {exc})")
                return (build or YOLO)(str(run_dir / "weights" / "best.pt")), done
            raise
        return m, done

    if done >= epochs:
        print(f"  -> da du {done}/{epochs} epoch, bo qua train")
        return (build or YOLO)(str(run_dir / "weights" / "best.pt")), done

    if done:
        print(f"  (co {done} epoch cu nhung thieu last.pt/args.yaml -> train lai tu dau)")
    m = (build or YOLO)(str(spec_weights))
    m.train(**kw)
    return m, done_epochs(run_dir)


def best_of(trainer):
    """Duong dan best.pt sau khi train xong."""
    best = getattr(trainer, "best", None)
    return str(best) if best and Path(best).exists() else None


# ───────────────────────────── stages ─────────────────────────────

def stage_baseline(a, man):
    """Train yolo26m tren VOC tu trong so COCO."""
    banner(f"STAGE 1/6  BASELINE  ({a.pretrained} -> {a.epochs} epoch)")
    from ultralytics import YOLO

    kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
              device=a.device, seed=a.seed, project=a.project, name="baseline",
              exist_ok=True)
    run_dir = Path(a.project) / "baseline"
    model, ep = train_or_resume(a.pretrained, run_dir, kw, a.epochs)

    if ep < a.epochs:
        print(f"\n!! Moi train {ep}/{a.epochs} epoch - chay lai de train tiep.")
        return

    best = best_of(getattr(model, "trainer", None)) or str(run_dir / "weights" / "best.pt")
    if not Path(best).exists():
        raise SystemExit("!! Train baseline xong nhung khong tim thay best.pt")

    dst = ROOT / "weights" / f"yolo26{a.model_size}_baseline.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    print(f"\n  baseline -> {dst}")
    record(man, "baseline", weights=str(dst), run=best, epochs=a.epochs)
    free_gpu()


def stage_prune(a, man):
    """L1-norm structured pruning. Goi thang prune_common nhu iterative_prune.py."""
    banner(f"STAGE 2/6  PRUNE  (L1-norm uniform {a.prune_ratio:.0%}, div{a.divisor})")
    from prune_common import create_masks, finalize_pruning, load_and_prepare
    from prune_l1norm import compute_l1norm_importance

    src = a.baseline_weights or need(man, "baseline", "weights")
    model, bn_dict, ignore_bn_list, _chunk, layer_ratio_cfg, pruned_yaml = \
        load_and_prepare(src, a.cfg, a.model_size, a.layer_ratio)

    # Repo clone ve KHONG co thu muc weights/ (bi gitignore), ma finalize_pruning
    # goi torch.save thang -> "Parent directory ... does not exist".
    (ROOT / "weights").mkdir(parents=True, exist_ok=True)

    importance = compute_l1norm_importance(model, bn_dict, ignore_bn_list)
    maskbndict = create_masks(importance, model, ignore_bn_list, layer_ratio_cfg,
                              a.prune_ratio, a.divisor)
    pruned = finalize_pruning(model, maskbndict, pruned_yaml, ignore_bn_list,
                              src, str(ROOT / "weights"), a.divisor, a.prune_ratio,
                              method_name="l1norm")

    # finalize_pruning dat ten theo <stem>_l1norm_div<N>.pt - KHONG co ratio trong ten,
    # nen quet nhieu ratio se de len nhau. Doi ten theo tag.
    tagged = Path(pruned).with_name(f"{Path(pruned).stem}_{a.tag}.pt")
    Path(pruned).replace(tagged)
    print(f"  -> {tagged}")

    record(man, skey(a, "prune"), weights=str(tagged), source=str(src),
           prune_ratio=a.prune_ratio, divisor=a.divisor,
           layer_ratio=a.layer_ratio or "uniform")
    free_gpu()


def stage_finetune(a, man):
    """Finetune model da prune, co CWD distillation tu baseline."""
    banner(f"STAGE 3/6  FINETUNE + {a.kd_method.upper()}  ({a.epochs} epoch)")
    from ultralytics import YOLO

    # DDP KHONG dung duoc voi pipeline nay.
    # ultralytics/utils/dist.py:generate_ddp_file() chi serialize vars(trainer.args),
    # trong khi model.py gan finetune/kd/kd_teacher/maskbndict thang len OBJECT trainer
    # (khong nam trong args, khong co trong default.yaml). Tien trinh con DDP dung lai
    # trainer tu args -> trainer.py:478 `self.kd_enabled = getattr(self, 'kd', False)`
    # thanh False -> train 100 epoch KHONG he co CWD ma KHONG bao loi gi.
    if a.kd_method != "none" and isinstance(a.device, (list, tuple)):
        raise SystemExit(
            "!! device={} (DDP) + kd={} => CWD se bi TAT AM THAM trong tien trinh con.\n"
            "   Dung --device 0 (mot GPU). Xem ultralytics/utils/dist.py:generate_ddp_file."
            .format(a.device, a.kd_method)
        )

    pruned = a.pruned_weights or need(man, skey(a, "prune"), "weights")
    teacher = a.teacher_weights or need(man, "baseline", "weights")

    kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
              device=a.device, seed=a.seed, project=a.project, name=f"finetune_{a.tag}",
              exist_ok=True, finetune=True)
    if a.kd_method != "none":
        kw.update(kd=True, kd_teacher=teacher, kd_method=a.kd_method,
                  kd_lambda=a.kd_lambda, kd_layers=a.kd_layers, kd_warmup=a.kd_warmup)
        if a.kd_method == "cwd":
            kw["cwd_temperature"] = a.cwd_temperature

    run_dir = Path(a.project) / f"finetune_{a.tag}"
    model, ep = train_or_resume(pruned, run_dir, kw, a.epochs)

    if ep < a.epochs:
        print(f"\n!! Moi train {ep}/{a.epochs} epoch - phien bi cat ngang.")
        print("   KHONG ghi ket qua (tranh lay nham best.pt cua epoch dang do).")
        print("   Chay lai notebook, nho Add Data output lan nay -> se train tiep.")
        return

    best = best_of(getattr(model, "trainer", None)) or str(run_dir / "weights" / "best.pt")
    if not Path(best).exists():
        raise SystemExit("!! Finetune xong nhung khong tim thay best.pt")

    dst = ROOT / "weights" / f"yolo26{a.model_size}_pruned_{a.kd_method}_{a.tag}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dst)
    print(f"\n  final -> {dst}")
    record(man, skey(a, "finetune"), weights=str(dst), run=best, teacher=str(teacher),
           kd_method=a.kd_method, kd_layers=a.kd_layers, kd_warmup=a.kd_warmup,
           kd_lambda=a.kd_lambda, cwd_temperature=a.cwd_temperature)
    free_gpu()


def one_device(dev):
    """Val/do dac chi chay mot GPU; truyen list vao validator khong dam bao chay."""
    return dev[0] if isinstance(dev, (list, tuple)) and dev else dev


def stage_val(a, man):
    """Do AP tren VOC2007 test cho ca baseline lan model cuoi."""
    banner("STAGE 4/6  VAL")
    from ultralytics import YOLO

    targets = []
    if man.get("baseline", {}).get("weights"):
        targets.append(("baseline", man["baseline"]["weights"]))
    if man.get(skey(a, "finetune"), {}).get("weights"):
        targets.append(("pruned", man[skey(a, "finetune")]["weights"]))
    if a.val_weights:
        targets = [(Path(w).stem, w) for w in a.val_weights]
    if not targets:
        raise SystemExit("!! Khong co checkpoint nao de val. Chay stage baseline/finetune truoc.")

    out = {}
    for tag, w in targets:
        print(f"\n  [{tag}] {w}")
        m = YOLO(w)
        r = m.val(data=a.data, imgsz=a.imgsz, batch=a.batch, device=one_device(a.device))
        # Do dac phu: hong thi bo qua, KHONG duoc lam mat ket qua AP
        try:
            params = sum(p.numel() for p in m.model.parameters()) / 1e6
        except Exception:
            params = float("nan")
        out[tag] = {"weights": str(w),
                    "AP50": round(float(r.box.map50) * 100, 2),
                    "AP50_95": round(float(r.box.map) * 100, 2),
                    "params_M": round(params, 2)}
        print(f"    AP50={out[tag]['AP50']}  AP50-95={out[tag]['AP50_95']}  "
              f"{out[tag]['params_M']}M")
        free_gpu()

    record(man, skey(a, "val"), **out)


def stage_export(a, man):
    """Export ONNX (+ TensorRT FP16 neu co). Chay tien trinh rieng de khong dung VRAM cu.

    Khong goi tools/export_for_benchmark.py vi file do chi export TensorRT du
    docstring ghi 'ONNX + TorchScript'. Bai co so ONNX nen phai export ONNX that.
    """
    banner("STAGE 5/6  EXPORT")
    final = a.export_weights or need(man, skey(a, "finetune"), "weights")
    save_dir = Path(a.export_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    produced = {}
    for fmt in a.export_format:
        print(f"\n  [{fmt}] {final}")
        snippet = (
            f"import sys; sys.path.insert(0, {str(ROOT)!r});"
            f"from ultralytics import YOLO;"
            f"p = YOLO({str(final)!r}).export(format={fmt!r}, imgsz={a.imgsz},"
            f" half={fmt == 'engine'});"
            f"print('EXPORTED:' + str(p))"
        )
        r = subprocess.run([sys.executable, "-c", snippet], cwd=str(ROOT),
                           capture_output=True, text=True)
        line = next((l for l in r.stdout.splitlines() if l.startswith("EXPORTED:")), None)
        if r.returncode != 0 or not line:
            print(f"    !! export {fmt} that bai (bo qua):")
            print("    " + "\n    ".join((r.stderr or r.stdout).strip().splitlines()[-8:]))
            continue
        src = Path(line.split("EXPORTED:", 1)[1].strip())
        dst = save_dir / f"{Path(final).stem}{src.suffix}"
        if src.resolve() != dst.resolve():
            shutil.move(str(src), dst)
        print(f"    -> {dst}  ({dst.stat().st_size/1e6:.1f} MB)")
        produced[fmt] = str(dst)

    if not produced:
        raise SystemExit("!! Khong export duoc dinh dang nao.")
    record(man, skey(a, "export"), **produced)


def stage_bench(a, man):
    """Do toc do bang tools/speed_kaggle.py (tien trinh rieng, VRAM sach)."""
    banner(f"STAGE 6/6  BENCH  (mode: {' '.join(a.bench_mode)})")
    weights = a.bench_weights or [
        w for w in (man.get("baseline", {}).get("weights"),
                    man.get(skey(a, "finetune"), {}).get("weights")) if w
    ]
    if not weights:
        raise SystemExit("!! Khong co checkpoint nao de bench.")

    log = Path(a.project) / f"bench_{a.tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "tools" / "speed_kaggle.py"),
           "--weights", *map(str, weights),
           "--imgsz", str(a.imgsz), "--mode", *a.bench_mode,
           "--threads", str(a.threads)]
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    log.write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    print(r.stdout[-4000:] if r.stdout else r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit(f"!! bench that bai, xem log: {log}")
    record(man, skey(a, "bench"), log=str(log), modes=a.bench_mode, weights=list(map(str, weights)))


RUNNERS = {"baseline": stage_baseline, "prune": stage_prune, "finetune": stage_finetune,
           "val": stage_val, "export": stage_export, "bench": stage_bench}

# --resume: stage coi nhu xong khi key nay da co trong manifest va file con ton tai
DONE_KEY = {"baseline": "weights", "prune": "weights", "finetune": "weights",
            "val": "pruned", "export": "onnx", "bench": "log"}


def is_done(man, stage, a=None):
    entry = man.get(skey(a, stage) if a else stage, {})
    val = entry.get(DONE_KEY[stage])
    if not val:
        return False
    if stage == "val":
        return True                      # val luu dict metric, khong phai duong dan
    return Path(val).exists()


# ───────────────────────────── main ─────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Chay end-to-end: baseline -> prune -> finetune+CWD -> val -> export -> bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--stage", nargs="+", default=["all"], choices=STAGES + ["all"])
    p.add_argument("--resume", action="store_true", help="bo qua stage da co output")
    p.add_argument("--dry-run", action="store_true", help="chi in ke hoach, khong chay")
    p.add_argument("--tag", default=None,
                   help="nhan cho lan chay nay (mac dinh r<ratio>, vd r50). Quet nhieu "
                        "ratio thi moi ratio mot tag -> khong ghi de len nhau.")

    g = p.add_argument_group("du lieu / model")
    g.add_argument("--data", default="VOC.yaml")
    g.add_argument("--cfg", default=str(ROOT / "cfg" / "yolo26m.yaml"))
    g.add_argument("--model-size", default="m", choices=list("nsmlx"))
    g.add_argument("--pretrained", default="yolo26m.pt", help="trong so COCO cho stage baseline")
    g.add_argument("--project", default=str(ROOT / "runs" / "e2e"))

    g = p.add_argument_group("train")
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--imgsz", type=int, default=640)
    g.add_argument("--batch", type=int, default=16)
    g.add_argument("--device", default="0", help='"0" mot GPU, "0,1" dung DDP 2 GPU')
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("prune")
    g.add_argument("--prune-ratio", type=float, default=0.5)
    g.add_argument("--divisor", type=int, default=8, choices=[8, 16])
    g.add_argument("--layer-ratio", default=None, help="YAML per-layer ratio (DMS)")

    g = p.add_argument_group("distillation")
    g.add_argument("--kd-method", default="cwd", choices=["cwd", "response", "fitnets", "mgd", "none"])
    g.add_argument("--kd-lambda", type=float, default=0.5)
    g.add_argument("--kd-layers", default="neck")
    g.add_argument("--kd-warmup", type=int, default=5)
    g.add_argument("--cwd-temperature", type=float, default=9.0)

    g = p.add_argument_group("export / bench")
    g.add_argument("--export-format", nargs="+", default=["onnx", "engine"])
    g.add_argument("--export-dir", default=str(ROOT / "export_out"))
    g.add_argument("--bench-mode", nargs="+", default=["gpu_fp32", "gpu_fp16", "tensorrt"],
                   choices=["gpu_fp32", "gpu_fp16", "tensorrt", "onnx_cpu", "cpu"])
    g.add_argument("--threads", type=int, default=2, help="so luong CPU cho mode onnx_cpu/cpu")

    g = p.add_argument_group("vao thang mot stage (bo qua manifest)")
    g.add_argument("--baseline-weights", default=None)
    g.add_argument("--pruned-weights", default=None)
    g.add_argument("--teacher-weights", default=None)
    g.add_argument("--val-weights", nargs="+", default=None)
    g.add_argument("--export-weights", default=None)
    g.add_argument("--bench-weights", nargs="+", default=None)

    a = p.parse_args()
    a.stage = STAGES if "all" in a.stage else [s for s in STAGES if s in a.stage]
    if not a.tag:
        a.tag = f"r{int(round(a.prune_ratio * 100))}"
    # "0" -> 0 ; "0,1" -> [0, 1] (DDP). Ultralytics chia batch cho so GPU, khong nhan len:
    # batch=16 voi 2 GPU van la batch hieu dung 16, moi GPU 8 mau.
    if isinstance(a.device, str):
        if "," in a.device:
            a.device = [int(x) for x in a.device.split(",") if x.strip() != ""]
        elif a.device.isdigit():
            a.device = int(a.device)
    return a


def main():
    a = parse_args()
    man = load_manifest()

    plan = [s for s in a.stage if not (a.resume and is_done(man, s, a))]
    skipped = [s for s in a.stage if s not in plan]

    banner("KE HOACH")
    for s in a.stage:
        mark = "BO QUA (da xong)" if s in skipped else "chay"
        print(f"  {s:<10} {mark}")
    print(f"\n  data={a.data}  epochs={a.epochs}  imgsz={a.imgsz}  batch={a.batch}  device={a.device}")
    print(f"  prune={a.prune_ratio:.0%} div{a.divisor}  kd={a.kd_method} "
          f"lambda={a.kd_lambda} layers={a.kd_layers} warmup={a.kd_warmup} tau={a.cwd_temperature}")
    print(f"  tag={a.tag}   manifest={MANIFEST}")

    if a.dry_run:
        print("\n  --dry-run: dung o day.")
        return
    if not plan:
        print("\n  Khong con stage nao phai chay.")
        return

    t_all = time.time()
    for s in plan:
        t0 = time.time()
        RUNNERS[s](a, man)
        print(f"\n  [{s}] xong sau {(time.time()-t0)/3600:.2f}h")

    banner(f"HOAN TAT sau {(time.time()-t_all)/3600:.2f}h")
    print(json.dumps(man, indent=2, ensure_ascii=False))
    print(f"\nManifest: {MANIFEST}")


if __name__ == "__main__":
    main()
