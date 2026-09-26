# -*- coding: utf-8 -*-
"""Chay thu logic resume cua notebooks/share_visdrone/vd_common.py.

Dung lai cay thu muc /kaggle/input gia roi goi dung ham that, doi moi
"/kaggle/input/" thanh thu muc tam. Bat lai dung lop loi da tung gap:
chep nham ban it epoch hon, khong nhan thu muc chi co best.pt, va goi
resume tren mot run da ket thuc.
"""
import pathlib
import shutil
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks" / "share_visdrone"))

import torch

SRC = (ROOT / "notebooks" / "share_visdrone" / "vd_common.py").read_text(encoding="utf-8")


def load(input_dir):
    """Nap vd_common voi /kaggle/input tro toi thu muc tam."""
    mod = types.ModuleType("vd_common_test")
    mod.__file__ = "vd_common_test"
    code = SRC.replace('"/kaggle/input/', '"' + str(input_dir).replace("\\", "/") + "/")
    exec(compile(code, "vd_common_test", "exec"), mod.__dict__)
    return mod


def fake_run(path, n_epoch, files=("last.pt",), csv=True, total=100):
    d = pathlib.Path(path)
    (d / "weights").mkdir(parents=True, exist_ok=True)
    for f in files:
        torch.save({"epoch": n_epoch - 1 if n_epoch else -1,
                    "train_args": {"epochs": total}}, d / "weights" / f)
    if csv and n_epoch:
        rows = ["epoch,x"] + ["{},0".format(i) for i in range(1, n_epoch + 1)]
        (d / "results.csv").write_text("\n".join(rows))
    return d


def case(title, build, expect_base, expect_ours, manual=None):
    tmp = pathlib.Path(tempfile.mkdtemp())
    inp, repo = tmp / "input", tmp / "repo"
    inp.mkdir()
    (repo / "weights").mkdir(parents=True)
    build(inp)

    V = load(inp)
    V.init(REPO_DIR=repo, DATA="VisDrone.yaml", EPOCHS=100, BATCH=16, IMGSZ=640,
           DEVICE="0", COS_LR=False, PATIENCE=100, WARMUP=3.0, STOP_AFTER_H=10.0)
    V.restore("vd_yolo26m", "vd_oursm", repo / "weights" / "p.pt", manual or {})

    gb, go = V.done_epochs("vd_yolo26m"), V.done_epochs("vd_oursm")
    best_ok = (repo / "runs" / "vd_yolo26m" / "weights" / "best.pt").exists()
    ok = gb == expect_base and go == expect_ours
    print("  -> baseline {}/100, ours {}/100, co best.pt cho prune: {}   {}".format(
        gb, go, best_ok, "OK" if ok else "SAI (mong doi {} / {})".format(
            expect_base, expect_ours)))
    shutil.rmtree(tmp, ignore_errors=True)
    return ok


results = []

print("1. Goi resume that: baseline chi co best.pt, ours co last.pt")
results.append(case(
    "", lambda i: (fake_run(i / "ds/vd_resume_m/vd_yolo26m", 0, ("best.pt",), csv=False),
                   fake_run(i / "ds/vd_resume_m/vd_oursm", 80, ("last.pt",), csv=False)),
    100, 80))

print("2. Add Data output lan truoc: day du runs/<name>/")
results.append(case(
    "", lambda i: (fake_run(i / "d/yolo/runs/vd_yolo26m", 100, ("last.pt", "best.pt")),
                   fake_run(i / "d/yolo/runs/vd_oursm", 42, ("last.pt",))),
    100, 42))

print("3. Hai ban ours, phai lay ban nhieu epoch nhat")
results.append(case(
    "", lambda i: (fake_run(i / "a/vd_oursm", 42, ("last.pt",)),
                   fake_run(i / "b/vd_oursm", 71, ("last.pt",))),
    0, 71))

print("4. Chi co mot file last.pt roi le -> duong dan thu cong")
def _manual(i):
    (i / "up").mkdir(parents=True)
    torch.save({"epoch": 87, "train_args": {"epochs": 100}}, i / "up" / "last.pt")
tmpdir = None
results.append(case("", _manual, 0, 0))  # khong co manual -> khong tim thay

print("5. Khong co gi trong input -> tat ca ve 0, khong no")
results.append(case("", lambda i: None, 0, 0))

print()
print("TONG:", sum(results), "/", len(results), "truong hop dung")
sys.exit(0 if all(results) else 1)
