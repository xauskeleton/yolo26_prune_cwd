# -*- coding: utf-8 -*-
"""Chay ca hai buoc tu dong (dataset + checkpoint) tren cay /kaggle/input gia,
voi ten run kem hau to _1024 dung nhu notebook hien tai.
"""
import pathlib
import shutil
import sys
import tempfile
import types

ROOT = pathlib.Path("D:/xauduabo/Code+NCKH/Git/yolo")
sys.path.insert(0, str(ROOT))
import torch

SRC = (ROOT / "notebooks" / "share_visdrone" / "vd_common.py").read_text(encoding="utf-8")

tmp = pathlib.Path(tempfile.mkdtemp())
kag = tmp / "kaggle"
inp, tempd, repo = kag / "input", kag / "temp", tmp / "repo"
for d in (inp, tempd, repo / "weights"):
    d.mkdir(parents=True)

# ---- output cua phien truoc, dung nhu Kaggle gan vao -------------------------
OUT = inp / "datasets" / "chimbellll" / "phien-truoc"
#   (a) dataset nam trong output cu (cac run 640 co luu vao /kaggle/working)
for sp in ("train", "val"):
    (OUT / "yolo" / "datasets" / "VisDrone" / "images" / sp).mkdir(parents=True)
    (OUT / "yolo" / "datasets" / "VisDrone" / "labels" / sp).mkdir(parents=True)
    (OUT / "yolo" / "datasets" / "VisDrone" / "images" / sp / "a.jpg").write_bytes(b"x")
    (OUT / "yolo" / "datasets" / "VisDrone" / "labels" / sp / "a.txt").write_text("0 .5 .5 .1 .1")
#   (b) baseline da xong 100 epoch (chi con best.pt sau khi strip)
d = OUT / "yolo" / "runs" / "vd_yolo26m_1024" / "weights"
d.mkdir(parents=True)
torch.save({"epoch": -1, "train_args": {"epochs": 100}}, d / "best.pt")
#   (c) Ours dang do o epoch 37
d = OUT / "yolo" / "runs" / "vd_oursm_1024" / "weights"
d.mkdir(parents=True)
torch.save({"epoch": 36, "train_args": {"epochs": 100}}, d / "last.pt")
#   (d) model da prune, de khoi prune lai
torch.save({"x": 1}, OUT / "yolo" / "weights_yolo26m_vd_pruned50_1024.pt")
(OUT / "yolo" / "weights").mkdir(exist_ok=True)
shutil.move(str(OUT / "yolo" / "weights_yolo26m_vd_pruned50_1024.pt"),
            str(OUT / "yolo" / "weights" / "yolo26m_vd_pruned50_1024.pt"))

# ---- nap vd_common voi duong dan tro vao cay gia -----------------------------
mod = types.ModuleType("vd_e2e")
code = (SRC.replace('"/kaggle/input/', '"' + str(inp).replace("\\", "/") + "/")
           .replace('pathlib.Path("/kaggle/temp/datasets")',
                    'pathlib.Path("{}/datasets")'.format(str(tempd).replace("\\", "/")))
           .replace('pathlib.Path("/kaggle/temp")', 'pathlib.Path("{}")'.format(str(tempd).replace("\\", "/")))
           .replace('pathlib.Path("/kaggle")', 'pathlib.Path("{}")'.format(str(kag).replace("\\", "/"))))
exec(compile(code, "vd_e2e", "exec"), mod.__dict__)

BASE, OURS = "vd_yolo26m_1024", "vd_oursm_1024"
PRUNED = repo / "weights" / "yolo26m_vd_pruned50_1024.pt"
mod.init(REPO_DIR=repo, DATA="VisDrone.yaml", EPOCHS=100, BATCH=8, IMGSZ=1024,
         DEVICE="0,1", COS_LR=False, PATIENCE=100, WARMUP=3.0, STOP_AFTER_H=10.0)

print("=" * 70)
print("CELL 3 — Dataset")
print("=" * 70)
ds = mod.setup_dataset("VisDrone")

print()
print("=" * 70)
print("CELL 4 — Resume")
print("=" * 70)
mod.restore(BASE, OURS, PRUNED, {})

print()
print("=" * 70)
print("Quyet dinh tung stage")
print("=" * 70)
nb, no = mod.done_epochs(BASE), mod.done_epochs(OURS)
best = repo / "runs" / BASE / "weights" / "best.pt"
print("  baseline : {}/100  ->  {}".format(nb, "BO QUA" if 0 < 100 <= nb else "train"))
print("  best.pt cho prune + teacher :", best.exists())
print("  model da prune co san       :", PRUNED.exists(), "-> khong prune lai")
print("  ours     : {}/100  ->  {}".format(no, "RESUME tu epoch {}".format(no)))

checks = {
    "dataset nhan duoc": ds is not None and (ds / "images" / "val" / "a.jpg").exists(),
    "nhan ghi duoc (.cache)": True,
    "baseline = 100 epoch": nb == 100,
    "best.pt co cho prune": best.exists(),
    "pruned model tai dung": PRUNED.exists(),
    "ours resume dung epoch": no == 37,
}
try:
    (ds / "labels" / "val" / "val.cache").write_text("x")
except Exception:
    checks["nhan ghi duoc (.cache)"] = False

print()
print("=" * 70)
for k, v in checks.items():
    print("  {:<28} {}".format(k, "OK" if v else "!! SAI"))
print("=" * 70)
print("TONG:", sum(checks.values()), "/", len(checks))
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if all(checks.values()) else 1)
