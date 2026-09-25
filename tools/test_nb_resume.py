# -*- coding: utf-8 -*-
"""Chay thu cell resume cua nb_m tren mot cay thu muc gia.

Lay dung source trong notebook ra, chi doi "/kaggle/input/" thanh thu muc tam.
"""
import glob
import json
import pathlib
import shutil
import tempfile

import torch

NB = pathlib.Path("notebooks/share_visdrone/nb_m.ipynb")
cells = [c for c in json.loads(NB.read_text(encoding="utf-8"))["cells"]
         if c["cell_type"] == "code"]
HELPERS = "".join(cells[2]["source"])   # cell 3. Ham dung chung
RESUME = "".join(cells[3]["source"])    # cell 4. Resume

TMP = pathlib.Path(tempfile.mkdtemp())
INPUT = TMP / "input"


def fake_run(path, n_epoch, with_csv=True):
    d = pathlib.Path(path)
    (d / "weights").mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": n_epoch - 1, "train_args": {"epochs": 100}},
               d / "weights" / "last.pt")
    if with_csv:
        rows = ["epoch,x"] + ["{},0".format(i) for i in range(1, n_epoch + 1)]
        (d / "results.csv").write_text("\n".join(rows))
    return d


def run_case(title, build, manual=None, expect=None):
    for p in (INPUT, TMP / "repo"):
        shutil.rmtree(p, ignore_errors=True)
    INPUT.mkdir(parents=True)
    (TMP / "repo" / "weights").mkdir(parents=True)
    build()

    ns = {"pathlib": pathlib, "glob": glob, "shutil": shutil, "torch": torch,
          "REPO_DIR": TMP / "repo",
          "BASE_NAME": "vd_yolo26m", "OURS_NAME": "vd_oursm",
          "EPOCHS": 100,
          "PRUNED": TMP / "repo" / "weights" / "yolo26m_vd_pruned50.pt",
          "MANUAL_LAST": manual or {"vd_yolo26m": "", "vd_oursm": ""}}
    src = (HELPERS + "\n" + RESUME).replace('"/kaggle/input/', '"' + str(INPUT).replace("\\", "/") + "/")
    print("=" * 62)
    print(title)
    exec(compile(src, "<resume>", "exec"), ns)
    got = ns["epochs_of"](TMP / "repo" / "runs" / "vd_yolo26m")
    ok = "OK" if got == expect else "SAI (mong doi {})".format(expect)
    print("  -> lay duoc {} epoch   {}".format(got, ok))
    return got == expect


results = []

results.append(run_case(
    "1. Add Data output lan truoc (yolo/runs/<name>)",
    lambda: fake_run(INPUT / "ds1" / "yolo" / "runs" / "vd_yolo26m", 42),
    expect=42))

results.append(run_case(
    "2. Hai output, phai lay ban NHIEU epoch nhat",
    lambda: (fake_run(INPUT / "ds1" / "yolo" / "runs" / "vd_yolo26m", 42),
             fake_run(INPUT / "ds2" / "yolo" / "runs" / "vd_yolo26m", 71)),
    expect=71))

results.append(run_case(
    "3. Upload thu cong ca thu muc, KHONG co runs/ bao ngoai",
    lambda: fake_run(INPUT / "vd-resume" / "vd_yolo26m", 63),
    expect=63))

results.append(run_case(
    "4. Thu muc khong co results.csv (chi last.pt)",
    lambda: fake_run(INPUT / "vd-resume" / "vd_yolo26m", 55, with_csv=False),
    expect=55))

results.append(run_case(
    "5. Chi co mot file last.pt roi le -> MANUAL_LAST",
    lambda: torch.save({"epoch": 87, "train_args": {"epochs": 100}},
                       (INPUT / "up").mkdir(parents=True) or (INPUT / "up" / "last.pt")),
    manual={"vd_yolo26m": str(INPUT / "up" / "last.pt"), "vd_oursm": ""},
    expect=88))

print("=" * 62)
print("TONG:", sum(results), "/", len(results), "truong hop dung")
shutil.rmtree(TMP, ignore_errors=True)
