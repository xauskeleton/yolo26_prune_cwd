# -*- coding: utf-8 -*-
"""Dong goi checkpoint thanh zip de upload lam Kaggle dataset.

Cau truc ben trong PHAI la <run_name>/weights/last.pt thi cell resume cua
notebook moi tu tim thay (glob "/kaggle/input/**/<name>"), khong phai dien
MANUAL_LAST.

Moi size mot zip, de hai nguoi upload song song thay vi mot goi 200 MB.
"""
import pathlib
import shutil
import sys
import zipfile

sys.path.insert(0, ".")
import torch

CK = pathlib.Path("results/visdrone/ckpt")
OUT = pathlib.Path("results/visdrone/upload")
OUT.mkdir(parents=True, exist_ok=True)

SIZES = sys.argv[1:] or ["m", "l"]


def find(kind, size):
    d = CK / kind / size
    fs = [f for f in d.glob("*.pt")]
    return fs[0] if fs else None


for size in SIZES:
    base = find("baseline", size)
    ours = find("pruned", size)
    if not base:
        print("{}: thieu baseline, bo qua".format(size))
        continue

    zp = OUT / "vd_resume_{}.zip".format(size)
    members = []

    # Baseline chi can best.pt: stage prune doc no va no la teacher cho CWD.
    # Cell resume nhan ca thu muc chi co best.pt, va epochs_of() doc epoch = -1
    # (da strip optimizer = train xong) -> tra ve train_args.epochs = 100 nen
    # train() bo qua, khong goi resume tren run da ket thuc.
    members.append((base, "vd_yolo26{}/weights/best.pt".format(size)))

    if ours:
        members.append((ours, "vd_ours{}/weights/last.pt".format(size)))

    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        for src, arc in members:
            z.write(src, arc)

    print("{}  ->  {:.1f} MB".format(zp, zp.stat().st_size / 1e6))
    for _, arc in members:
        print("     ", arc)

    # Kiem tra lai: mo zip ra doc epoch de chac khong nham file
    with zipfile.ZipFile(zp) as z:
        for arc in {a for _, a in members}:
            tmp = OUT / "_chk.pt"
            with z.open(arc) as fh, open(tmp, "wb") as o:
                shutil.copyfileobj(fh, o)
            ck = torch.load(tmp, map_location="cpu", weights_only=False)
            run = (ck.get("train_args") or {}).get("name")
            ep = ck.get("epoch")
            want = arc.split("/")[0]
            flag = "OK" if run == want else "!! LECH (run={})".format(run)
            print("     kiem tra {:<34} epoch={:<4} {}".format(arc, ep, flag))
            del ck
            tmp.unlink()
    print()
