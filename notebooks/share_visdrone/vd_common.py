# -*- coding: utf-8 -*-
"""Logic dung chung cho 4 notebook VisDrone (n / s / m / l).

De o day chu khong nhet vao cell notebook: cell notebook nam trong file .ipynb
tren Kaggle nen `git clone` KHONG va duoc - moi lan sua lai phai bat 4 nguoi
import lai notebook. File nay di theo repo nen sua mot cho la ca nhom co ngay.
"""
import glob
import pathlib
import shutil

import torch
from ultralytics import YOLO

CFG = {}


def init(**kw):
    """Nhan cau hinh tu cell 2 cua notebook."""
    CFG.update(kw)
    CFG["REPO_DIR"] = pathlib.Path(CFG["REPO_DIR"])
    return CFG


def _run(name):
    return CFG["REPO_DIR"] / "runs" / name


def run_csv(name):
    return _run(name) / "results.csv"


def epochs_of(run_dir):
    """So epoch da train xong cua mot thu muc run.

    results.csv truoc; khong co thi doc thang checkpoint. Thu ca last.pt lan
    best.pt: mot run da xong co the chi con best.pt (vd goi resume gui tay).
    epoch = -1 nghia la da strip optimizer, tuc train xong -> lay so epoch muc
    tieu trong train_args.
    """
    run_dir = pathlib.Path(run_dir)
    csv = run_dir / "results.csv"
    if csv.exists():
        rows = [r for r in csv.read_text().strip().splitlines()[1:] if r.strip()]
        if rows:
            return int(float(rows[-1].split(",")[0]))
    for f in (run_dir / "weights" / "last.pt", run_dir / "weights" / "best.pt"):
        if not f.exists():
            continue
        try:
            ck = torch.load(f, map_location="cpu", weights_only=False)
            ep = int(ck.get("epoch", -1))
            total = int((ck.get("train_args") or {}).get("epochs", 0) or 0)
            del ck
            return (ep + 1) if ep >= 0 else total
        except Exception:
            pass
    return 0


def done_epochs(name):
    return epochs_of(_run(name))


def bring_back(name, manual=""):
    """Tim checkpoint cu trong /kaggle/input va chep ve runs/<name>."""
    dst = _run(name)
    if dst.exists():
        return

    cands = []
    # (a) Add Data output cua lan chay truoc: .../yolo/runs/<name>/
    cands += glob.glob("/kaggle/input/**/runs/" + name, recursive=True)
    # (b) Upload thu cong: thu muc ten <name> nam bat ky dau, khong can co runs/.
    cands += glob.glob("/kaggle/input/**/" + name, recursive=True)
    # Nhan ca thu muc chi co best.pt: goi resume cua baseline da xong khong can
    # kem last.pt (tiet kiem ~50 MB moi goi).
    cands = [c for c in set(cands)
             if pathlib.Path(c, "weights", "last.pt").exists()
             or pathlib.Path(c, "weights", "best.pt").exists()]

    if cands:
        # Gan nhieu output (phien 1, phien 2, ...) thi phai lay ban NHIEU EPOCH
        # NHAT. Lay "cai dau tien tim thay" la sai: thu tu glob khong xac dinh,
        # co the chep nham ban cu va mat vai gio train ma khong he biet.
        best = max(cands, key=epochs_of)
        shutil.copytree(best, dst)
        print("  {}: {} epoch  <- {}".format(name, epochs_of(best), best))
        return

    # (c) Chi co moi file last.pt roi le -> duong dan truyen vao qua manual.
    if manual and pathlib.Path(manual).exists():
        (dst / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copy2(manual, dst / "weights" / "last.pt")
        print("  {}: {} epoch  <- (thu cong) {}".format(name, epochs_of(dst), manual))
    elif manual:
        print("  {}: !! duong dan khong ton tai: {}".format(name, manual))


def restore(base_name, ours_name, pruned, manual=None):
    """Chep het ve, roi in tien do."""
    manual = manual or {}
    for n in (base_name, ours_name):
        bring_back(n, manual.get(n, ""))

    pruned = pathlib.Path(pruned)
    if not pruned.exists():
        for src in glob.glob("/kaggle/input/**/" + pruned.name, recursive=True):
            pruned.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, pruned)
            print("  {} <- {}".format(pruned.name, src))
            break

    print()
    for label, n in (("baseline", base_name), ("ours", ours_name)):
        print("{:<9}: {} / {} epoch".format(label, done_epochs(n), CFG["EPOCHS"]))


def train(name, weights, **extra):
    """Train moi, hoac train TIEP neu phien truoc bi cat ngang.

    Co last.pt la resume. Khong lay so epoch lam dieu kien de QUYET DINH resume:
    doc that bai thi se am tham train lai tu dau, mat ca chuc gio ma khong bao gi.
    """
    n_done = done_epochs(name)

    # Da chay du roi thi KHONG duoc goi resume: Ultralytics bao "nothing to
    # resume" va dung han. Gap khi phien truoc train xong baseline, phien sau
    # chep runs/ ve roi chay lai tu dau notebook.
    if 0 < CFG["EPOCHS"] <= n_done:
        print("[{}] da du {}/{} epoch, bo qua".format(name, n_done, CFG["EPOCHS"]))
        return n_done

    last = _run(name) / "weights" / "last.pt"
    if last.exists():
        print("[{}] resume tu epoch {}".format(name, n_done))
        YOLO(str(last)).train(resume=True, stop_after_h=CFG["STOP_AFTER_H"])
    else:
        print("[{}] train tu dau".format(name))
        YOLO(weights).train(
            data=CFG["DATA"], epochs=CFG["EPOCHS"], batch=CFG["BATCH"],
            imgsz=CFG["IMGSZ"], device=CFG["DEVICE"], seed=0,
            cos_lr=CFG["COS_LR"], patience=CFG["PATIENCE"],
            warmup_epochs=CFG["WARMUP"],
            project=str(CFG["REPO_DIR"] / "runs"), name=name, exist_ok=True,
            stop_after_h=CFG["STOP_AFTER_H"], **extra)
    return done_epochs(name)


def prune50(size, src_weights, out_path, ratio=0.5, divisor=8):
    """L1-norm uniform prune. Goi thang cac ham trong pruning/."""
    from prune_common import load_and_prepare, create_masks, finalize_pruning
    from prune_l1norm import compute_l1norm_importance

    out_path = pathlib.Path(out_path)
    if out_path.exists():
        print("da co", out_path.name)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    m0, bn_dict, ignore_bn, _c, layer_cfg, pruned_yaml = load_and_prepare(
        str(src_weights), str(CFG["REPO_DIR"] / "cfg" / "yolo26m.yaml"), size, None)
    imp = compute_l1norm_importance(m0, bn_dict, ignore_bn)
    masks = create_masks(imp, m0, ignore_bn, layer_cfg, ratio, divisor)
    p = finalize_pruning(m0, masks, pruned_yaml, ignore_bn, str(src_weights),
                         str(CFG["REPO_DIR"] / "weights"), divisor, ratio,
                         method_name="l1norm")
    pathlib.Path(p).replace(out_path)
    print("->", out_path)
    return out_path


def measure(name):
    """Do AP cua mot run da xong. Tra ve (params_M, AP50, AP50-95, ten file)."""
    if done_epochs(name) < CFG["EPOCHS"]:
        return None
    # best.pt co the KHONG ton tai: khi resume, Ultralytics chi ghi best.pt luc
    # fitness vuot ky luc cu, ma ky luc do nam trong checkpoint tu phien truoc.
    # Neu cac epoch cuoi khong vuot thi khong co best.pt -> lui ve last.pt.
    w = _run(name) / "weights" / "best.pt"
    if not w.exists():
        w = _run(name) / "weights" / "last.pt"
        if not w.exists():
            return None
        print("  ({}: khong co best.pt, do tren last.pt)".format(name))
    m = YOLO(str(w))
    r = m.val(data=CFG["DATA"], imgsz=CFG["IMGSZ"], batch=CFG["BATCH"],
              device=str(CFG["DEVICE"]).split(",")[0])
    return (sum(p.numel() for p in m.model.parameters()) / 1e6,
            r.box.map50 * 100, r.box.map * 100, w)


def report(size, base_name, ours_name):
    """In bang 2 dong, va gom file lai dung layout cua repo."""
    rows = [("YOLO26-" + size.upper(), measure(base_name)),
            ("Ours-" + size.upper(), measure(ours_name))]
    print()
    if not all(v for _, v in rows):
        for label, n in (("baseline", base_name), ("ours", ours_name)):
            print("  {:<9} {} / {} epoch".format(label, done_epochs(n), CFG["EPOCHS"]))
        print()
        print("CHUA XONG - Add Data output lan nay roi Save & Run All lai.")
        return

    print("| Model | Params (M) | AP50 | AP50-95 |")
    print("|---|---:|---:|---:|")
    for label, (par, ap50, ap, _w) in rows:
        print("| {} | {:.2f} | {:.2f} | {:.2f} |".format(label, par, ap50, ap))

    vd = CFG["REPO_DIR"] / "results" / "visdrone"
    saved = []
    for name, kind, stem in ((base_name, "baseline", "yolo26{}_vd_baseline".format(size)),
                             (ours_name, "pruned", "yolo26{}_vd_ours50".format(size))):
        ck, lg = vd / "ckpt" / kind / size, vd / "logs" / kind / size
        ck.mkdir(parents=True, exist_ok=True)
        lg.mkdir(parents=True, exist_ok=True)
        src = dict(rows)[("YOLO26-" if kind == "baseline" else "Ours-") + size.upper()][3]
        shutil.copy2(src, ck / (stem + ".pt"))
        saved.append(ck / (stem + ".pt"))
        # results.csv co the khong co: run duoc khoi phuc tu goi resume chi kem
        # weights/, khong kem csv. Khong duoc de cho nay lam hong ca buoc bao cao.
        if run_csv(name).exists():
            shutil.copy2(run_csv(name), lg / (stem + ".csv"))
            saved.append(lg / (stem + ".csv"))

    print()
    print("XONG ca hai. Gui lai bang tren, kem cac file nay:")
    for f in saved:
        print("   {:<58} {:.1f} MB".format(
            str(f.relative_to(CFG["REPO_DIR"])), f.stat().st_size / 1e6))
