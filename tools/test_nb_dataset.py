# -*- coding: utf-8 -*-
"""Chay thu vd_common.setup_dataset() tren cay /kaggle/input gia.

Kiem tra ba truong hop: da co san, tim thay trong input (lien ket anh + chep
nhan), va khong co gi trong input.
"""
import pathlib
import shutil
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = (ROOT / "notebooks" / "share_visdrone" / "vd_common.py").read_text(encoding="utf-8")


def load(input_dir, temp_dir):
    """Nap vd_common voi /kaggle/... tro toi thu muc tam."""
    mod = types.ModuleType("vd_common_ds")
    code = (SRC.replace('"/kaggle/input/', '"' + str(input_dir).replace("\\", "/") + "/")
               .replace('Path("/kaggle/temp")', 'Path("{}")'.format(str(temp_dir).replace("\\", "/")))
               .replace('Path("/kaggle")', 'Path("{}")'.format(str(temp_dir.parent).replace("\\", "/")))
               .replace('Path("/kaggle/temp/datasets")',
                        'Path("{}/datasets")'.format(str(temp_dir).replace("\\", "/"))))
    exec(compile(code, "vd_common_ds", "exec"), mod.__dict__)
    return mod


def make_src(root, splits=("train", "val")):
    for sp in splits:
        (root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sp).mkdir(parents=True, exist_ok=True)
        (root / "images" / sp / "a.jpg").write_bytes(b"x")
        (root / "labels" / sp / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
    return root


def case(title, build):
    tmp = pathlib.Path(tempfile.mkdtemp())
    kag = tmp / "kaggle"
    inp, tempd = kag / "input", kag / "temp"
    inp.mkdir(parents=True)
    tempd.mkdir(parents=True)
    build(inp, tempd)

    V = load(inp, tempd)
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    dst = V.setup_dataset("VisDrone")

    ok = True
    if dst is not None:
        val_img = dst / "images" / "val" / "a.jpg"
        val_lab = dst / "labels" / "val" / "a.txt"
        lab_writable = False
        try:
            (dst / "labels" / "val" / "val.cache").write_text("x")
            lab_writable = True
        except Exception:
            pass
        print("   anh val doc duoc  :", val_img.exists())
        print("   nhan val doc duoc :", val_lab.exists())
        print("   nhan val GHI duoc :", lab_writable, " (can cho file .cache)")
        ok = val_img.exists() and val_lab.exists() and lab_writable
    shutil.rmtree(tmp, ignore_errors=True)
    return ok, dst


res = []

ok, dst = case("1. Tim thay trong /kaggle/input -> dung lai",
               lambda inp, td: make_src(inp / "ds" / "yolo" / "datasets" / "VisDrone"))
res.append(ok and dst is not None)

ok, dst = case("2. Khong co gi trong input -> tra ve None de Ultralytics tai",
               lambda inp, td: None)
res.append(dst is None)

ok, dst = case("3. Da co san o datasets_dir -> khong lam gi",
               lambda inp, td: make_src(td / "datasets" / "VisDrone"))
res.append(ok and dst is not None)

print("\n" + "=" * 68)
print("TONG:", sum(res), "/", len(res), "truong hop dung")
sys.exit(0 if all(res) else 1)
