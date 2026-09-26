# -*- coding: utf-8 -*-
"""Do AP theo kich thuoc vat the (AP_small / medium / large) cho cac checkpoint.

Ultralytics khong in AP_small mac dinh - no chi cho AP50, AP50-95 va per-class.
Script nay chay val voi save_json roi cham lai bang pycocotools de lay day du
sau chi so COCO, trong do co AP_small la cot ma reviewer tim khi doc bang
VisDrone (bai doi chung arXiv 2509.12918 co cot nay).

    python tools/val_apsmall.py                       # 8 ckpt trong results/visdrone/ckpt
    python tools/val_apsmall.py --data VisDrone.yaml --imgsz 640
    python tools/val_apsmall.py --weights a.pt b.pt

Khong train gi, chi val lai. Can pycocotools va dataset da tai ve.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_coco_gt(val_images, names, out_json):
    """Dung file GT dang COCO tu nhan YOLO.

    Phai khop dung quy uoc cua Ultralytics khi ghi predictions:
      - image_id  = stem cua anh (chuoi) -> o day doi sang so nguyen, va
                    predictions cung duoc doi theo cung bang anh xa
      - category_id = chi so lop + 1  (val.py:91 dung list(range(1, nc+1)))
      - bbox      = [x_goc_trai, y_tren, w, h] tinh bang pixel anh goc
    """
    from PIL import Image

    val_images = Path(val_images)
    imgs = sorted([p for p in val_images.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not imgs:
        raise SystemExit("Khong thay anh nao trong {}".format(val_images))

    stem2id, images, anns = {}, [], []
    ann_id = 1
    for i, im in enumerate(imgs, start=1):
        stem2id[im.stem] = i
        w, h = Image.open(im).size
        images.append({"id": i, "file_name": im.name, "width": w, "height": h})

        lab = Path(str(im.parent).replace("images", "labels")) / (im.stem + ".txt")
        if not lab.exists():
            continue
        for line in lab.read_text().splitlines():
            f = line.split()
            if len(f) < 5:
                continue
            c, cx, cy, bw, bh = int(f[0]), *[float(x) for x in f[1:5]]
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            bw, bh = bw * w, bh * h
            anns.append({"id": ann_id, "image_id": i, "category_id": c + 1,
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            ann_id += 1

    gt = {"images": images,
          "annotations": anns,
          "categories": [{"id": k + 1, "name": v} for k, v in sorted(names.items())]}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(gt))
    print("GT: {} anh, {} vat the -> {}".format(len(images), len(anns), out_json))
    return stem2id


def coco_eval(gt_json, pred_json, stem2id, max_det=300):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    preds = json.loads(Path(pred_json).read_text())
    for p in preds:
        # Ultralytics ghi image_id = stem (chuoi) khi dataset khong phai COCO.
        # pycocotools can id so nguyen nen doi theo dung bang anh xa cua GT.
        if not isinstance(p["image_id"], int):
            p["image_id"] = stem2id[str(p["image_id"])]
    tmp = Path(pred_json).with_name("predictions_coco.json")
    tmp.write_text(json.dumps(preds))

    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(tmp))
    e = COCOeval(coco_gt, coco_dt, "bbox")
    # COCOeval mac dinh maxDets=100, Ultralytics dung max_det=300. VisDrone
    # trung binh ~70 vat/anh va nhieu anh vuot 100, nen cat o 100 lam tut AP
    # khoang 2.4 diem so voi so Ultralytics -> phai cho khop, neu khong cot
    # APs khong cung thang do voi cot AP50 trong bang chinh.
    e.params.maxDets = [1, 10, max_det]
    e.evaluate()
    e.accumulate()
    e.summarize()
    #  0:AP  1:AP50  2:AP75  3:APs  4:APm  5:APl
    # BO stats[0]: pycocotools tinh no bang _summarize(1) voi maxDets mac dinh
    # = 100, ma ta da doi params.maxDets thanh [1,10,300] nen 100 khong con
    # trong danh sach -> tra ve -1. (Khong va duoc: _summarize la ham long ben
    # trong summarize(), khong phai method.) stats[1..5] deu dung maxDets[2]
    # nen chinh xac. AP50-95 tong the lay tu bang chinh (so Ultralytics).
    return [round(float(v) * 100, 2) for v in e.stats[1:6]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", default=None)
    ap.add_argument("--data", default="VisDrone.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=300,
                    help="phai bang max_det cua Ultralytics (mac dinh 300)")
    a = ap.parse_args()

    if a.weights:
        ckpts = [Path(w) for w in a.weights]
    else:
        ckpts = sorted((ROOT / "results" / "visdrone" / "ckpt").rglob("*.pt"))
    ckpts = [c for c in ckpts if c.exists()]
    if not ckpts:
        raise SystemExit("Khong thay checkpoint nao.")

    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    info = check_det_dataset(a.data)
    gt_json = ROOT / "results" / "visdrone" / "logs" / "coco_gt_val.json"
    names = info["names"]
    stem2id = build_coco_gt(info["val"], names, gt_json)

    rows = []
    for c in ckpts:
        print("\n" + "=" * 70)
        print(c.relative_to(ROOT) if ROOT in c.parents else c)
        print("=" * 70)
        m = YOLO(str(c))
        r = m.val(data=a.data, imgsz=a.imgsz, batch=a.batch, device=a.device,
                  max_det=a.max_det, save_json=True, verbose=False)
        pred = Path(r.save_dir) / "predictions.json"
        if not pred.exists():
            print("  !! khong sinh duoc predictions.json, bo qua")
            continue
        stats = coco_eval(gt_json, pred, stem2id, a.max_det)
        par = sum(p.numel() for p in m.model.parameters()) / 1e6
        rows.append((c.stem, round(par, 2), *stats))

    print("\n" + "=" * 92)
    print("{:<30} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
        "model", "par M", "AP50", "AP75", "APs", "APm", "APl"))
    print("-" * 84)
    for r in rows:
        print("{:<30} {:>8.2f} {:>7} {:>7} {:>7} {:>7} {:>7}".format(*r))

    out = ROOT / "results" / "visdrone" / "logs" / "ap_by_size.json"
    out.write_text(json.dumps(
        [dict(zip(("model", "params_M", "AP50", "AP75", "APs", "APm", "APl"), r))
         for r in rows], indent=2))
    print("\n->", out)


if __name__ == "__main__":
    main()
