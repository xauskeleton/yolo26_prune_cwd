# -*- coding: utf-8 -*-
"""Thu build_coco_gt + coco_eval tren mot dataset gia nho.

Kiem tra ba cho de sai: doi image_id chuoi -> so, category_id lech 1, va
chuyen bbox YOLO chuan hoa -> COCO pixel.
"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, "tools")
sys.path.insert(0, ".")
from PIL import Image

from val_apsmall import build_coco_gt, coco_eval

TMP = pathlib.Path(tempfile.mkdtemp())
IMG = TMP / "images" / "val"
LAB = TMP / "labels" / "val"
IMG.mkdir(parents=True)
LAB.mkdir(parents=True)

W, H = 200, 100
# Ba anh, moi anh mot hop, ba kich thuoc khac nhau de rot vao 3 dai dien tich:
#   small  < 32^2 = 1024      medium 32^2..96^2      large > 96^2 = 9216
BOXES = {
    "aaa_001": (0.5, 0.5, 20 / W, 20 / H),      # 20x20  = 400   -> small
    "bbb_002": (0.5, 0.5, 50 / W, 50 / H),      # 50x50  = 2500  -> medium
    "ccc_003": (0.5, 0.5, 150 / W, 90 / H),     # 150x90 = 13500 -> large
}
for stem, (cx, cy, bw, bh) in BOXES.items():
    Image.new("RGB", (W, H)).save(IMG / (stem + ".jpg"))
    (LAB / (stem + ".txt")).write_text("0 {} {} {} {}".format(cx, cy, bw, bh))

names = {0: "obj"}
gt_json = TMP / "gt.json"
stem2id = build_coco_gt(IMG, names, gt_json)

gt = json.loads(gt_json.read_text())
print("\nKiem tra GT:")
for a in gt["annotations"]:
    print("   image_id={}  category_id={}  bbox={}  area={:.0f}".format(
        a["image_id"], a["category_id"],
        [round(x, 1) for x in a["bbox"]], a["area"]))
assert all(a["category_id"] == 1 for a in gt["annotations"]), "category_id phai 1-based"
areas = sorted(a["area"] for a in gt["annotations"])
assert areas == [400.0, 2500.0, 13500.0], areas
print("   -> category_id 1-based OK, dien tich OK")

# predictions.json dung dinh dang Ultralytics: image_id la CHUOI (stem),
# category_id 1-based, bbox goc trai + wh. Du doan trung khop GT.
preds = []
for stem, (cx, cy, bw, bh) in BOXES.items():
    x = (cx - bw / 2) * W
    y = (cy - bh / 2) * H
    preds.append({"image_id": stem, "file_name": stem + ".jpg", "category_id": 1,
                  "bbox": [x, y, bw * W, bh * H], "score": 0.9})
pred_json = TMP / "predictions.json"
pred_json.write_text(json.dumps(preds))

print("\nCham diem (du doan trung khop hoan toan -> moi chi so phai la 100):")
stats = coco_eval(gt_json, pred_json, stem2id)
print("\n   AP={} AP50={} AP75={} APs={} APm={} APl={}".format(*stats))
ok = all(abs(s - 100.0) < 1e-6 for s in stats)
print("\n==>", "OK - duong ong cham diem dung" if ok else "!! SAI")
sys.exit(0 if ok else 1)
