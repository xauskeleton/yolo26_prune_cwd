# Patch cho torch-pruning 1.6.0

`torch-pruning-1.6.0-yolo-c3k2.patch` — sua hai loi lam DepGraph suy sai nhom
tren khoi C3k2 cua YOLO.

## Ap dung

```bash
git clone --depth 1 https://github.com/VainF/Torch-Pruning "Tai lieu/Torch-Pruning"
cd "Tai lieu/Torch-Pruning" && git apply ../../patches/torch-pruning-1.6.0-yolo-c3k2.patch
```

`pruning/prune_depgraph.py` tu uu tien ban clone nay neu co, va in ro dang dung
ban nao.

## Loi gi

C3k2 lam: `y = list(cv1(x).chunk(2, 1)); y.extend(m(y[-1])); cv2(cat(y, 1))`
— mot manh cua chunk vua chay thang vao concat, vua di qua bottleneck.

**1. Nhanh xu ly `_saved_split_size` khong bao gio chay toi.** Dieu kien ngoai
trong `shape_infer.init_shape_information` chi kiem `_saved_self_sizes` hoac
`_saved_split_sizes` (so nhieu). PyTorch 2.6 sinh `SplitBackward0` cho `chunk()`,
chi co `_saved_split_size` (so it). Moi thu roi xuong nhanh suy nguoc qua concat
va cho ra kich thuoc rac — vd `[768, 768, 256, 256]` cho mot chunk that su la
`[256, 256]`.

**2. Dung `len(node.outputs)` lam so manh.** `node.outputs` dem ben TIEU THU chu
khong phai so manh. Mot manh nuoi nhieu noi la lech ngay.

Do that tren yolo26m, so lop prunable tinh duoc diem nhom:

| | lop |
|---|---|
| truoc khi va | 76/90 |
| sau khi va | **83/90** |

Sau khi va, moi nut `chunk` bao dung kich thuoc that: `[64,64]`, `[128,128]`,
`[256,256]` — doi chieu bang hook vao forward.

## Chua va duoc

`index_mapping.update_split_index_mapping` cung lap theo ben tieu thu:
`offsets[i:i+2]`. Voi 4 consumer nhung offsets dai 3 thi `i=2` cho `[128]` va
`i=3` cho `[]`, roi `offset[1]` vuot bien.

Sua dung phai biet **consumer doc manh nao** — quan he ma torch-pruning khong ghi
lai luc trace. Do la thay doi thiet ke chu khong phai vet va, nen 7 lop
`cvN.cv1.bn` con lai lui ve L1. Co y KHONG vet tam bang cach noi rong offset:
mot nhom sai am tham con nguy hiem hon la bao loi.
