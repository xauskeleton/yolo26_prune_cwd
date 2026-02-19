tôi đang làm một project pruning + cwd cho yolo26 , code sẽ gồm source code full của ultralytics,các file sẽ chỉnh sửa
các file chính tôi sẽ làm việc là :
prune.py file chính để gọi
ultralytics/nn/modules/block_pruned.py : file này được custom chỉnh sủa theo file block.py nhằm thêm các khối prune custom
ultralytics/nn/modules/head_pruned.py : file này được chỉnh theo head.py để viết detection head
ultralytics/nn/tasks_pruned.py : thực hiện chính các tính toán
ultralytics\cfg\models\26\yolo26.yaml : file yaml cua yolo26
hiện tại chỉ triển khai phần pruning, distilation sẽ được thực hiện bởi người khác

python prune.py --weights weights/best.pt --cfg cfg/yolo26m.yaml --prune-ratio 0.3

lệnh chính sẽ được thực hiện nhiều lần 




YOLOv26 Model Pruning - Project Context
Dự án
Implement channel pruning cho YOLOv26 object detection model để giảm kích thước model và tăng tốc độ inference.
Cấu trúc
ultralytics-main/
├── prune.py                           # Script chính chạy pruning
├── ultralytics/nn/
│   ├── tasks_pruned.py                # Build pruned model từ masks
│   └── modules/
│       ├── block_pruned.py            # Pruned modules (C3kPruned, C3k2Pruned, etc.)
│       └── head_pruned.py             # DetectPruned head
Workflow

