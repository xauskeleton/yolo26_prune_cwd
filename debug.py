from ultralytics import YOLO

# Load model (bạn có thể dùng file .yaml định nghĩa cấu trúc hoặc file trọng số .pt)
model = YOLO('yolo26m.pt') # Thay bằng tên file model bạn đang dùng (vd: yolo26.yaml, yolov10n.pt...)

# Hiển thị bảng tóm tắt chi tiết cấu trúc mạng
model.info(detailed=False)