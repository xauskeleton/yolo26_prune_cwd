from ultralytics import YOLO


def main():
    # 1. Khởi tạo mô hình (đường dẫn tới file weights YOLO26m của bạn)
    model = YOLO('yolo26m.pt')

    # 2. Bắt đầu quá trình huấn luyện với các tham số custom
    print("Bắt đầu train mô hình...")
    results = model.train(
        data='coco8.yaml',  # Dataset
        epochs=5,  # Số vòng lặp
        dms=True,  # Tham số custom (Dynamic Model Sparsity)
        dms_target=0.5,  # Tham số custom
        dms_lambda=1.0,  # Tham số custom
        dms_importance='taylor',  # Taylor importance

    )

    print("Train xong! Kết quả lưu tại:", results.save_dir)


if __name__ == '__main__':
    # Bắt buộc phải có block if __name__ == '__main__' khi chạy đa luồng (multi-worker) trên Windows/WSL
    main()