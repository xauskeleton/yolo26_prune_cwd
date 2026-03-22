from ultralytics import YOLO


def main():
    # 1. Khởi tạo mô hình (đường dẫn tới file weights YOLO26m của bạn)
    model = YOLO('weights/yolo26m_baseline.pt')

    # 2. Bắt đầu quá trình huấn luyện với các tham số custom
    print("Bắt đầu train mô hình...")
    results = model.train(
        data='VOC.yaml',  # Dataset
        batch=32,
        patience=30,
        epochs=30,
        dms=True,
        dms_importance="l1",
        dms_lambda=10,
        dms_target=0.65,
        device=0,
        dms_lr=0.02,
        project="dms_train",   # luu thang vao day
        name="l1_norm",                         # ten folder con
    )

    print("Train xong! Kết quả lưu tại:", results.save_dir)


if __name__ == '__main__':
    # Bắt buộc phải có block if __name__ == '__main__' khi chạy đa luồng (multi-worker) trên Windows/WSL
    main()