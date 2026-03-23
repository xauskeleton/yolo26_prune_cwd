from ultralytics import YOLO


def main():
    # 1. Khởi tạo mô hình (đường dẫn tới file weights YOLO26m của bạn)
    model = YOLO('ok/runs/detect/dms_train/l1_norm/weights/last.pt')

    # 2. Bắt đầu quá trình huấn luyện với các tham số custom
    print("Bắt đầu train mô hình...")
    results = model.train(
        data='VOC.yaml',  # Dataset
        batch=32,
        patience=10,
        epochs=10,
        dms=True,
        dms_importance="l1",
        dms_lambda=30,
        dms_target=0.7,
        device=0,
        dms_lr=0.02,
        dms_warmup=0,
        project="dms_train",   # luu thang vao day
        name="l1_norm",                         # ten folder con
        save_period=1
    )

    print("Train xong! Kết quả lưu tại:", results.save_dir)


if __name__ == '__main__':
    # Bắt buộc phải có block if __name__ == '__main__' khi chạy đa luồng (multi-worker) trên Windows/WSL
    main()