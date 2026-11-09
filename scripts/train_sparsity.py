from ultralytics import YOLO


def main():
    # 1. Khởi tạo mô hình (đường dẫn tới file weights YOLO26m của bạn)
    model = YOLO("weights/best.pt")

    # 2. Bắt đầu quá trình huấn luyện với các than số custom
    print("Bắt đầu train mô hình...")
    results = model.train(
        data="VOC.yaml",  # Dataset
        epochs=50,  # Số vòng lặp
        patience=50,
        sr=1e-2,
        batch=16,
        optimizer="SGD",
        lr0=1e-3,
        resume=True,
    )

    print("Train xong! Kết quả lưu tại:", results.save_dir)


if __name__ == "__main__":
    # Bắt buộc phải có block if __name__ == '__main__' khi chạy đa luồng (multi-worker) trên Windows/WSL
    main()
