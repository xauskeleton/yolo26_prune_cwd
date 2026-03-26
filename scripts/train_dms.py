from ultralytics import YOLO


def main():
    model = YOLO('weights/yolo26m_baseline.pt')

    results = model.train(
        data='VOC.yaml',
        batch=32,
        epochs=10,
        device=0,
        # DMS
        dms=True,
        dms_taylor_type="taylor",
        dms_target=0.70,
        dms_lambda=1.0,
        dms_lr=2e-5,
        dms_decay_ratio=0.8,
        dms_refine_ratio=0.2,
        # Save
        project="dms_train",
        name="taylor",
        save_period=2,
    )

    print("Train xong! Ket qua luu tai:", results.save_dir)


if __name__ == '__main__':
    main()
