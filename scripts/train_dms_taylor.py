from ultralytics import YOLO


def main():
    model = YOLO('weights/yolo26m_baseline.pt')

    results = model.train(
        data='VOC.yaml',
        batch=32,
        epochs=30,
        device=0,
        # DMS
        dms=True,
        dms_taylor_type="taylor",
        dms_target=0.70,
        dms_lambda=50.0,
        dms_lr=1e-2,
        dms_grad_scale=-1,    # -1=OFF (default, match paper), >=0=ON auto-balance
        # Save
        # project="dms_train",
        # name="taylor",
        save_period=10,

    )

    print("Train xong! Ket qua luu tai:", results.save_dir)


if __name__ == '__main__':
    main()
