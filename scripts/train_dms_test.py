from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    results = model.train(
        data="coco8.yaml",
        batch=64,
        epochs=30,
        device=0,
        # DMS
        dms=True,
        dms_taylor_type="taylor",
        dms_target=0.70,
        dms_lambda=40.0,
        dms_lr=1e-2,
        dms_grad_scale=-1,  # -1=OFF (default, match paper), >=0=ON auto-balance
        # Save
        project="dms_train",
        name="taylor",
        save_period=5,
        workers=2,
    )

    print("Train xong! Ket qua luu tai:", results.save_dir)


if __name__ == "__main__":
    main()
