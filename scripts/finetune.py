from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="coco8.yaml", epochs=5,
        finetune=True,
        kd=False, kd_teacher="yolo26m.pt",
        kd_lambda=0.5,
        cwd_temperature=9.0,
        kd_layers="all",
    )

