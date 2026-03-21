from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26n.pt")
    model.train(
        data="coco8.yaml",
        epochs=30,
        dms=True,
        dms_target=0.65,
        dms_importance="gamma",
        batch=16,
        project="/kaggle/working/results_dms_gammma",
        name="train",
    )
