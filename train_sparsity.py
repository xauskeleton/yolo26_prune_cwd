#sua 2 file ultralytics/engine/model.py ultralytics/engine/trainer.py
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/best.pt")
    model.train(
        sr=5e-4,
        lr0=1e-3,
        data="ultralytics/cfg/datasets/coco8.yaml",
        epochs=50,
        patience=50,
        project='.',
        name='weights/train-sparsity',
        batch=8,
        device=0
    )
