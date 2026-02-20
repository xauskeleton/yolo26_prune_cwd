#ultralytics/engine/model.py
#ultralytics/engine/trainer.py
from ultralytics import YOLO

model = YOLO("weights/best.pt")
model.train(
    sr=5e-4,
    lr0=1e-3,
    data="/kaggle/input/datasets/kushagrapandya/visdrone-dataset/VisDrone.yaml",
    epochs=50,
    patience=50,
    project='.',
    name='weights/train-sparsity',
    batch=48,
    device=0
)