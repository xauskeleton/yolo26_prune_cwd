from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("weights/pruned_div8.pt")
    model.train(
        data="coco8.yaml", epochs=5,
        finetune=True,
        cwd=False, cwd_teacher="yolo26m.pt",
        cwd_lambda=0.5,
        cwd_temperature="dynamic",
        tau_max=10.0, tau_min=1.0,
        cwd_layers="all",
        cwd_layer_weights={
            2: 0.3, 4: 0.3, 6: 0.5, 8: 0.5,
            13: 1.0, 16: 1.0, 19: 1.0, 22: 1.5,
        },
    )