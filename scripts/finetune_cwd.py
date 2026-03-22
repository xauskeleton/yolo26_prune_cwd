from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("weights/yolo26m_pruned_div8.pt")
    model.train(
        data="VOC.yaml",
        epochs=100,
        finetune=True,
        kd=True,
        kd_teacher="weights/yolo26m_sparsed.pt",
        kd_layers="all",
        kd_warmup=3,
        batch=16,
        device=0,
    )