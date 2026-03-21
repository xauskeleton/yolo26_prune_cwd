from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("weights/yolo26m_pruned_div8.pt")
    model.train(
        data="VOC.yaml",
        epochs=100,
        finetune=True,
        cwd=True,
        cwd_teacher="weights/yolo26m_sparsed.pt",
        cwd_layers="all",
        cwd_warmup=3,
        batch=16,
        device=0,
    )