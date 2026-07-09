from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("weights/yolo26m_pruned_div8.pt")

    # ============================================================
    # 1. CWD (mac dinh) - Channel-Wise Distillation
    # ============================================================
    model.train(
        imgsz=160,
        data="VOC.yaml",
        epochs=100,
        patience=10,
        finetune=True,
        kd=True,
        kd_teacher="weights/yolo26m_sparsed.pt",
        kd_method="response",
        #   kd_method="fitnets",
        #   fitnets_normalize=True,        # L2 normalize truoc MSE
        #   kd_method="mgd",
        kd_warmup=3,  # so epoch warmup truoc khi bat KD
        batch=1,
    )
