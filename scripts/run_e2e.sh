#!/usr/bin/env bash
# Chay ca pipeline: prune -> finetune + CWD -> val.
# Sua cac bien duoi day, hoac dat truoc lenh: TAG=r30 RATIO=0.3 ./scripts/run_e2e.sh
set -e

DATA=${DATA:-VOC.yaml}
TAG=${TAG:-r50}
RATIO=${RATIO:-0.5}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-16}
IMGSZ=${IMGSZ:-640}
DEVICE=${DEVICE:-0}
STOP_AFTER_H=${STOP_AFTER_H:-10}

# De trong -> tu train baseline tu trong so COCO.
BASELINE=${BASELINE:-weights/yolo26m_baseline.pt}

cd "$(dirname "$0")/.."

if [ -f "$BASELINE" ]; then
  STAGES="prune finetune val"
  EXTRA="--baseline-weights $BASELINE --teacher-weights $BASELINE"
else
  echo "Khong thay $BASELINE -> train baseline truoc."
  STAGES="baseline prune finetune val"
  EXTRA=""
fi

python scripts/run_e2e.py \
  --stage $STAGES \
  --data "$DATA" --tag "$TAG" --prune-ratio "$RATIO" \
  --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ" \
  --device "$DEVICE" --stop-after-h "$STOP_AFTER_H" \
  --resume $EXTRA
