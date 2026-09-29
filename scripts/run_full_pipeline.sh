#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:?Usage: scripts/run_full_pipeline.sh DATASET_DIR LABEL_DIR [CVAT_DIR] [INPUT_FORMAT]}"
LABEL_DIR="${2:?Usage: scripts/run_full_pipeline.sh DATASET_DIR LABEL_DIR [CVAT_DIR] [INPUT_FORMAT]}"
CVAT_DIR="${3:-}"
INPUT_FORMAT="${4:-npz}"

python scripts/extract_all_recordings.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir outputs/features \
  --config configs/paper.yaml \
  --input-format "$INPUT_FORMAT"

bash scripts/train_all_folds.sh outputs/features "$LABEL_DIR" outputs/tabpfn_oof
bash scripts/evaluate_all_folds.sh \
  outputs/tabpfn_oof/oof_predictions.csv \
  outputs/features \
  "$CVAT_DIR" \
  outputs/evaluation

