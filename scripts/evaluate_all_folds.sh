#!/usr/bin/env bash
set -euo pipefail

OOF_PATH="${1:-outputs/tabpfn_oof/oof_predictions.csv}"
TRACK_DIR="${2:-outputs/features}"
CVAT_DIR="${3:-}"
OUTPUT_DIR="${4:-outputs/evaluation}"
mkdir -p "$OUTPUT_DIR"

tade-ef evidence "$OOF_PATH" \
  --config configs/paper.yaml \
  --output "$OUTPUT_DIR/oof_evidence.csv"

tade-ef evaluate-track "$OUTPUT_DIR/oof_evidence.csv" \
  --score-field accumulated_evidence \
  --identity-field identity \
  --threshold 0.42 \
  --output "$OUTPUT_DIR/track_metrics.json"

python scripts/summarize_oof.py \
  --predictions "$OOF_PATH" \
  --evidence "$OUTPUT_DIR/oof_evidence.csv" \
  --folds configs/folds.yaml \
  --fixed-threshold 0.42 \
  --output-dir "$OUTPUT_DIR"

if [[ -n "$CVAT_DIR" ]]; then
  python scripts/evaluate_frame_oof.py \
    --predictions "$OUTPUT_DIR/oof_evidence.csv" \
    --track-dir "$TRACK_DIR" \
    --cvat-dir "$CVAT_DIR" \
    --confidence-threshold 0.42 \
    --output-dir "$OUTPUT_DIR"
fi

