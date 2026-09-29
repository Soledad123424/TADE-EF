#!/usr/bin/env bash
set -euo pipefail

FEATURE_DIR="${1:-outputs/features}"
LABEL_DIR="${2:-outputs/labels}"
OUTPUT_DIR="${3:-outputs/tabpfn_oof}"

tade-ef train \
  --feature-dir "$FEATURE_DIR" \
  --label-dir "$LABEL_DIR" \
  --folds configs/folds.yaml \
  --config configs/paper.yaml \
  --output-dir "$OUTPUT_DIR"

