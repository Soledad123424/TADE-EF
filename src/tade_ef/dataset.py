"""Validated 36-D feature/label loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tade_ef.io import read_csv
from tade_ef.schema import FEATURE_NAMES


@dataclass(frozen=True)
class LabeledDataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_ids: np.ndarray
    parent_ids: np.ndarray
    start_us: np.ndarray
    end_us: np.ndarray


def load_labeled_dataset(feature_dir: Path, label_dir: Path) -> LabeledDataset:
    features: dict[str, dict[str, str]] = {}
    for path in sorted(feature_dir.rglob("*_features.csv")):
        for row in read_csv(path):
            sample_id = row["sample_id"]
            if sample_id in features:
                raise ValueError(f"Duplicate feature sample_id: {sample_id}")
            features[sample_id] = row
    labels: dict[str, str] = {}
    for path in sorted(label_dir.rglob("*_labels.csv")):
        for row in read_csv(path):
            label = row.get("label", "").strip()
            if label in {"drone", "non_drone"}:
                labels[row["sample_id"]] = label
    sample_ids = sorted(set(features) & set(labels))
    if not sample_ids:
        raise ValueError("No matched labeled samples")
    matrix, targets, groups, parents, starts, ends = [], [], [], [], [], []
    for sample_id in sample_ids:
        row = features[sample_id]
        vector = [float(row[name]) for name in FEATURE_NAMES]
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite features for {sample_id}")
        matrix.append(vector)
        targets.append(1 if labels[sample_id] == "drone" else 0)
        groups.append(row["recording_id"])
        parents.append(f"{row['recording_id']}:{row['track_id']}")
        starts.append(int(row["start_us"]))
        ends.append(int(row["end_us"]))
    return LabeledDataset(
        np.asarray(matrix, dtype=float),
        np.asarray(targets, dtype=np.int8),
        np.asarray(groups),
        np.asarray(sample_ids),
        np.asarray(parents),
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
    )

