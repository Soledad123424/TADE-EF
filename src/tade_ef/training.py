"""Explicit recording-level three-fold TabPFN training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tade_ef.dataset import LabeledDataset
from tade_ef.io import write_csv
from tade_ef.schema import FEATURE_NAMES


def verify_sha256(path: Path, expected: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest.lower() != expected.lower():
        raise ValueError(f"Checkpoint hash mismatch: expected {expected}, got {digest}")
    return digest


def load_folds(path: Path, groups: np.ndarray) -> list[dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    folds = payload.get("folds") if isinstance(payload, dict) else None
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("Exactly three folds are required")
    expected = set(groups.tolist())
    seen: set[str] = set()
    for fold in folds:
        test = set(fold["test"])
        if seen & test:
            raise ValueError("A recording appears in multiple test folds")
        seen |= test
    if seen != expected:
        raise ValueError(f"Fold recordings differ from dataset: missing={expected-seen}, extra={seen-expected}")
    return folds


def train_oof(
    dataset: LabeledDataset,
    folds: list[dict[str, object]],
    *,
    output_dir: Path,
    model_factory: Callable[[], Any],
    save_model: Callable[[Any, Path], None] | None = None,
) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = np.full(dataset.y.shape, np.nan, dtype=float)
    fold_reports = []
    for fold_index, fold in enumerate(folds, start=1):
        test_groups = set(str(item) for item in fold["test"])
        test_mask = np.asarray([group in test_groups for group in dataset.groups])
        train_mask = ~test_mask
        if not np.any(test_mask) or len(set(dataset.y[train_mask])) < 2:
            raise ValueError(f"Fold {fold_index} has invalid train/test composition")
        model = model_factory()
        model.fit(dataset.X[train_mask], dataset.y[train_mask])
        probabilities[test_mask] = model.predict_proba(dataset.X[test_mask])[:, 1]
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if save_model is not None:
            save_model(model, fold_dir / "model.tabpfn_fit")
        report = {
            "fold": fold_index,
            "train_recordings": sorted(set(dataset.groups[train_mask].tolist())),
            "test_recordings": sorted(test_groups),
            "train_samples": int(np.count_nonzero(train_mask)),
            "test_samples": int(np.count_nonzero(test_mask)),
        }
        (fold_dir / "split.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        fold_reports.append(report)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF prediction vector is incomplete")
    rows = []
    for index, probability in enumerate(probabilities):
        rows.append({
            "sample_id": dataset.sample_ids[index],
            "recording_id": dataset.groups[index],
            "parent_track_id": dataset.parent_ids[index],
            "start_us": int(dataset.start_us[index]),
            "end_us": int(dataset.end_us[index]),
            "label": int(dataset.y[index]),
            "probability": float(probability),
        })
    write_csv(output_dir / "oof_predictions.csv", rows)
    (output_dir / "training_report.json").write_text(
        json.dumps(
            {"feature_names": FEATURE_NAMES, "folds": fold_reports},
            indent=2,
        ),
        encoding="utf-8",
    )
    return probabilities


def tabpfn_factory(*, checkpoint: Path, device: str) -> Callable[[], Any]:
    def create() -> Any:
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise RuntimeError("Install the project with the tabpfn extra") from exc
        return TabPFNClassifier(device=device, model_path=str(checkpoint))
    return create


def save_tabpfn(model: Any, path: Path) -> None:
    from tabpfn.model_loading import save_fitted_tabpfn_model

    save_fitted_tabpfn_model(model, path)

