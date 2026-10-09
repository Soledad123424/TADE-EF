import csv
from math import exp

import pytest

from tade_ef.evidence import EvidenceConfig
from tade_ef.inference import apply_evidence_to_oof


def test_oof_evidence_is_applied_in_chronological_order(tmp_path) -> None:
    source = tmp_path / "oof.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "parent_track_id", "end_us", "label", "probability"])
        writer.writeheader()
        writer.writerow({"sample_id": "late", "parent_track_id": "r:1", "end_us": 100, "label": 1, "probability": 0.9})
        writer.writerow({"sample_id": "early", "parent_track_id": "r:1", "end_us": 0, "label": 1, "probability": 0.9})
    rows = apply_evidence_to_oof(
        source,
        tmp_path / "evidence.csv",
        EvidenceConfig(drone_threshold=100),
    )
    assert [row["sample_id"] for row in rows] == ["early", "late"]


def test_oof_uses_each_recordings_fold_threshold(tmp_path) -> None:
    source = tmp_path / "oof.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "parent_track_id", "recording_id", "end_us", "label", "probability"])
        writer.writeheader()
        for fold in (1, 2, 3):
            writer.writerow(dict(sample_id=f"r{fold}:1:0", parent_track_id=f"r{fold}:1",
                                 recording_id=f"r{fold}", end_us=300000, label=1,
                                 probability=1 / (1 + exp(-0.43))))
    config = EvidenceConfig(drone_threshold_by_fold={"fold_1": .42, "fold_2": .45, "fold_3": .36})
    rows = apply_evidence_to_oof(source, tmp_path / "evidence.csv", config,
                                recording_folds={f"r{i}": f"fold_{i}" for i in (1, 2, 3)})
    assert [row["identity"] for row in rows] == ["drone", "undecided", "drone"]
    assert [row["uav_evidence_threshold"] for row in rows] == [.42, .45, .36]
    with pytest.raises(ValueError, match="known test fold"):
        apply_evidence_to_oof(source, tmp_path / "missing.csv", config)

