import csv

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

