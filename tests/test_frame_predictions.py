from pathlib import Path

from tade_ef.frame_predictions import build_frame_predictions
from tade_ef.io import write_csv


def test_evidence_is_applied_only_from_its_update_time_forward(tmp_path: Path) -> None:
    recording = "sample"
    track_dir = tmp_path / "tracks" / recording
    manifest_dir = tmp_path / "cvat" / recording
    write_csv(
        track_dir / "sample_tracks.csv",
        [
            {"recording_id": recording, "track_id": 3, "timestamp_us": 20, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"recording_id": recording, "track_id": 3, "timestamp_us": 40, "x1": 1, "y1": 0, "x2": 11, "y2": 10},
        ],
    )
    write_csv(
        manifest_dir / "manifest.csv",
        [
            {"frame_index": 0, "window_end_timestamp_us": 20},
            {"frame_index": 1, "window_end_timestamp_us": 40},
        ],
    )
    predictions = build_frame_predictions(
        [{
            "sample_id": "sample:3:000",
            "recording_id": recording,
            "parent_track_id": "sample:3",
            "start_us": "0",
            "end_us": "40",
            "label": "1",
            "probability": "0.8",
            "accumulated_evidence": "0.7",
            "identity": "drone",
        }],
        tmp_path / "tracks",
        tmp_path / "cvat",
    )
    assert [row["frame_index"] for row in predictions] == [1]
    assert predictions[0]["score"] == 0.7
    assert predictions[0]["identity"] == "drone"
