from pathlib import Path

from tade_ef.frame_predictions import build_frame_predictions
from tade_ef.io import write_csv


def test_segment_probability_maps_to_matching_window_end(tmp_path: Path) -> None:
    recording = "sample"
    track_dir = tmp_path / "tracks" / recording
    manifest_dir = tmp_path / "cvat" / recording
    write_csv(
        track_dir / "sample_tracks.csv",
        [{"recording_id": recording, "track_id": 3, "timestamp_us": 20, "x1": 0, "y1": 0, "x2": 10, "y2": 10}],
    )
    write_csv(
        manifest_dir / "manifest.csv",
        [{"frame_index": 0, "window_end_timestamp_us": 20}],
    )
    predictions = build_frame_predictions(
        [{"sample_id": "sample:3:000", "recording_id": recording, "parent_track_id": "sample:3", "start_us": "0", "end_us": "20", "label": "1", "probability": "0.8"}],
        tmp_path / "tracks",
        tmp_path / "cvat",
    )
    assert predictions[0]["frame_index"] == 0
    assert predictions[0]["score"] == 0.8
