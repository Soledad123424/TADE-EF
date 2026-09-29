from tade_ef.label_migration import (
    migrate_labels,
    normalize_old_segment_rows,
    temporal_coverage,
)


def test_legacy_millisecond_intervals_are_normalized() -> None:
    rows = normalize_old_segment_rows([{"start_ms": "1.25", "end_ms": "2.5"}])
    assert rows[0]["start_us"] == 1250
    assert rows[0]["end_us"] == 2500


def test_temporal_coverage_does_not_penalize_larger_reference_segment() -> None:
    assert temporal_coverage((20, 40), (0, 100)) == 1.0


def test_clear_spatiotemporal_match_is_accepted() -> None:
    old_segment = [{"sample_id": "old", "recording_id": "r", "track_id": 1, "start_us": 0, "end_us": 100, "label": "drone"}]
    new_segment = [{"sample_id": "new", "recording_id": "r", "track_id": 2, "segment_index": 0, "start_us": 0, "end_us": 100}]
    old_tracks = [{"track_id": 1, "timestamp_us": 50, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    new_tracks = [{"track_id": 2, "timestamp_us": 50, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    accepted, review = migrate_labels(old_segment, old_tracks, new_segment, new_tracks)
    assert accepted[0]["label"] == "drone"
    assert review == []

