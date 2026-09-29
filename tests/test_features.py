import numpy as np

from tade_ef.features import FeatureConfig, extract_segment_features, split_track
from tade_ef.schema import FEATURE_NAMES
from tade_ef.types import Box, Track, TrackObservation


def _observations() -> list[TrackObservation]:
    result = []
    for index in range(20):
        timestamp = index * 20_000
        times = np.arange(timestamp, timestamp + 20_000, 1000, dtype=np.int64)
        phase = 2 * np.pi * 100 * ((times - timestamp) / 1_000_000.0)
        result.append(
            TrackObservation(
                timestamp,
                Box(10 + index, 10, 16 + index, 16),
                13 + index + np.cos(phase),
                13 + np.sin(phase),
                times,
                np.where(np.sin(phase) >= 0, 1.0, -1.0),
            )
        )
    return result


def test_segment_feature_vector_matches_paper_schema() -> None:
    features = extract_segment_features(
        _observations(),
        recording_id="synthetic",
        track_id=1,
        segment_index=0,
        config=FeatureConfig(min_frequency_events=8, min_quadrant_events=1),
    )
    assert tuple(features.values) == FEATURE_NAMES
    assert len(features.values) == 36
    assert features.values["local_speed_mean_window_mean"] > 0
    assert np.isfinite(list(features.values.values())).all()


def test_track_first_emits_at_300_ms_then_updates_causally() -> None:
    observations = []
    for index in range(76):
        timestamp = index * 20_000
        observations.append(
            TrackObservation(
                timestamp,
                Box(index, 0, index + 4, 4),
                np.asarray([index], dtype=float),
                np.asarray([0.0]),
                np.asarray([timestamp], dtype=np.int64),
                np.asarray([1.0]),
            )
        )
    segments = split_track(
        Track(track_id=1, observations=observations),
        min_duration_ms=300.0,
        max_duration_ms=1000.0,
        update_interval_ms=100.0,
    )
    assert [segment[-1].timestamp_us for segment in segments[:3]] == [
        300_000,
        400_000,
        500_000,
    ]
    assert all(segment[-1].timestamp_us - segment[0].timestamp_us <= 1_000_000 for segment in segments)

