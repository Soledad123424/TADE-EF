import numpy as np

from tade_ef.features import FeatureConfig, extract_segment_features
from tade_ef.schema import FEATURE_NAMES
from tade_ef.types import Box, TrackObservation


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

