import numpy as np
from sklearn.linear_model import HuberRegressor

from tade_ef.alignment import align_events, fit_huber_velocity
from tade_ef.types import Box, TrackObservation


def _observation(timestamp_us: int, center_x: int, event_x: float) -> TrackObservation:
    return TrackObservation(
        timestamp_us,
        Box(center_x - 1, 0, center_x + 1, 2),
        np.asarray([event_x]),
        np.asarray([1.0]),
        np.asarray([timestamp_us]),
        np.asarray([1.0]),
    )


def test_huber_fit_recovers_velocity_despite_outlier() -> None:
    observations = [_observation(i * 100_000, i * 10, i * 10.0) for i in range(5)]
    observations[2] = _observation(200_000, 100, 20.0)
    velocity_x, velocity_y = fit_huber_velocity(observations)
    assert abs(velocity_x - 100.0) < 5.0
    assert abs(velocity_y) < 1.0e-6


def test_alignment_moves_old_events_to_reference_position() -> None:
    observations = [_observation(0, 1, 1.0), _observation(100_000, 11, 11.0)]
    result = align_events(observations)
    assert np.allclose(result.x, [11.0, 11.0], atol=0.2)
    assert result.gain > 1.0


def test_cached_coordinate_fit_matches_original_estimator_exactly() -> None:
    observations = [_observation(i * 20_000, i % 7, float(i)) for i in range(51)]
    observations[12] = _observation(240_000, 100, 12.0)
    reference_us = observations[-1].timestamp_us
    times = np.asarray([
        (item.timestamp_us - reference_us) / 1e6 for item in observations
    ]).reshape(-1, 1)
    centers = np.asarray([item.box.center for item in observations], dtype=float)
    expected = tuple(float(HuberRegressor(epsilon=1.35, fit_intercept=True)
                           .fit(times, centers[:, axis]).coef_[0]) for axis in range(2))
    assert fit_huber_velocity(observations) == expected
    assert fit_huber_velocity(observations) == expected

