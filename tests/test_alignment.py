import numpy as np

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

