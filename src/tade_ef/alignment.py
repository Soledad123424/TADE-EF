"""Trajectory-guided event alignment from manuscript Eqs. (7)-(14)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.linear_model import HuberRegressor

from tade_ef.types import TrackObservation


@lru_cache(maxsize=4096)
def _fit_coordinate(times_bytes: bytes, target_bytes: bytes, epsilon: float) -> float:
    """Memoize identical solver inputs, without changing the Huber estimator."""
    times = np.frombuffer(times_bytes, dtype=np.float64).reshape(-1, 1)
    # Match the original column view's stride, including its floating-point path.
    targets = np.empty((times.shape[0], 2), dtype=np.float64)
    targets[:, 0] = np.frombuffer(target_bytes, dtype=np.float64)
    model = HuberRegressor(epsilon=epsilon, fit_intercept=True)
    model.fit(times, targets[:, 0])
    return float(model.coef_[0])


@dataclass(frozen=True)
class AlignmentResult:
    x: np.ndarray
    y: np.ndarray
    t_us: np.ndarray
    polarity: np.ndarray
    velocity_x_px_s: float
    velocity_y_px_s: float
    gain: float


def fit_huber_velocity(
    observations: list[TrackObservation],
    *,
    epsilon: float = 1.35,
) -> tuple[float, float]:
    if len(observations) < 2:
        return (0.0, 0.0)
    reference_us = observations[-1].timestamp_us
    times = np.asarray(
        [(item.timestamp_us - reference_us) / 1_000_000.0 for item in observations]
    ).reshape(-1, 1)
    if float(np.ptp(times)) <= 1.0e-12:
        return (0.0, 0.0)
    centers = np.asarray([item.box.center for item in observations], dtype=float)
    velocities = []
    times_bytes = times.tobytes()
    for coordinate in range(2):
        velocities.append(_fit_coordinate(
            times_bytes, centers[:, coordinate].tobytes(), epsilon,
        ))
    return (velocities[0], velocities[1])


def align_events(
    observations: list[TrackObservation],
    *,
    epsilon: float = 1.35,
    delta: float = 1.0e-9,
) -> AlignmentResult:
    available = [item for item in observations if item.event_t_us.size]
    if not available:
        empty = np.asarray([], dtype=float)
        return AlignmentResult(empty, empty, empty.astype(np.int64), empty, 0.0, 0.0, 0.0)
    velocity_x, velocity_y = fit_huber_velocity(observations, epsilon=epsilon)
    x = np.concatenate([item.event_x for item in available]).astype(float)
    y = np.concatenate([item.event_y for item in available]).astype(float)
    t_us = np.concatenate([item.event_t_us for item in available]).astype(np.int64)
    polarity = np.concatenate([item.event_polarity for item in available]).astype(float)
    reference_us = observations[-1].timestamp_us
    dt_s = (reference_us - t_us.astype(float)) / 1_000_000.0
    aligned_x = x + velocity_x * dt_s
    aligned_y = y + velocity_y * dt_s
    before = float(np.var(x) + np.var(y)) if x.size > 1 else 0.0
    after = (
        float(np.var(aligned_x) + np.var(aligned_y))
        if aligned_x.size > 1
        else 0.0
    )
    gain = before / max(after, delta) if before > 0.0 else 0.0
    return AlignmentResult(
        aligned_x,
        aligned_y,
        t_us,
        polarity,
        velocity_x,
        velocity_y,
        gain,
    )

