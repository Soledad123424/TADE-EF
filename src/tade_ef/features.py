"""Dual-timescale 36-D segment features from manuscript Eqs. (21)-(24)."""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from math import acos, hypot
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from tade_ef.alignment import AlignmentResult, align_events
from tade_ef.schema import FEATURE_NAMES
from tade_ef.spectral import SpectralResult, extract_spectral_features
from tade_ef.types import SegmentFeatures, Track, TrackObservation

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class FeatureCache:
    """Bounded, track-local memoization for identical observation prefixes."""

    def __init__(self, config: FeatureConfig, max_entries: int = 128) -> None:
        self.config = config
        self.max_entries = max_entries
        self.alignments: OrderedDict[tuple[int, ...], AlignmentResult] = OrderedDict()
        self.dynamics: OrderedDict[tuple[int, ...], dict[str, float]] = OrderedDict()
        self.elongations: OrderedDict[int, float] = OrderedDict()
        self.observations: dict[int, TrackObservation] = {}
        self.hits = 0
        self.misses = 0

    def get_or_compute(
        self, store: OrderedDict[_Key, _Value], key: _Key,
        compute: Callable[[], _Value],
    ) -> _Value:
        if key in store:
            self.hits += 1
            store.move_to_end(key)
            return store[key]
        self.misses += 1
        value = compute()
        store[key] = value
        if len(store) > self.max_entries:
            store.popitem(last=False)
        return value


@dataclass(frozen=True)
class FeatureConfig:
    alignment_history_ms: float = 1000.0
    short_window_ms: float = 100.0
    frequency_update_interval_ms: float = 100.0
    min_frequency_events: int = 12
    frequency_min_hz: float = 20.0
    frequency_max_hz: float = 1200.0
    frequency_step_hz: float = 20.0
    harmonics: int = 4
    harmonic_half_width_bins: int = 1
    min_quadrant_events: int = 6
    spectral_epsilon: float = 1.0e-12
    huber_epsilon: float = 1.35
    delta: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.alignment_history_ms <= 0:
            raise ValueError("Alignment history must be positive")
        if self.spectral_epsilon <= 0:
            raise ValueError("Spectral epsilon must be positive")

    @property
    def frequencies_hz(self) -> np.ndarray:
        return np.arange(
            self.frequency_min_hz,
            self.frequency_max_hz + self.frequency_step_hz / 2.0,
            self.frequency_step_hz,
        )


def extract_track_features(
    segments: list[list[TrackObservation]], *, recording_id: str,
    track_id: int, config: FeatureConfig,
) -> list[SegmentFeatures]:
    """Process one track with a private cache; safe for independent CPU workers."""
    cache = FeatureCache(config)
    return [
        extract_segment_features(
            segment, recording_id=recording_id, track_id=track_id,
            segment_index=index, config=config, cache=cache,
        )
        for index, segment in enumerate(segments)
    ]


def split_track(
    track: Track,
    *,
    min_duration_ms: float,
    max_duration_ms: float,
    update_interval_ms: float,
) -> list[list[TrackObservation]]:
    if (
        min_duration_ms <= 0
        or max_duration_ms < min_duration_ms
        or update_interval_ms <= 0
    ):
        raise ValueError("Invalid segment duration limits")
    observations = sorted(track.observations, key=lambda item: item.timestamp_us)
    if not observations:
        return []
    max_duration_us = int(max_duration_ms * 1000.0)
    min_duration_us = int(min_duration_ms * 1000.0)
    update_interval_us = int(update_interval_ms * 1000.0)
    segments: list[list[TrackObservation]] = []
    track_start_us = observations[0].timestamp_us
    next_update_us = track_start_us + min_duration_us
    for end_index, observation in enumerate(observations):
        if observation.timestamp_us < next_update_us:
            continue
        history_start_us = max(track_start_us, observation.timestamp_us - max_duration_us)
        history = [
            item
            for item in observations[: end_index + 1]
            if item.timestamp_us >= history_start_us
        ]
        if history[-1].timestamp_us - history[0].timestamp_us >= min_duration_us:
            segments.append(history)
        elapsed_after_first = observation.timestamp_us - (track_start_us + min_duration_us)
        update_count = elapsed_after_first // update_interval_us + 1
        next_update_us = (
            track_start_us + min_duration_us + update_count * update_interval_us
        )
    return segments


def extract_segment_features(
    observations: list[TrackObservation],
    *,
    recording_id: str,
    track_id: int,
    segment_index: int,
    config: FeatureConfig,
    cache: FeatureCache | None = None,
) -> SegmentFeatures:
    if cache is not None and cache.config != config:
        raise ValueError("Feature cache belongs to a different configuration")
    ordered = sorted(observations, key=lambda item: item.timestamp_us)
    if cache is not None:
        # Keep identities alive while memoized prefixes refer to them.
        cache.observations.update((id(item), item) for item in ordered)
    if len(ordered) < 2:
        raise ValueError("A segment needs at least two observations")

    alignment_gains: list[float] = []
    spectral_results: list[SpectralResult] = []
    elongations: list[float] = []
    elongation_cvs: list[float] = []
    area_cvs: list[float] = []
    aspect_cvs: list[float] = []
    local_speed_means: list[float] = []
    local_speed_stds: list[float] = []
    local_acceleration_means: list[float] = []
    local_acceleration_stds: list[float] = []
    local_jerk_means: list[float] = []
    local_curvature_means: list[float] = []
    last_frequency_us: int | None = None
    last_spectral: SpectralResult | None = None

    for end_index, current in enumerate(ordered):
        prefix = ordered[: end_index + 1]
        alignment_start_us = current.timestamp_us - int(
            config.alignment_history_ms * 1000.0
        )
        alignment_prefix = [
            item for item in prefix if item.timestamp_us >= alignment_start_us
        ]
        def compute_alignment():
            return align_events(
                alignment_prefix, epsilon=config.huber_epsilon, delta=config.delta,
            )

        alignment = (
            compute_alignment() if cache is None else cache.get_or_compute(
                cache.alignments, tuple(id(item) for item in alignment_prefix),
                compute_alignment,
            )
        )
        alignment_gains.append(alignment.gain)
        if alignment.t_us.size:
            recent = alignment.t_us >= current.timestamp_us - int(config.short_window_ms * 1000.0)
            due = (
                last_frequency_us is None
                or current.timestamp_us - last_frequency_us
                >= int(config.frequency_update_interval_ms * 1000.0)
            )
            if due and int(np.count_nonzero(recent)) >= config.min_frequency_events:
                last_spectral = extract_spectral_features(
                    alignment.x[recent],
                    alignment.y[recent],
                    alignment.t_us[recent],
                    alignment.polarity[recent],
                    config.frequencies_hz,
                    harmonics=config.harmonics,
                    half_width_bins=config.harmonic_half_width_bins,
                    min_quadrant_events=config.min_quadrant_events,
                    epsilon=config.spectral_epsilon,
                )
                last_frequency_us = current.timestamp_us
        if last_spectral is not None:
            spectral_results.append(last_spectral)

        def compute_elongation():
            return _elongation(current.event_x, current.event_y, config.delta)

        elongations.append(
            compute_elongation() if cache is None else cache.get_or_compute(
                cache.elongations, id(current), compute_elongation,
            )
        )
        areas = [float(item.box.area) for item in prefix]
        aspects = [item.box.width / item.box.height for item in prefix]
        elongation_cvs.append(_cv(elongations, config.delta))
        area_cvs.append(_cv(areas, config.delta))
        aspect_cvs.append(_cv(aspects, config.delta))
        dynamics = (
            _local_dynamics(prefix, config.delta) if cache is None
            else cache.get_or_compute(
                cache.dynamics, tuple(id(item) for item in prefix),
                lambda: _local_dynamics(prefix, config.delta),
            )
        )
        local_speed_means.append(dynamics["speed_mean"])
        local_speed_stds.append(dynamics["speed_std"])
        local_acceleration_means.append(dynamics["acceleration_mean"])
        local_acceleration_stds.append(dynamics["acceleration_std"])
        local_jerk_means.append(dynamics["jerk_mean"])
        local_curvature_means.append(dynamics["curvature_mean"])

    centers = np.asarray([item.box.center for item in ordered], dtype=float)
    timestamps = np.asarray([item.timestamp_us for item in ordered], dtype=float)
    displacement_vectors = np.diff(centers, axis=0)
    step_lengths = np.linalg.norm(displacement_vectors, axis=1)
    path = float(np.sum(step_lengths))
    endpoint = float(np.linalg.norm(centers[-1] - centers[0]))
    dt_s = np.diff(timestamps) / 1_000_000.0
    valid = dt_s > 0
    speeds = step_lengths[valid] / dt_s[valid]
    areas = [float(item.box.area) for item in ordered]
    aspects = [item.box.width / item.box.height for item in ordered]
    event_counts = [float(item.event_t_us.size) for item in ordered]

    values = {
        "alignment_gain_mean": _mean(alignment_gains),
        "alignment_gain_max": _maximum(alignment_gains),
        "peak_frequency_mean_hz": _mean([item.peak_frequency_hz for item in spectral_results]),
        "peak_frequency_std_hz": _std([item.peak_frequency_hz for item in spectral_results]),
        "peak_frequency_cv": _cv([item.peak_frequency_hz for item in spectral_results], config.delta),
        "spectral_flatness_mean": _mean([item.spectral_flatness for item in spectral_results]),
        "spectral_flatness_std": _std([item.spectral_flatness for item in spectral_results]),
        "harmonic_score_mean": _mean([item.harmonic_score for item in spectral_results]),
        "harmonic_score_max": _maximum([item.harmonic_score for item in spectral_results]),
        "spatial_phase_concentration_mean": _mean([item.phase_concentration for item in spectral_results]),
        "spatial_phase_concentration_max": _maximum([item.phase_concentration for item in spectral_results]),
        "elongation_mean": _mean(elongations),
        "elongation_std": _std(elongations),
        "elongation_cv_mean": _mean(elongation_cvs),
        "elongation_cv_max": _maximum(elongation_cvs),
        "box_area_cv_mean": _mean(area_cvs),
        "box_area_cv_max": _maximum(area_cvs),
        "box_aspect_cv_mean": _mean(aspect_cvs),
        "box_aspect_cv_max": _maximum(aspect_cvs),
        "path_length_px": path,
        "endpoint_displacement_px": endpoint,
        "straightness": endpoint / path if path > config.delta else 0.0,
        "speed_mean_px_per_s": float(np.mean(speeds)) if speeds.size else 0.0,
        "speed_max_px_per_s": float(np.max(speeds)) if speeds.size else 0.0,
        "local_speed_mean_window_mean": _mean(local_speed_means),
        "local_speed_std_window_mean": _mean(local_speed_stds),
        "local_acceleration_mean_window_mean": _mean(local_acceleration_means),
        "local_acceleration_std_window_mean": _mean(local_acceleration_stds),
        "local_jerk_mean_window_mean": _mean(local_jerk_means),
        "local_curvature_mean_window_mean": _mean(local_curvature_means),
        "duration_ms": (ordered[-1].timestamp_us - ordered[0].timestamp_us) / 1000.0,
        "roi_event_count_mean": _mean(event_counts),
        "box_area_mean_px2": _mean(areas),
        "box_area_std_px2": _std(areas),
        "box_aspect_mean": _mean(aspects),
        "box_aspect_std": _std(aspects),
    }
    if tuple(values) != FEATURE_NAMES:
        raise RuntimeError("Feature extraction order differs from the paper schema")
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Feature vector contains a non-finite value")
    return SegmentFeatures(
        recording_id,
        track_id,
        segment_index,
        ordered[0].timestamp_us,
        ordered[-1].timestamp_us,
        values,
    )


def _elongation(x: np.ndarray, y: np.ndarray, delta: float) -> float:
    if x.size < 2:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(np.cov(np.vstack((x, y)), bias=True))
    return float(eigenvalues[-1] / max(float(eigenvalues[0]), delta))


def _local_dynamics(
    observations: list[TrackObservation],
    delta: float,
) -> dict[str, float]:
    if len(observations) < 2:
        return {key: 0.0 for key in (
            "speed_mean", "speed_std", "acceleration_mean",
            "acceleration_std", "jerk_mean", "curvature_mean",
        )}
    centers = np.asarray([item.box.center for item in observations], dtype=float)
    timestamps = np.asarray([item.timestamp_us for item in observations], dtype=float)
    dt = np.diff(timestamps) / 1_000_000.0
    valid = dt > 0
    displacements = np.diff(centers, axis=0)[valid]
    dt = dt[valid]
    velocities = displacements / dt[:, None]
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = _derivative(velocities, dt)
    acceleration_norms = np.linalg.norm(accelerations, axis=1)
    jerks = _derivative(accelerations, _centered_dt(dt))
    jerk_norms = np.linalg.norm(jerks, axis=1)
    curvatures = []
    for previous, current in zip(displacements, displacements[1:]):
        previous_norm, current_norm = np.linalg.norm(previous), np.linalg.norm(current)
        if previous_norm <= delta or current_norm <= delta:
            continue
        cosine = float(np.clip(np.dot(previous, current) / (previous_norm * current_norm), -1, 1))
        curvatures.append(acos(cosine) / max(float((previous_norm + current_norm) / 2), delta))
    return {
        "speed_mean": float(np.mean(speeds)) if speeds.size else 0.0,
        "speed_std": float(np.std(speeds)) if speeds.size else 0.0,
        "acceleration_mean": float(np.mean(acceleration_norms)) if acceleration_norms.size else 0.0,
        "acceleration_std": float(np.std(acceleration_norms)) if acceleration_norms.size else 0.0,
        "jerk_mean": float(np.mean(jerk_norms)) if jerk_norms.size else 0.0,
        "curvature_mean": _mean(curvatures),
    }


def _centered_dt(dt: np.ndarray) -> np.ndarray:
    return (dt[:-1] + dt[1:]) / 2.0 if dt.size >= 2 else np.asarray([], dtype=float)


def _derivative(vectors: np.ndarray, intervals: np.ndarray) -> np.ndarray:
    if vectors.shape[0] < 2 or intervals.size < 2:
        return np.empty((0, 2), dtype=float)
    derivative_dt = (intervals[:-1] + intervals[1:]) / 2.0
    valid = derivative_dt > 0
    return np.diff(vectors, axis=0)[valid] / derivative_dt[valid, None]


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def _maximum(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _cv(values: list[float], delta: float) -> float:
    mean = _mean(values)
    return _std(values) / abs(mean) if abs(mean) > delta else 0.0

