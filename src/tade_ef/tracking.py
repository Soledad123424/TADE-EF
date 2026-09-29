"""Deterministic track association from manuscript Eqs. (5)-(6)."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np
from scipy.optimize import linear_sum_assignment

from tade_ef.types import Box, Candidate, EventWindow, Track, TrackObservation


@dataclass(frozen=True)
class TrackingConfig:
    lambda_iou: float = 0.5
    lambda_distance: float = 0.5
    max_center_distance_px: float = 60.0
    max_cost: float = 0.9
    max_missed_windows: int = 8
    min_observations: int = 4
    min_path_length_px: float = 2.0
    roi_margin_px: int = 4

    def __post_init__(self) -> None:
        if self.lambda_iou < 0 or self.lambda_distance < 0:
            raise ValueError("Association weights must be non-negative")
        if self.lambda_iou + self.lambda_distance <= 0:
            raise ValueError("At least one association weight must be positive")
        if self.max_center_distance_px <= 0 or self.max_cost < 0:
            raise ValueError("Association limits must be positive")


def box_iou(first: Box, second: Box) -> float:
    x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
    x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = first.area + second.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def association_cost(
    current: Box,
    previous: Box,
    config: TrackingConfig,
) -> float:
    distance = hypot(
        current.center[0] - previous.center[0],
        current.center[1] - previous.center[1],
    )
    if distance > config.max_center_distance_px:
        return float("inf")
    return (
        config.lambda_iou * (1.0 - box_iou(current, previous))
        + config.lambda_distance * distance / config.max_center_distance_px
    )


def path_length(track: Track) -> float:
    return sum(
        hypot(
            current.box.center[0] - previous.box.center[0],
            current.box.center[1] - previous.box.center[1],
        )
        for previous, current in zip(track.observations, track.observations[1:])
    )


def is_valid_motion_track(track: Track, config: TrackingConfig) -> bool:
    return (
        len(track.observations) >= config.min_observations
        and path_length(track) >= config.min_path_length_px
    )


class TrackManager:
    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self._next_id = 1

    def update(
        self,
        window: EventWindow,
        candidates: list[Candidate],
    ) -> list[tuple[int, Candidate]]:
        assignments = self._assign(candidates)
        matched_tracks = set(assignments.values())
        matched_candidates = set(assignments)
        output: list[tuple[int, Candidate]] = []
        for candidate_index, track_index in assignments.items():
            candidate = candidates[candidate_index]
            track = self.active[track_index]
            track.observations.append(_observation(window, candidate, self.config))
            track.missed_windows = 0
            output.append((track.track_id, candidate))
        for index, track in enumerate(self.active):
            if index not in matched_tracks:
                track.missed_windows += 1
        for index, candidate in enumerate(candidates):
            if index in matched_candidates:
                continue
            track = Track(
                track_id=self._next_id,
                observations=[_observation(window, candidate, self.config)],
            )
            self._next_id += 1
            self.active.append(track)
            output.append((track.track_id, candidate))
        retained: list[Track] = []
        for track in self.active:
            if track.missed_windows > self.config.max_missed_windows:
                self.finished.append(track)
            else:
                retained.append(track)
        self.active = retained
        return sorted(output, key=lambda item: item[0])

    def finalize(self) -> list[Track]:
        tracks = [*self.finished, *self.active]
        self.finished = []
        self.active = []
        return sorted(tracks, key=lambda track: track.track_id)

    def _assign(self, candidates: list[Candidate]) -> dict[int, int]:
        if not candidates or not self.active:
            return {}
        costs = np.full((len(candidates), len(self.active)), np.inf)
        for candidate_index, candidate in enumerate(candidates):
            for track_index, track in enumerate(self.active):
                costs[candidate_index, track_index] = association_cost(
                    candidate.box,
                    track.last.box,
                    self.config,
                )
        finite = np.isfinite(costs)
        if not np.any(finite):
            return {}
        safe = np.where(finite, costs, 1.0e12)
        rows, columns = linear_sum_assignment(safe)
        return {
            int(row): int(column)
            for row, column in zip(rows, columns)
            if finite[row, column] and costs[row, column] <= self.config.max_cost
        }


def _observation(
    window: EventWindow,
    candidate: Candidate,
    config: TrackingConfig,
) -> TrackObservation:
    box = candidate.box
    x1 = max(0, box.x1 - config.roi_margin_px)
    y1 = max(0, box.y1 - config.roi_margin_px)
    x2 = min(window.width, box.x2 + config.roi_margin_px)
    y2 = min(window.height, box.y2 + config.roi_margin_px)
    selected = (
        (window.x >= x1)
        & (window.x < x2)
        & (window.y >= y1)
        & (window.y < y2)
    )
    return TrackObservation(
        candidate.timestamp_us,
        box,
        window.x[selected].astype(float),
        window.y[selected].astype(float),
        window.t_us[selected].astype(np.int64),
        window.polarity[selected].astype(float),
    )

