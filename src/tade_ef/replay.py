"""Sequential causal replay, with optional event-time pacing and no backfill."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Iterable, Iterator
from math import hypot
import time

import numpy as np

from tade_ef.candidates import detect_candidates
from tade_ef.config import PipelineConfig
from tade_ef.evidence import EvidenceAccumulator, EvidenceRecord
from tade_ef.features import FeatureCache, extract_segment_features
from tade_ef.schema import FEATURE_NAMES
from tade_ef.tracking import TrackManager
from tade_ef.types import EventWindow, SegmentFeatures, TrackObservation


@dataclass
class TrackState:
    birth_us: int
    next_update_us: int
    accumulator: EvidenceAccumulator
    cache: FeatureCache
    observations: int = 0
    path_px: float = 0.0
    last_center: tuple[float, float] | None = None
    segment_index: int = 0
    latest: EvidenceRecord | None = None
    latest_sample_id: str = ""


@dataclass(frozen=True)
class ReplayWindow:
    window_index: int
    start_us: int
    end_us: int
    processing_ms: float
    waiting_ms: float | None
    completion_lag_ms: float | None
    updates: list[dict[str, object]]
    boxes: list[dict[str, object]]
    features: list[SegmentFeatures] = field(repr=False)


class SequentialReplay:
    """A fitted model consumes only feature updates from the current window."""

    def __init__(self, config: PipelineConfig, model, *, recording_id: str,
                 candidate_detector=detect_candidates):
        self.config = config
        self.model = model
        self.recording_id = recording_id
        self.detector = candidate_detector
        self.manager = TrackManager(config.tracking)
        self.states: dict[int, TrackState] = {}
        self.last_end_us: int | None = None

    def process(self, window: EventWindow):
        if (window.width, window.height) != (self.config.sensor_width, self.config.sensor_height):
            raise ValueError("Window sensor dimensions differ from configuration")
        if self.last_end_us is not None and window.start_us < self.last_end_us:
            raise ValueError("Replay windows must be chronological and non-overlapping")
        self.last_end_us = window.end_us
        cfg = self.config
        assignments = self.manager.update(window, self.detector(window, cfg.candidate))
        active = {track.track_id: track for track in self.manager.active}
        self.states = {key: value for key, value in self.states.items() if key in active}
        self.manager.finished.clear()
        due: list[SegmentFeatures] = []
        min_us = int(cfg.segmentation.min_duration_ms * 1000)
        max_us = int(cfg.segmentation.max_duration_ms * 1000)
        interval_us = int(cfg.segmentation.update_interval_ms * 1000)
        for track_id, _ in assignments:
            track = active[track_id]
            obs = track.last
            if track_id not in self.states:
                self.states[track_id] = TrackState(
                    obs.timestamp_us, obs.timestamp_us + min_us,
                    EvidenceAccumulator(cfg.evidence), FeatureCache(cfg.features),
                )
            state = self.states[track_id]
            center = obs.box.center
            if state.last_center is not None:
                state.path_px += hypot(center[0] - state.last_center[0],
                                       center[1] - state.last_center[1])
            state.last_center = center
            state.observations += 1
            track.observations[:] = [item for item in track.observations
                                     if item.timestamp_us >= obs.timestamp_us - max_us]
            self._prune_cache(state.cache, track.observations)
            if obs.timestamp_us < state.next_update_us:
                continue
            count = (obs.timestamp_us - state.birth_us - min_us) // interval_us + 1
            state.next_update_us = state.birth_us + min_us + count * interval_us
            if (state.observations < cfg.tracking.min_observations
                    or state.path_px < cfg.tracking.min_path_length_px
                    or obs.timestamp_us - track.observations[0].timestamp_us < min_us):
                continue
            feature = extract_segment_features(
                track.observations, recording_id=self.recording_id,
                track_id=track_id, segment_index=state.segment_index,
                config=cfg.features, cache=state.cache,
            )
            due.append(feature)
            state.segment_index += 1

        updates = []
        if due:
            matrix = np.asarray([[item.values[name] for name in FEATURE_NAMES]
                                 for item in due], dtype=float)
            probabilities = np.asarray(self.model.predict_proba(matrix))
            if probabilities.shape != (len(due), 2):
                raise ValueError("Classifier must return two-class probabilities")
            for feature, probability in zip(due, probabilities[:, 1], strict=True):
                state = self.states[feature.track_id]
                evidence = state.accumulator.update(float(probability), feature.end_us)
                state.latest = evidence
                state.latest_sample_id = (
                    f"{self.recording_id}:{feature.track_id}:{feature.segment_index:03d}"
                )
                updates.append({
                    "sample_id": state.latest_sample_id,
                    "recording_id": self.recording_id,
                    "parent_track_id": f"{self.recording_id}:{feature.track_id}",
                    "start_us": feature.start_us, "end_us": feature.end_us,
                    "probability": float(probability),
                    "accumulated_evidence": evidence.accumulated,
                    "identity": evidence.identity, "locked_now": evidence.locked_now,
                    "track_age_ms": (feature.end_us - state.birth_us) / 1000,
                })
        boxes = []
        for track_id, candidate in assignments:
            state = self.states[track_id]
            evidence = state.latest
            box = candidate.box
            boxes.append({
                "recording_id": self.recording_id, "timestamp_us": window.end_us,
                "track_id": track_id, "x1": box.x1, "y1": box.y1,
                "x2": box.x2, "y2": box.y2,
                "score": None if evidence is None else evidence.accumulated,
                "probability": None if evidence is None else evidence.probability,
                "identity": "undecided" if evidence is None else evidence.identity,
                "has_evidence": evidence is not None,
                "sample_id": state.latest_sample_id,
            })
        return due, updates, boxes

    @staticmethod
    def _prune_cache(cache: FeatureCache, observations: list[TrackObservation]):
        keep = {id(item) for item in observations}
        for store in (cache.alignments, cache.dynamics):
            for key in list(store):
                if not set(key).issubset(keep):
                    del store[key]
        for key in list(cache.elongations):
            if key not in keep:
                del cache.elongations[key]
        cache.observations = {key: item for key, item in cache.observations.items()
                              if key in keep}

    def run(self, windows: Iterable[EventWindow], *, paced: bool = False,
            clock: Callable[[], float] = time.perf_counter,
            sleep: Callable[[float], None] = time.sleep) -> Iterator[ReplayWindow]:
        """Yield immediately after each window; lateness includes prior consumer work."""
        origin_wall = None
        origin_event = None
        for index, window in enumerate(windows):
            if origin_wall is None:
                origin_wall = clock()
                origin_event = window.start_us
            deadline = origin_wall + (window.end_us - origin_event) / 1e6
            if paced:
                remaining = deadline - clock()
                if remaining > 0:
                    sleep(remaining)
            start = clock()
            features, updates, boxes = self.process(window)
            end = clock()
            yield ReplayWindow(
                index, window.start_us, window.end_us, (end - start) * 1000,
                max(0.0, (start - deadline) * 1000) if paced else None,
                max(0.0, (end - deadline) * 1000) if paced else None,
                updates, boxes, features,
            )


def stream_event_windows(x, y, t_us, polarity, *, width: int, height: int,
                         window_us: int) -> Iterator[EventWindow]:
    """Generate views of decoded, sorted input without constructing future windows."""
    if window_us <= 0 or not (len(x) == len(y) == len(t_us) == len(polarity)):
        raise ValueError("Invalid event array sizes or window duration")
    if not len(t_us):
        return
    if np.any(t_us[1:] < t_us[:-1]):
        raise ValueError("Replay input must already be timestamp-sorted")
    start = int(t_us[0])
    final = int(t_us[-1])
    while start <= final:
        end = start + window_us
        left, right = np.searchsorted(t_us, [start, end], side="left")
        yield EventWindow(start, end, x[left:right], y[left:right],
                          t_us[left:right], polarity[left:right], width, height)
        start = end
