"""Core immutable data types shared by the TADE-EF pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EventWindow:
    start_us: int
    end_us: int
    x: np.ndarray
    y: np.ndarray
    t_us: np.ndarray
    polarity: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        sizes = {self.x.size, self.y.size, self.t_us.size, self.polarity.size}
        if len(sizes) != 1:
            raise ValueError("Event arrays must have equal lengths")
        if self.end_us <= self.start_us:
            raise ValueError("Event window end must follow start")


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Box must have positive width and height")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class Candidate:
    timestamp_us: int
    box: Box
    event_count: int


@dataclass(frozen=True)
class TrackObservation:
    timestamp_us: int
    box: Box
    event_x: np.ndarray = field(repr=False)
    event_y: np.ndarray = field(repr=False)
    event_t_us: np.ndarray = field(repr=False)
    event_polarity: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        sizes = {
            self.event_x.size,
            self.event_y.size,
            self.event_t_us.size,
            self.event_polarity.size,
        }
        if len(sizes) != 1:
            raise ValueError("Observation event arrays must have equal lengths")


@dataclass
class Track:
    track_id: int
    observations: list[TrackObservation]
    missed_windows: int = 0

    @property
    def last(self) -> TrackObservation:
        return self.observations[-1]


@dataclass(frozen=True)
class SegmentFeatures:
    recording_id: str
    track_id: int
    segment_index: int
    start_us: int
    end_us: int
    values: dict[str, float]

