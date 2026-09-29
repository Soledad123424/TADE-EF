"""Event activity candidate generation from manuscript Eqs. (2)-(4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from tade_ef.types import Box, Candidate, EventWindow


@dataclass(frozen=True)
class CandidateConfig:
    dilation_radius_px: int = 1
    min_component_area_px: int = 9
    min_width_px: int = 3
    min_height_px: int = 3
    min_events: int = 8
    roi_padding_px: int = 2

    def __post_init__(self) -> None:
        if self.dilation_radius_px < 0:
            raise ValueError("Dilation radius must be non-negative")
        if min(
            self.min_component_area_px,
            self.min_width_px,
            self.min_height_px,
            self.min_events,
        ) <= 0:
            raise ValueError("Candidate filtering thresholds must be positive")


def circular_kernel(radius_px: int) -> np.ndarray:
    """Return the circular structuring element B_r in Eq. (4)."""
    if radius_px < 0:
        raise ValueError("Radius must be non-negative")
    coordinates = np.arange(-radius_px, radius_px + 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    return ((xx * xx + yy * yy) <= radius_px * radius_px).astype(np.uint8)


def activity_map(window: EventWindow) -> np.ndarray:
    mask = np.zeros((window.height, window.width), dtype=np.uint8)
    if window.x.size:
        valid = (
            (window.x >= 0)
            & (window.x < window.width)
            & (window.y >= 0)
            & (window.y < window.height)
        )
        mask[window.y[valid].astype(int), window.x[valid].astype(int)] = 1
    return mask


def detect_candidates(
    window: EventWindow,
    config: CandidateConfig,
) -> list[Candidate]:
    mask = activity_map(window)
    if config.dilation_radius_px:
        mask = ndimage.binary_dilation(
            mask,
            structure=circular_kernel(config.dilation_radius_px),
            iterations=1,
        ).astype(np.uint8)
    labels, component_count = ndimage.label(
        mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    result: list[Candidate] = []
    component_slices = ndimage.find_objects(labels)
    for component_id in range(1, component_count + 1):
        component_slice = component_slices[component_id - 1]
        if component_slice is None:
            continue
        y_slice, x_slice = component_slice
        x, y = int(x_slice.start), int(y_slice.start)
        width, height = int(x_slice.stop - x), int(y_slice.stop - y)
        area = int(np.count_nonzero(labels[component_slice] == component_id))
        if area < config.min_component_area_px:
            continue
        if width < config.min_width_px or height < config.min_height_px:
            continue
        event_labels = labels[window.y.astype(int), window.x.astype(int)]
        event_count = int(np.count_nonzero(event_labels == component_id))
        if event_count < config.min_events:
            continue
        box = Box(
            max(0, x - config.roi_padding_px),
            max(0, y - config.roi_padding_px),
            min(window.width, x + width + config.roi_padding_px),
            min(window.height, y + height + config.roi_padding_px),
        )
        result.append(Candidate(window.end_us, box, event_count))
    return sorted(result, key=lambda candidate: (candidate.box.x1, candidate.box.y1))


def iter_event_windows(
    x: np.ndarray,
    y: np.ndarray,
    t_us: np.ndarray,
    polarity: np.ndarray,
    *,
    width: int,
    height: int,
    window_us: int,
) -> list[EventWindow]:
    """Split sorted events into non-overlapping windows as in Eq. (2)."""
    if window_us <= 0:
        raise ValueError("window_us must be positive")
    if not (x.size == y.size == t_us.size == polarity.size):
        raise ValueError("Event arrays must have equal lengths")
    if t_us.size == 0:
        return []
    order = np.argsort(t_us, kind="stable")
    x, y, t_us, polarity = x[order], y[order], t_us[order], polarity[order]
    start = int(t_us[0])
    final = int(t_us[-1])
    windows: list[EventWindow] = []
    while start <= final:
        end = start + window_us
        left = int(np.searchsorted(t_us, start, side="left"))
        right = int(np.searchsorted(t_us, end, side="left"))
        windows.append(
            EventWindow(
                start,
                end,
                x[left:right],
                y[left:right],
                t_us[left:right],
                polarity[left:right],
                width,
                height,
            )
        )
        start = end
    return windows

