"""Conservative old-to-new trajectory label migration."""

from __future__ import annotations

from collections import defaultdict

from tade_ef.tracking import box_iou
from tade_ef.types import Box


def normalize_old_segment_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Normalize legacy millisecond labels to the microsecond project schema."""
    normalized: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        if "start_us" not in item and "start_ms" in item:
            item["start_us"] = round(float(item["start_ms"]) * 1000.0)
        if "end_us" not in item and "end_ms" in item:
            item["end_us"] = round(float(item["end_ms"]) * 1000.0)
        if "start_us" not in item or "end_us" not in item:
            raise ValueError("Label rows require start/end in either us or ms")
        normalized.append(item)
    return normalized


def temporal_overlap(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def temporal_coverage(query: tuple[int, int], reference: tuple[int, int]) -> float:
    """Fraction of the query segment covered by the reference segment."""
    intersection = max(0, min(query[1], reference[1]) - max(query[0], reference[0]))
    duration = query[1] - query[0]
    return intersection / duration if duration > 0 else 0.0


def mean_nearest_box_iou(
    old_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
    *,
    interval: tuple[int, int] | None = None,
) -> float:
    if not old_rows or not new_rows:
        return 0.0
    if interval is not None:
        old_rows = [
            row for row in old_rows
            if interval[0] <= int(row["timestamp_us"]) <= interval[1]
        ]
        new_rows = [
            row for row in new_rows
            if interval[0] <= int(row["timestamp_us"]) <= interval[1]
        ]
    if not old_rows or not new_rows:
        return 0.0
    old_sorted = sorted(old_rows, key=lambda row: int(row["timestamp_us"]))
    values = []
    for new in new_rows:
        nearest = min(
            old_sorted,
            key=lambda row: abs(int(row["timestamp_us"]) - int(new["timestamp_us"])),
        )
        values.append(box_iou(_box(new), _box(nearest)))
    return sum(values) / len(values)


def migrate_labels(
    old_segments: list[dict[str, object]],
    old_tracks: list[dict[str, object]],
    new_segments: list[dict[str, object]],
    new_tracks: list[dict[str, object]],
    *,
    minimum_score: float = 0.65,
    ambiguity_margin: float = 0.10,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    old_by_track = _group_tracks(old_tracks)
    new_by_track = _group_tracks(new_tracks)
    accepted, review = [], []
    for new in new_segments:
        candidates = []
        new_interval = (int(new["start_us"]), int(new["end_us"]))
        new_id = int(new["track_id"])
        for old in old_segments:
            old_interval = (int(old["start_us"]), int(old["end_us"]))
            coverage = temporal_coverage(new_interval, old_interval)
            if coverage <= 0:
                continue
            intersection = (
                max(new_interval[0], old_interval[0]),
                min(new_interval[1], old_interval[1]),
            )
            spatial = mean_nearest_box_iou(
                old_by_track[int(old["track_id"])],
                new_by_track[new_id],
                interval=intersection,
            )
            candidates.append((0.55 * coverage + 0.45 * spatial, old))
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score = candidates[0][0] if candidates else 0.0
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        target = {
            "sample_id": new["sample_id"],
            "recording_id": new["recording_id"],
            "track_id": new["track_id"],
            "segment_index": new["segment_index"],
            "start_us": new["start_us"],
            "end_us": new["end_us"],
            "label": candidates[0][1]["label"] if candidates else "",
            "migration_score": best_score,
            "source_sample_id": candidates[0][1]["sample_id"] if candidates else "",
        }
        if best_score >= minimum_score and best_score - second_score >= ambiguity_margin:
            accepted.append(target)
        else:
            target["label"] = ""
            review.append(target)
    return accepted, review


def _group_tracks(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["track_id"])].append(row)
    return grouped


def _box(row: dict[str, object]) -> Box:
    return Box(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))

