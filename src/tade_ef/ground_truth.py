"""CVAT ground truth parsing and conservative segment adjudication."""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tade_ef.tracking import box_iou
from tade_ef.types import Box


@dataclass(frozen=True)
class SegmentGroundTruthEvidence:
    decision: str
    observation_count: int
    matched_count: int
    match_fraction: float
    maximum_iou: float
    minimum_normalized_center_distance: float


def load_cvat_boxes(
    annotation_zip: Path,
    manifest_path: Path,
) -> dict[int, list[Box]]:
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    timestamps = [int(row["window_start_timestamp_us"]) for row in manifest]
    with zipfile.ZipFile(annotation_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith(".xml")]
        if len(names) != 1:
            raise ValueError(f"Expected one CVAT XML in {annotation_zip}")
        root = ET.fromstring(archive.read(names[0]))
    boxes: dict[int, list[Box]] = defaultdict(list)
    for track in root.findall("track"):
        if track.attrib.get("label") != "drone":
            continue
        for element in track.findall("box"):
            if element.attrib.get("outside", "0") == "1":
                continue
            frame = int(element.attrib["frame"])
            if not 0 <= frame < len(timestamps):
                raise ValueError(f"CVAT frame {frame} is outside the manifest")
            boxes[timestamps[frame]].append(
                Box(
                    int(math.floor(float(element.attrib["xtl"]))),
                    int(math.floor(float(element.attrib["ytl"]))),
                    int(math.ceil(float(element.attrib["xbr"]))),
                    int(math.ceil(float(element.attrib["ybr"]))),
                )
            )
    return dict(boxes)


def load_cvat_evaluation_rows(
    annotation_zip: Path,
    manifest_path: Path,
    recording_id: str,
) -> list[dict[str, object]]:
    """Return CVAT UAV boxes in the common frame-evaluation row schema."""
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))
    with zipfile.ZipFile(annotation_zip) as archive:
        names = [name for name in archive.namelist() if name.endswith(".xml")]
        if len(names) != 1:
            raise ValueError(f"Expected one CVAT XML in {annotation_zip}")
        root = ET.fromstring(archive.read(names[0]))
    rows: list[dict[str, object]] = []
    for track in root.findall("track"):
        if track.attrib.get("label") != "drone":
            continue
        for element in track.findall("box"):
            if element.attrib.get("outside", "0") == "1":
                continue
            frame = int(element.attrib["frame"])
            if not 0 <= frame < len(manifest):
                raise ValueError(f"CVAT frame {frame} is outside the manifest")
            rows.append(
                {
                    "recording_id": recording_id,
                    "frame_index": frame,
                    "x1": float(element.attrib["xtl"]),
                    "y1": float(element.attrib["ytl"]),
                    "x2": float(element.attrib["xbr"]),
                    "y2": float(element.attrib["ybr"]),
                }
            )
    return rows


def adjudicate_segment(
    segment: dict[str, object],
    track_rows: list[dict[str, object]],
    ground_truth: dict[int, list[Box]],
    *,
    timestamp_tolerance_us: int = 30_000,
    match_iou: float = 0.05,
    match_center_distance: float = 0.75,
    drone_fraction: float = 0.20,
    clear_negative_distance: float = 1.5,
) -> SegmentGroundTruthEvidence:
    start_us, end_us = int(segment["start_us"]), int(segment["end_us"])
    observations = [
        row for row in track_rows
        if start_us <= int(row["timestamp_us"]) <= end_us
    ]
    gt_timestamps = sorted(ground_truth)
    matched = 0
    maximum_iou = 0.0
    minimum_distance = math.inf
    for observation in observations:
        timestamp = int(observation["timestamp_us"])
        candidates = [
            gt_time for gt_time in gt_timestamps
            if abs(gt_time - timestamp) <= timestamp_tolerance_us
        ]
        if not candidates:
            continue
        gt_time = min(candidates, key=lambda item: abs(item - timestamp))
        observed_box = _box(observation)
        observation_match = False
        for gt_box in ground_truth[gt_time]:
            iou = box_iou(observed_box, gt_box)
            distance = _normalized_center_distance(observed_box, gt_box)
            maximum_iou = max(maximum_iou, iou)
            minimum_distance = min(minimum_distance, distance)
            if iou >= match_iou or distance <= match_center_distance:
                observation_match = True
        matched += int(observation_match)
    count = len(observations)
    fraction = matched / count if count else 0.0
    if matched and fraction >= drone_fraction:
        decision = "drone"
    elif matched == 0 and (
        not math.isfinite(minimum_distance)
        or minimum_distance >= clear_negative_distance
    ):
        decision = "non_drone"
    else:
        decision = "review"
    return SegmentGroundTruthEvidence(
        decision=decision,
        observation_count=count,
        matched_count=matched,
        match_fraction=fraction,
        maximum_iou=maximum_iou,
        minimum_normalized_center_distance=(
            minimum_distance if math.isfinite(minimum_distance) else -1.0
        ),
    )


def _normalized_center_distance(first: Box, second: Box) -> float:
    dx = first.center[0] - second.center[0]
    dy = first.center[1] - second.center[1]
    diagonal = math.hypot(second.width, second.height)
    return math.hypot(dx, dy) / max(diagonal, 1.0)


def _box(row: dict[str, object]) -> Box:
    return Box(
        int(float(row["x1"])),
        int(float(row["y1"])),
        int(float(row["x2"])),
        int(float(row["y2"])),
    )
