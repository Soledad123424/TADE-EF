"""Event and tabular artifact I/O."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tade_ef.schema import FEATURE_NAMES
from tade_ef.types import SegmentFeatures, Track


_RAW_FORMAT_PATTERN = re.compile(
    r"format\s+(?P<format>EVT\d+);\s*height\s*=\s*(?P<height>\d+);\s*width\s*=\s*(?P<width>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RawHeader:
    event_format: str
    width: int
    height: int
    header_size_bytes: int


def load_events(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            return (
                np.asarray(data["x"]),
                np.asarray(data["y"]),
                np.asarray(data["t_us"]),
                np.asarray(data["polarity"]),
            )
    if path.suffix.lower() == ".raw":
        try:
            return _load_raw_with_dvsense(path)
        except ImportError as exc:
            raise RuntimeError(
                "DVSense RAW decoding requires dvsense_driver. Convert RAW files "
                "to NPZ on the acquisition workstation before using Linux servers."
            ) from exc
    raise ValueError(f"Unsupported event file format: {path.suffix}")


def parse_raw_header(path: Path) -> RawHeader:
    lines: list[str] = []
    header_size = 0
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line or not line.startswith(b"%"):
                break
            header_size += len(line)
            lines.append(line.decode("ascii", errors="replace").strip())
    match = _RAW_FORMAT_PATTERN.search("\n".join(lines))
    if match is None:
        raise ValueError(f"Could not parse RAW metadata from {path}")
    return RawHeader(
        event_format=match.group("format").upper(),
        width=int(match.group("width")),
        height=int(match.group("height")),
        header_size_bytes=header_size,
    )


def _load_raw_with_dvsense(
    path: Path,
    *,
    chunk_size: int = 500_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from dvsense_driver.raw_file_reader import RawFileReader
    except ImportError as exc:
        raise ImportError("dvsense_driver is not installed") from exc

    reader = RawFileReader(str(path))
    if not reader.load_file():
        raise RuntimeError(f"DVSense driver failed to load RAW file: {path}")
    chunks: list[np.ndarray] = []
    while not reader.reached_end_of_events():
        chunk = reader.get_n_events(chunk_size)
        if chunk.size == 0:
            break
        chunks.append(chunk.copy())
    if not chunks:
        empty = np.asarray([], dtype=np.int64)
        return empty, empty, empty, empty
    events = np.concatenate(chunks)
    required = {"x", "y", "polarity", "timestamp"}
    if events.dtype.names is None or not required.issubset(events.dtype.names):
        raise ValueError("Unexpected DVSense event dtype")
    return (
        events["x"].astype(np.int32, copy=False),
        events["y"].astype(np.int32, copy=False),
        events["timestamp"].astype(np.int64, copy=False),
        np.where(events["polarity"] > 0, 1, -1).astype(np.int8),
    )


def _decode_evt3_relative_time_unsafe(
    path: Path,
    header: RawHeader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Debug-only decoder; EVT3 absolute time reconstruction is vendor-specific."""
    if header.event_format != "EVT3":
        raise ValueError(f"Only EVT3 RAW is supported, got {header.event_format}")
    payload = path.read_bytes()[header.header_size_bytes :]
    words = np.frombuffer(payload[: len(payload) - len(payload) % 2], dtype="<u2")
    xs: list[int] = []
    ys: list[int] = []
    timestamps: list[int] = []
    polarities: list[int] = []
    current_y: int | None = None
    current_time_high = 0
    current_timestamp = 0
    for raw_word in words:
        word = int(raw_word)
        event_type = (word >> 12) & 0xF
        payload_bits = word & 0x0FFF
        if event_type == 0x8:
            current_time_high = payload_bits
        elif event_type == 0x6:
            current_timestamp = (current_time_high << 12) | payload_bits
        elif event_type == 0x0:
            current_y = word & 0x07FF
        elif event_type == 0x2 and current_y is not None:
            x = word & 0x07FF
            if x >= header.width or current_y >= header.height:
                continue
            xs.append(x)
            ys.append(current_y)
            timestamps.append(current_timestamp)
            polarities.append(1 if word & 0x0800 else -1)
    return (
        np.asarray(xs, dtype=np.int32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(timestamps, dtype=np.int64),
        np.asarray(polarities, dtype=np.int8),
    )


def write_features(path: Path, features: Iterable[SegmentFeatures]) -> None:
    rows = list(features)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id", "recording_id", "track_id", "segment_index",
        "start_us", "end_us", *FEATURE_NAMES,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": f"{row.recording_id}:{row.track_id}:{row.segment_index:03d}",
                    "recording_id": row.recording_id,
                    "track_id": row.track_id,
                    "segment_index": row.segment_index,
                    "start_us": row.start_us,
                    "end_us": row.end_us,
                    **row.values,
                }
            )


def write_tracks(path: Path, tracks: Iterable[Track], recording_id: str) -> None:
    rows: list[dict[str, object]] = []
    for track in tracks:
        for observation in track.observations:
            rows.append(
                {
                    "recording_id": recording_id,
                    "track_id": track.track_id,
                    "timestamp_us": observation.timestamp_us,
                    "x1": observation.box.x1,
                    "y1": observation.box.y1,
                    "x2": observation.box.x2,
                    "y2": observation.box.y2,
                }
            )
    if rows:
        write_csv(path, rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames = list(rows[0])
    seen = set(fieldnames)
    for row in rows[1:]:
        for name in row:
            if name not in seen:
                fieldnames.append(name)
                seen.add(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

