"""Map causal evidence states forward to 20-ms trajectory boxes."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from tade_ef.io import read_csv


def build_frame_predictions(
    oof_rows: list[dict[str, str]],
    track_dir: Path,
    manifest_root: Path,
) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    by_recording: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in oof_rows:
        by_recording[row["recording_id"]].append(row)
    for recording, segments in by_recording.items():
        tracks = read_csv(track_dir / recording / f"{recording}_tracks.csv")
        tracks_by_id: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in tracks:
            tracks_by_id[int(row["track_id"])].append(row)
        frame_by_end = _frame_by_window_end(
            manifest_root / _safe_name(recording) / "manifest.csv"
        )
        evidence_by_track: dict[int, list[dict[str, str]]] = defaultdict(list)
        for segment in segments:
            track_id = int(segment["parent_track_id"].rsplit(":", 1)[1])
            evidence_by_track[track_id].append(segment)
        for track_id, updates in evidence_by_track.items():
            ordered_updates = sorted(updates, key=lambda item: int(item["end_us"]))
            latest: dict[str, str] | None = None
            update_index = 0
            for observation in sorted(
                tracks_by_id[track_id], key=lambda item: int(item["timestamp_us"])
            ):
                timestamp = int(observation["timestamp_us"])
                while (
                    update_index < len(ordered_updates)
                    and int(ordered_updates[update_index]["end_us"]) <= timestamp
                ):
                    latest = ordered_updates[update_index]
                    update_index += 1
                if latest is None:
                    continue
                frame_index = frame_by_end.get(timestamp)
                if frame_index is None:
                    continue
                predictions.append(
                    {
                        "recording_id": recording,
                        "frame_index": frame_index,
                        "timestamp_us": timestamp,
                        "track_id": track_id,
                        "x1": observation["x1"],
                        "y1": observation["y1"],
                        "x2": observation["x2"],
                        "y2": observation["y2"],
                        "score": float(latest["accumulated_evidence"]),
                        "identity": latest["identity"],
                        "label": int(latest["label"]),
                        "sample_id": latest["sample_id"],
                    }
                )
    return predictions


def _frame_by_window_end(path: Path) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["window_end_timestamp_us"]): int(row["frame_index"])
        for row in rows
    }


def _safe_name(recording: str) -> str:
    return recording.replace(" ", "_")
