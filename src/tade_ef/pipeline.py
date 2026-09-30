"""End-to-end event-to-36D feature extraction."""

from __future__ import annotations

from pathlib import Path
from joblib import Parallel, delayed, parallel_config

from tade_ef.candidates import detect_candidates, iter_event_windows
from tade_ef.config import PipelineConfig
from tade_ef.features import extract_track_features, split_track
from tade_ef.io import load_events, write_features, write_tracks
from tade_ef.tracking import TrackManager, is_valid_motion_track
from tade_ef.types import SegmentFeatures, Track


def extract_recording_features(
    event_path: Path,
    *,
    recording_id: str,
    config: PipelineConfig,
) -> list[SegmentFeatures]:
    _, features = extract_recording(
        event_path,
        recording_id=recording_id,
        config=config,
    )
    return features


def extract_recording(
    event_path: Path,
    *,
    recording_id: str,
    config: PipelineConfig,
    feature_workers: int = 4,
) -> tuple[list[Track], list[SegmentFeatures]]:
    if feature_workers < 1:
        raise ValueError("Feature workers must be positive")
    x, y, t_us, polarity = load_events(event_path)
    windows = iter_event_windows(
        x,
        y,
        t_us,
        polarity,
        width=config.sensor_width,
        height=config.sensor_height,
        window_us=int(config.window_ms * 1000.0),
    )
    manager = TrackManager(config.tracking)
    for window in windows:
        manager.update(window, detect_candidates(window, config.candidate))
    tracks = manager.finalize()
    tasks = []
    for track in tracks:
        if not is_valid_motion_track(track, config.tracking):
            continue
        segments = split_track(
            track,
            min_duration_ms=config.segmentation.min_duration_ms,
            max_duration_ms=config.segmentation.max_duration_ms,
            update_interval_ms=config.segmentation.update_interval_ms,
        )
        tasks.append((track.track_id, segments))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=min(feature_workers, max(1, len(tasks))))(
            delayed(extract_track_features)(
                segments, recording_id=recording_id, track_id=track_id,
                config=config.features,
            )
            for track_id, segments in tasks
        )
    features = [item for result in results for item in result]
    return tracks, features


def extract_to_csv(
    event_path: Path,
    output_path: Path,
    *,
    recording_id: str,
    config: PipelineConfig,
) -> int:
    tracks, features = extract_recording(
        event_path,
        recording_id=recording_id,
        config=config,
    )
    write_features(output_path, features)
    write_tracks(
        output_path.with_name(output_path.name.replace("_features.csv", "_tracks.csv")),
        tracks,
        recording_id,
    )
    return len(features)

