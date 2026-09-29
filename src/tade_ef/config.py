"""Typed configuration loading with unknown-key rejection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from tade_ef.candidates import CandidateConfig
from tade_ef.evidence import EvidenceConfig
from tade_ef.features import FeatureConfig
from tade_ef.tracking import TrackingConfig


@dataclass(frozen=True)
class SegmentConfig:
    min_duration_ms: float = 300.0
    max_duration_ms: float = 1000.0
    update_interval_ms: float = 100.0

    def __post_init__(self) -> None:
        if self.min_duration_ms <= 0.0:
            raise ValueError("Minimum segment duration must be positive")
        if self.max_duration_ms < self.min_duration_ms:
            raise ValueError("Maximum history must be at least the minimum duration")
        if self.update_interval_ms <= 0.0:
            raise ValueError("Segment update interval must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    window_ms: float
    sensor_width: int
    sensor_height: int
    candidate: CandidateConfig
    tracking: TrackingConfig
    features: FeatureConfig
    segmentation: SegmentConfig
    evidence: EvidenceConfig
    tabpfn: dict[str, object]


def load_config(path: Path) -> PipelineConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    candidate_payload = _mapping(payload, "candidate")
    tracking_payload = _mapping(payload, "tracking")
    alignment_payload = _mapping(payload, "alignment")
    frequency_payload = _mapping(payload, "frequency")
    segment_payload = _mapping(payload, "segmentation")
    evidence_payload = _mapping(payload, "evidence")
    tracking_payload["roi_margin_px"] = alignment_payload["roi_margin_px"]
    feature_payload = {
        "alignment_history_ms": alignment_payload["history_ms"],
        "short_window_ms": frequency_payload["short_window_ms"],
        "frequency_update_interval_ms": frequency_payload["update_interval_ms"],
        "min_frequency_events": frequency_payload["min_roi_events"],
        "frequency_min_hz": frequency_payload["min_hz"],
        "frequency_max_hz": frequency_payload["max_hz"],
        "frequency_step_hz": frequency_payload["step_hz"],
        "harmonics": frequency_payload["harmonics"],
        "harmonic_half_width_bins": frequency_payload["harmonic_half_width_bins"],
        "min_quadrant_events": frequency_payload["min_quadrant_events"],
        "spectral_epsilon": frequency_payload["epsilon"],
        "huber_epsilon": alignment_payload["huber_epsilon"],
        "delta": alignment_payload["delta"],
    }
    return PipelineConfig(
        window_ms=float(payload["window_ms"]),
        sensor_width=int(payload["sensor_width"]),
        sensor_height=int(payload["sensor_height"]),
        candidate=CandidateConfig(**candidate_payload),
        tracking=TrackingConfig(**tracking_payload),
        features=FeatureConfig(**feature_payload),
        segmentation=SegmentConfig(**segment_payload),
        evidence=EvidenceConfig(**evidence_payload),
        tabpfn=dict(_mapping(payload, "tabpfn")),
    )


def _mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration field {key!r} must be a mapping")
    return dict(value)

