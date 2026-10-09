"""Sequential evidence accumulation and permanent identity locking."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import log


@dataclass(frozen=True)
class EvidenceConfig:
    classification_threshold: float = 0.5
    decay: float = 0.95
    drone_threshold: float = 0.42
    non_drone_threshold: float = 0.1
    non_drone_lock_ms: float = 500.0
    epsilon: float = 1.0e-6
    drone_threshold_by_fold: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.classification_threshold < 1:
            raise ValueError("Classification threshold must be in (0, 1)")
        if not 0 <= self.decay <= 1:
            raise ValueError("Evidence decay must be in [0, 1]")
        if self.non_drone_threshold >= self.drone_threshold:
            raise ValueError("Non-UAV threshold must be below UAV threshold")
        for fold, threshold in self.drone_threshold_by_fold.items():
            if fold not in {"fold_1", "fold_2", "fold_3"}:
                raise ValueError(f"Unknown evidence fold: {fold}")
            if self.non_drone_threshold >= threshold:
                raise ValueError("Non-UAV threshold must be below every fold UAV threshold")

    def for_fold(self, fold: str | int | None) -> "EvidenceConfig":
        if not self.drone_threshold_by_fold:
            return self
        key = str(fold)
        if key in {"1", "2", "3"}:
            key = f"fold_{key}"
        if key not in self.drone_threshold_by_fold:
            raise ValueError("Fold-specific UAV thresholds require fold_1, fold_2 or fold_3")
        return replace(self, drone_threshold=self.drone_threshold_by_fold[key],
                       drone_threshold_by_fold={})


@dataclass(frozen=True)
class EvidenceRecord:
    timestamp_us: int
    probability: float
    log_odds: float
    accumulated: float
    identity: str
    locked_now: bool


class EvidenceAccumulator:
    def __init__(self, config: EvidenceConfig) -> None:
        if config.drone_threshold_by_fold:
            raise ValueError("Resolve the fold-specific evidence configuration before accumulation")
        self.config = config
        self.accumulated = 0.0
        self.identity = "undecided"
        self._below_drone_since_us: int | None = None

    def update(self, probability: float, timestamp_us: int) -> EvidenceRecord:
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be in [0, 1]")
        epsilon = self.config.epsilon
        probability = min(max(probability, epsilon), 1 - epsilon)
        threshold = min(max(self.config.classification_threshold, epsilon), 1 - epsilon)
        score = log(probability / (1 - probability)) - log(threshold / (1 - threshold))
        self.accumulated = self.config.decay * self.accumulated + score
        if self.identity != "undecided":
            return EvidenceRecord(
                timestamp_us,
                probability,
                score,
                self.accumulated,
                self.identity,
                False,
            )
        if self.accumulated < self.config.drone_threshold:
            if self._below_drone_since_us is None:
                self._below_drone_since_us = timestamp_us
        else:
            self._below_drone_since_us = None
        locked_now = False
        if self.accumulated >= self.config.drone_threshold:
            self.identity = "drone"
            locked_now = True
        elif self.accumulated <= self.config.non_drone_threshold:
            duration_us = (
                0
                if self._below_drone_since_us is None
                else timestamp_us - self._below_drone_since_us
            )
            if duration_us >= int(self.config.non_drone_lock_ms * 1000.0):
                self.identity = "non_drone"
                locked_now = True
        return EvidenceRecord(
            timestamp_us,
            probability,
            score,
            self.accumulated,
            self.identity,
            locked_now,
        )

