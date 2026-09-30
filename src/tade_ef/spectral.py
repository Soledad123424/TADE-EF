"""Short-term spatially resolved spectral and spatial-phase features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralResult:
    peak_frequency_hz: float
    spectral_flatness: float
    harmonic_score: float
    phase_concentration: float


def ndft(
    times_s: np.ndarray,
    polarity: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    if times_s.size == 0:
        return np.zeros_like(frequencies_hz, dtype=np.complex128)
    phases = -2j * np.pi * frequencies_hz[:, None] * times_s[None, :]
    return (np.exp(phases) @ polarity.astype(float)) / times_s.size


def spectral_flatness(power: np.ndarray, epsilon: float = 1.0e-12) -> float:
    safe = power.astype(float) + epsilon
    arithmetic = float(np.mean(safe))
    return float(np.exp(np.mean(np.log(safe))) / arithmetic) if arithmetic > 0 else 1.0


def spatial_power_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    times_s: np.ndarray,
    polarity: np.ndarray,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    """Sum quadrant powers, normalizing each response by all ROI events."""
    power = np.zeros_like(frequencies_hz, dtype=float)
    if times_s.size == 0:
        return power
    center_x, center_y = float(np.median(x)), float(np.median(y))
    quadrant = (x >= center_x).astype(np.int8) + 2 * (y >= center_y).astype(np.int8)
    total_events = float(times_s.size)
    for quadrant_id in range(4):
        selected = quadrant == quadrant_id
        event_count = int(np.count_nonzero(selected))
        if event_count == 0:
            continue
        response = ndft(times_s[selected], polarity[selected], frequencies_hz)
        response *= event_count / total_events
        power += np.abs(response) ** 2
    return power


def harmonic_alignment_score(
    power: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    harmonics: int,
    half_width_bins: int,
    epsilon: float,
) -> float:
    baseline = float(np.median(power) + epsilon)
    best = 0.0
    for fundamental in frequencies_hz:
        peaks: list[float] = []
        for harmonic in range(1, harmonics + 1):
            target = float(fundamental * harmonic)
            if target > float(frequencies_hz[-1]):
                break
            index = int(np.argmin(np.abs(frequencies_hz - target)))
            start = max(0, index - half_width_bins)
            stop = min(power.size, index + half_width_bins + 1)
            peaks.append(float(np.max(power[start:stop])))
        if peaks:
            best = max(best, float(np.mean(peaks) / baseline))
    return float(np.log1p(best) * (1.0 - spectral_flatness(power, epsilon)))


def phase_concentration(
    x: np.ndarray,
    y: np.ndarray,
    times_s: np.ndarray,
    polarity: np.ndarray,
    *,
    frequency_hz: float,
    min_quadrant_events: int,
    epsilon: float,
) -> float:
    center_x, center_y = float(np.median(x)), float(np.median(y))
    quadrant = (x >= center_x).astype(np.int8) + 2 * (y >= center_y).astype(np.int8)
    phases: list[complex] = []
    for quadrant_id in range(4):
        selected = quadrant == quadrant_id
        if np.count_nonzero(selected) < min_quadrant_events:
            continue
        response = np.sum(
            polarity[selected]
            * np.exp(-2j * np.pi * frequency_hz * times_s[selected])
        )
        if abs(response) > epsilon:
            phases.append(response / abs(response))
    return float(abs(sum(phases) / len(phases))) if phases else 0.0


def extract_spectral_features(
    x: np.ndarray,
    y: np.ndarray,
    t_us: np.ndarray,
    polarity: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    harmonics: int = 4,
    half_width_bins: int = 1,
    min_quadrant_events: int = 6,
    epsilon: float = 1.0e-12,
) -> SpectralResult:
    if t_us.size == 0:
        return SpectralResult(0.0, 1.0, 0.0, 0.0)
    times_s = (t_us.astype(float) - float(np.min(t_us))) / 1_000_000.0
    weights = np.where(polarity >= 0, 1.0, -1.0)
    power = spatial_power_spectrum(x, y, times_s, weights, frequencies_hz)
    peak_index = int(np.argmax(power))
    peak = float(frequencies_hz[peak_index])
    flatness = spectral_flatness(power, epsilon)
    harmonic = harmonic_alignment_score(
        power,
        frequencies_hz,
        harmonics=harmonics,
        half_width_bins=half_width_bins,
        epsilon=epsilon,
    )
    phase = phase_concentration(
        x,
        y,
        times_s,
        weights,
        frequency_hz=peak,
        min_quadrant_events=min_quadrant_events,
        epsilon=epsilon,
    )
    return SpectralResult(peak, flatness, harmonic, phase)

