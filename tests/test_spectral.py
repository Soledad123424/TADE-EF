import numpy as np

from tade_ef.spectral import (
    extract_spectral_features, harmonic_alignment_score, ndft,
    spatial_power_spectrum, spectral_flatness,
)


def test_ndft_peak_recovers_periodic_event_frequency() -> None:
    frequency = 100.0
    times_s = np.arange(0.0, 0.1, 1.0 / 2000.0)
    polarity = np.sign(np.sin(2.0 * np.pi * frequency * times_s))
    polarity[polarity == 0] = 1
    t_us = (times_s * 1_000_000).astype(np.int64)
    x = np.tile(np.asarray([0.0, 10.0]), t_us.size // 2)
    y = np.tile(np.asarray([0.0, 10.0]), t_us.size // 2)
    result = extract_spectral_features(
        x,
        y,
        t_us,
        polarity,
        np.arange(20.0, 301.0, 20.0),
        min_quadrant_events=4,
    )
    assert result.peak_frequency_hz == 100.0
    assert result.spectral_flatness < 0.5


def test_spatial_power_preserves_local_opposed_periodic_signals() -> None:
    times = np.tile(np.arange(20, dtype=float) * 0.005, 2)
    local_polarity = np.where(np.arange(20) % 2 == 0, 1.0, -1.0)
    polarity = np.concatenate((local_polarity, -local_polarity))
    x = np.repeat([-1.0, 1.0], 20)
    y = np.zeros_like(x)
    frequencies = np.asarray([100.0])
    pooled = np.abs(ndft(times, polarity, frequencies)) ** 2
    spatial = spatial_power_spectrum(x, y, times, polarity, frequencies)
    np.testing.assert_allclose(pooled, [0.0], atol=1.0e-12)
    np.testing.assert_allclose(spatial, [0.5], atol=1.0e-12)
    np.testing.assert_allclose(
        spatial_power_spectrum(x + 100.0, y - 40.0, times, polarity, frequencies),
        spatial,
    )


def test_spatial_power_empty_and_single_quadrant() -> None:
    frequencies = np.asarray([20.0, 100.0])
    empty = np.empty(0, dtype=float)
    np.testing.assert_array_equal(
        spatial_power_spectrum(empty, empty, empty, empty, frequencies),
        np.zeros(2),
    )
    times = np.arange(20, dtype=float) * 0.005
    polarity = np.where(np.arange(20) % 2 == 0, 1.0, -1.0)
    positions = np.zeros_like(times)
    np.testing.assert_allclose(
        spatial_power_spectrum(positions, positions, times, polarity, frequencies),
        np.abs(ndft(times, polarity, frequencies)) ** 2,
    )


def test_vectorized_harmonics_match_original_loops() -> None:
    rng = np.random.default_rng(23)
    frequencies = np.arange(20.0, 1201.0, 20.0)
    for harmonics in (1, 2, 3, 4, 5):
        for half_width in (0, 1, 3):
            power = rng.random(frequencies.size)
            baseline = float(np.median(power) + 1e-12)
            best = 0.0
            for fundamental in frequencies:
                peaks = []
                for harmonic in range(1, harmonics + 1):
                    target = float(fundamental * harmonic)
                    if target > frequencies[-1]:
                        break
                    index = int(np.argmin(np.abs(frequencies - target)))
                    peaks.append(float(np.max(power[
                        max(0, index - half_width):min(power.size, index + half_width + 1)
                    ])))
                best = max(best, float(np.mean(peaks) / baseline))
            expected = float(np.log1p(best) * (1 - spectral_flatness(power, 1e-12)))
            actual = harmonic_alignment_score(
                power, frequencies, harmonics=harmonics,
                half_width_bins=half_width, epsilon=1e-12,
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

