import numpy as np

from tade_ef.spectral import extract_spectral_features


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

