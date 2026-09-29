import numpy as np

from tade_ef.candidates import CandidateConfig, circular_kernel, detect_candidates
from tade_ef.types import EventWindow


def test_circular_kernel_is_not_square() -> None:
    kernel = circular_kernel(2)
    assert kernel.shape == (5, 5)
    assert kernel[2, 2] == 1
    assert kernel[0, 0] == 0


def test_connected_activity_produces_candidate() -> None:
    x = np.asarray([4, 5, 4, 5, 4, 5, 4, 5])
    y = np.asarray([4, 4, 5, 5, 6, 6, 7, 7])
    window = EventWindow(0, 20_000, x, y, np.arange(8), np.ones(8), 20, 20)
    config = CandidateConfig(
        dilation_radius_px=1,
        min_component_area_px=4,
        min_width_px=2,
        min_height_px=2,
        min_events=8,
        roi_padding_px=0,
    )
    candidates = detect_candidates(window, config)
    assert len(candidates) == 1
    assert candidates[0].event_count == 8

