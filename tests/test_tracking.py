import numpy as np

from tade_ef.tracking import TrackingConfig, association_cost, box_iou
from tade_ef.types import Box


def test_association_cost_combines_iou_and_distance() -> None:
    config = TrackingConfig(lambda_iou=0.5, lambda_distance=0.5, max_center_distance_px=10)
    first = Box(0, 0, 4, 4)
    second = Box(2, 0, 6, 4)
    expected = 0.5 * (1.0 - box_iou(first, second)) + 0.5 * 0.2
    assert association_cost(first, second, config) == expected


def test_association_rejects_excessive_displacement() -> None:
    config = TrackingConfig(max_center_distance_px=5)
    assert np.isinf(association_cost(Box(0, 0, 2, 2), Box(10, 0, 12, 2), config))

