from tade_ef.ground_truth import adjudicate_segment


def test_segment_near_ground_truth_is_labeled_drone() -> None:
    segment = {"start_us": 0, "end_us": 100}
    tracks = [
        {"timestamp_us": 20, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"timestamp_us": 40, "x1": 1, "y1": 0, "x2": 11, "y2": 10},
    ]
    from tade_ef.types import Box
    evidence = adjudicate_segment(segment, tracks, {20: [Box(0, 0, 10, 10)], 40: [Box(1, 0, 11, 10)]})
    assert evidence.decision == "drone"


def test_segment_far_from_ground_truth_is_labeled_non_drone() -> None:
    segment = {"start_us": 0, "end_us": 100}
    tracks = [{"timestamp_us": 20, "x1": 100, "y1": 100, "x2": 110, "y2": 110}]
    from tade_ef.types import Box
    evidence = adjudicate_segment(segment, tracks, {20: [Box(0, 0, 10, 10)]})
    assert evidence.decision == "non_drone"
