import numpy as np

from tade_ef.evaluation import frame_detection_metrics, track_metrics


def test_track_metrics_perfect_predictions() -> None:
    result = track_metrics(np.asarray([0, 1]), np.asarray([0.1, 0.9]), threshold=0.5)
    assert result["f1"] == 1.0
    assert result["APtrack"] == 1.0


def test_frame_metrics_include_background_false_positive() -> None:
    ground_truth = [{"recording_id": "r", "frame_index": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    predictions = [
        {"recording_id": "r", "frame_index": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.9},
        {"recording_id": "r", "frame_index": 1, "x1": 0, "y1": 0, "x2": 5, "y2": 5, "score": 0.8},
    ]
    result = frame_detection_metrics(ground_truth, predictions, iou_threshold=0.5, confidence_threshold=0.5)
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_frame_metrics_accept_tied_prediction_scores() -> None:
    ground_truth = [{"recording_id": "r", "frame_index": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]
    predictions = [
        {"recording_id": "r", "frame_index": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.5},
        {"recording_id": "r", "frame_index": 0, "x1": 20, "y1": 20, "x2": 30, "y2": 30, "score": 0.5},
    ]
    result = frame_detection_metrics(
        ground_truth,
        predictions,
        iou_threshold=0.5,
        confidence_threshold=0.5,
    )
    assert result["tp"] == 1.0
    assert result["fp"] == 1.0

