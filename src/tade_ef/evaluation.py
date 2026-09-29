"""Track classification and frame detection metrics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from tade_ef.tracking import box_iou
from tade_ef.types import Box


def track_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    predictions = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    beta2 = 4.0
    f2 = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if beta2 * precision + recall > 0
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "APtrack": float(average_precision_score(labels, scores)),
    }


def frame_detection_metrics(
    ground_truth: list[dict[str, object]],
    predictions: list[dict[str, object]],
    *,
    iou_threshold: float,
    confidence_threshold: float,
) -> dict[str, float]:
    by_frame_gt: dict[tuple[str, int], list[Box]] = defaultdict(list)
    by_frame_prediction: dict[tuple[str, int], list[tuple[float, Box]]] = defaultdict(list)
    for row in ground_truth:
        by_frame_gt[(str(row["recording_id"]), int(row["frame_index"]))].append(_box(row))
    for row in predictions:
        by_frame_prediction[(str(row["recording_id"]), int(row["frame_index"]))].append(
            (float(row["score"]), _box(row))
        )
    scores_and_labels: list[tuple[float, int]] = []
    false_negatives = 0
    for frame in sorted(set(by_frame_gt) | set(by_frame_prediction)):
        gt = by_frame_gt[frame]
        used: set[int] = set()
        for score, prediction in sorted(
            by_frame_prediction[frame],
            key=lambda item: item[0],
            reverse=True,
        ):
            best_index, best_iou = -1, 0.0
            for index, truth in enumerate(gt):
                if index in used:
                    continue
                overlap = box_iou(prediction, truth)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            matched = best_index >= 0 and best_iou >= iou_threshold
            if matched:
                used.add(best_index)
            scores_and_labels.append((score, int(matched)))
        false_negatives += len(gt) - len(used)
    selected = [item for item in scores_and_labels if item[0] >= confidence_threshold]
    true_positives = sum(label for _, label in selected)
    false_positives = len(selected) - true_positives
    total_gt = sum(len(items) for items in by_frame_gt.values())
    recall = true_positives / total_gt if total_gt else 0.0
    precision = true_positives / len(selected) if selected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ap = _interpolated_ap(scores_and_labels, total_gt)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "AP": ap,
        "tp": float(true_positives),
        "fp": float(false_positives),
        "fn": float(total_gt - true_positives),
        "conditional_unmatched_gt": float(false_negatives),
    }


def _interpolated_ap(scores_and_labels: list[tuple[float, int]], total_gt: int) -> float:
    if total_gt == 0 or not scores_and_labels:
        return 0.0
    ordered = sorted(scores_and_labels, reverse=True)
    labels = np.asarray([label for _, label in ordered], dtype=float)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.mean([
        np.max(precision[recall >= level]) if np.any(recall >= level) else 0.0
        for level in np.linspace(0, 1, 101)
    ]))


def _box(row: dict[str, object]) -> Box:
    return Box(int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"]))

