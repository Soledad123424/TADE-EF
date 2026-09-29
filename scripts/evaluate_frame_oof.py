from __future__ import annotations

import argparse
import json
from pathlib import Path

from tade_ef.evaluation import frame_detection_metrics
from tade_ef.frame_predictions import build_frame_predictions
from tade_ef.ground_truth import load_cvat_evaluation_rows
from tade_ef.io import read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--cvat-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    oof_rows = read_csv(args.predictions)
    frame_predictions = build_frame_predictions(
        oof_rows,
        args.track_dir,
        args.cvat_dir,
    )
    recordings = sorted({row["recording_id"] for row in oof_rows})
    ground_truth: list[dict[str, object]] = []
    for recording in recordings:
        safe_name = recording.replace(" ", "_")
        ground_truth.extend(
            load_cvat_evaluation_rows(
                args.cvat_dir / "annotations" / f"{safe_name}_annotations.zip",
                args.cvat_dir / safe_name / "manifest.csv",
                recording,
            )
        )
    report = {
        "confidence_threshold": args.confidence_threshold,
        "iou_0.3": frame_detection_metrics(
            ground_truth,
            frame_predictions,
            iou_threshold=0.3,
            confidence_threshold=args.confidence_threshold,
        ),
        "iou_0.5": frame_detection_metrics(
            ground_truth,
            frame_predictions,
            iou_threshold=0.5,
            confidence_threshold=args.confidence_threshold,
        ),
        "ground_truth_boxes": len(ground_truth),
        "prediction_boxes": len(frame_predictions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frame_predictions.csv", frame_predictions)
    (args.output_dir / "frame_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
