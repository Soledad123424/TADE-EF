from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tade_ef.evaluation import track_metrics
from tade_ef.io import read_csv, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import yaml

    rows = read_csv(args.predictions)
    labels = np.asarray([int(row["label"]) for row in rows])
    scores = np.asarray([float(row["probability"]) for row in rows])
    fixed = track_metrics(labels, scores, threshold=args.fixed_threshold)
    thresholds = np.unique(np.concatenate(([0.0], scores, [1.0])))
    sweep = [
        {"threshold": float(value), **track_metrics(labels, scores, threshold=float(value))}
        for value in thresholds
    ]
    best = max(sweep, key=lambda item: (item["f1"], item["precision"], item["threshold"]))

    fold_payload = yaml.safe_load(args.folds.read_text(encoding="utf-8"))
    fold_rows: list[dict[str, object]] = []
    for index, fold in enumerate(fold_payload["folds"], start=1):
        mask = np.asarray([row["recording_id"] in set(fold["test"]) for row in rows])
        fold_rows.append(
            {
                "fold": index,
                "test_recordings": " | ".join(fold["test"]),
                "samples": int(mask.sum()),
                **track_metrics(labels[mask], scores[mask], threshold=args.fixed_threshold),
            }
        )

    evidence_rows = read_csv(args.evidence)
    identities: dict[str, int] = {}
    for row in evidence_rows:
        identity = row["identity"]
        identities[identity] = identities.get(identity, 0) + 1
    report = {
        "sample_count": len(rows),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "fixed_threshold": args.fixed_threshold,
        "fixed_threshold_metrics": fixed,
        "diagnostic_oof_best_f1": best,
        "diagnostic_warning": (
            "The best-F1 threshold is selected on OOF test predictions and must not "
            "be reported as a validation-selected operating point."
        ),
        "evidence_identity_counts": identities,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "fold_metrics_fixed_threshold.csv", fold_rows)
    write_csv(args.output_dir / "threshold_sweep.csv", sweep)
    (args.output_dir / "oof_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
