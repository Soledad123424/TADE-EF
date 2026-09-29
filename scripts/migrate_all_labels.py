from __future__ import annotations

import argparse
import json
from pathlib import Path

from tade_ef.io import read_csv, write_csv
from tade_ef.label_migration import migrate_labels, normalize_old_segment_rows


RECORDINGS = (
    "35 500",
    "35 650",
    "35 680",
    "35mm 500m 1 20m-s",
    "50 270",
    "50 500",
    "50mm 500m 1 30m-s",
    "50mm 500m 1.1 10m-s",
    "50mm 500m 1.2 10m-s",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-label-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-score", type=float, default=0.65)
    parser.add_argument("--ambiguity-margin", type=float, default=0.10)
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    all_accepted: list[dict[str, object]] = []
    all_review: list[dict[str, object]] = []
    for index, recording in enumerate(RECORDINGS):
        legacy = args.legacy_label_dir / str(index)
        rebuilt = args.feature_dir / recording
        accepted, review = migrate_labels(
            normalize_old_segment_rows(
                read_csv(legacy / f"{recording}_track_labels.csv")
            ),
            read_csv(legacy / f"{recording}_detections.csv"),
            read_csv(rebuilt / f"{recording}_features.csv"),
            read_csv(rebuilt / f"{recording}_tracks.csv"),
            minimum_score=args.minimum_score,
            ambiguity_margin=args.ambiguity_margin,
        )
        all_accepted.extend(accepted)
        all_review.extend(review)
        summary_rows.append(
            {
                "recording_id": recording,
                "accepted": len(accepted),
                "review": len(review),
                "total": len(accepted) + len(review),
            }
        )
        print(f"{recording}: accepted={len(accepted)}, review={len(review)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "migration_by_recording.csv", summary_rows)
    if all_accepted:
        write_csv(args.output_dir / "labels_accepted.csv", all_accepted)
    if all_review:
        write_csv(args.output_dir / "labels_review.csv", all_review)
    summary = {
        "accepted": len(all_accepted),
        "review": len(all_review),
        "total": len(all_accepted) + len(all_review),
        "minimum_score": args.minimum_score,
        "ambiguity_margin": args.ambiguity_margin,
    }
    (args.output_dir / "migration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
