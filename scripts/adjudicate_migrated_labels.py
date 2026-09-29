from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tade_ef.ground_truth import adjudicate_segment, load_cvat_boxes
from tade_ef.io import read_csv, write_csv


SAFE_NAMES = {
    "35 500": "35_500",
    "35 650": "35_650",
    "35 680": "35_680",
    "35mm 500m 1 20m-s": "35mm_500m_1_20m-s",
    "50 270": "50_270",
    "50 500": "50_500",
    "50mm 500m 1 30m-s": "50mm_500m_1_30m-s",
    "50mm 500m 1.1 10m-s": "50mm_500m_1.1_10m-s",
    "50mm 500m 1.2 10m-s": "50mm_500m_1.2_10m-s",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--migration-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--cvat-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-overrides", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("labels_resolved.csv", "labels_manual_review.csv"):
        stale_path = args.output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    accepted = read_csv(args.migration_dir / "labels_accepted.csv")
    review = read_csv(args.migration_dir / "labels_review.csv")
    review_by_recording: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in review:
        review_by_recording[row["recording_id"]].append(row)
    resolved: list[dict[str, object]] = [dict(row) for row in accepted]
    unresolved: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for recording, rows in review_by_recording.items():
        safe_name = SAFE_NAMES[recording]
        gt = load_cvat_boxes(
            args.cvat_dir / "annotations" / f"{safe_name}_annotations.zip",
            args.cvat_dir / safe_name / "manifest.csv",
        )
        tracks = read_csv(
            args.feature_dir / recording / f"{recording}_tracks.csv"
        )
        tracks_by_id: dict[int, list[dict[str, str]]] = defaultdict(list)
        for track_row in tracks:
            tracks_by_id[int(track_row["track_id"])].append(track_row)
        for row in rows:
            evidence = adjudicate_segment(
                row,
                tracks_by_id[int(row["track_id"])],
                gt,
            )
            output = dict(row)
            output.update(
                {
                    "cvat_decision": evidence.decision,
                    "cvat_observation_count": evidence.observation_count,
                    "cvat_matched_count": evidence.matched_count,
                    "cvat_match_fraction": evidence.match_fraction,
                    "cvat_maximum_iou": evidence.maximum_iou,
                    "cvat_minimum_normalized_center_distance": (
                        evidence.minimum_normalized_center_distance
                    ),
                }
            )
            audit.append(output)
            if evidence.decision == "review":
                unresolved.append(output)
            else:
                output["label"] = evidence.decision
                output["label_source"] = "cvat_ground_truth"
                resolved.append(output)

    for row in resolved[: len(accepted)]:
        row["label_source"] = "legacy_manual_unambiguous"
    if args.manual_overrides:
        overrides = {
            row["sample_id"]: row for row in read_csv(args.manual_overrides)
        }
        still_unresolved: list[dict[str, object]] = []
        for row in unresolved:
            override = overrides.get(str(row["sample_id"]))
            if override is None:
                still_unresolved.append(row)
                continue
            output = dict(row)
            output["label"] = override["label"]
            output["label_source"] = "manual_override"
            output["override_reason"] = override.get("reason", "")
            resolved.append(output)
        unresolved = still_unresolved
    write_csv(args.output_dir / "final_labels.csv", resolved)
    write_csv(args.output_dir / "cvat_adjudication_audit.csv", audit)
    if unresolved:
        write_csv(args.output_dir / "labels_manual_review.csv", unresolved)
    summary = {
        "legacy_unambiguous": len(accepted),
        "cvat_resolved": sum(
            row.get("label_source") == "cvat_ground_truth" for row in resolved
        ),
        "manual_overrides": sum(
            row.get("label_source") == "manual_override" for row in resolved
        ),
        "manual_review": len(unresolved),
        "total": len(accepted) + len(review),
    }
    (args.output_dir / "adjudication_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
