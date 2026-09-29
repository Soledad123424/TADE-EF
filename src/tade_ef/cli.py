"""Command-line interface for extraction, training, evidence, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tade_ef.config import load_config
from tade_ef.dataset import load_labeled_dataset
from tade_ef.evaluation import track_metrics
from tade_ef.inference import apply_evidence_to_oof
from tade_ef.io import read_csv
from tade_ef.io import write_csv
from tade_ef.label_migration import migrate_labels, normalize_old_segment_rows
from tade_ef.pipeline import extract_to_csv
from tade_ef.training import (
    load_folds,
    save_tabpfn,
    tabpfn_factory,
    train_oof,
    verify_sha256,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract paper-defined 36-D features")
    extract.add_argument("event_file", type=Path)
    extract.add_argument("--recording-id", required=True)
    extract.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    extract.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="Run recording-level three-fold TabPFN OOF training")
    train.add_argument("--feature-dir", type=Path, required=True)
    train.add_argument("--label-dir", type=Path, required=True)
    train.add_argument("--folds", type=Path, default=Path("configs/folds.yaml"))
    train.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    train.add_argument("--output-dir", type=Path, required=True)

    evidence = subparsers.add_parser("evidence", help="Apply Eqs. (25)-(27) to OOF probabilities")
    evidence.add_argument("oof_predictions", type=Path)
    evidence.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    evidence.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate-track", help="Evaluate track-level scores")
    evaluate.add_argument("predictions", type=Path)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--score-field", default="probability")
    evaluate.add_argument("--identity-field")
    evaluate.add_argument("--output", type=Path, required=True)

    migrate = subparsers.add_parser("migrate-labels", help="Map legacy labels to rebuilt tracks")
    migrate.add_argument("--old-labels", type=Path, required=True)
    migrate.add_argument("--old-tracks", type=Path, required=True)
    migrate.add_argument("--new-features", type=Path, required=True)
    migrate.add_argument("--new-tracks", type=Path, required=True)
    migrate.add_argument("--output-dir", type=Path, required=True)
    migrate.add_argument("--minimum-score", type=float, default=0.65)
    migrate.add_argument("--ambiguity-margin", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "extract":
        count = extract_to_csv(
            args.event_file,
            args.output,
            recording_id=args.recording_id,
            config=load_config(args.config),
        )
        print(f"Extracted {count} valid segments to {args.output}")
        return
    if args.command == "train":
        config = load_config(args.config)
        dataset = load_labeled_dataset(args.feature_dir, args.label_dir)
        folds = load_folds(args.folds, dataset.groups)
        checkpoint = Path(str(config.tabpfn["checkpoint"]))
        verify_sha256(checkpoint, str(config.tabpfn["checkpoint_sha256"]))
        train_oof(
            dataset,
            folds,
            output_dir=args.output_dir,
            model_factory=tabpfn_factory(
                checkpoint=checkpoint,
                device=str(config.tabpfn["device"]),
            ),
            save_model=save_tabpfn,
        )
        print(f"OOF predictions written to {args.output_dir}")
        return
    if args.command == "evidence":
        rows = apply_evidence_to_oof(
            args.oof_predictions,
            args.output,
            load_config(args.config).evidence,
        )
        print(f"Wrote {len(rows)} chronological evidence records")
        return
    if args.command == "evaluate-track":
        rows = read_csv(args.predictions)
        metrics = track_metrics(
            np.asarray([int(row["label"]) for row in rows]),
            np.asarray([float(row[args.score_field]) for row in rows]),
            threshold=args.threshold,
            predictions=(
                None
                if args.identity_field is None
                else np.asarray([row[args.identity_field] == "drone" for row in rows])
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "migrate-labels":
        accepted, review = migrate_labels(
            normalize_old_segment_rows(read_csv(args.old_labels)),
            read_csv(args.old_tracks),
            read_csv(args.new_features),
            read_csv(args.new_tracks),
            minimum_score=args.minimum_score,
            ambiguity_margin=args.ambiguity_margin,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if accepted:
            write_csv(args.output_dir / "labels_accepted.csv", accepted)
        if review:
            write_csv(args.output_dir / "labels_review.csv", review)
        summary = {"accepted": len(accepted), "review": len(review)}
        (args.output_dir / "migration_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

