"""Apply OOF probabilities through manuscript Eqs. (25)-(27)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tade_ef.evidence import EvidenceAccumulator, EvidenceConfig
from tade_ef.io import read_csv, write_csv


def apply_evidence_to_oof(
    oof_path: Path,
    output_path: Path,
    config: EvidenceConfig,
    *,
    recording_folds: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(oof_path):
        grouped[row["parent_track_id"]].append(row)
    output: list[dict[str, object]] = []
    for parent_id, rows in sorted(grouped.items()):
        if config.drone_threshold_by_fold:
            folds = {
                row.get("fold") or (recording_folds or {}).get(row.get("recording_id", ""))
                for row in rows
            }
            if None in folds or len(folds) != 1:
                raise ValueError(f"Track {parent_id} must belong to one known test fold")
            track_config = config.for_fold(folds.pop())
        else:
            track_config = config
        accumulator = EvidenceAccumulator(track_config)
        for row in sorted(rows, key=lambda item: int(item["end_us"])):
            record = accumulator.update(float(row["probability"]), int(row["end_us"]))
            output.append(
                {
                    **row,
                    "log_odds": record.log_odds,
                    "accumulated_evidence": record.accumulated,
                    "identity": record.identity,
                    "locked_now": int(record.locked_now),
                    "uav_evidence_threshold": track_config.drone_threshold,
                    "non_uav_evidence_threshold": track_config.non_drone_threshold,
                    "non_uav_lock_ms": track_config.non_drone_lock_ms,
                }
            )
    write_csv(output_path, output)
    return output

