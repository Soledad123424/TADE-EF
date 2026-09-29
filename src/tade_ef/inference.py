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
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(oof_path):
        grouped[row["parent_track_id"]].append(row)
    output: list[dict[str, object]] = []
    for parent_id, rows in sorted(grouped.items()):
        accumulator = EvidenceAccumulator(config)
        for row in sorted(rows, key=lambda item: int(item["end_us"])):
            record = accumulator.update(float(row["probability"]), int(row["end_us"]))
            output.append(
                {
                    **row,
                    "log_odds": record.log_odds,
                    "accumulated_evidence": record.accumulated,
                    "identity": record.identity,
                    "locked_now": int(record.locked_now),
                }
            )
    write_csv(output_path, output)
    return output

