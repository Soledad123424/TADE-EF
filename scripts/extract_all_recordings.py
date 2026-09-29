from __future__ import annotations

import argparse
from pathlib import Path

from tade_ef.config import load_config
from tade_ef.pipeline import extract_to_csv


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
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--input-format", choices=("raw", "npz"), default="raw")
    args = parser.parse_args()
    config = load_config(args.config)
    for recording in RECORDINGS:
        event_path = args.dataset_dir / f"{recording}.{args.input_format}"
        output_path = args.output_dir / recording / f"{recording}_features.csv"
        count = extract_to_csv(
            event_path,
            output_path,
            recording_id=recording,
            config=config,
        )
        print(f"{recording}: {count} segments", flush=True)


if __name__ == "__main__":
    main()

