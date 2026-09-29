from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tade_ef.io import load_events, parse_raw_header


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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for recording in RECORDINGS:
        raw_path = args.dataset_dir / f"{recording}.raw"
        header = parse_raw_header(raw_path)
        x, y, t_us, polarity = load_events(raw_path)
        output = args.output_dir / f"{recording}.npz"
        np.savez_compressed(
            output,
            x=x,
            y=y,
            t_us=t_us,
            polarity=polarity,
            width=np.asarray(header.width, dtype=np.int32),
            height=np.asarray(header.height, dtype=np.int32),
        )
        print(f"{recording}: {len(t_us)} events -> {output}", flush=True)


if __name__ == "__main__":
    main()
