"""Run a fitted TabPFN classifier on an NPZ recording window by window."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from tade_ef.config import load_config
from tade_ef.replay import SequentialReplay, stream_event_windows


def load_fitted_model(path: Path, checkpoint: Path, device: str):
    from tabpfn.model_loading import load_fitted_tabpfn_model

    with tempfile.TemporaryDirectory() as folder:
        temp = Path(folder)
        extracted = temp / "model"
        extracted.mkdir()
        with zipfile.ZipFile(path) as archive:
            archive.extractall(extracted)
        params_path = extracted / "init_params.json"
        params = json.loads(params_path.read_text())
        params["model_path"] = str(checkpoint.resolve())
        params_path.write_text(json.dumps(params))
        patched = shutil.make_archive(str(temp / "patched"), "zip", extracted)
        return load_fitted_tabpfn_model(Path(patched), device=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", type=Path)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paced", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checksum != config.tabpfn["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum differs from the configured model")
    model = load_fitted_model(args.model, args.checkpoint, args.device)
    with np.load(args.event_file) as payload:
        x, y = np.asarray(payload["x"]), np.asarray(payload["y"])
        t = np.asarray(payload["t_us"] if "t_us" in payload else payload["t"])
        p = np.asarray(payload["polarity"] if "polarity" in payload else payload["p"])
        if "width" in payload and int(payload["width"]) != config.sensor_width:
            raise ValueError("Input sensor width differs from configuration")
        if "height" in payload and int(payload["height"]) != config.sensor_height:
            raise ValueError("Input sensor height differs from configuration")
    windows = stream_event_windows(x, y, t, p, width=config.sensor_width,
                                  height=config.sensor_height,
                                  window_us=int(config.window_ms * 1000))
    replay = SequentialReplay(config, model, recording_id=args.recording_id)
    args.output.mkdir(parents=True, exist_ok=False)
    writers, handles = {}, []
    try:
        for name in ("boxes", "updates", "windows"):
            handle = (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8")
            handles.append(handle)
            writers[name] = (handle, None)
        for result in replay.run(windows, paced=args.paced):
            groups = {"boxes": result.boxes, "updates": result.updates,
                      "windows": [{"window_index": result.window_index,
                                   "start_us": result.start_us, "end_us": result.end_us,
                                   "processing_ms": result.processing_ms,
                                   "waiting_ms": result.waiting_ms,
                                   "completion_lag_ms": result.completion_lag_ms}]}
            for name, rows in groups.items():
                handle, writer = writers[name]
                for row in rows:
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writers[name] = (handle, writer)
                    writer.writerow(row)
                handle.flush()
            if result.updates:
                print(json.dumps({"end_us": result.end_us, "updates": result.updates}), flush=True)
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
