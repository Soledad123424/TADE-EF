# TADE-EF

## Installation

```bash
git clone https://github.com/Soledad123424/TADE-EF.git
cd TADE-EF
conda env create -f environment.yml
conda activate event_cam_tade_ef
```

Place the TabPFN checkpoint at the path specified in `configs/paper.yaml`.

## Data and Configuration

NPZ recordings must contain `x`, `y`, `t_us`, and `polarity` arrays. RAW input
requires a compatible decoder; DVSense EVT3 files use `dvsense_driver`.

Set sensor dimensions and processing parameters in `configs/paper.yaml`.
Configure your recording IDs and test splits in `configs/folds.yaml`.
Use consistent recording IDs in feature files, labels, and split definitions.

## Feature Extraction

Run the following command for each recording:

```bash
tade-ef extract /path/to/recording.npz \
  --recording-id recording_001 \
  --config configs/paper.yaml \
  --output outputs/features/recording_001/recording_001_features.csv
```

Extraction also produces a `*_tracks.csv` file.

## Track-Segment Labeling

1. Open `*_features.csv` and identify each segment by `sample_id`, `track_id`,
   `start_us`, and `end_us` (timestamps are in microseconds).
2. Inspect the corresponding event interval and boxes in `*_tracks.csv`.
   Assign `drone` to UAV segments and `non_drone` to non-UAV segments.
   Leave uncertain segments unlabeled; do not label from model predictions alone.
3. Save a `*_labels.csv` file under your label directory. Copy each `sample_id`
   exactly from the feature CSV, for example:

```csv
sample_id,label
recording_001:1:000,drone
recording_001:2:000,non_drone
```

Only feature samples with a matching valid label are used for training.

## Training

After configuring the three folds, train the models:

```bash
bash scripts/train_all_folds.sh \
  outputs/features \
  /path/to/labels \
  outputs/tabpfn_oof
```

Models and `oof_predictions.csv` are saved under the output directory.
Use features extracted with the same settings for training and inference.

## Sequential Replay

```bash
tade-ef-replay /path/to/recording.npz \
  --recording-id recording_001 \
  --model outputs/tabpfn_oof/fold_1/model.tabpfn_fit \
  --checkpoint models/tabpfn-v2-classifier-v2_default.ckpt \
  --config configs/paper.yaml \
  --fold fold_1 \
  --output outputs/replay/recording_001 \
  --paced
```

Choose the model and fold corresponding to the recording's test split.
Omit `--paced` for unpaced replay; add `--device cpu` for CPU inference.
Use a new output directory. Results are saved as `boxes.csv`, `updates.csv`,
and `windows.csv`; UAV predictions have `identity == drone`.

## Evaluation

### Track-Level Evaluation

Evaluate labeled track segments without frame annotations:

```bash
bash scripts/evaluate_all_folds.sh \
  outputs/tabpfn_oof/oof_predictions.csv \
  outputs/features \
  "" \
  outputs/evaluation
```

`track_metrics.json` contains precision, recall, F1, F2, and APtrack.
P/R/F1/F2 use the evidence-based UAV identity; APtrack uses accumulated evidence.
`fold_metrics_fixed_threshold.csv` contains per-fold results, and
`oof_evidence.csv` retains the sample-level evidence and identities.

### Frame-Level Evaluation

Add CVAT UAV box annotations to run both track-level and frame-level evaluation:

```bash
bash scripts/evaluate_all_folds.sh \
  outputs/tabpfn_oof/oof_predictions.csv \
  outputs/features \
  /path/to/cvat_annotations \
  outputs/evaluation
```

For each recording, the annotation directory must contain
`annotations/recording_001_annotations.zip` (CVAT XML with label `drone`) and
`recording_001/manifest.csv`, mapping frame indices to event-window timestamps.
The manifest must include `frame_index` and `window_end_timestamp_us`.
For recording IDs containing spaces, use underscores in these paths.

`frame_metrics.json` contains detection metrics at IoU 0.3 and 0.5;
`frame_predictions.csv` retains the evaluated boxes and scores.

For CPU training, set `tabpfn.device` to `cpu` in `configs/paper.yaml`.
