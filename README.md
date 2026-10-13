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

Extraction also produces a `*_tracks.csv` file. Prepare `*_labels.csv` files
with `sample_id` and `label` columns; valid labels are `drone` and `non_drone`.

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

Run trajectory-level and frame-level evaluation using CVAT box annotations:

```bash
bash scripts/evaluate_all_folds.sh \
  outputs/tabpfn_oof/oof_predictions.csv \
  outputs/features \
  /path/to/cvat_annotations \
  outputs/evaluation
```

Metrics are saved under the evaluation directory. For CPU training, set
`tabpfn.device` to `cpu` in `configs/paper.yaml`.
