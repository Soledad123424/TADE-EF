# TADE-EF

## Installation

```bash
git clone https://github.com/Soledad123424/TADE-EF.git
cd TADE-EF
conda env create -f environment.yml
conda activate event_cam_tade_ef
pytest -q
```

Place the TabPFN checkpoint at:

```text
models/tabpfn-v2-classifier-v2_default.ckpt
```

The expected SHA-256 is:

```text
cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49
```

## Input Data

NPZ event files must contain the arrays `x`, `y`, `t_us`, and `polarity`.
DVSense EVT3 RAW files require the vendor-provided `dvsense_driver` package.

Convert the nine RAW recordings to portable NPZ files on a machine with the
DVSense driver installed:

```bash
python scripts/convert_all_raw_to_npz.py \
  --dataset-dir /path/to/raw-recordings \
  --output-dir outputs/npz_events
```

## Feature Extraction

Extract one recording:

```bash
tade-ef extract "/path/to/recording.npz" \
  --recording-id "35 500" \
  --config configs/paper.yaml \
  --output "outputs/features/35 500/35 500_features.csv"
```

Extract all nine recordings:

```bash
python scripts/extract_all_recordings.py \
  --dataset-dir outputs/npz_events \
  --output-dir outputs/features \
  --config configs/paper.yaml \
  --input-format npz
```

Each extraction also creates a `*_tracks.csv` file.

## Label Migration

Migrate legacy trajectory labels to the rebuilt tracks:

```bash
python scripts/migrate_all_labels.py \
  --legacy-label-dir /path/to/legacy-labels \
  --feature-dir outputs/features \
  --output-dir outputs/migrated_labels
```

Review `labels_review.csv` before training. If CVAT annotations are available,
the ambiguous labels can be adjudicated with:

```bash
python scripts/adjudicate_migrated_labels.py \
  --migration-dir outputs/migrated_labels \
  --feature-dir outputs/features \
  --cvat-dir /path/to/cvat_drone_gt_20ms \
  --output-dir outputs/labels_final
```

The training label directory must contain a file ending in `*_labels.csv` with
the columns `sample_id` and `label`. Valid labels are `drone` and `non_drone`.

## Three-Fold Training

Frequency features use a 2 x 2 spatial power sum. After changing from the
pooled ROI spectrum, regenerate the feature CSVs and rerun three-fold fitting
before inference. Previously fitted models contain the pooled-spectrum training
features and must not be mixed with spatial-spectrum test features.

```bash
bash scripts/train_all_folds.sh \
  outputs/features \
  outputs/labels_final \
  outputs/tabpfn_oof
```

The OOF probabilities are written to:

```text
outputs/tabpfn_oof/oof_predictions.csv
```

## Evaluation

Run trajectory-level and frame-level evaluation:

```bash
bash scripts/evaluate_all_folds.sh \
  outputs/tabpfn_oof/oof_predictions.csv \
  outputs/features \
  /path/to/cvat_drone_gt_20ms \
  outputs/evaluation
```

Run the complete NPZ workflow:

```bash
bash scripts/run_full_pipeline.sh \
  outputs/npz_events \
  outputs/labels_final \
  /path/to/cvat_drone_gt_20ms \
  npz
```

To use CPU inference, change `tabpfn.device` from `cuda` to `cpu` in
`configs/paper.yaml`.
