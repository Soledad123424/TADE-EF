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

Feature extraction uses up to four CPU worker processes for independent tracks.
The bounded caches and vectorized harmonic search preserve the existing feature
definitions. Spatial-spectrum feature CSVs and fitted models remain compatible;
this engineering optimization does not require refitting.

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

### Sequential Replay

Replay an NPZ recording window by window using a fitted spatial-spectrum model:

```bash
tade-ef-replay /path/to/recording.npz \
  --recording-id "35 500" \
  --model outputs/tabpfn_oof/fold_1/model.tabpfn_fit \
  --checkpoint models/tabpfn-v2-classifier-v2_default.ckpt \
  --config configs/paper.yaml \
  --output outputs/replay/35_500 \
  --paced
```

Use the held-out fold's model for cross-validation. Omit `--paced` to process
windows sequentially as quickly as possible; use `--device cpu` for CPU inference.
The output directory must not already exist. Server-cache NPZ arrays `t` and `p`
are also accepted in place of `t_us` and `polarity`.

Outputs are flushed after each window: `boxes.csv` retains candidate boxes,
`updates.csv` retains probabilities and accumulated evidence, and `windows.csv`
retains processing time and paced waiting/completion delays. Box `score` is
accumulated evidence. Filter `identity == drone` for locked UAV detections;
`has_evidence == False` denotes a candidate awaiting its first feature update.
No later label is backfilled into earlier frames.

The first eligible feature update occurs at/after 300 ms of observed track history,
followed by 100-ms updates with at most 1000 ms of retained history. Motion
eligibility is checked using only past observations. These are feature-update
times, not guaranteed firm-label or wall-clock detection times. Existing evidence
thresholds and permanent identity locking are unchanged.

Paced replay releases complete 20-ms windows at their event-time deadlines without
dropping windows when processing falls behind. This is recorded-event replay, not
a live-camera driver or a guarantee of real-time throughput. Decoded NPZ input
remains resident in memory; active track histories/caches are pruned. Small-batch
TabPFN probabilities can differ slightly from recording-level batch predictions.

### Batch Evaluation

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
