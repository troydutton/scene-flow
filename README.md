# Rigid Scene Flow

Object-aware 3D scene flow correction using rigid-body constraints. Corrects the "jelly effect" in per-point scene flow by enforcing consistent rigid motion per detected object via median pooling or SVD-based fitting.

This repository also contains the historical **ZeroFlow** stack under `zeroflow/`; see `CLAUDE.md` for training, Docker, and legacy workflows.

## Overview

Given two consecutive LiDAR frames from the Waymo Open Dataset, the pipeline:

1. **Parses** Waymo tfrecord files, extracting point clouds, ego poses, and GT bounding boxes
2. **Computes GT scene flow** from object velocities and ego-motion compensation
3. **Assigns points to boxes** using CPU-friendly axis-aligned containment checks (GT boxes by default, or predicted boxes from a Waymo detection `.bin`)
4. **Aggregates per-object flow** via median pooling (default) or SVD rigid fitting
5. **Evaluates** using endpoint error (EPE) with speed, class, predicted vs true foreground, and rigidity breakdowns

## Project structure

```
src/rigid_flow/
  core/types.py              # Shared dataclasses: BoundingBox, SceneFlowPair, FlowResult
  data/
    waymo_parser.py           # Waymo tfrecord parser (protobuf-only, no Waymo SDK needed)
    waymo_protos/             # Compiled Waymo protobuf definitions
    pred_boxes.py             # Index for predicted detection Objects (.bin)
  geometry/
    se3.py                    # SE3 rigid transform (pure NumPy)
    points_in_boxes.py        # CPU point-in-box assignment + ego-motion compensation
  aggregation/
    rigid_aggregation.py      # Median/SVD rigid flow correction
  eval/metrics.py             # EPE evaluation with speed/class breakdowns
  pipeline.py                 # End-to-end orchestrator + CLI
  visualization/              # BEV, flow, and correction plots
```

## Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4) — tested on M4 Mac Pro
- [Miniforge](https://github.com/conda-forge/miniforge) or Mambaforge installed

### Environment

```bash
# Create the conda environment
mamba env create -f mac_environment.yaml

# Activate
mamba activate flow

# Install tensorflow-macos (ARM64 compatible) and grpcio-tools for proto compilation
pip install "tensorflow-macos>=2.16"
pip install grpcio-tools==1.62.3

# Install the package in editable mode
pip install --editable .
```

> **Note:** The `waymo-open-dataset` pip package has no ARM64 wheels. Instead, we use
> pre-compiled protobuf definitions bundled in `src/rigid_flow/data/waymo_protos/`.
> If you need to recompile them (e.g., for a newer Waymo dataset version):
>
> ```bash
> git clone --depth 1 https://github.com/waymo-research/waymo-open-dataset.git /tmp/waymo_protos
> python -m grpc_tools.protoc \
>   --python_out=src/rigid_flow/data/waymo_protos \
>   -I /tmp/waymo_protos/src \
>   /tmp/waymo_protos/src/waymo_open_dataset/dataset.proto \
>   /tmp/waymo_protos/src/waymo_open_dataset/label.proto \
>   /tmp/waymo_protos/src/waymo_open_dataset/protos/*.proto
> ```

### Data

1. Download `validation_0000.tar` from the [Waymo Open Dataset](https://waymo.com/open/) (requires registration)
2. Place it at `../data/validation_0000.tar` (one level above the repo root)
3. Extract:

```bash
mkdir -p ../data/validation_0000
cd ../data/validation_0000
tar xf ../validation_0000.tar
cd -
```

The archive contains ~29 `.tfrecord` files, each representing a driving sequence of ~200 frames at 10 Hz.

## Usage

### Run the full pipeline (Phase A — GT boxes)

Ground-truth flow always uses labels from the tfrecord. Omit `--pred-boxes-bin` to use **GT boxes** for assignment, pooling, and metrics.

```bash
# Process all frame pairs from extracted tfrecords
python -m rigid_flow.pipeline \
  --data-root ../data/validation_0000 \
  --output-dir results/ \
  --method median

# Limit to first 10 pairs for quick testing
python -m rigid_flow.pipeline \
  --data-root ../data/validation_0000 \
  --output-dir results/ \
  --method median \
  --max-pairs 10
```

### Phase B — predicted detection boxes

GT flow is still computed from tfrecord labels. Pass a Waymo **detection submission** `.bin` (serialized `Objects`) so **predicted** boxes drive point assignment and rigid pooling; optional `--score-threshold` filters low-confidence boxes.

```bash
python -m rigid_flow.pipeline \
  --data-root ../data/validation_0000 \
  --output-dir results/phase_b_gt_flow_pred_boxes/thr_0.50 \
  --method median \
  --pred-boxes-bin /path/to/detection_pred.bin \
  --score-threshold 0.5
```

### Score-threshold sweep

`scripts/run_pred_box_sweep.py` runs several confidence cutoffs, writes `thr_X.XX/` subdirectories, and compares each aggregate to a Phase A `aggregate_metrics.json`.

```bash
python scripts/run_pred_box_sweep.py \
  --data-root ../data/validation_0000 \
  --pred-boxes-bin results/2_stage/detection_pred.bin \
  --output-root results/phase_b_gt_flow_pred_boxes \
  --gt-aggregate results/phase_a_gt_flow_gt_boxes/aggregate_metrics.json \
  --thresholds 0.1 0.3 0.5
```

### CLI arguments (`python -m rigid_flow.pipeline`)

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Path to directory containing `.tfrecord` files |
| `--output-dir` | No | None | Directory to save JSON metric files |
| `--method` | No | `median` | Aggregation method: `median` or `svd` |
| `--max-pairs` | No | None | Limit number of frame pairs to process |
| `--pred-boxes-bin` | No | None | Waymo detection `.bin`; when set, predicted boxes replace GT for pooling and predicted-foreground metrics |
| `--score-threshold` | No | `0.0` | Minimum detection score (only with `--pred-boxes-bin`) |

### Output

When `--output-dir` is specified, two JSON files are saved:

- `per_frame_metrics.json` — per-pair metric breakdown
- `aggregate_metrics.json` — means over frames (`avg_*` keys)

### Evaluation metrics

Full key definitions live in the module docstring in `src/rigid_flow/eval/metrics.py`. In short:

| Metric | Description |
|--------|-------------|
| `epe_mean` | Mean endpoint error across all points |
| `epe_background` / `epe_foreground` | EPE for points outside / inside **evaluation** boxes (GT or predicted) |
| `epe_true_foreground` | EPE on every point inside a **GT** box (detection-agnostic; important for Phase B) |
| `epe_static` / `epe_slow` / `epe_fast` | Predicted-foreground points bucketed by **‖gt_flow‖/dt** (not detector metadata speed) |
| `epe_all_*` | All points bucketed by GT flow speed |
| `acc_*`, `out3d*` | Accuracy and outlier rates (global, foreground, background) |
| `flow_variance_mean`, `dist_preservation_mean` | Rigidity-style structure within evaluation boxes |

## Visualization

Generate visualizations for any frame pair to inspect data quality and pipeline behavior:

```bash
python -m rigid_flow.visualization.visualize \
  --data-root ../data/validation_0000 \
  --output-dir figures/ \
  --frame-index 0 \
  --method median
```

Add `--show` to display plots interactively instead of just saving.

This produces 5 PNG files:

| File | Description |
|------|-------------|
| `bev_boxes.png` | Bird's-eye view of point cloud colored by height, with oriented bounding boxes overlaid by class |
| `flow_magnitude.png` | BEV heatmap of per-point flow magnitude (L2 norm) |
| `flow_quiver.png` | Subsampled flow arrows colored by foreground/background, with box outlines |
| `correction_comparison.png` | 3-panel comparison: raw flow, corrected flow, and correction residual magnitude |
| `per_object_histograms.png` | Per-object histograms of flow magnitude before (blue) and after (red) rigid correction |

### Visualization CLI arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Path to directory containing `.tfrecord` files |
| `--output-dir` | Yes | — | Directory to save PNG figures |
| `--frame-index` | No | `0` | Which frame pair to visualize (0-indexed) |
| `--method` | No | `median` | Aggregation method: `median` or `svd` |
| `--show` | No | off | Display plots interactively |

## Design

### Module independence

The three core modules (data, geometry, aggregation) have **zero inter-module imports**. They communicate exclusively through the shared dataclasses in `core/types.py`. Only `pipeline.py` imports from all modules.

### Key design decisions

- **Median pooling** as default aggregation — robust to outliers, directly addresses the jelly effect
- **Pure NumPy geometry** — no CUDA, no torch_scatter, no OpenPCDet ops. Point-in-box uses vectorized rotation into box-local frame + axis-aligned bounds check
- **Protobuf-only Waymo parsing** — no dependency on the Waymo SDK (which lacks ARM64 wheels). Range images are converted to point clouds using spherical-to-Cartesian math with beam inclination tables from the frame calibration
- **GT flow convention** — flow is physical displacement in the ego_t0 coordinate frame. Static background points have zero flow; moving objects have flow = ego compensation + velocity × dt
