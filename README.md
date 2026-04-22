# Rigid Scene Flow

Object-aware 3D scene flow correction using rigid-body constraints. Corrects the "jelly effect" in per-point scene flow by enforcing consistent rigid motion per detected object via median-pooling or SVD-based fitting.

## Overview

Given two consecutive LiDAR frames from the Waymo Open Dataset, the pipeline:

1. **Parses** Waymo tfrecord files, extracting point clouds, ego poses, and GT bounding boxes
2. **Computes GT scene flow** from object velocities and ego-motion compensation
3. **Assigns points to boxes** using CPU-friendly axis-aligned containment checks
4. **Aggregates per-object flow** via median pooling (default) or SVD rigid fitting
5. **Evaluates** using End-Point Error (EPE) with speed, class, and fg/bg breakdowns

## Project Structure

```
src/rigid_flow/
  core/types.py              # Shared dataclasses: BoundingBox, SceneFlowPair, FlowResult
  data/
    waymo_parser.py           # Waymo tfrecord parser (protobuf-only, no Waymo SDK needed)
    waymo_protos/             # Compiled Waymo protobuf definitions
  geometry/
    se3.py                    # SE3 rigid transform (pure NumPy)
    points_in_boxes.py        # CPU point-in-box assignment + ego-motion compensation
  aggregation/
    rigid_aggregation.py      # Median/SVD rigid flow correction
  eval/metrics.py             # EPE evaluation with speed/class breakdowns
  pipeline.py                 # End-to-end orchestrator + CLI
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
pip install tensorflow-macos>=2.16
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

### Run the full pipeline

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

### CLI arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Path to directory containing `.tfrecord` files |
| `--output-dir` | No | None | Directory to save JSON metric files |
| `--method` | No | `median` | Aggregation method: `median` or `svd` |
| `--max-pairs` | No | None | Limit number of frame pairs to process |

### Output

When `--output-dir` is specified, two JSON files are saved:

- `per_frame_metrics.json` — per-pair EPE breakdown
- `aggregate_metrics.json` — averaged metrics across all pairs

### Evaluation metrics

| Metric | Description |
|--------|-------------|
| `epe_mean` | Mean endpoint error across all points |
| `epe_background` | EPE for points not in any bounding box |
| `epe_foreground` | EPE for points inside a bounding box |
| `epe_static` | EPE for objects with speed < 0.5 m/s |
| `epe_slow` | EPE for objects with 0.5 <= speed < 2.0 m/s |
| `epe_fast` | EPE for objects with speed >= 2.0 m/s |
| `epe_vehicle` | EPE for vehicle points |
| `epe_pedestrian` | EPE for pedestrian points |
| `epe_cyclist` | EPE for cyclist points |

## Design

### Module independence

The three core modules (data, geometry, aggregation) have **zero inter-module imports**. They communicate exclusively through the shared dataclasses in `core/types.py`. Only `pipeline.py` imports from all modules.

### Key design decisions

- **Median pooling** as default aggregation — robust to outliers, directly addresses the jelly effect
- **Pure NumPy geometry** — no CUDA, no torch_scatter, no OpenPCDet ops. Point-in-box uses vectorized rotation into box-local frame + axis-aligned bounds check
- **Protobuf-only Waymo parsing** — no dependency on the Waymo SDK (which lacks ARM64 wheels). Range images are converted to point clouds using spherical-to-Cartesian math with beam inclination tables from the frame calibration
- **GT flow convention** — flow is physical displacement in the ego_t0 coordinate frame. Static background points have zero flow; moving objects have flow = ego_compensation + velocity * dt
