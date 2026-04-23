# Rigid Scene Flow

Object-aware 3D scene flow on Waymo: we build **ground-truth scene flow** from labels, optionally **enforce rigid motion per 3D box** (median or SVD), and report **endpoint error and related metrics**. This corrects inconsistent per-point flow (“jelly”) inside objects when you pool to a single rigid motion per box.

The repo also ships the historical **ZeroFlow** stack under `zeroflow/` (training, Docker, checkpoints). See [`CLAUDE.md`](CLAUDE.md) for that workflow.

---

## Contents

1. [What runs end-to-end](#what-runs-end-to-end)
2. [Layout](#layout)
3. [Setup](#setup)
4. [Data](#data)
5. [How to run](#how-to-run)
6. [Outputs](#outputs)
7. [Metrics reference](#metrics-reference)
8. [Visualization](#visualization)
9. [Design notes](#design-notes)

---

## What runs end-to-end

For each consecutive LiDAR pair `(t, t+1)`:

1. **Load** points, ego poses, and **GT boxes** from Waymo `.tfrecord` files.
2. **Compute GT flow** at every point: object motion from box velocities + ego frame convention (background ≈ zero object motion).
3. **Choose evaluation boxes**: either the same **GT boxes** (Phase A) or **predicted boxes** from a Waymo detection `.bin` (Phase B).
4. **Assign** each point to at most one box (axis-aligned check in box frame) or background (`-1`).
5. **Rigid step**: inside each box with enough points, replace per-point flow with the **median** vector (default) or an **SVD** rigid fit; background keeps raw flow (here: still GT flow).
6. **Score**: compare the **refined** flow to **GT flow** and write JSON metrics.

So the “prediction” being scored is the **post-processed** flow (piecewise-constant inside evaluation boxes), not a neural scene flow model—unless you later plug one in upstream.

---

## Layout

```
src/rigid_flow/
  core/types.py              # BoundingBox, SceneFlowPair, FlowResult
  data/
    waymo_parser.py          # tfrecord → frame pairs (protobuf-only)
    waymo_protos/            # Vendored Waymo protos
    pred_boxes.py            # Detection .bin → PredBoxIndex
  geometry/                  # SE3, point-in-box, ego helpers
  aggregation/
    rigid_aggregation.py     # Median / SVD pooling per box
  eval/metrics.py            # EPE, accuracy, rigidity, breakdowns
  pipeline.py                # CLI + run_pipeline()
  visualization/             # BEV, quiver, correction figures
scripts/
  run_pred_box_sweep.py      # Phase B over several score thresholds
```

---

## Setup

### Prerequisites

- **macOS Apple Silicon** path below is the one documented in-repo (`mac_environment.yaml`). For Linux + CUDA, adapt from `environment.yaml` and install PyTorch for your stack.
- [Miniforge](https://github.com/conda-forge/miniforge) or Mambaforge recommended.

### Environment (macOS / Apple Silicon)

```bash
mamba env create -f mac_environment.yaml
mamba activate flow

pip install "tensorflow-macos>=2.16"
pip install grpcio-tools==1.62.3
pip install --editable .
```

**Waymo SDK on ARM64:** the `waymo-open-dataset` pip wheel is not used for tfrecord parsing. We bundle generated protos under `src/rigid_flow/data/waymo_protos/`. To regenerate from upstream protos:

```bash
git clone --depth 1 https://github.com/waymo-research/waymo-open-dataset.git /tmp/waymo_protos
python -m grpc_tools.protoc \
  --python_out=src/rigid_flow/data/waymo_protos \
  -I /tmp/waymo_protos/src \
  /tmp/waymo_protos/src/waymo_open_dataset/dataset.proto \
  /tmp/waymo_protos/src/waymo_open_dataset/label.proto \
  /tmp/waymo_protos/src/waymo_open_dataset/protos/*.proto
```

---

## Data

1. Register and download a Waymo Open Dataset shard (e.g. `validation_0000.tar`).
2. Extract so you have a directory tree containing `*.tfrecord` files (nested dirs are fine).
3. Pass that directory as `--data-root` to the pipeline or sweep.

Example:

```bash
mkdir -p /path/to/waymo/validation_0000
tar xf validation_0000.tar -C /path/to/waymo/validation_0000
```

---

## How to run

### Phase A — GT flow, GT boxes

GT flow always comes from tfrecord labels. **Evaluation boxes = GT boxes.** Good baseline: refined flow should nearly match per-point GT inside each GT object.

```bash
python -m rigid_flow.pipeline \
  --data-root /path/to/waymo/validation_0000 \
  --output-dir results/phase_a_gt_flow_gt_boxes \
  --method median
```

Quick test on a few pairs:

```bash
python -m rigid_flow.pipeline \
  --data-root /path/to/waymo/validation_0000 \
  --output-dir results/debug \
  --method median \
  --max-pairs 10
```

### Phase B — GT flow, predicted boxes

GT flow is still label-derived. **Evaluation boxes** come from a Waymo **detection submission** `.bin` (`Objects`). Use `--score-threshold` to drop low-confidence boxes.

```bash
python -m rigid_flow.pipeline \
  --data-root /path/to/waymo/validation_0000 \
  --output-dir results/phase_b_gt_flow_pred_boxes/thr_0.50 \
  --method median \
  --pred-boxes-bin /path/to/detection_pred.bin \
  --score-threshold 0.5
```

### Sweep several score thresholds

Runs Phase B once per threshold, writes `thr_X.XX/` under `--output-root`, and emits `compare_vs_gt.json` with per-key deltas vs your Phase A aggregate.

```bash
python scripts/run_pred_box_sweep.py \
  --data-root /path/to/waymo/validation_0000 \
  --pred-boxes-bin results/2_stage/detection_pred.bin \
  --output-root results/phase_b_gt_flow_pred_boxes \
  --gt-aggregate results/phase_a_gt_flow_gt_boxes/aggregate_metrics.json \
  --thresholds 0.1 0.3 0.5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | *(script default; override)* | Root directory searched recursively for `*.tfrecord` |
| `--pred-boxes-bin` | `results/2_stage/detection_pred.bin` | Detection `.bin` |
| `--gt-aggregate` | `results/aggregate_metrics.json` | Phase A `aggregate_metrics.json` for deltas |
| `--output-root` | `results/phase_b_gt_flow_pred_boxes` | Parent dir for `thr_*` folders |
| `--thresholds` | `0.1 0.3 0.5` | Score cutoffs to sweep |
| `--method` | `median` | `median` or `svd` |
| `--max-pairs` | unlimited | Optional cap |

### `python -m rigid_flow.pipeline` CLI

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Directory tree containing Waymo `.tfrecord` files |
| `--output-dir` | No | None | If set, writes JSON metrics here |
| `--method` | No | `median` | `median` or `svd` rigid aggregation |
| `--max-pairs` | No | None | Stop after *N* frame pairs |
| `--pred-boxes-bin` | No | None | If set, use predicted boxes for assignment + pooling + predicted-fg metrics |
| `--score-threshold` | No | `0.0` | Min detection score (only with `--pred-boxes-bin`) |

---

## Outputs

With `--output-dir`:

| File | Content |
|------|---------|
| `per_frame_metrics.json` | List of dicts, one per frame pair: all metric keys below plus `sequence_id`, `frame_index`, `num_gt_boxes`, `num_pred_boxes` (-1 in Phase A), `score_threshold` (NaN in Phase A) |
| `aggregate_metrics.json` | Mean over frames: every numeric per-frame key is prefixed with `avg_`, plus `total_frames` |

---

## Metrics reference

Unless noted, **EPE** is the **L2 endpoint error** per point: the Euclidean norm of `(refined_flow - gt_flow)` at that point. Here `refined_flow` is the flow **after** rigid pooling; `gt_flow` is label-derived ground truth.

**Relative error** (for accuracy bands) is `EPE / ||gt_flow||`, defined as 0 when both are zero, and treated as infinite when GT flow is zero but EPE is not.

**Speed** for all speed buckets is **`||gt_flow|| / dt`** in m/s (not detector box metadata), with default thresholds **0.5** and **2.0** m/s between static / slow / fast splits.

### Core EPE (all points)

| Key | Meaning |
|-----|---------|
| `epe_mean` | Mean EPE over **all** points in the frame |

### Predicted foreground / background

Membership uses **evaluation** boxes (GT in Phase A, predicted in Phase B).

| Key | Meaning |
|-----|---------|
| `epe_background` | Mean EPE where point is **outside** evaluation boxes |
| `epe_foreground` | Mean EPE where point is **inside** an evaluation box |

### True foreground (GT boxes, detection-agnostic)

Computed from a **separate** GT point-to-box map. In Phase A this matches predicted-fg; in Phase B it does **not** if detectors miss or misalign objects.

| Key | Meaning |
|-----|---------|
| `epe_true_foreground` | Mean EPE over every point inside any **GT** box |
| `epe_true_fg_static` / `epe_true_fg_slow` / `epe_true_fg_fast` | Same as above, split by GT-flow speed using the same 0.5 / 2.0 m/s thresholds |

### Speed buckets on predicted-foreground only

| Key | Meaning |
|-----|---------|
| `epe_static` | Mean EPE for evaluation-foreground points with GT speed below the low threshold (0.5 m/s default) |
| `epe_slow` | Foreground points with GT speed between low and high thresholds |
| `epe_fast` | Foreground points with GT speed at or above the high threshold (2.0 m/s default) |

### Speed buckets on the full cloud

| Key | Meaning |
|-----|---------|
| `epe_all_static` / `epe_all_slow` / `epe_all_fast` | Same thresholds, evaluated on **all** points (background included) using GT-flow speed |

### Per-class (evaluation boxes)

Uses the **class label on the evaluation box** each point is assigned to (Waymo types: e.g. vehicle = 1, pedestrian = 2, cyclist = 4).

| Key | Meaning |
|-----|---------|
| `epe_vehicle` | Mean EPE over points in vehicle **evaluation** boxes |
| `epe_pedestrian` | Pedestrian boxes |
| `epe_cyclist` | Cyclist boxes |

### Accuracy and outliers

Percentages are **of points in the stated subset** that satisfy the condition.

| Key | Condition (per point) | Subset |
|-----|------------------------|--------|
| `acc_strict` | EPE under 0.05 m **or** relative error under 5% | all points |
| `acc_relaxed` | EPE under 0.10 m **or** relative error under 10% | all points |
| `out3d` | EPE over 0.30 m **or** relative error over 10% | all points |
| `acc_strict_fg` / `acc_relaxed_fg` / `out3d_fg` | same rules | predicted-foreground only |
| `acc_strict_bg` / `acc_relaxed_bg` / `out3d_bg` | same rules | predicted-background only |

### Rigidity (structure inside evaluation boxes)

| Key | Meaning |
|-----|---------|
| `flow_variance_mean` | Mean over boxes of trace of covariance of corrected flow (0 if every point in a box shares the same flow vector) |
| `dist_preservation_mean` | Mean over boxes of mean absolute pairwise distance change after warping points by corrected flow |

### Counts

| Key | Meaning |
|-----|---------|
| `num_points` | Total LiDAR points at the first frame of the pair |
| `num_foreground` | Points inside evaluation boxes |
| `num_background` | Points outside evaluation boxes |
| `num_true_foreground` | Points inside **GT** boxes |

Authoritative names and edge cases live in the docstring of [`src/rigid_flow/eval/metrics.py`](src/rigid_flow/eval/metrics.py).

---

## Visualization

Inspect one frame pair (writes PNGs under `--output-dir`):

```bash
python -m rigid_flow.visualization.visualize \
  --data-root /path/to/waymo/validation_0000 \
  --output-dir figures/ \
  --frame-index 0 \
  --method median
```

Add `--show` for interactive windows.

| Output | Description |
|--------|-------------|
| `bev_boxes.png` | BEV height-colored points + oriented boxes by class |
| `flow_magnitude.png` | BEV heatmap of flow L2 norm |
| `flow_quiver.png` | Subsampled arrows; fg/bg |
| `correction_comparison.png` | Raw vs corrected flow vs residual |
| `per_object_histograms.png` | Per-object flow magnitude before vs after correction |

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | `.tfrecord` directory tree |
| `--output-dir` | Yes | — | PNG output directory |
| `--frame-index` | No | `0` | Which pair index to render |
| `--method` | No | `median` | `median` or `svd` |
| `--show` | No | off | Show plots |

---

## Design notes

- **Module boundaries:** `data/`, `geometry/`, and `aggregation/` avoid importing each other; they share types via `core/types.py`. `pipeline.py` wires everything.
- **Median default:** robust pooling to a single translation per box (or SVD for full rigid).
- **NumPy geometry:** point-in-box via yaw rotation + axis-aligned bounds; no CUDA / torch_scatter / OpenPCDet ops in this path.
- **Protobuf tfrecords:** range image → points using calibration; no heavy Waymo pip dependency for parsing on ARM64.
- **Flow convention:** displacement in the ego frame at the first timestamp; static world points get approximately zero GT object motion after the pose and velocity recipe.

---

## Legacy ZeroFlow

Training, evaluation, Docker, and dataset mounts for the original scene flow codebase: see [`CLAUDE.md`](CLAUDE.md).
