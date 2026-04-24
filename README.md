# Rigid Scene Flow

Object-aware 3D scene flow on Waymo: we **enforce rigid motion per 3D bounding box** to correct the "jelly effect" — the physically impossible per-point flow inconsistency produced by naive learned models such as ZeroFlow. A deterministic post-processing step pools all per-point flow vectors inside each box into a single representative translation (or full rigid transform), then broadcasts that result back.

We validate this in a four-phase ablation study that incrementally swaps ground-truth (GT) components for predicted ones (ZeroFlow predictions + CenterPoint detections), isolating the contribution of each component.

See [`SUMMARY.md`](SUMMARY.md) for the full project writeup, methods, and results.

---

## Contents

1. [Repository layout](#repository-layout)
2. [Setup](#setup)
3. [Data](#data)
4. [Generating ZeroFlow predictions](#generating-zeroflow-predictions)
5. [Generating CenterPoint detections](#generating-centerpoint-detections)
6. [How to run](#how-to-run)
7. [Full method sweep](#full-method-sweep)
8. [Visualization](#visualization)
9. [Outputs](#outputs)
10. [Metrics reference](#metrics-reference)
11. [Design notes](#design-notes)

---

## Repository layout

```
scene-flow/
├── src/rigid_flow/                  # Main Python package
│   ├── core/types.py                # BoundingBox, SceneFlowPair, FlowResult
│   ├── data/
│   │   ├── waymo_parser.py          # tfrecord → SceneFlowPair (protobuf-only, ARM64 safe)
│   │   ├── waymo_protos/            # Vendored Waymo proto stubs
│   │   ├── pred_boxes.py            # Detection .bin → PredBoxIndex
│   │   └── zeroflow_loader.py       # .pkl + .feather + tfrecord → ZeroFlowDataSource
│   ├── geometry/
│   │   ├── se3.py                   # SE3 rigid transforms (NumPy)
│   │   └── points_in_boxes.py       # Point-in-box assignment + ego compensation
│   ├── aggregation/
│   │   └── rigid_aggregation.py     # none / mean / median / weighted_median / geometric_median / svd
│   ├── eval/
│   │   └── metrics.py               # EPE, accuracy, rigidity, per-class, speed buckets
│   ├── pipeline.py                  # run_pipeline() + run_zeroflow_pipeline() + CLI
│   └── visualization/               # BEV, quiver, correction figures (Matplotlib)
│       ├── visualize.py             # CLI entry point
│       ├── bev.py                   # Bird's-eye-view plots
│       ├── flow_comparison.py       # Raw vs corrected vs residual panels
│       └── per_object.py            # Per-object flow histograms
│
├── scripts/
│   ├── run_all_methods.sh           # Sweep all 6 methods × 4 phases
│   ├── run_phase_cd.py              # Phase C (ZeroFlow + GT boxes) and D (ZeroFlow + pred boxes)
│   ├── run_pred_box_sweep.py        # Phase B threshold sweep
│   └── visualize_phase_c_jelly.py  # Find and render frames with largest EPE gain
│
├── visualize_predictions.py         # Interactive Plotly 3D: GT vs raw vs geometric_median
├── viz_zeroflow_predictions.py      # Matplotlib BEV + stats for a single ZeroFlow segment
│
├── SceneFlowZoo/                    # ZeroFlow model (external; generates .pkl + .feather)
├── OpenPCDet/                       # CenterPoint detector (external; generates detection_pred.bin)
│
├── mac_environment.yaml             # Conda environment (macOS Apple Silicon)
├── environment.yaml                 # Conda environment (Linux + CUDA)
└── pyproject.toml                   # src/ layout, rigid_flow package
```

---

## Setup

### macOS Apple Silicon

```bash
mamba env create -f mac_environment.yaml
mamba activate flow
pip install "tensorflow-macos>=2.16"
pip install grpcio-tools==1.62.3
pip install --editable .
```

### Linux + CUDA

```bash
conda env create -f environment.yaml
conda activate flow
pip install --editable .
```

**Waymo SDK note:** The `waymo-open-dataset` pip wheel is not used for tfrecord parsing. We bundle generated proto stubs under `src/rigid_flow/data/waymo_protos/`. To regenerate from upstream protos:

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

Download a Waymo Open Dataset validation shard and extract so you have a directory containing `*.tfrecord` files (nested directories are fine):

```bash
mkdir -p /path/to/waymo/validation
tar xf validation_0000.tar -C /path/to/waymo/validation
```

Pass that directory as `--data-root` (Phases A/B) or `--tfrecord-root` (Phases C/D).

---

## Generating ZeroFlow predictions

Phases C and D require ZeroFlow `.pkl` + `.feather` files precomputed by `SceneFlowZoo/`. Follow [`SceneFlowZoo/GETTING_STARTED.md`](SceneFlowZoo/GETTING_STARTED.md) for model weights and config, then run inference on the Waymo validation set:

```bash
cd SceneFlowZoo
bash run_zeroflow_waymo.sh   # or: docker ./launch.sh then python test_pl.py <config> <checkpoint>
```

This writes per-segment directories under `results/zeroflow/`:
- `results/zeroflow/validation/<segment>/<frame_idx:06d>.pkl` — preprocessed point cloud
- `results/zeroflow/sequence_len_002/<segment>/<pair_idx:010d>.feather` — predicted flow vectors

---

## Generating CenterPoint detections

Phases B and D require a CenterPoint detection `.bin` in Waymo submission format. Follow [`OpenPCDet/docs/GETTING_STARTED.md`](OpenPCDet/docs/GETTING_STARTED.md) for the Waymo config and checkpoint, then run:

```bash
cd OpenPCDet
python tools/test.py \
  --cfg_file tools/cfgs/waymo_models/centerpoint.yaml \
  --ckpt /path/to/centerpoint.pth \
  --save_to_file
```

The output `.bin` (e.g. `results/2_stage/detection_pred.bin`) is passed as `--pred-boxes-bin`.

---

## How to run

### Phase A — GT flow + GT boxes (sanity check)

GT flow is derived analytically from box velocities. Evaluation boxes are the same GT boxes. Result: EPE ≈ 0 for all methods.

```bash
python -m rigid_flow.pipeline \
  --data-root /path/to/waymo/validation \
  --output-dir results/phase_a_gt_flow_gt_boxes \
  --method median
```

Quick smoke test on a few pairs:

```bash
python -m rigid_flow.pipeline \
  --data-root /path/to/waymo/validation \
  --output-dir results/debug \
  --method median \
  --max-pairs 10
```

### Phase B — GT flow + predicted boxes

GT flow source unchanged. Evaluation boxes come from a CenterPoint `.bin`. Use `--score-threshold` to filter low-confidence boxes (recommended: 0.50).

```bash
python -m rigid_flow.pipeline \
  --data-root /path/to/waymo/validation \
  --output-dir results/phase_b_gt_flow_pred_boxes/thr_0.50 \
  --method median \
  --pred-boxes-bin results/2_stage/detection_pred.bin \
  --score-threshold 0.50
```

**Sweep multiple thresholds** and compare to Phase A:

```bash
python scripts/run_pred_box_sweep.py \
  --data-root /path/to/waymo/validation \
  --pred-boxes-bin results/2_stage/detection_pred.bin \
  --output-root results/phase_b_gt_flow_pred_boxes \
  --gt-aggregate results/phase_a_gt_flow_gt_boxes/aggregate_metrics.json \
  --thresholds 0.1 0.3 0.5 \
  --method median
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | required | Directory tree with `*.tfrecord` files |
| `--pred-boxes-bin` | `results/2_stage/detection_pred.bin` | CenterPoint `.bin` |
| `--gt-aggregate` | `results/aggregate_metrics.json` | Phase A aggregate for delta comparison |
| `--output-root` | `results/phase_b_gt_flow_pred_boxes` | Parent dir; writes `thr_X.XX/` subdirs |
| `--thresholds` | `0.1 0.3 0.5` | Score thresholds to sweep |
| `--method` | `median` | Aggregation method |
| `--max-pairs` | unlimited | Optional cap |

### Phase C — ZeroFlow predictions + GT boxes (core experiment)

This is the primary experiment. ZeroFlow per-point flow is noisy within objects; rigid pooling corrects it.

```bash
python scripts/run_phase_cd.py \
  --tfrecord-root /path/to/waymo/validation \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --output-root results \
  --method median \
  --skip-phase-d
```

### Phase D — ZeroFlow predictions + predicted boxes (fully predicted)

Real-world setting: no GT information used at inference.

```bash
python scripts/run_phase_cd.py \
  --tfrecord-root /path/to/waymo/validation \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --pred-boxes-bin results/2_stage/detection_pred.bin \
  --score-threshold 0.50 \
  --output-root results \
  --method median \
  --skip-phase-c
```

**`run_phase_cd.py` arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--tfrecord-root` | required | Directory tree with `*.tfrecord` files |
| `--pkl-root` | required | Root of per-segment `.pkl` directories |
| `--feather-root` | required | Root of per-segment `.feather` directories |
| `--pred-boxes-bin` | `results/2_stage/detection_pred.bin` | Detection `.bin` for Phase D |
| `--score-threshold` | `0.50` | Score threshold for Phase D boxes |
| `--output-root` | `results` | Parent dir; writes `phase_c_*/` and `phase_d_*/` |
| `--method` | `median` | Aggregation method |
| `--max-pairs` | unlimited | Optional cap |
| `--skip-phase-c` | off | Skip Phase C |
| `--skip-phase-d` | off | Skip Phase D |

### `python -m rigid_flow.pipeline` CLI reference

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | Directory tree containing Waymo `.tfrecord` files |
| `--output-dir` | No | None | If set, writes JSON metrics here |
| `--method` | No | `median` | One of: `none`, `mean`, `median`, `weighted_median`, `geometric_median`, `svd` |
| `--max-pairs` | No | None | Stop after N frame pairs |
| `--pred-boxes-bin` | No | None | If set, use predicted boxes (Phase B/D mode) |
| `--score-threshold` | No | `0.0` | Min detection score (requires `--pred-boxes-bin`) |

---

## Full method sweep

Run all 6 aggregation methods across all 4 phases in one go:

```bash
bash scripts/run_all_methods.sh \
  --tfrecord-root /path/to/waymo/validation \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --pred-boxes-bin results/2_stage/detection_pred.bin \
  --score-threshold 0.50
```

Results are written to `results/{method}/phase_{a,b,c,d}_*/`. The script accepts `--max-pairs N` to cap the number of frame pairs for a quick test.

---

## Visualization

### Interactive Plotly 3D — GT vs ZeroFlow vs rigid correction

Writes HTML files viewable in any browser. With `--tfrecord-root`, produces two overlay files per pair: GT vs raw ZeroFlow and GT vs geometric median corrected.

```bash
python visualize_predictions.py \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --tfrecord-root /path/to/waymo/validation \
  --segment segment-10203656353524179475_7625_000_7645_000_with_camera_labels \
  --output-dir figures/zeroflow_plotly \
  --pair-indices 41 193
```

Without `--tfrecord-root`, writes a single baseline ZeroFlow HTML (no GT overlay).

| Argument | Default | Description |
|----------|---------|-------------|
| `--pkl-root` | `results/zeroflow/validation` | Root of per-segment `.pkl` directories |
| `--feather-root` | `results/zeroflow/sequence_len_002` | Root of per-segment `.feather` directories |
| `--segment` | (default segment) | Waymo segment name |
| `--pair-indices` | `193 41 49` | Pair indices to render |
| `--output-dir` | `figures/zeroflow_plotly` | Output directory for HTML files |
| `--tfrecord-root` | None | If set, enables GT overlays |
| `--max-pts` | `75000` | Point cloud render limit |
| `--line-limit` | `5000` | Max flow arrows per field |

**Output files per pair (with `--tfrecord-root`):**
- `pair_XXXX_overlay_gt_vs_zeroflow_raw.html`
- `pair_XXXX_overlay_gt_vs_geometric_median.html`

### BEV figures — single frame pair

Writes 5 PNGs using Matplotlib:

```bash
python -m rigid_flow.visualization.visualize \
  --data-root /path/to/waymo/validation \
  --output-dir figures/ \
  --frame-index 0 \
  --method median
```

Add `--show` for interactive windows.

| Output | Description |
|--------|-------------|
| `bev_boxes.png` | BEV height-colored points + oriented boxes by class |
| `flow_magnitude.png` | BEV heatmap of flow L2 norm |
| `flow_quiver.png` | Subsampled arrows; fg/bg distinction |
| `correction_comparison.png` | Raw vs corrected flow vs residual (3 panels) |
| `per_object_histograms.png` | Per-object flow magnitude before vs after correction |

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data-root` | Yes | — | `.tfrecord` directory tree |
| `--output-dir` | Yes | — | PNG output directory |
| `--frame-index` | No | `0` | Which pair index to render |
| `--method` | No | `median` | Aggregation method |
| `--show` | No | off | Show plots interactively |

### EPE improvement maps — Phase C jelly analysis

Find and render frames where rigid pooling provides the largest EPE gain:

```bash
# Rank all 197 frames by improvement
python scripts/visualize_phase_c_jelly.py print-top \
  --results-root results/full_run \
  --top 15

# Render BEV EPE maps for a specific pair
python scripts/visualize_phase_c_jelly.py render \
  --tfrecord-root /path/to/waymo/validation \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --pair-index 193 \
  --output-dir figures/jelly_pair193
```

`render` subcommand outputs:
- `epe_baseline_none.png` — per-point EPE (no aggregation)
- `epe_after_geometric_median.png` — per-point EPE after rigid pooling
- `epe_improvement_delta.png` — ΔEPE heatmap (green = pooling helped, red = hurt)
- `flow_correction_geometric_median.png` — raw / corrected / magnitude comparison
- `flow_mag_zeroflow_input.png` — ZeroFlow input flow magnitude BEV
- `flow_mag_after_geometric_median.png` — corrected flow magnitude BEV
- `summary.json` — mean EPE before/after + improvement

---

## Outputs

With `--output-dir` set, each pipeline run writes:

| File | Content |
|------|---------|
| `per_frame_metrics.json` | List of dicts, one per frame pair: all metric keys below plus `sequence_id`, `frame_index`, `num_gt_boxes`, `num_pred_boxes` (-1 in Phase A), `score_threshold` (NaN in Phase A) |
| `aggregate_metrics.json` | Mean over frames: every numeric per-frame key prefixed with `avg_`, plus `total_frames` |

---

## Metrics reference

**EPE** = L2 norm of `(refined_flow − gt_flow)` per point (meters). Lower is better.

**Relative error** = `EPE / ‖gt_flow‖`; defined as 0 when both are zero, ∞ when GT is zero but EPE is not.

**Speed** for bucketing = `‖gt_flow‖ / dt` (m/s); thresholds 0.5 m/s (static/slow) and 2.0 m/s (slow/fast).

### Core EPE

| Key | Meaning |
|-----|---------|
| `epe_mean` | Mean EPE over **all** points |
| `epe_foreground` | Mean EPE — points inside evaluation boxes |
| `epe_background` | Mean EPE — points outside evaluation boxes |
| `epe_true_foreground` | Mean EPE — points inside any **GT** box (missed detections scored against zero-flow baseline) |

### Speed-bucketed EPE

| Key | Subset |
|-----|--------|
| `epe_static` / `epe_slow` / `epe_fast` | Evaluation-foreground only |
| `epe_all_static` / `epe_all_slow` / `epe_all_fast` | All points |
| `epe_true_fg_static` / `epe_true_fg_slow` / `epe_true_fg_fast` | GT-foreground only |

### Per-class EPE

| Key | Meaning |
|-----|---------|
| `epe_vehicle` | Points inside vehicle evaluation boxes |
| `epe_pedestrian` | Pedestrian evaluation boxes |
| `epe_cyclist` | Cyclist evaluation boxes |

### Accuracy and outliers

| Key | Condition | Subset |
|-----|-----------|--------|
| `acc_strict` | EPE < 0.05 m **or** rel\_err < 5% | all points |
| `acc_relaxed` | EPE < 0.10 m **or** rel\_err < 10% | all points |
| `out3d` | EPE > 0.30 m **or** rel\_err > 10% | all points |
| `acc_strict_fg` / `acc_relaxed_fg` / `out3d_fg` | same | predicted-foreground |
| `acc_strict_bg` / `acc_relaxed_bg` / `out3d_bg` | same | predicted-background |

### Rigidity

| Key | Meaning |
|-----|---------|
| `flow_variance_mean` | Mean over boxes of trace of covariance of corrected flow vectors (0 = perfect rigidity) |
| `dist_preservation_mean` | Mean over boxes of mean absolute pairwise distance change after warping (0 = perfect rigidity) |

### Counts

| Key | Meaning |
|-----|---------|
| `num_points` | Total LiDAR points at t0 |
| `num_foreground` | Points inside evaluation boxes |
| `num_background` | Points outside evaluation boxes |
| `num_true_foreground` | Points inside GT boxes |

---

## Design notes

- **Module boundaries:** `data/`, `geometry/`, and `aggregation/` do not import each other; all share types through `core/types.py`. `pipeline.py` wires them together.
- **Six aggregation methods:** `none` (passthrough), `mean`, `median`, `weighted_median` (by distance to box center), `geometric_median` (Weiszfeld), `svd` (Kabsch/Procrustes full rigid fit). Boxes with fewer than 3 assigned points are skipped.
- **ARM64-compatible tfrecord parsing:** pure-Python protobuf stubs under `waymo_protos/`; no Waymo pip wheel dependency.
- **Flow convention:** displacement in the ego frame at t0; background points get zero GT object motion after ego-pose + velocity recipe.
- **GT flow derivation:** `velocity * dt` per object in global frame, then SE3 transform back to ego frame; not a neural estimate.
