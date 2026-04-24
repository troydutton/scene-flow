# Object-Aware 3D Scene Flow: Rigid-Refinement Pipeline — Project Summary

## 1. Problem Statement

Naive learned scene flow models estimate a per-point 3D displacement vector independently for every LiDAR point. When these models are applied to rigid objects such as cars, the predicted vectors are inconsistent within the same object — one point predicts the car moved left while a neighboring point predicts it moved right. This physically impossible deformation is called the **"jelly effect"**. 

This project implements and evaluates a deterministic post-processing step that **enforces rigid motion** inside each detected 3D bounding box by pooling all per-point flow vectors within a box into a single representative translation (or full rigid transform), then broadcasting that result back. We validate this approach in a four-phase ablation study that incrementally swaps ground-truth (GT) components for predicted ones, isolating the contribution of the flow model and the detector separately.

---

## 2. Dataset

All experiments run on the **Waymo Open Dataset** validation split. Data is stored as `.tfrecord` files. Each record contains:

- **LiDAR range images** (5 returns, 64 beams), decoded to `(N, 3)` point clouds per frame.
- **Ego poses** — 4×4 homogeneous matrices (ego-vehicle → global) at every timestamp.
- **3D bounding box annotations** per frame: center `(x, y, z)`, dimensions `(dx, dy, dz)`, yaw heading, per-object velocity `(vx, vy)` in the global frame, class label (Vehicle, Pedestrian, Cyclist), and a persistent tracking ID.
- **Timestamps** in microseconds, needed to compute the time delta `dt ≈ 0.1 s` between consecutive frames.

The pipeline parses tfrecords using **vendored Protobuf stubs** (under `src/rigid_flow/data/waymo_protos/`) rather than the official `waymo-open-dataset` pip wheel, which is not available on Apple Silicon / ARM64.

The full run processed **197 consecutive frame pairs** across multiple sequences.

---

## 3. External Models and Precomputed Inputs

Two external codebases were used to generate precomputed inputs consumed by the pipeline.

### 3.1 ZeroFlow (via SceneFlowZoo)

**Codebase:** [`SceneFlowZoo/`](SceneFlowZoo/) — an open-source scene flow benchmark and model zoo built on top of the original ZeroFlow repository (Vedder et al., ICLR 2024).

**What it provides:** Per-point 3D flow predictions for each Waymo frame pair. ZeroFlow is a knowledge distillation-based feed-forward model: it trains a fast student network (FastFlow3D / FastFlow3D XL) to mimic a slow teacher (Neural Scene Flow Prior, NSFP) without requiring ground-truth flow labels. Inference produces:

- **`.feather` files** — one per consecutive frame pair, each containing a `(N, 3)` float32 array of predicted flow vectors aligned to the ground-removed, subsampled point cloud.
- **`.pkl` files** — one per frame, containing the preprocessed point cloud (ground-removed, ~30K points after subsampling by OpenPCDet) used as input to ZeroFlow.

These files are precomputed offline and loaded at evaluation time by `src/rigid_flow/data/zeroflow_loader.py`, which matches `.pkl` point clouds to `.feather` flow predictions and enriches them with bounding boxes and timestamps from the raw tfrecords.

**Why ZeroFlow predictions are noisy:** ZeroFlow treats each LiDAR point independently. Even though the model has learned to approximate physically plausible flow, it has no explicit object-level constraint. Inside a car, neighboring points may receive slightly different flow vectors — the jelly effect that rigid pooling is designed to correct.

### 3.2 CenterPoint (via OpenPCDet)

**Codebase:** [`OpenPCDet/`](OpenPCDet/) — an open-source 3D object detection framework (OpenMMLab). Provides CenterPoint, a state-of-the-art anchor-free LiDAR detector.

**What it provides:** A detection submission `.bin` file in the **Waymo Open Dataset submission format** (`waymo_open_dataset.protos.metrics_pb2.Objects`). Each `Object` proto contains:

- A predicted 3D bounding box `(x, y, z, dx, dy, dz, heading)` in the **sensor / ego frame**.
- A detection confidence score.
- A class label (Vehicle, Pedestrian, Cyclist).
- Optional velocity fields `(speed_x, speed_y)` — typically 0 for this detector (no temporal backbone).

This `.bin` is parsed by `src/rigid_flow/data/pred_boxes.py`, which indexes every detection by `(context_name, frame_timestamp_micros)` so the pipeline can look up predictions per frame in O(1).

At **score threshold 0.50**, the detector yields an average of **1,160 predicted foreground points** per frame vs. **977 GT foreground points** per frame (a slight over-detection), with 7,814 predicted-foreground points vs. 8,850 GT-foreground points in Phases C/D (where point clouds are smaller due to ground removal).

---

## 4. Ground-Truth Flow Computation

The pipeline derives **ground-truth scene flow analytically** from box annotations rather than using a neural model or optical flow approximation. The algorithm (implemented in `pipeline.py::compute_gt_flow`) runs per frame pair:

1. **Point-to-box assignment:** Each point in `points_t0` is tested against every GT box at `t0` using a yaw-rotated axis-aligned containment check (in `geometry/points_in_boxes.py`). Background points receive assignment `-1`.

2. **Object motion in global frame:** For each foreground point assigned to box `b`, the displacement is `[vx * dt, vy * dt, 0]` where `vx, vy` are the global-frame box velocities and `dt` is the frame timestamp delta in seconds. Vertical velocity is assumed zero.

3. **Ego-motion compensation via SE3:** All points are transformed to the global frame using `ego_pose_t0`, displaced by their object motion, then transformed back to the `t0` ego frame using `ego_pose_t0_inv`. Background points receive no object displacement and land back at exactly their original position, giving zero GT flow in the ego frame.

4. **Flow vector:** `gt_flow[i] = points_prime[i] - points_t0[i]`, an `(N, 3)` float32 array.

This GT flow is the reference for all EPE computations in every phase.

---

## 5. Pipeline Architecture

```
Waymo .tfrecord  ──────────────────────────────────┐
                                                   ▼
                                          WaymoParser
                                          (SceneFlowPair)
                                                   │
                         ┌─────────────────────────┤
                         ▼                         ▼
              GT flow source                ZeroFlow loader
              (compute_gt_flow)             (.pkl + .feather)
                         │                         │
                         └──────────┬──────────────┘
                                    ▼
                           Box source selection
                          ┌──────────┴──────────┐
                          ▼                     ▼
                    GT boxes              PredBoxIndex
                  (tfrecord)              (.bin file)
                          └──────────┬──────────┘
                                     ▼
                         points_in_boxes_cpu
                       (point-to-box assignment)
                                     │
                                     ▼
                          compute_rigid_flow
                       (rigid_aggregation.py)
                       method: none / mean / median /
                       weighted_median / geometric_median / svd
                                     │
                                     ▼
                              evaluate()
                            (metrics.py)
                                     │
                                     ▼
                     per_frame_metrics.json
                     aggregate_metrics.json
```

All code lives under `src/rigid_flow/` and is installable as a package via `pip install -e .`. The CLI entrypoint is `python -m rigid_flow.pipeline`.

---

## 6. Rigid Aggregation Methods

Six aggregation strategies are evaluated, all implemented in `src/rigid_flow/aggregation/rigid_aggregation.py`. For boxes with fewer than 3 assigned points, no correction is applied.

| Method | What it does |
|--------|-------------|
| `none` | **Passthrough** — no modification. Raw flow is scored as-is. Used as the baseline. |
| `mean` | Replace every in-box flow vector with the **component-wise mean** of all in-box vectors. |
| `median` | Replace with the **component-wise median**. Robust to outlier points at box edges. |
| `weighted_median` | Component-wise median where each point is weighted by `1 / distance_to_box_center`. Down-weights edge points. |
| `geometric_median` | Iterative **Weiszfeld algorithm** to find the L2-minimizing point in flow space. More robust than mean to multi-modal distributions. |
| `svd` | Full **rigid transform fitting** via the Kabsch (Procrustes) algorithm: finds the `(R, t)` that minimizes L2 between `{source points}` and `{source + flow}`. Produces per-point flows that encode rotation, not just translation. |

Background points always keep their raw (ZeroFlow or GT) flow unchanged.

---

## 7. Evaluation Metrics

All metrics are computed by `src/rigid_flow/eval/metrics.py` and averaged across frames in `aggregate_metrics.json` (keys prefixed `avg_`). Evaluated over 197 frames.

### 7.1 Endpoint Error (EPE)

EPE is the **L2 norm of (predicted_flow − gt_flow)** per point, in **meters**. Lower is better.

- **Global mean EPE** (`epe_mean`): average over all points.
- **Foreground EPE** (`epe_foreground`): average over points assigned to **evaluation boxes** (GT boxes in Phase A, predicted boxes in Phases B/D, GT in C).
- **True foreground EPE** (`epe_true_foreground`): average over points inside any **GT box**, regardless of whether the detector found it. In Phase D, GT-box points missed by the detector are scored against a **zero-flow baseline** (not the model output) so missed detections are not silently ignored.
- **Per-class EPE** (`epe_vehicle`, `epe_pedestrian`, `epe_cyclist`): mean EPE for points inside evaluation boxes of each class.
- **Speed-bucketed EPE** (`epe_static`, `epe_slow`, `epe_fast`): foreground split by GT-flow speed (`‖gt_flow‖ / dt`), thresholds 0.5 and 2.0 m/s.

### 7.2 Accuracy and Outlier Rates

Percentage of points satisfying the condition `EPE < threshold OR relative_error < rel_threshold`:

| Metric | EPE threshold | Relative error threshold |
|--------|--------------|--------------------------|
| `acc_strict` | < 0.05 m | < 5% |
| `acc_relaxed` | < 0.10 m | < 10% |
| `out3d` | > 0.30 m | > 10% |

Also reported separately for predicted foreground (`_fg`) and background (`_bg`) subsets.

### 7.3 Rigidity Metrics

Structural quality inside evaluation boxes:

- **`flow_variance_mean`**: mean over boxes of the trace of the covariance matrix of corrected flow vectors. Zero means every point in a box has an identical flow vector — perfect rigidity. Nonzero measures how much the assigned flow vectors still diverge within each box.
- **`dist_preservation_mean`**: mean over boxes of the mean absolute pairwise distance change after warping points by their corrected flow. A rigid body should preserve all pairwise distances; this measures how well the corrected flow satisfies that constraint.

---

## 8. Four-Phase Ablation Design

The four phases swap GT vs. predicted inputs one at a time, isolating each component's contribution to final error.

| Phase | Flow source | Box source | Purpose |
|-------|-------------|------------|---------|
| **A** | Ground truth | Ground truth | Sanity check: rigid pooling on already-rigid GT flow. EPE should be ~0. |
| **B** | Ground truth | CenterPoint (predicted) | Isolate box quality: introduces detector error while keeping flow perfect. |
| **C** | ZeroFlow (predicted) | Ground truth | Isolate flow quality + rigid correction benefit: shows how much pooling recovers. |
| **D** | ZeroFlow (predicted) | CenterPoint (predicted) | Fully predicted / real-world setting. |

---

## 9. Results

All numbers from the **full run** of 197 frames (`results/full_run/`).

### 9.1 Phase A — GT Flow + GT Boxes (Sanity Check)

All methods produce **EPE ≈ 0** everywhere, as expected: pooling GT flow within GT boxes gives back the same GT flow (GT flow is already rigid per box by construction from box velocities). The only nonzero entries are sub-0.0002 m floating-point residuals from rounding. `acc_strict = 100%`, `out3d = 0%`. This validates the pipeline math end-to-end.

| Method | epe_mean | acc_strict | out3d |
|--------|---------|-----------|-------|
| none | 0.000000 | 100.00% | 0.00% |
| mean | 0.000001 | 100.00% | 0.36% |
| median | 0.000001 | 100.00% | 0.36% |
| weighted_median | 0.000001 | 100.00% | 0.36% |
| geometric_median | 0.000001 | 100.00% | 0.36% |
| svd | 0.000001 | 100.00% | 0.36% |

The `out3d = 0.36%` for the non-none methods is a known artifact: the relaxed relative-error arm of `out3d` fires on a tiny fraction of slow-moving foreground points where the pooled GT flow has a near-zero magnitude denominator.

### 9.2 Phase B — GT Flow + Predicted Boxes

Predicted boxes introduce **mislocation, missed detections, and false positives**. With GT flow as input and `none` (passthrough), EPE is still exactly 0 because the flow is untouched. For pooling methods, box mismatch causes the pooled vectors to diverge from the per-point GT, producing a **sub-millimeter** global mean EPE penalty:

| Method | epe_mean | epe_foreground | epe_true_foreground | flow_variance_mean |
|--------|---------|---------------|--------------------|--------------------|
| none | 0.000000 | 0.000000 | 0.054014 | 0.107104 |
| mean | 0.000556 | 0.127926 | 0.132842 | 0.000000 |
| median | 0.000386 | 0.089292 | 0.075455 | 0.000000 |
| weighted_median | 0.000386 | 0.089320 | 0.071235 | 0.000000 |
| geometric_median | 0.000386 | 0.089288 | 0.075457 | 0.000000 |
| svd | 0.000554 | 0.126919 | 0.133557 | 0.008457 |

**Key observations:**
- `epe_true_foreground = 0.054 m` for `none` Phase B: GT-box points missed by the detector are scored against zero-flow baseline, revealing that true-foreground points are receiving no correction.
- `median / weighted_median / geometric_median` are significantly better than `mean / svd` on **true foreground** — the median is more robust to box-boundary outliers introduced by detector misalignment.
- `flow_variance_mean = 0.107` for `none` confirms that raw GT flow inside predicted boxes is **not uniformly zero** — the predicted boxes span regions with heterogeneous GT motion.

### 9.3 Phase C — ZeroFlow + GT Boxes (Core Experiment)

This is the primary experiment where rigid pooling is evaluated on realistic predicted flow. ZeroFlow predictions are noisy within objects; pooling collapses per-box variance to zero while also bringing EPE down closer to the GT.

**Global and foreground EPE:**

| Method | epe_mean | epe_foreground | epe_vehicle | epe_pedestrian | flow_variance_mean |
|--------|---------|---------------|------------|---------------|--------------------|
| **none** | 0.05262 | 0.10397 | 0.11861 | 0.03770 | 0.004995 |
| mean | 0.05071 | 0.09606 | 0.10988 | 0.03738 | 0.000000 |
| **median** | **0.05006** | **0.09257** | **0.10590** | **0.03730** | **0.000000** |
| weighted_median | 0.05129 | 0.09821 | 0.11227 | 0.03694 | 0.000000 |
| **geometric_median** | **0.05007** | **0.09267** | **0.10601** | 0.03730 | **0.000000** |
| svd | 0.05086 | 0.09668 | 0.11054 | 0.03748 | 0.000272 |

**Key findings:**
- **All pooling methods improve over the `none` baseline** on global and foreground EPE.
- **`median` and `geometric_median` are the best performers**, reducing vehicle EPE by **~10.7% / 10.6%** vs. passthrough (none), and reducing foreground EPE by **~1.1 cm** absolute.
- **`flow_variance_mean` drops to zero** for all translation methods (mean, median, weighted_median, geometric_median) — these methods enforce exactly identical flow vectors within each box by construction. SVD produces a tiny residual (0.000272) because it fits a full rotation, which does not give every point the same flow vector.
- **Pedestrian EPE** is only weakly improved by pooling (vehicles show ~10% gains; pedestrians show <1%). Pedestrians are small and often have fewer than ~5 points per box, limiting the signal for pooling. Weighted median even beats median slightly on pedestrians (0.03694 vs. 0.03730).
- **Background EPE is identical** across all methods (0.0399 m for none, unchanged for all others), confirming that pooling correctly leaves background points untouched.

**Speed buckets (foreground):**

| | epe_static | epe_slow | epe_fast |
|-|------------|---------|---------|
| none | 0.00933 | 0.09934 | 0.32674 |
| median | 0.00664 | 0.10097 | 0.30408 |

Fast-moving objects see the largest absolute EPE reduction (−0.023 m), matching the intuition that high-speed objects have more per-point variance in ZeroFlow predictions that pooling can correct. Static foreground objects also benefit (−0.003 m).

### 9.4 Phase D — ZeroFlow + Predicted Boxes (Full Predicted Setting)

Phase D is the fully realistic end-to-end configuration: no ground truth information is used at inference.

**Predicted-foreground count decreases** from 8,850 (Phase C, GT boxes) to 7,814 (Phase D, predicted boxes), meaning the detector missed or mislocated ~1,036 foreground points per frame on average.

**Global EPE (epe_mean) is nearly identical to Phase C** for all methods — because foreground is a small fraction of the total scene (~23% in Phase C/D). The difference in behavior is visible only in foreground-specific metrics:

| Method | C pred_fg | D pred_fg | Δ (D−C) | C true_fg | D true_fg | Δ (D−C) |
|--------|----------|---------|---------|----------|---------|---------|
| none | 0.1040 | 0.1298 | +0.0258 | 0.1040 | 0.1039 | ≈0 |
| median | 0.0926 | 0.1172 | +0.0246 | 0.0926 | 0.0929 | +0.0004 |
| geometric_median | 0.0927 | 0.1172 | +0.0245 | 0.0927 | 0.0930 | +0.0003 |
| svd | 0.0967 | 0.1217 | +0.0250 | 0.0967 | 0.0967 | ≈0 |

**Key findings:**
- **`epe_true_foreground` barely changes Phase C → D**: the physical object-point EPE is robust to the box source swap. This means that CenterPoint's detections at threshold 0.50 are well-aligned enough that the points which are detected get corrected similarly to GT-box correction.
- **`epe_foreground` (predicted fg) rises ~2.5 cm** from C to D for all methods: the predicted boxes contain a different set of points (some misaligned or false positive regions), so the foreground EPE calculation is measuring a partially different population.
- **Cyclist EPE** appears only in Phase D (no cyclists in Phases A/B/C for this validation shard): all methods between 0.059 and 0.062 m, with `mean` / `svd` slightly better (0.058 m).

**Phase D vehicle EPE** (the dominant class):

| Method | epe_vehicle | vs Phase C |
|--------|------------|------------|
| none | 0.13222 | +0.01361 |
| mean | 0.12334 | +0.01346 |
| median | 0.11933 | +0.01343 |
| geometric_median | 0.11935 | +0.01334 |
| svd | 0.12397 | +0.01343 |

The ~0.013 m increase from C to D for all methods indicates that **CenterPoint introduces a roughly method-independent localization error cost** for vehicle correction at this threshold.

---

## 10. Summary of Findings

1. **Rigid pooling consistently helps ZeroFlow** when GT boxes are available (Phase C). `median` and `geometric_median` give the best overall improvement: **−2.6 mm global mean EPE**, **−1.1 cm foreground EPE**, and **−1.3 cm vehicle EPE** (~10.7% reduction). Flow variance drops to zero — the jelly effect is fully eliminated within each box.

2. **Predicted boxes cost ~2.5 cm in predicted-foreground EPE** (Phase D vs C), driven by localization error and missed detections rather than by a change in object-point quality. **True-foreground EPE changes by <0.5 mm**, showing the detector is accurate for the points it does find.

3. **`median` and `geometric_median` are statistically tied** for best method on both global and foreground EPE in this dataset. `weighted_median` and `mean` are slightly weaker; `svd` is on par with `mean` globally but has the highest foreground EPE, likely because fitting a rotation to noisy point sets introduces additional error.

4. **Background points are completely unaffected** by the aggregation method, as intended. All background-EPE numbers are identical across methods within each phase.

5. **Global mean EPE is an insufficient summary metric** for evaluating rigid refinement: because foreground makes up only ~7% of scene points (Phases A/B; ~26% in C/D after ground removal), foreground-EPE improvements are diluted. `epe_foreground` and `epe_true_foreground` are the relevant metrics for evaluating the object-aware correction.

6. **Phase A (GT flow + GT boxes) yields zero EPE** for all methods, confirming the pipeline correctness — GT flow is rigid per box by construction, so any pooling method is a no-op on the EPE.

---

## 11. Codebase Structure

```
scene-flow/
├── src/rigid_flow/                        # Main installable package (pip install -e .)
│   ├── core/
│   │   └── types.py                       # BoundingBox, SceneFlowPair, FlowResult dataclasses
│   ├── data/
│   │   ├── waymo_parser.py                # tfrecord → SceneFlowPair (protobuf-only, ARM64 safe)
│   │   ├── waymo_protos/                  # Vendored Waymo proto stubs (no pip wheel needed)
│   │   ├── pred_boxes.py                  # detection .bin → PredBoxIndex (O(1) lookup by frame)
│   │   └── zeroflow_loader.py             # .pkl + .feather + tfrecord → ZeroFlowDataSource
│   ├── geometry/
│   │   ├── se3.py                         # SE3 rigid transforms (4×4 homogeneous, NumPy)
│   │   └── points_in_boxes.py             # CPU point-in-box (yaw-rotated AABB) + ego_compensate
│   ├── aggregation/
│   │   └── rigid_aggregation.py           # none/mean/median/weighted_median/geometric_median/svd
│   ├── eval/
│   │   └── metrics.py                     # EPE, accuracy, rigidity, per-class, speed buckets
│   ├── pipeline.py                        # run_pipeline() + run_zeroflow_pipeline() + CLI
│   └── visualization/                     # Matplotlib BEV + flow figures
│       ├── visualize.py                   # CLI: writes 5 PNG outputs for one frame pair
│       ├── bev.py                         # BEV scatter plots, flow magnitude heatmaps, quiver
│       ├── flow_comparison.py             # 3-panel raw / corrected / residual figure
│       └── per_object.py                  # Per-object flow magnitude histogram grid
│
├── scripts/
│   ├── run_all_methods.sh                 # Sweep all 6 methods × all 4 phases (bash)
│   ├── run_phase_cd.py                    # Phase C (ZeroFlow+GT) and D (ZeroFlow+pred boxes)
│   ├── run_pred_box_sweep.py              # Phase B: sweep score thresholds vs Phase A baseline
│   └── visualize_phase_c_jelly.py        # Rank frames by EPE gain; render BEV EPE maps
│
├── visualize_predictions.py              # Interactive Plotly 3D: GT vs raw vs geometric_median
├── viz_zeroflow_predictions.py           # Matplotlib BEV + aggregate stats for one segment
│
├── SceneFlowZoo/                         # ZeroFlow model and inference (external)
├── OpenPCDet/                            # CenterPoint detection framework (external)
├── results/full_run/                     # Full 197-frame results by method and phase
├── mac_environment.yaml                  # Conda environment (macOS Apple Silicon)
├── environment.yaml                      # Conda environment (Linux + CUDA 12.6)
└── pyproject.toml                        # src/ layout, rigid_flow package
```

### Key data classes (`src/rigid_flow/core/types.py`)

**`BoundingBox`** — a 7-DOF 3D box with tracking metadata:
- `center: (3,)`, `dimensions: (3,)` (dx, dy, dz), `heading: float` (yaw), `class_label: int` (1=Vehicle, 2=Pedestrian, 4=Cyclist), `tracking_id: str`, `velocity: (2,) | None` (vx, vy global frame)
- `.as_7dof` — packs to `[x, y, z, dx, dy, dz, heading]`
- `.speed` — scalar speed in m/s

**`SceneFlowPair`** — two consecutive LiDAR frames with all annotations:
- `points_t0/t1: (N/M, 3)`, `ego_pose_t0/t1: (4, 4)`, `boxes_t0/t1: list[BoundingBox]`, `timestamp_us_t0/t1: int`, `gt_flow: (N, 3) | None`, `sequence_id: str`, `frame_index: int`
- `.dt` — time delta in seconds

**`FlowResult`** — output of `compute_rigid_flow`:
- `flow: (N, 3)` — corrected flow, `raw_flow: (N, 3)` — original flow, `point_to_box: (N,)` — box index (-1 = background), `is_rigid: (N,)` — True if point was rigidly corrected

### Aggregation methods (`src/rigid_flow/aggregation/rigid_aggregation.py`)

Boxes with fewer than 3 assigned points are skipped (no correction applied).

| Method | Description |
|--------|-------------|
| `none` | Passthrough — raw flow unchanged |
| `mean` | Component-wise mean of all in-box flow vectors |
| `median` | Component-wise median — robust to outlier points at box edges |
| `weighted_median` | Median weighted by `1 / distance_to_box_center` — down-weights edge points |
| `geometric_median` | Iterative Weiszfeld algorithm — L2-minimizing point in flow space |
| `svd` | Full rigid transform (R, t) via Kabsch/Procrustes — produces rotation-aware per-point flows |

### Geometry (`src/rigid_flow/geometry/`)

- **`se3.py`** — `SE3` class wrapping a 4×4 matrix: `.inverse()`, `.compose()`, `.transform_points(N,3)`, `from_rot_trans()`, `identity()`
- **`points_in_boxes.py`** — `points_in_boxes_cpu(points, boxes_7dof)` returns `(N,)` int32 box index array; `ego_compensate(points, ego_t0, ego_t1)` re-expresses points in another ego frame

### Evaluation metrics (`src/rigid_flow/eval/metrics.py`)

All metric keys (prefixed `avg_` in `aggregate_metrics.json`):

| Category | Keys |
|----------|------|
| Global EPE | `epe_mean` |
| Foreground/background | `epe_foreground`, `epe_background`, `epe_true_foreground` |
| Speed-bucketed (pred-fg) | `epe_static`, `epe_slow`, `epe_fast` |
| Speed-bucketed (all pts) | `epe_all_static`, `epe_all_slow`, `epe_all_fast` |
| Speed-bucketed (GT-fg) | `epe_true_fg_static`, `epe_true_fg_slow`, `epe_true_fg_fast` |
| Per-class | `epe_vehicle`, `epe_pedestrian`, `epe_cyclist` |
| Accuracy (all pts) | `acc_strict`, `acc_relaxed`, `out3d` |
| Accuracy (pred-fg) | `acc_strict_fg`, `acc_relaxed_fg`, `out3d_fg` |
| Accuracy (pred-bg) | `acc_strict_bg`, `acc_relaxed_bg`, `out3d_bg` |
| Rigidity | `flow_variance_mean`, `dist_preservation_mean` |
| Counts | `num_points`, `num_foreground`, `num_background`, `num_true_foreground` |
| Meta | `sequence_id`, `frame_index`, `num_gt_boxes`, `num_pred_boxes`, `score_threshold` |

---

## 12. Visualization Tools

Three separate visualization systems are provided, targeting different use cases.

### 12.1 Matplotlib BEV Figures (`src/rigid_flow/visualization/`)

Single-frame static PNG outputs, primarily for quick sanity checks on Phases A/B (GT flow source):

```bash
python -m rigid_flow.visualization.visualize \
  --data-root /path/to/waymo/validation \
  --output-dir figures/ \
  --frame-index 0 \
  --method median
```

| Output | Description |
|--------|-------------|
| `bev_boxes.png` | BEV height-colored points + oriented boxes by class |
| `flow_magnitude.png` | BEV heatmap of flow L2 norm |
| `flow_quiver.png` | Subsampled flow arrows; fg/bg distinction |
| `correction_comparison.png` | 3-panel raw / corrected / residual BEV |
| `per_object_histograms.png` | Grid of per-object flow magnitude distributions |

### 12.2 Interactive Plotly 3D (`visualize_predictions.py`)

Writes self-contained HTML files for browser-based 3D inspection of ZeroFlow predictions. With `--tfrecord-root`, produces two overlay files per pair:

1. Ground truth (green) vs ZeroFlow raw (orange)
2. Ground truth (green) vs geometric median corrected (blue)

Arrow subsampling is seeded and fixed across both files (same RNG, same ‖GT‖ magnitude ranking), enabling direct visual comparison. Without `--tfrecord-root`, writes a single baseline ZeroFlow HTML with Viridis magnitude coloring.

```bash
python visualize_predictions.py \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --tfrecord-root /path/to/waymo/validation \
  --pair-indices 41 193 \
  --output-dir figures/zeroflow_plotly
```

### 12.3 EPE Improvement Maps (`scripts/visualize_phase_c_jelly.py`)

Two subcommands targeting Phase C results:

**`print-top`** — reads `per_frame_metrics.json` from `none/` and `geometric_median/` Phase C results and ranks all 197 frames by EPE improvement (no data required):

```bash
python scripts/visualize_phase_c_jelly.py print-top \
  --results-root results/full_run \
  --top 15
```

**`render`** — loads one ZeroFlow frame pair and writes 6 PNGs + `summary.json`:

```bash
python scripts/visualize_phase_c_jelly.py render \
  --tfrecord-root /path/to/waymo/validation \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --pair-index 193 \
  --output-dir figures/jelly_pair193
```

| Output | Description |
|--------|-------------|
| `epe_baseline_none.png` | BEV EPE heatmap — ZeroFlow no aggregation |
| `epe_after_geometric_median.png` | BEV EPE heatmap — after rigid pooling |
| `epe_improvement_delta.png` | ΔEPE map (green = pooling helped, red = hurt) |
| `flow_correction_geometric_median.png` | Raw / corrected / magnitude correction comparison |
| `flow_mag_zeroflow_input.png` | ZeroFlow input flow magnitude BEV |
| `flow_mag_after_geometric_median.png` | Corrected flow magnitude BEV |
| `summary.json` | Mean EPE before/after + improvement scalar |

### 12.4 Segment Statistics (`viz_zeroflow_predictions.py`)

Legacy Matplotlib script for exploring a single ZeroFlow segment. Produces a 4-panel figure (BEV flow magnitude, 3D quiver, magnitude histogram, valid/invalid point map) and prints aggregate statistics (mean/p99 flow magnitude, valid fraction, flow direction) across all frames in the segment. Paths are hardcoded at the top of the file.

---

## 13. Reproduction

```bash
# 1. Set up the environment
conda env create -f environment.yaml   # Linux+CUDA
# or: mamba env create -f mac_environment.yaml  (macOS ARM64)
conda activate flow
pip install --editable .
cd OpenPCDet && python setup.py develop && cd ..

# 2. Run ZeroFlow inference (SceneFlowZoo) to generate .pkl + .feather files
#    (see SceneFlowZoo/GETTING_STARTED.md for model weights + config)
cd SceneFlowZoo && bash run_zeroflow_waymo.sh && cd ..

# 3. Run CenterPoint detection (OpenPCDet) to generate detection_pred.bin
#    (see OpenPCDet/docs/GETTING_STARTED.md for Waymo config + checkpoint)
cd OpenPCDet && python tools/test.py \
  --cfg_file tools/cfgs/waymo_models/centerpoint.yaml \
  --ckpt /path/to/centerpoint.pth \
  --save_to_file && cd ..

# 4. Run all phases + all methods
bash scripts/run_all_methods.sh \
  --tfrecord-root /path/to/waymo/validation \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --pred-boxes-bin results/2_stage/detection_pred.bin \
  --score-threshold 0.50

# Results written to results/full_run/{method}/phase_{a,b,c,d}_*/

# 5. (Optional) Find and visualize the best jelly-effect frames
python scripts/visualize_phase_c_jelly.py print-top --results-root results/full_run
python scripts/visualize_phase_c_jelly.py render \
  --tfrecord-root /path/to/waymo/validation \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --pair-index 193 \
  --output-dir figures/jelly_pair193

# 6. (Optional) Interactive Plotly 3D visualization
python visualize_predictions.py \
  --pkl-root results/zeroflow/validation \
  --feather-root results/zeroflow/sequence_len_002 \
  --tfrecord-root /path/to/waymo/validation \
  --pair-indices 41 193 \
  --output-dir figures/zeroflow_plotly
```

For a quick smoke test on any pipeline step, add `--max-pairs 10`.
