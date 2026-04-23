# Object-Aware 3D Scene Flow: Rigid-Refinement Pipeline

## Context

Naive deep learning scene flow models treat LiDAR points independently, causing rigid objects (cars) to appear to deform ("jelly effect"). This pipeline applies a deterministic post-processing step: median-pool flow vectors within each bounding box to enforce rigid motion. We validate incrementally by swapping GT vs predicted components across four phases.

---

## Project Structure

All new code under `src/rigid_flow/`, importable via `pip install -e .` (pyproject.toml already configured for `src/` layout).

```
src/
  rigid_flow/
    __init__.py
    core/
      __init__.py
      types.py                 # Dataclasses: SceneFlowPair, FlowResult, BoxResult, RefinementResult
      ego_compensation.py      # Ego-motion removal using pose matrices
      rigid_refine.py          # scatter_median rigid aggregation (THE core algorithm)
    data/
      __init__.py
      waymo_adapter.py         # Wraps OpenPCDet Waymo infos into SceneFlowPair pairs
      flow_sources.py          # FlowSource protocol + GTFlowSource + DeFlowSource
      box_sources.py           # BoxSource protocol + GTBoxSource + CenterPointBoxSource
    eval/
      __init__.py
      metrics.py               # EPE computation: vanilla vs refined, per-class breakdowns
    pipeline/
      __init__.py
      runner.py                # Hydra entrypoint: data → flow → boxes → refine → eval → W&B
    utils/
      __init__.py
      distributed.py           # DDP setup/teardown helpers

configs/
  config.yaml                  # Hydra root (defaults list)
  phase/
    a_gt_gt.yaml               # Phase A: GT Flow + GT Boxes
    b_gt_centerpoint.yaml      # Phase B: GT Flow + CenterPoint Boxes
    c_deflow_gt.yaml           # Phase C: DeFlow Flow + GT Boxes
    d_deflow_centerpoint.yaml  # Phase D: DeFlow Flow + CenterPoint Boxes
  data/
    waymo.yaml                 # Dataset paths, split, class names
  model/
    deflow.yaml                # DeFlow checkpoint + SceneFlowZoo config path
    centerpoint.yaml           # CenterPoint checkpoint + OpenPCDet config path
  eval/
    default.yaml               # W&B project name, metric settings
```

---

## Implementation Steps

### Step 0: Environment & Dependencies

**Files:** `environment.yaml`, `scripts/setup.sh`, `pyproject.toml`

- Add missing pip deps to `environment.yaml`: `torch_scatter`, `pytorch-lightning`, `mmengine`, `omegaconf`
- Clone SceneFlowZoo back into repo: `git clone https://github.com/kylevedder/SceneFlowZoo`
- Download CenterPoint config YAMLs into `OpenPCDet/tools/cfgs/` (currently missing from clone)
- Update `scripts/setup.sh` to install SceneFlowZoo in dev mode
- Verify `python setup.py develop` works for OpenPCDet CUDA ops

### Step 1: Core Data Types

**File:** `src/rigid_flow/core/types.py`

```python
@dataclass
class SceneFlowPair:
    pc_t0: torch.Tensor          # (N, 3+) points at t
    pc_t1: torch.Tensor          # (M, 3+) points at t+1
    pose_t0: torch.Tensor        # (4, 4) ego→global at t
    pose_t1: torch.Tensor        # (4, 4) ego→global at t+1
    gt_boxes: torch.Tensor       # (K, 9) GT boxes with velocities [x,y,z,dx,dy,dz,heading,vx,vy]
    gt_labels: torch.Tensor      # (K,) class labels
    sequence_name: str
    sample_idx: int

@dataclass
class FlowResult:
    flow: torch.Tensor            # (N, 3)
    valid_mask: torch.Tensor      # (N,) bool

@dataclass
class BoxResult:
    boxes: torch.Tensor           # (K, 7) [x,y,z,dx,dy,dz,heading]
    scores: torch.Tensor          # (K,)
    labels: torch.Tensor          # (K,)

@dataclass
class RefinementResult:
    vanilla_flow: torch.Tensor    # (N, 3)
    refined_flow: torch.Tensor    # (N, 3)
    point_to_box: torch.Tensor    # (N,) box idx, -1=background
    box_median_flow: torch.Tensor # (K, 3)
```

### Step 2: Waymo Data Adapter

**File:** `src/rigid_flow/data/waymo_adapter.py`

- Read OpenPCDet's waymo info pickle files directly (same format as `WaymoDataset`)
- Load `.npy` point clouds via same logic as `WaymoDataset.get_lidar()` (lines 196-208): filter NLZ, tanh intensity
- Extract ego poses from `info['pose']` (4×4 matrix)
- Extract GT boxes from `info['annos']['gt_boxes_lidar']` (M, 9 with velocities)
- Build consecutive-pair index: each `__getitem__` returns a `SceneFlowPair` (frame t, frame t+1)
- **Key decision**: Bypass `DatasetTemplate.prepare_data()` entirely — no voxelization or augmentation at dataset level

### Step 3: Ego-Motion Compensation

**File:** `src/rigid_flow/core/ego_compensation.py`

```python
def compensate_ego_motion(points_t0, flow, pose_t0, pose_t1) -> torch.Tensor:
    # T_ego = pose_t1^{-1} @ pose_t0
    # ego_flow_at_p = T_ego @ p - p
    # object_flow = total_flow - ego_flow
```

Follows same transform pattern as `WaymoDataset.get_sequence_data` lines 301-304.

### Step 4: Rigid Refinement (Core Algorithm)

**File:** `src/rigid_flow/core/rigid_refine.py`

1. **Point-to-box assignment**: Use `points_in_boxes_gpu` from `OpenPCDet/pcdet/ops/roiaware_pool3d/roiaware_pool3d_utils.py`
   - Input: points `(1, N, 3)`, boxes `(1, K, 7)` → output: `(1, N)` with box index per point (-1 = background)
2. **Median flow per box**: `torch_scatter.scatter_median(in_box_flow, box_indices, dim=0, dim_size=K)` → `(K, 3)`
3. **Broadcast back**: Replace each in-box point's flow with its box's median flow
4. Background points keep original flow unchanged

### Step 5: Flow Sources

**File:** `src/rigid_flow/data/flow_sources.py`

- `FlowSource` protocol with `predict(pair: SceneFlowPair) -> FlowResult`
- **`GTFlowSource`**: Compute per-point GT flow from box velocities:
  - Assign points to GT boxes via `points_in_boxes_gpu`
  - For point in box b: `flow = [vx*dt, vy*dt, 0]` (dt=0.1s)
  - Background points: zero flow (static after ego compensation)
- **`DeFlowSource`**: Wrap SceneFlowZoo's DeFlow model
  - Convert `SceneFlowPair` → `TorchFullFrameInputSequence`
  - Run inference, extract `ego_flows` from output
  - Requires SceneFlowZoo to be cloned and importable

### Step 6: Box Sources

**File:** `src/rigid_flow/data/box_sources.py`

- `BoxSource` protocol with `predict(pair: SceneFlowPair) -> BoxResult`
- **`GTBoxSource`**: Return `pair.gt_boxes[:, :7]` and `pair.gt_labels` directly
- **`CenterPointBoxSource`**: Run OpenPCDet CenterPoint inference
  - Load model via `build_network()` + checkpoint
  - Create minimal `DataProcessor` for voxelization (no augmentation)
  - Forward pass returns `pred_boxes`, `pred_scores`, `pred_labels`
  - **Note**: Needs CenterPoint config YAML (currently missing from `OpenPCDet/tools/cfgs/`)

### Step 7: Evaluation Metrics

**File:** `src/rigid_flow/eval/metrics.py`

- `compute_epe(pred, gt, mask)` → mean/median EPE on masked points
- `compute_rigid_epe(refinement, gt_flow, valid_mask)` → vanilla vs refined EPE on in-box points + improvement %
- Per-class breakdown (Vehicle, Pedestrian, Cyclist) using box labels

### Step 8: Pipeline Runner + Hydra Configs

**File:** `src/rigid_flow/pipeline/runner.py`

Hydra entrypoint that:
1. Initializes DDP (`torch.distributed`)
2. Initializes W&B (rank 0 only)
3. Builds dataset + `DistributedSampler` + `DataLoader`
4. Instantiates `FlowSource` and `BoxSource` via `hydra.utils.instantiate` (config-driven)
5. Loops over frame pairs: flow → ego_comp → boxes → rigid_refine → eval → W&B log
6. Aggregates final metrics across all samples

**Hydra configs**: Each phase YAML sets `_target_` for flow_source and box_source, enabling `hydra.utils.instantiate` to build the correct classes.

### Step 9: DDP + W&B Integration

**File:** `src/rigid_flow/utils/distributed.py`

- `setup_distributed()` / `cleanup_distributed()` using `torch.distributed`
- `DistributedSampler` for data splitting across 8 GPUs
- W&B logging on rank 0 only; `dist.reduce` for metric aggregation

---

## Critical Files to Modify/Reference

| File | Role |
|------|------|
| `OpenPCDet/pcdet/datasets/waymo/waymo_dataset.py` | Reference for data loading patterns (get_lidar, info dict, poses) |
| `OpenPCDet/pcdet/ops/roiaware_pool3d/roiaware_pool3d_utils.py` | `points_in_boxes_gpu` — core point-to-box assignment |
| `OpenPCDet/pcdet/models/detectors/centerpoint.py` | CenterPoint inference interface |
| `OpenPCDet/pcdet/models/detectors/detector3d_template.py` | Model building + post_processing pipeline |
| `environment.yaml` | Add torch_scatter, pytorch-lightning, mmengine |
| `pyproject.toml` | Verify src/ layout discovery works for rigid_flow |

---

## Verification Plan

1. **Unit test rigid_refine**: Synthetic points + boxes, verify median broadcast is correct
2. **Phase A end-to-end**: GT flow + GT boxes on a small Waymo subset — refined EPE should equal vanilla EPE (GT flow is already perfectly rigid per-box)
3. **Phase B**: Swap to CenterPoint boxes — slight EPE degradation expected from box mismatch
4. **Phase C**: Swap to DeFlow flow — significant improvement expected (jelly → rigid)
5. **Phase D**: Both predicted — real-world performance
6. **DDP check**: Run on 2+ GPUs, verify metrics match single-GPU
7. **W&B dashboard**: Confirm all metrics, phase comparisons, and per-class breakdowns appear

---

## Known Risks

- **Missing CenterPoint configs**: `OpenPCDet/tools/cfgs/` is empty — need to download waymo CenterPoint YAML configs from OpenPCDet releases
- **torch_scatter compatibility**: Must match PyTorch 2.11.0 + CUDA 12.6 exactly
- **OpenPCDet CUDA ops**: `points_in_boxes_gpu` requires compiled CUDA extension (`python setup.py develop`)
- **SceneFlowZoo → Waymo**: DeFlow was primarily trained on Argoverse 2; Waymo inference may need config adjustments for point cloud range/voxel size
