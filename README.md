# scene-flow

Object-aware scene flow prediction. This repository includes the historical ZeroFlow stack under `zeroflow/` (see `CLAUDE.md`) and a newer **`rigid_flow`** package under `src/rigid_flow/` for Waymo-centric rigid refinement and evaluation.

## rigid_flow (Waymo)

`rigid_flow` loads consecutive Waymo LiDAR frames from `.tfrecord` files, builds **ground-truth scene flow** from box velocities, optionally **median- or SVD-pools** flow inside 3D boxes, and writes **per-frame and aggregate metrics** (EPE, accuracy bands, rigidity, and related breakdowns).

### Install

From the repository root, install the package in editable mode so `rigid_flow` is importable:

```bash
pip install -e .
```

Dependencies are not fully pinned in `pyproject.toml`; use `environment.yaml` (conda) plus any Waymo/protobuf packages you need for your setup. Parsing tfrecords uses vendored protobufs under `src/rigid_flow/data/waymo_protos/`; loading a Waymo **detection submission** `.bin` may require a compatible `waymo-open-dataset` / `protobuf` install if your generated `metrics_pb2` imports `waymo_open_dataset`.

### Phase A — GT flow, GT boxes

Rigid refinement and metrics use **the same** ground-truth boxes as in the tfrecord. Omit `--pred-boxes-bin`.

```bash
python -m rigid_flow.pipeline \
  --data-root /path/to/waymo/tfrecords \
  --output-dir results/phase_a_gt_flow_gt_boxes \
  --method median
```

Use `--max-pairs N` for a short debug run.

### Phase B — GT flow, predicted boxes

Pass a Waymo **detection** `.bin` (serialized `Objects` predictions). GT flow is still computed from tfrecord labels; only box assignment and pooling use predictions.

```bash
python -m rigid_flow.pipeline \
  --data-root /path/to/waymo/tfrecords \
  --output-dir results/phase_b_gt_flow_pred_boxes/thr_0.50 \
  --pred-boxes-bin /path/to/detection_pred.bin \
  --score-threshold 0.5 \
  --method median
```

### Score-threshold sweep (Phase B)

`scripts/run_pred_box_sweep.py` runs the pipeline at several confidence thresholds, writes one subdirectory per threshold, and a comparison JSON against a Phase A aggregate.

```bash
python scripts/run_pred_box_sweep.py \
  --data-root /path/to/waymo/tfrecords \
  --pred-boxes-bin results/2_stage/detection_pred.bin \
  --output-root results/phase_b_gt_flow_pred_boxes \
  --gt-aggregate results/phase_a_gt_flow_gt_boxes/aggregate_metrics.json \
  --thresholds 0.1 0.3 0.5
```

Outputs:

- `results/phase_b_gt_flow_pred_boxes/thr_X.XX/aggregate_metrics.json`
- `results/phase_b_gt_flow_pred_boxes/compare_vs_gt.json`

### Metrics reference

See the module docstring in `src/rigid_flow/eval/metrics.py` for metric key definitions (EPE splits, accuracy / outlier rates, true-foreground vs predicted-foreground, rigidity statistics). Aggregate JSON files prefix frame-level keys with `avg_`.

### Design notes

Higher-level pipeline phases and file layout are described in `resilient-fluttering-octopus.md` (working design doc; some paths there refer to a planned Hydra layout; the runnable entrypoint is `python -m rigid_flow.pipeline` as above).

## Legacy ZeroFlow

Training and evaluation for the original ZeroFlow system live under `zeroflow/`; see `CLAUDE.md` for commands and Docker workflows.
