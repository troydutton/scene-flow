"""End-to-end rigid scene flow pipeline.

Orchestrates data loading, ground-truth flow computation, rigid aggregation,
and evaluation.  This is the only module that imports from all sub-packages.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from rigid_flow.aggregation.rigid_aggregation import compute_rigid_flow
from rigid_flow.core.types import BoundingBox, SceneFlowPair
from rigid_flow.data.pred_boxes import PredBoxIndex
from rigid_flow.data.waymo_parser import WaymoParser
from rigid_flow.data.zeroflow_loader import ZeroFlowDataSource
from rigid_flow.eval.metrics import evaluate
from rigid_flow.geometry.points_in_boxes import points_in_boxes_cpu
from rigid_flow.geometry.se3 import SE3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth flow computation
# ---------------------------------------------------------------------------


def compute_gt_flow(pair: SceneFlowPair) -> SceneFlowPair:
    """Compute ground-truth scene flow for every point in ``pair.points_t0``.

    The flow is defined in the ego frame at t0: adding the flow vector to a
    point in ``points_t0`` yields where that point moved to (still expressed
    in the t0 ego frame) at time t1.

    Algorithm (fully vectorized):
        1. Transform all points to the global frame via ``ego_pose_t0``.
        2. Add per-object velocity * dt for foreground points (background
           points receive zero object motion).
        3. Transform all points back to the ego-t0 frame.
        4. Flow = displaced position - original position.

    Parameters
    ----------
    pair:
        A :class:`SceneFlowPair` with ``gt_flow`` either ``None`` or to be
        overwritten.

    Returns
    -------
    SceneFlowPair
        A **new** (frozen dataclass) instance identical to *pair* but with
        ``gt_flow`` populated as an ``(N, 3)`` float32 array.
    """
    points = pair.points_t0  # (N, 3)
    N = points.shape[0]
    dt = pair.dt

    # -- Step 1: assign points to boxes --
    if pair.boxes_t0:
        boxes_array = np.stack([b.as_7dof for b in pair.boxes_t0], axis=0)  # (M, 7)
    else:
        boxes_array = np.empty((0, 7), dtype=np.float32)

    point_to_box = points_in_boxes_cpu(points, boxes_array)  # (N,)

    # -- Step 2: transform all points to global frame --
    ego_t0 = SE3(pair.ego_pose_t0)
    points_global = ego_t0.transform_points(points)  # (N, 3) float32

    # -- Step 3: build per-point object motion in global frame --
    object_motion = np.zeros((N, 3), dtype=np.float32)

    for k, box in enumerate(pair.boxes_t0):
        mask = point_to_box == k
        if not np.any(mask):
            continue
        if box.velocity is not None:
            vx, vy = box.velocity
            object_motion[mask, 0] = vx * dt
            object_motion[mask, 1] = vy * dt
            # vz assumed 0 → object_motion[:, 2] stays 0

    # Add object motion in global frame.
    points_global_displaced = points_global + object_motion  # (N, 3)

    # -- Step 4: transform back to ego-t0 frame --
    # Flow convention: physical displacement in ego_t0 coordinates.
    # For static background (zero object motion) this yields flow = 0.
    ego_t0_inv = ego_t0.inverse()
    points_prime = ego_t0_inv.transform_points(points_global_displaced)  # (N, 3)

    # -- Step 5: flow = displaced - original --
    gt_flow = (points_prime - points).astype(np.float32)

    return replace(pair, gt_flow=gt_flow)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    data_root: Path,
    output_dir: Path | None = None,
    method: str = "median",
    max_pairs: int | None = None,
    pred_boxes_bin: Path | None = None,
    score_threshold: float = 0.0,
) -> list[dict[str, float]]:
    """Run the full rigid scene flow evaluation pipeline.

    Parameters
    ----------
    data_root:
        Directory containing Waymo ``.tfrecord`` files.
    output_dir:
        If provided, per-frame and aggregate results are saved as JSON here.
    method:
        Aggregation method passed to :func:`compute_rigid_flow`
        (``"median"`` or ``"svd"``).
    max_pairs:
        If set, stop after processing this many frame pairs.
    pred_boxes_bin:
        Optional path to a Waymo detection-submission ``.bin`` file.  When
        supplied, the boxes used for rigid aggregation and evaluation are
        looked up from this file by ``(sequence_id, timestamp_us_t0)`` instead
        of using ``pair.boxes_t0``.  Ground-truth flow is still computed from
        the GT boxes in the tfrecord — only the refinement/evaluation boxes
        change.
    score_threshold:
        Minimum ``Object.score`` to accept when selecting predicted boxes.
        Ignored when ``pred_boxes_bin`` is ``None``.

    Returns
    -------
    list of per-frame metric dictionaries (same keys as :func:`evaluate`).
    """
    parser = WaymoParser(data_root)
    pred_index = PredBoxIndex(pred_boxes_bin) if pred_boxes_bin is not None else None
    if pred_index is not None:
        logger.info(
            "Using predicted boxes from %s (score_threshold=%.2f)",
            pred_boxes_bin,
            score_threshold,
        )
    all_metrics: list[dict[str, float]] = []

    for idx, pair in enumerate(parser.iterate_pairs()):
        if max_pairs is not None and idx >= max_pairs:
            break

        logger.info(
            "Processing pair %d  (seq=%s, frame=%d, N=%d)",
            idx,
            pair.sequence_id,
            pair.frame_index,
            pair.num_points_t0,
        )

        # (a) Compute ground-truth flow (always uses GT boxes from tfrecord).
        pair = compute_gt_flow(pair)

        # (b) GT box assignments — needed for epe_true_foreground regardless of mode.
        if pair.boxes_t0:
            gt_boxes_array = np.stack([b.as_7dof for b in pair.boxes_t0], axis=0)
        else:
            gt_boxes_array = np.empty((0, 7), dtype=np.float32)
        gt_point_to_box = points_in_boxes_cpu(pair.points_t0, gt_boxes_array)

        # (c) Pick the boxes used for rigid refinement + evaluation.
        if pred_index is not None:
            boxes_to_use: list[BoundingBox] = pred_index.get(
                pair.sequence_id, pair.timestamp_us_t0, score_threshold
            )
        else:
            boxes_to_use = list(pair.boxes_t0)

        # (d) Assign points to the chosen (predicted or GT) boxes.
        if boxes_to_use:
            boxes_array = np.stack([b.as_7dof for b in boxes_to_use], axis=0)  # (M, 7)
        else:
            boxes_array = np.empty((0, 7), dtype=np.float32)
        point_to_box = points_in_boxes_cpu(pair.points_t0, boxes_array)

        # (e) Rigid aggregation.
        result = compute_rigid_flow(
            pair.points_t0,
            pair.gt_flow,
            point_to_box,
            boxes_to_use,
            method,
        )

        # (f) Evaluate.  Pass gt_point_to_box so epe_true_foreground is always
        # computed against GT box membership regardless of the box source.
        metrics = evaluate(
            result.flow,
            pair.gt_flow,
            point_to_box,
            boxes_to_use,
            pair.points_t0,
            pair.dt,
            gt_point_to_box=gt_point_to_box,
        )

        # (g) Log per-frame metrics.
        logger.info(
            "  Pair %d — EPE mean=%.4f  fg=%.4f  true_fg=%.4f  bg=%.4f  "
            "acc_strict=%.1f%%  acc_relaxed=%.1f%%  out3d=%.1f%%  "
            "flow_var=%.6f  (N=%d, pred_boxes=%d, gt_boxes=%d)",
            idx,
            metrics["epe_mean"],
            metrics["epe_foreground"],
            metrics["epe_true_foreground"],
            metrics["epe_background"],
            metrics["acc_strict"],
            metrics["acc_relaxed"],
            metrics["out3d"],
            metrics["flow_variance_mean"],
            metrics["num_points"],
            len(boxes_to_use),
            len(pair.boxes_t0),
        )

        # (h) Collect.
        metrics["sequence_id"] = pair.sequence_id  # type: ignore[assignment]
        metrics["frame_index"] = pair.frame_index  # type: ignore[assignment]
        metrics["num_gt_boxes"] = int(len(pair.boxes_t0))
        metrics["num_pred_boxes"] = int(len(boxes_to_use)) if pred_index is not None else -1
        metrics["score_threshold"] = float(score_threshold) if pred_index is not None else float("nan")
        all_metrics.append(metrics)

    # -- Aggregate metrics across all frames --
    if all_metrics:
        numeric_keys = [
            k
            for k in all_metrics[0]
            if isinstance(all_metrics[0][k], (int, float))
        ]
        aggregate: dict[str, float] = {}
        for key in numeric_keys:
            values = [
                m[key]
                for m in all_metrics
                if not (isinstance(m[key], float) and math.isnan(m[key]))
            ]
            if values:
                aggregate[f"avg_{key}"] = float(np.mean(values))

        aggregate["total_frames"] = float(len(all_metrics))
        logger.info("=== Aggregate metrics over %d frames ===", len(all_metrics))
        for k, v in sorted(aggregate.items()):
            logger.info("  %s: %.4f", k, v)
    else:
        aggregate = {}
        logger.warning("No frame pairs were processed.")

    # -- Optionally save results --
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        per_frame_path = output_dir / "per_frame_metrics.json"
        with open(per_frame_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info("Per-frame metrics saved to %s", per_frame_path)

        aggregate_path = output_dir / "aggregate_metrics.json"
        with open(aggregate_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        logger.info("Aggregate metrics saved to %s", aggregate_path)

    return all_metrics


# ---------------------------------------------------------------------------
# ZeroFlow predicted-flow pipeline
# ---------------------------------------------------------------------------


def run_zeroflow_pipeline(
    tfrecord_root: Path,
    pkl_root: Path,
    feather_root: Path,
    output_dir: Path | None = None,
    method: str = "median",
    max_pairs: int | None = None,
    pred_boxes_bin: Path | None = None,
    score_threshold: float = 0.0,
) -> list[dict[str, float]]:
    """Run rigid scene flow evaluation using ZeroFlow predicted flow.

    Unlike :func:`run_pipeline` which uses GT flow as the refinement input,
    this function feeds **ZeroFlow predictions** into ``compute_rigid_flow``
    and evaluates the rigidified output against GT flow derived from tfrecord
    bounding box velocities.

    The point clouds come from the pkl files (ground-removed, subsampled)
    rather than from the raw tfrecord range images.

    Parameters
    ----------
    tfrecord_root:
        Directory containing Waymo ``.tfrecord`` files (for boxes / timestamps).
    pkl_root:
        Directory with per-segment subdirectories of ``{idx:06d}.pkl`` files.
    feather_root:
        Directory with per-segment subdirectories of ``{idx:010d}.feather`` files.
    output_dir:
        If provided, per-frame and aggregate results are saved as JSON here.
    method:
        Aggregation method (``"median"`` or ``"svd"``).
    max_pairs:
        Stop after this many frame pairs.
    pred_boxes_bin:
        Optional detection ``.bin`` file for Phase D (predicted boxes).
    score_threshold:
        Minimum detection score for predicted boxes.
    """
    source = ZeroFlowDataSource(pkl_root, feather_root, tfrecord_root)
    pred_index = PredBoxIndex(pred_boxes_bin) if pred_boxes_bin is not None else None
    if pred_index is not None:
        logger.info(
            "Using predicted boxes from %s (score_threshold=%.2f)",
            pred_boxes_bin,
            score_threshold,
        )

    all_metrics: list[dict[str, float]] = []

    for idx, (pair, pred_flow, is_valid) in enumerate(source.iterate_pairs()):
        if max_pairs is not None and idx >= max_pairs:
            break

        logger.info(
            "Processing pair %d  (seq=%s, frame=%d, N=%d, valid=%d)",
            idx,
            pair.sequence_id,
            pair.frame_index,
            pair.num_points_t0,
            int(is_valid.sum()),
        )

        # (a) Compute GT flow on the pkl point cloud.
        pair = compute_gt_flow(pair)

        # (b) Build the flow input: use ZeroFlow where valid, zero elsewhere.
        flow_input = np.zeros_like(pred_flow)
        flow_input[is_valid] = pred_flow[is_valid]

        # (c) GT box assignments for epe_true_foreground.
        if pair.boxes_t0:
            gt_boxes_array = np.stack([b.as_7dof for b in pair.boxes_t0], axis=0)
        else:
            gt_boxes_array = np.empty((0, 7), dtype=np.float32)
        gt_point_to_box = points_in_boxes_cpu(pair.points_t0, gt_boxes_array)

        # (d) Pick boxes for rigid refinement + evaluation.
        if pred_index is not None:
            boxes_to_use: list[BoundingBox] = pred_index.get(
                pair.sequence_id, pair.timestamp_us_t0, score_threshold
            )
        else:
            boxes_to_use = list(pair.boxes_t0)

        # (e) Assign points to the chosen boxes.
        if boxes_to_use:
            boxes_array = np.stack([b.as_7dof for b in boxes_to_use], axis=0)
        else:
            boxes_array = np.empty((0, 7), dtype=np.float32)
        point_to_box = points_in_boxes_cpu(pair.points_t0, boxes_array)

        # (f) Rigid aggregation with predicted flow as input (not GT).
        result = compute_rigid_flow(
            pair.points_t0,
            flow_input,
            point_to_box,
            boxes_to_use,
            method,
        )

        # (g) Evaluate rigidified predicted flow against GT flow.
        metrics = evaluate(
            result.flow,
            pair.gt_flow,
            point_to_box,
            boxes_to_use,
            pair.points_t0,
            pair.dt,
            gt_point_to_box=gt_point_to_box,
        )

        logger.info(
            "  Pair %d — EPE mean=%.4f  fg=%.4f  true_fg=%.4f  bg=%.4f  "
            "acc_strict=%.1f%%  acc_relaxed=%.1f%%  out3d=%.1f%%  "
            "flow_var=%.6f  (N=%d, pred_boxes=%d, gt_boxes=%d)",
            idx,
            metrics["epe_mean"],
            metrics["epe_foreground"],
            metrics["epe_true_foreground"],
            metrics["epe_background"],
            metrics["acc_strict"],
            metrics["acc_relaxed"],
            metrics["out3d"],
            metrics["flow_variance_mean"],
            metrics["num_points"],
            len(boxes_to_use),
            len(pair.boxes_t0),
        )

        # (h) Collect.
        metrics["sequence_id"] = pair.sequence_id  # type: ignore[assignment]
        metrics["frame_index"] = pair.frame_index  # type: ignore[assignment]
        metrics["num_gt_boxes"] = int(len(pair.boxes_t0))
        metrics["num_pred_boxes"] = int(len(boxes_to_use)) if pred_index is not None else -1
        metrics["score_threshold"] = float(score_threshold) if pred_index is not None else float("nan")
        all_metrics.append(metrics)

    # -- Aggregate --
    if all_metrics:
        numeric_keys = [
            k
            for k in all_metrics[0]
            if isinstance(all_metrics[0][k], (int, float))
        ]
        aggregate: dict[str, float] = {}
        for key in numeric_keys:
            values = [
                m[key]
                for m in all_metrics
                if not (isinstance(m[key], float) and math.isnan(m[key]))
            ]
            if values:
                aggregate[f"avg_{key}"] = float(np.mean(values))

        aggregate["total_frames"] = float(len(all_metrics))
        logger.info("=== Aggregate metrics over %d frames ===", len(all_metrics))
        for k, v in sorted(aggregate.items()):
            logger.info("  %s: %.4f", k, v)
    else:
        aggregate = {}
        logger.warning("No frame pairs were processed.")

    # -- Save --
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        per_frame_path = output_dir / "per_frame_metrics.json"
        with open(per_frame_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        logger.info("Per-frame metrics saved to %s", per_frame_path)

        aggregate_path = output_dir / "aggregate_metrics.json"
        with open(aggregate_path, "w") as f:
            json.dump(aggregate, f, indent=2)
        logger.info("Aggregate metrics saved to %s", aggregate_path)

    return all_metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rigid scene flow evaluation pipeline for Waymo Open Dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to directory containing .tfrecord files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save JSON result files.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="median",
        choices=["none", "mean", "median", "weighted_median", "geometric_median", "svd"],
        help="Aggregation method (default: median).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of frame pairs to process.",
    )
    parser.add_argument(
        "--pred-boxes-bin",
        type=Path,
        default=None,
        help="Optional Waymo detection-submission .bin file; when given, its "
        "boxes replace GT boxes for rigid aggregation + evaluation.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Score threshold applied to predicted boxes (ignored without "
        "--pred-boxes-bin).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_pipeline(
        data_root=args.data_root,
        output_dir=args.output_dir,
        method=args.method,
        max_pairs=args.max_pairs,
        pred_boxes_bin=args.pred_boxes_bin,
        score_threshold=args.score_threshold,
    )
