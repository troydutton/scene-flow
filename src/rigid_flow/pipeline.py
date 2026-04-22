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
from rigid_flow.data.waymo_parser import WaymoParser
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

    Returns
    -------
    list of per-frame metric dictionaries (same keys as :func:`evaluate`).
    """
    parser = WaymoParser(data_root)
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

        # (a) Compute ground-truth flow.
        pair = compute_gt_flow(pair)

        # (b) Assign points to boxes.
        if pair.boxes_t0:
            boxes_array = np.stack(
                [b.as_7dof for b in pair.boxes_t0], axis=0
            )  # (M, 7)
        else:
            boxes_array = np.empty((0, 7), dtype=np.float32)
        point_to_box = points_in_boxes_cpu(pair.points_t0, boxes_array)

        # (c) Rigid aggregation.
        result = compute_rigid_flow(
            pair.points_t0,
            pair.gt_flow,
            point_to_box,
            pair.boxes_t0,
            method,
        )

        # (d) Evaluate.
        metrics = evaluate(
            result.flow,
            pair.gt_flow,
            point_to_box,
            pair.boxes_t0,
        )

        # (e) Log per-frame metrics.
        logger.info(
            "  Pair %d — EPE mean=%.4f  fg=%.4f  bg=%.4f  (N=%d)",
            idx,
            metrics["epe_mean"],
            metrics["epe_foreground"],
            metrics["epe_background"],
            metrics["num_points"],
        )

        # (f) Collect.
        metrics["sequence_id"] = pair.sequence_id  # type: ignore[assignment]
        metrics["frame_index"] = pair.frame_index  # type: ignore[assignment]
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
        choices=["median", "svd"],
        help="Aggregation method (default: median).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of frame pairs to process.",
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
    )
