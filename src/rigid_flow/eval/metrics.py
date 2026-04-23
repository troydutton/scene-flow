"""Scene flow evaluation metrics.

Computes endpoint error (EPE) broken down by motion speed, semantic class,
and foreground/background membership, plus standard accuracy/outlier metrics
and rigidity structural metrics.

Metric key reference
--------------------
epe_mean                 Global mean EPE over all N points.
epe_background           Mean EPE for background (no-box) points.
epe_foreground           Mean EPE for foreground (in-box) points.  [Rigid-EPE]
epe_static               Mean EPE for fg points with box speed < low threshold.
epe_slow                 Mean EPE for fg points with low <= speed < high.
epe_fast                 Mean EPE for fg points with speed >= high threshold.
epe_vehicle              Mean EPE for points in vehicle boxes.
epe_pedestrian           Mean EPE for points in pedestrian boxes.
epe_cyclist              Mean EPE for points in cyclist boxes.
epe_all_static           Mean EPE for ALL points with ||gt_flow||/dt < low.
epe_all_slow             Mean EPE for ALL points with low <= ||gt_flow||/dt < high.
epe_all_fast             Mean EPE for ALL points with ||gt_flow||/dt >= high.
acc_strict               % of ALL points with EPE<0.05m OR rel_err<5%.
acc_relaxed              % of ALL points with EPE<0.10m OR rel_err<10%.
out3d                    % of ALL points with EPE>0.30m OR rel_err>10%.
acc_strict_fg            acc_strict restricted to foreground points.
acc_relaxed_fg           acc_relaxed restricted to foreground points.
out3d_fg                 out3d restricted to foreground points.
acc_strict_bg            acc_strict restricted to background points.
acc_relaxed_bg           acc_relaxed restricted to background points.
out3d_bg                 out3d restricted to background points.
flow_variance_mean       Mean intra-object flow variance (trace of covariance).
dist_preservation_mean   Mean pairwise distance preservation error (ΔD) per box.
num_points               Total number of points in the frame.
num_foreground           Number of foreground (in-box) points.
num_background           Number of background points.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rigid_flow.core.types import BoundingBox


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def endpoint_error(
    predicted_flow: NDArray[np.float32],  # (N, 3)
    gt_flow: NDArray[np.float32],  # (N, 3)
) -> NDArray[np.float32]:  # (N,)
    """Per-point L2 norm of the difference between predicted and ground-truth flow."""
    return np.linalg.norm(predicted_flow - gt_flow, axis=1).astype(np.float32)


def _safe_mean(epe: NDArray[np.float32], mask: NDArray[np.bool_]) -> float:
    """Mean EPE over masked points, returning NaN when no points match."""
    if not np.any(mask):
        return float("nan")
    return float(np.mean(epe[mask]))


def _safe_pct(num_mask: NDArray[np.bool_], denom_mask: NDArray[np.bool_]) -> float:
    """Percentage of denom_mask points that also satisfy num_mask.

    Returns NaN when no denominator points exist.
    """
    n_denom = int(np.count_nonzero(denom_mask))
    if n_denom == 0:
        return float("nan")
    n_num = int(np.count_nonzero(num_mask & denom_mask))
    return 100.0 * n_num / n_denom


def _relative_error(
    epe: NDArray[np.float32],
    gt_flow: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Per-point relative error: epe / ||gt_flow||.

    When ||gt_flow|| == 0 and epe == 0, relative error is defined as 0.
    When ||gt_flow|| == 0 and epe > 0, relative error is inf.
    """
    gt_norm = np.linalg.norm(gt_flow, axis=1).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(gt_norm == 0, np.where(epe == 0, 0.0, np.inf), epe / gt_norm)
    return rel.astype(np.float32)


# ---------------------------------------------------------------------------
# Standard accuracy / outlier metrics
# ---------------------------------------------------------------------------


def accuracy_metrics(
    epe: NDArray[np.float32],
    rel_err: NDArray[np.float32],
    scope_mask: NDArray[np.bool_],
) -> dict[str, float]:
    """Compute Acc@0.05, Acc@0.10, and Out3D for a given point subset.

    Parameters
    ----------
    epe : (N,) per-point endpoint error.
    rel_err : (N,) per-point relative error.
    scope_mask : (N,) boolean mask selecting the subset of interest.

    Returns
    -------
    dict with keys ``acc_strict``, ``acc_relaxed``, ``out3d``.
    """
    acc_strict_mask = (epe < 0.05) | (rel_err < 0.05)
    acc_relaxed_mask = (epe < 0.10) | (rel_err < 0.10)
    out3d_mask = (epe > 0.30) | (rel_err > 0.10)

    return {
        "acc_strict": _safe_pct(acc_strict_mask, scope_mask),
        "acc_relaxed": _safe_pct(acc_relaxed_mask, scope_mask),
        "out3d": _safe_pct(out3d_mask, scope_mask),
    }


# ---------------------------------------------------------------------------
# Full-scene threeway EPE
# ---------------------------------------------------------------------------


def threeway_epe_fullscene(
    epe: NDArray[np.float32],
    gt_flow: NDArray[np.float32],
    dt: float,
    thresholds: tuple[float, float] = (0.5, 2.0),
) -> dict[str, float]:
    """EPE split by per-point GT speed over ALL points (including background).

    Speed is derived from ground-truth flow magnitude: speed = ||gt_flow|| / dt.
    This gives a true full-scene threeway split that includes static background
    points in the static bucket, unlike the existing fg-only speed buckets.

    Parameters
    ----------
    epe : (N,) per-point endpoint error.
    gt_flow : (N, 3) ground-truth flow vectors.
    dt : time delta in seconds between frames.
    thresholds : (low, high) speed thresholds in m/s.

    Returns
    -------
    dict with keys ``epe_all_static``, ``epe_all_slow``, ``epe_all_fast``.
    """
    low, high = thresholds
    gt_speed = np.linalg.norm(gt_flow, axis=1).astype(np.float32)
    if dt > 0:
        gt_speed = gt_speed / dt

    static_mask = gt_speed < low
    slow_mask = (gt_speed >= low) & (gt_speed < high)
    fast_mask = gt_speed >= high

    return {
        "epe_all_static": _safe_mean(epe, static_mask),
        "epe_all_slow": _safe_mean(epe, slow_mask),
        "epe_all_fast": _safe_mean(epe, fast_mask),
    }


# ---------------------------------------------------------------------------
# Rigidity metrics
# ---------------------------------------------------------------------------


def rigidity_metrics(
    predicted_flow: NDArray[np.float32],  # (N, 3)
    points: NDArray[np.float32],          # (N, 3)
    point_to_box: NDArray[np.int32],      # (N,)
    boxes: list[BoundingBox],
) -> dict[str, float]:
    """Compute structural rigidity metrics for each foreground object.

    Intra-object flow variance
        For a perfectly rigid body under pure translation every point has the
        same flow vector, so the variance is zero.  We measure
        ``trace(Cov(flow_vectors))`` per box and average across boxes.

    Pairwise distance preservation (ΔD)
        For any two points p_i, p_j on a rigid body the distance
        ||p_i - p_j|| must be preserved after warping by the predicted flow.
        We compute the mean absolute discrepancy over all pairs within each
        box and average across boxes.

    Parameters
    ----------
    predicted_flow : (N, 3) corrected flow vectors.
    points : (N, 3) point positions at t0.
    point_to_box : (N,) box index per point; -1 = background.
    boxes : list of BoundingBox objects.

    Returns
    -------
    dict with keys ``flow_variance_mean`` and ``dist_preservation_mean``.
    """
    variances: list[float] = []
    dist_errors: list[float] = []

    unique_ids = np.unique(point_to_box)

    for box_idx in unique_ids:
        if box_idx == -1:
            continue

        mask = point_to_box == box_idx
        if int(mask.sum()) < 3:
            continue

        pts = points[mask].astype(np.float64)           # (K, 3)
        flow = predicted_flow[mask].astype(np.float64)  # (K, 3)

        # -- Intra-object flow variance --
        # trace(Cov) = sum of per-axis variances
        var = float(np.sum(np.var(flow, axis=0)))
        variances.append(var)

        # -- Pairwise distance preservation --
        # D_before[i,j] = ||p_i - p_j||  (upper triangle only)
        diff_before = pts[:, None, :] - pts[None, :, :]   # (K, K, 3)
        dist_before = np.linalg.norm(diff_before, axis=-1)  # (K, K)

        pts_warped = pts + flow
        diff_after = pts_warped[:, None, :] - pts_warped[None, :, :]  # (K, K, 3)
        dist_after = np.linalg.norm(diff_after, axis=-1)               # (K, K)

        # Upper triangle to avoid double-counting
        k = pts.shape[0]
        triu_idx = np.triu_indices(k, k=1)
        delta_d = float(np.mean(np.abs(dist_before[triu_idx] - dist_after[triu_idx])))
        dist_errors.append(delta_d)

    flow_variance_mean = float(np.mean(variances)) if variances else float("nan")
    dist_preservation_mean = float(np.mean(dist_errors)) if dist_errors else float("nan")

    return {
        "flow_variance_mean": flow_variance_mean,
        "dist_preservation_mean": dist_preservation_mean,
    }


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def evaluate(
    predicted_flow: NDArray[np.float32],  # (N, 3)
    gt_flow: NDArray[np.float32],         # (N, 3)
    point_to_box: NDArray[np.int32],      # (N,) box index, -1=background
    boxes: list[BoundingBox],
    points_t0: NDArray[np.float32],       # (N, 3)
    dt: float,                            # seconds between frames
    speed_thresholds: tuple[float, float] = (0.5, 2.0),
) -> dict[str, float]:
    """Compute all evaluation metrics broken down by speed, class, fg/bg, and rigidity.

    Parameters
    ----------
    predicted_flow : (N, 3) predicted scene flow vectors.
    gt_flow : (N, 3) ground-truth scene flow vectors.
    point_to_box : (N,) index into *boxes* for each point; -1 means background.
    boxes : list of BoundingBox objects for the current frame.
    points_t0 : (N, 3) point positions at t0 (needed for rigidity metrics).
    dt : time delta in seconds (needed for full-scene threeway EPE).
    speed_thresholds : (low, high) thresholds in m/s for static/slow/fast buckets.

    Returns
    -------
    dict with keys described in the module docstring.
    """
    epe = endpoint_error(predicted_flow, gt_flow)
    rel_err = _relative_error(epe, gt_flow)
    n = len(epe)

    bg_mask = point_to_box == -1
    fg_mask = ~bg_mask
    all_mask = np.ones(n, dtype=np.bool_)

    threshold_low, threshold_high = speed_thresholds

    # Pre-compute per-point speed and class_label for foreground points.
    point_speed = np.empty(n, dtype=np.float32)
    point_class = np.empty(n, dtype=np.int32)
    point_speed[bg_mask] = np.nan
    point_class[bg_mask] = -1

    if np.any(fg_mask):
        fg_indices = point_to_box[fg_mask]
        for i, box in enumerate(boxes):
            box_mask_in_fg = fg_indices == i
            if not np.any(box_mask_in_fg):
                continue
            full_mask = np.zeros(n, dtype=np.bool_)
            full_mask[fg_mask] = box_mask_in_fg
            point_speed[full_mask] = box.speed
            point_class[full_mask] = box.class_label

    # Speed buckets (foreground only, using box speed metadata).
    static_mask = fg_mask & (point_speed < threshold_low)
    slow_mask = fg_mask & (point_speed >= threshold_low) & (point_speed < threshold_high)
    fast_mask = fg_mask & (point_speed >= threshold_high)

    # Class buckets (foreground only).
    vehicle_mask = fg_mask & (point_class == 1)
    pedestrian_mask = fg_mask & (point_class == 2)
    cyclist_mask = fg_mask & (point_class == 4)

    num_fg = int(np.count_nonzero(fg_mask))
    num_bg = int(np.count_nonzero(bg_mask))

    # -- Standard accuracy / outlier metrics --
    acc_all = accuracy_metrics(epe, rel_err, all_mask)
    acc_fg = accuracy_metrics(epe, rel_err, fg_mask)
    acc_bg = accuracy_metrics(epe, rel_err, bg_mask)

    # -- Full-scene threeway EPE (all points by GT speed) --
    threeway = threeway_epe_fullscene(epe, gt_flow, dt, speed_thresholds)

    # -- Rigidity metrics --
    rigid = rigidity_metrics(predicted_flow, points_t0, point_to_box, boxes)

    return {
        # EPE breakdowns
        "epe_mean": _safe_mean(epe, all_mask),
        "epe_background": _safe_mean(epe, bg_mask),
        "epe_foreground": _safe_mean(epe, fg_mask),
        "epe_static": _safe_mean(epe, static_mask),
        "epe_slow": _safe_mean(epe, slow_mask),
        "epe_fast": _safe_mean(epe, fast_mask),
        "epe_vehicle": _safe_mean(epe, vehicle_mask),
        "epe_pedestrian": _safe_mean(epe, pedestrian_mask),
        "epe_cyclist": _safe_mean(epe, cyclist_mask),
        # Full-scene threeway EPE
        **threeway,
        # Standard accuracy / outlier
        "acc_strict": acc_all["acc_strict"],
        "acc_relaxed": acc_all["acc_relaxed"],
        "out3d": acc_all["out3d"],
        "acc_strict_fg": acc_fg["acc_strict"],
        "acc_relaxed_fg": acc_fg["acc_relaxed"],
        "out3d_fg": acc_fg["out3d"],
        "acc_strict_bg": acc_bg["acc_strict"],
        "acc_relaxed_bg": acc_bg["acc_relaxed"],
        "out3d_bg": acc_bg["out3d"],
        # Rigidity
        **rigid,
        # Counts
        "num_points": n,
        "num_foreground": num_fg,
        "num_background": num_bg,
    }
