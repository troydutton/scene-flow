"""Scene flow evaluation metrics.

Computes endpoint error (EPE) broken down by motion speed, semantic class,
and foreground/background membership.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rigid_flow.core.types import BoundingBox


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


def evaluate(
    predicted_flow: NDArray[np.float32],  # (N, 3)
    gt_flow: NDArray[np.float32],  # (N, 3)
    point_to_box: NDArray[np.int32],  # (N,) box index, -1=background
    boxes: list[BoundingBox],
    speed_thresholds: tuple[float, float] = (0.5, 2.0),
) -> dict[str, float]:
    """Compute evaluation metrics broken down by speed, class, and fg/bg.

    Parameters
    ----------
    predicted_flow : (N, 3) predicted scene flow vectors.
    gt_flow : (N, 3) ground-truth scene flow vectors.
    point_to_box : (N,) index into *boxes* for each point; -1 means background.
    boxes : list of BoundingBox objects for the current frame.
    speed_thresholds : (low, high) thresholds in m/s for static/slow/fast buckets.

    Returns
    -------
    dict with keys described in the module docstring.
    """
    epe = endpoint_error(predicted_flow, gt_flow)
    n = len(epe)

    bg_mask = point_to_box == -1
    fg_mask = ~bg_mask

    threshold_low, threshold_high = speed_thresholds

    # Pre-compute per-point speed and class_label for foreground points.
    point_speed = np.empty(n, dtype=np.float32)
    point_class = np.empty(n, dtype=np.int32)
    # Background points get values that won't match any foreground bucket.
    point_speed[bg_mask] = np.nan
    point_class[bg_mask] = -1

    if np.any(fg_mask):
        fg_indices = point_to_box[fg_mask]
        for i, box in enumerate(boxes):
            box_mask_in_fg = fg_indices == i
            if not np.any(box_mask_in_fg):
                continue
            # Expand back to full array
            full_mask = np.zeros(n, dtype=np.bool_)
            full_mask[fg_mask] = box_mask_in_fg
            point_speed[full_mask] = box.speed
            point_class[full_mask] = box.class_label

    # Speed buckets (foreground only).
    static_mask = fg_mask & (point_speed < threshold_low)
    slow_mask = fg_mask & (point_speed >= threshold_low) & (point_speed < threshold_high)
    fast_mask = fg_mask & (point_speed >= threshold_high)

    # Class buckets (foreground only).
    vehicle_mask = fg_mask & (point_class == 1)
    pedestrian_mask = fg_mask & (point_class == 2)
    cyclist_mask = fg_mask & (point_class == 4)

    num_fg = int(np.count_nonzero(fg_mask))
    num_bg = int(np.count_nonzero(bg_mask))

    return {
        "epe_mean": _safe_mean(epe, np.ones(n, dtype=np.bool_)),
        "epe_background": _safe_mean(epe, bg_mask),
        "epe_foreground": _safe_mean(epe, fg_mask),
        "epe_static": _safe_mean(epe, static_mask),
        "epe_slow": _safe_mean(epe, slow_mask),
        "epe_fast": _safe_mean(epe, fast_mask),
        "epe_vehicle": _safe_mean(epe, vehicle_mask),
        "epe_pedestrian": _safe_mean(epe, pedestrian_mask),
        "epe_cyclist": _safe_mean(epe, cyclist_mask),
        "num_points": n,
        "num_foreground": num_fg,
        "num_background": num_bg,
    }
