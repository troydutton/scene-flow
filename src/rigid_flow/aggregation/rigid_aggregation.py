"""Rigid aggregation module for scene flow correction.

Replaces noisy per-point flow vectors with rigid-body estimates for points
that belong to tracked bounding boxes. Background points keep their raw flow.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rigid_flow.core.types import BoundingBox, FlowResult


def fit_rigid_transform_svd(
    source: NDArray[np.float32],
    target: NDArray[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the optimal rigid transform (R, t) from source to target.

    Uses the Kabsch (Procrustes) algorithm via SVD.

    Parameters
    ----------
    source : (K, 3) array of source points.
    target : (K, 3) array of corresponding target points.

    Returns
    -------
    R : (3, 3) rotation matrix.
    t : (3,) translation vector such that target ≈ (R @ source.T).T + t.
    """
    centroid_s = source.mean(axis=0).astype(np.float64)
    centroid_t = target.mean(axis=0).astype(np.float64)

    src_c = source.astype(np.float64) - centroid_s
    tgt_c = target.astype(np.float64) - centroid_t

    H = src_c.T @ tgt_c  # (3, 3) cross-covariance

    U, _S, Vt = np.linalg.svd(H)

    # Assemble rotation, handling possible reflection
    V = Vt.T
    d = np.linalg.det(V @ U.T)
    if d < 0:
        V[:, -1] *= -1

    R = V @ U.T
    t = centroid_t - R @ centroid_s

    return R, t


def compute_rigid_flow(
    points: NDArray[np.float32],
    raw_flow: NDArray[np.float32],
    point_to_box: NDArray[np.int32],
    boxes: list[BoundingBox],
    method: str = "median",
) -> FlowResult:
    """Replace per-point flow with rigid-body estimates for each object.

    Parameters
    ----------
    points : (N, 3) point positions at t0.
    raw_flow : (N, 3) per-point flow vectors (potentially noisy).
    point_to_box : (N,) box assignment per point; -1 means background.
    boxes : list of bounding boxes at t0.
    method : ``"median"`` for median-translation aggregation, or ``"svd"``
        for full rigid (rotation + translation) fitting via Kabsch.

    Returns
    -------
    FlowResult with corrected flow, raw flow copy, and per-object metadata.
    """
    n = points.shape[0]
    corrected_flow = raw_flow.copy()
    is_rigid = np.zeros(n, dtype=np.bool_)
    per_object_translation: dict[str, NDArray[np.float32]] = {}

    unique_ids = np.unique(point_to_box)

    for box_idx in unique_ids:
        if box_idx == -1:
            continue

        mask = point_to_box == box_idx
        count = int(mask.sum())
        if count < 3:
            continue

        pts = points[mask]          # (K, 3)
        flow = raw_flow[mask]       # (K, 3)

        if method == "median":
            translation = np.median(flow, axis=0).astype(np.float32)  # (3,)
            corrected_flow[mask] = translation
            per_object_translation[boxes[box_idx].tracking_id] = translation

        elif method == "svd":
            source = pts
            target = pts + flow
            R, t = fit_rigid_transform_svd(source, target)
            # Corrected flow: (R @ p + t) - p for each point p
            fitted_target = (R @ pts.astype(np.float64).T).T + t  # (K, 3)
            corrected_flow[mask] = (fitted_target - pts.astype(np.float64)).astype(np.float32)
            per_object_translation[boxes[box_idx].tracking_id] = t.astype(np.float32)

        else:
            raise ValueError(f"Unknown method '{method}'. Expected 'median' or 'svd'.")

        is_rigid[mask] = True

    return FlowResult(
        flow=corrected_flow,
        raw_flow=raw_flow.copy(),
        point_to_box=point_to_box,
        is_rigid=is_rigid,
        per_object_translation=per_object_translation,
    )
