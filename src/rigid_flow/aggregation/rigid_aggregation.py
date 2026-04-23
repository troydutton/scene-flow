"""Rigid aggregation module for scene flow correction.

Replaces noisy per-point flow vectors with rigid-body estimates for points
that belong to tracked bounding boxes. Background points keep their raw flow.

Supported aggregation methods:
    - ``none``: passthrough (no correction)
    - ``mean``: per-box mean translation
    - ``median``: per-box component-wise median translation
    - ``weighted_median``: per-box median weighted by inverse distance to box center
    - ``geometric_median``: per-box geometric (spatial) median via Weiszfeld's algorithm
    - ``svd``: full rigid transform (rotation + translation) via Kabsch/Procrustes
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rigid_flow.core.types import BoundingBox, FlowResult

AGGREGATION_METHODS = frozenset(
    {"none", "mean", "median", "weighted_median", "geometric_median", "svd"}
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _weighted_median_1d(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    """Compute the weighted median along a 1-D array.

    Finds the value where cumulative normalised weight first reaches 0.5.
    """
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w)
    cum_w /= cum_w[-1]
    idx = np.searchsorted(cum_w, 0.5)
    return float(sorted_vals[min(idx, len(sorted_vals) - 1)])


def _geometric_median(
    vectors: NDArray[np.float32],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> NDArray[np.float32]:
    """Geometric (spatial) median via Weiszfeld's iterative algorithm.

    Minimises ``sum_i ||v_i - y||_2`` over y.  Falls back to the
    component-wise median if the algorithm does not converge.
    """
    y = np.median(vectors, axis=0).astype(np.float64)
    vecs = vectors.astype(np.float64)
    for _ in range(max_iter):
        dists = np.linalg.norm(vecs - y, axis=1, keepdims=True)
        dists = np.maximum(dists, 1e-10)
        weights = 1.0 / dists
        y_new = (weights * vecs).sum(axis=0) / weights.sum()
        if np.linalg.norm(y_new - y) < tol:
            return y_new.astype(np.float32)
        y = y_new
    return y.astype(np.float32)


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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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
    method :
        One of ``"none"``, ``"mean"``, ``"median"``,
        ``"weighted_median"``, ``"geometric_median"``, or ``"svd"``.

    Returns
    -------
    FlowResult with corrected flow, raw flow copy, and per-object metadata.
    """
    if method not in AGGREGATION_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of {sorted(AGGREGATION_METHODS)}."
        )

    n = points.shape[0]
    corrected_flow = raw_flow.copy()
    is_rigid = np.zeros(n, dtype=np.bool_)
    per_object_translation: dict[str, NDArray[np.float32]] = {}

    if method == "none":
        return FlowResult(
            flow=corrected_flow,
            raw_flow=raw_flow.copy(),
            point_to_box=point_to_box,
            is_rigid=is_rigid,
            per_object_translation=per_object_translation,
        )

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

        if method == "mean":
            translation = np.mean(flow, axis=0).astype(np.float32)
            corrected_flow[mask] = translation
            per_object_translation[boxes[box_idx].tracking_id] = translation

        elif method == "median":
            translation = np.median(flow, axis=0).astype(np.float32)
            corrected_flow[mask] = translation
            per_object_translation[boxes[box_idx].tracking_id] = translation

        elif method == "weighted_median":
            center = boxes[box_idx].center  # (3,)
            dists = np.linalg.norm(pts - center, axis=1).astype(np.float64)
            weights = 1.0 / np.maximum(dists, 1e-6)
            translation = np.array(
                [
                    _weighted_median_1d(flow[:, d].astype(np.float64), weights)
                    for d in range(3)
                ],
                dtype=np.float32,
            )
            corrected_flow[mask] = translation
            per_object_translation[boxes[box_idx].tracking_id] = translation

        elif method == "geometric_median":
            translation = _geometric_median(flow)
            corrected_flow[mask] = translation
            per_object_translation[boxes[box_idx].tracking_id] = translation

        elif method == "svd":
            source = pts
            target = pts + flow
            R, t = fit_rigid_transform_svd(source, target)
            fitted_target = (R @ pts.astype(np.float64).T).T + t  # (K, 3)
            corrected_flow[mask] = (fitted_target - pts.astype(np.float64)).astype(np.float32)
            per_object_translation[boxes[box_idx].tracking_id] = t.astype(np.float32)

        is_rigid[mask] = True

    return FlowResult(
        flow=corrected_flow,
        raw_flow=raw_flow.copy(),
        point_to_box=point_to_box,
        is_rigid=is_rigid,
        per_object_translation=per_object_translation,
    )
