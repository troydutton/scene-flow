"""CPU point-in-box assignment and ego-motion compensation — pure NumPy."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rigid_flow.geometry.se3 import SE3


def points_in_boxes_cpu(
    points: NDArray[np.float32],
    boxes: NDArray[np.float32],
) -> NDArray[np.int32]:
    """Assign each point to a box index, or -1 if it belongs to no box.

    Parameters
    ----------
    points : (N, 3) float32
        Point cloud in the same coordinate frame as the boxes.
    boxes : (M, 7) float32
        Each row is ``[x, y, z, dx, dy, dz, heading]``.
        Heading is the yaw angle from +x toward +y (counter-clockwise from above).

    Returns
    -------
    (N,) int32
        Box index for each point, or -1 for background.  If a point falls
        inside multiple boxes the highest-index box wins.
    """
    points = np.asarray(points, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got shape {points.shape}")

    N = points.shape[0]
    assignment = np.full(N, -1, dtype=np.int32)

    if boxes.ndim == 1:
        # Single box — reshape to (1, 7).
        boxes = boxes.reshape(1, 7)

    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"Expected (M, 7) boxes, got shape {boxes.shape}")

    M = boxes.shape[0]

    for i in range(M):
        cx, cy, cz, dx, dy, dz, heading = boxes[i]

        # Shift points to box center.
        shifted_x = points[:, 0] - cx
        shifted_y = points[:, 1] - cy
        shifted_z = points[:, 2] - cz

        # Rotate into box-local frame (rotate by -heading around z).
        cos_h = np.cos(-heading)
        sin_h = np.sin(-heading)
        local_x = shifted_x * cos_h - shifted_y * sin_h
        local_y = shifted_x * sin_h + shifted_y * cos_h
        # z is unchanged by yaw rotation.

        # Axis-aligned containment check (boundary-inclusive).
        inside = (
            (np.abs(local_x) <= dx / 2.0)
            & (np.abs(local_y) <= dy / 2.0)
            & (np.abs(shifted_z) <= dz / 2.0)
        )

        # Last (highest index) box wins for overlapping points.
        assignment[inside] = i

    return assignment


def ego_compensate(
    points: NDArray[np.float32],
    ego_t0: NDArray[np.float64],
    ego_t1: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Re-express *points* (in the ego frame at t0) into the ego frame at t1.

    Mathematically: ``p_t1 = ego_t1^{-1} @ ego_t0 @ p_t0``.

    Parameters
    ----------
    points : (N, 3) float32
        Points in the ego-vehicle coordinate frame at time t0.
    ego_t0 : (4, 4) float64
        Ego-to-global rigid transform at t0.
    ego_t1 : (4, 4) float64
        Ego-to-global rigid transform at t1.

    Returns
    -------
    (N, 3) float32
        The same points expressed in the ego frame at t1.
    """
    T0 = SE3(np.asarray(ego_t0, dtype=np.float64))
    T1 = SE3(np.asarray(ego_t1, dtype=np.float64))
    # ego_t1_inv @ ego_t0
    relative = T1.inverse().compose(T0)
    return relative.transform_points(np.asarray(points, dtype=np.float32))
