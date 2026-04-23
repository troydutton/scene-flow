from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BoundingBox:
    """A 7-DOF 3D bounding box with tracking metadata.

    Coordinate convention follows OpenPCDet/Waymo: heading is yaw angle
    from +x toward +y (counter-clockwise when viewed from above).
    """

    center: NDArray[np.float32]  # (3,) — x, y, z
    dimensions: NDArray[np.float32]  # (3,) — length (dx), width (dy), height (dz)
    heading: float  # radians, yaw around z-axis
    class_label: int  # Waymo: 1=Vehicle, 2=Pedestrian, 3=Sign, 4=Cyclist
    tracking_id: str  # unique object ID across frames in a sequence
    velocity: NDArray[np.float32] | None = None  # (2,) — vx, vy in global frame; vz assumed 0

    @property
    def as_7dof(self) -> NDArray[np.float32]:
        """Pack into [x, y, z, dx, dy, dz, heading] array."""
        return np.array([*self.center, *self.dimensions, self.heading], dtype=np.float32)

    @property
    def speed(self) -> float:
        """Scalar speed (m/s). Returns 0.0 if velocity is unknown."""
        if self.velocity is None:
            return 0.0
        return float(np.linalg.norm(self.velocity))


@dataclass(frozen=True)
class SceneFlowPair:
    """Two consecutive LiDAR frames with all annotations needed for scene flow.

    Points and poses are in the ego-vehicle frame at their respective timestamps.
    gt_flow is defined in the ego frame at t0: applying it to points_t0 yields where
    each point moved to (in the t0 coordinate frame) at time t1.
    """

    points_t0: NDArray[np.float32]  # (N, 3)
    points_t1: NDArray[np.float32]  # (M, 3)
    ego_pose_t0: NDArray[np.float64]  # (4, 4) ego-to-global at t0
    ego_pose_t1: NDArray[np.float64]  # (4, 4) ego-to-global at t1
    boxes_t0: list[BoundingBox]
    boxes_t1: list[BoundingBox]
    timestamp_us_t0: int  # microseconds
    timestamp_us_t1: int  # microseconds
    gt_flow: NDArray[np.float32] | None = None  # (N, 3), same length as points_t0
    sequence_id: str = ""
    frame_index: int = 0

    @property
    def dt(self) -> float:
        """Time delta in seconds between the two frames."""
        return (self.timestamp_us_t1 - self.timestamp_us_t0) / 1e6

    @property
    def num_points_t0(self) -> int:
        return self.points_t0.shape[0]


@dataclass
class FlowResult:
    """Per-point flow vectors with rigid-correction metadata.

    Produced by the aggregation module. Each point has both raw and corrected flow,
    plus information about which object it belongs to and whether its flow was
    replaced by a rigid estimate.
    """

    flow: NDArray[np.float32]  # (N, 3) — corrected flow
    raw_flow: NDArray[np.float32]  # (N, 3) — original per-point flow before correction
    point_to_box: NDArray[np.int32]  # (N,) — index into boxes list, -1 = background
    is_rigid: NDArray[np.bool_]  # (N,) — True if this point's flow was rigidly corrected
    per_object_translation: dict[str, NDArray[np.float32]] = field(default_factory=dict)  # tracking_id -> (3,)
