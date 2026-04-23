"""SE3 rigid body transform — pure NumPy, no CUDA dependencies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SE3:
    """A rigid body transformation in SE(3), stored as a 4x4 homogeneous matrix.

    Convention: the matrix maps points from the local frame to the parent frame,
    i.e.  p_parent = T @ p_local  (with p in homogeneous coordinates).
    """

    def __init__(self, matrix: NDArray[np.float64]) -> None:
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 matrix, got shape {matrix.shape}")
        self._matrix = matrix.astype(np.float64)

    # ------------------------------------------------------------------
    # Static constructors
    # ------------------------------------------------------------------

    @staticmethod
    def identity() -> SE3:
        """Return the identity transform."""
        return SE3(np.eye(4, dtype=np.float64))

    @staticmethod
    def from_rot_trans(rotation: NDArray, translation: NDArray) -> SE3:
        """Build an SE3 from a 3x3 rotation and a (3,) translation vector."""
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64).ravel()
        if rotation.shape != (3, 3):
            raise ValueError(f"Expected 3x3 rotation, got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"Expected (3,) translation, got {translation.shape}")
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rotation
        mat[:3, 3] = translation
        return SE3(mat)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rotation(self) -> NDArray[np.float64]:
        """Return a copy of the 3x3 rotation sub-matrix."""
        return self._matrix[:3, :3].copy()

    @property
    def translation(self) -> NDArray[np.float64]:
        """Return a copy of the (3,) translation vector."""
        return self._matrix[:3, 3].copy()

    @property
    def matrix(self) -> NDArray[np.float64]:
        """Return the full 4x4 homogeneous matrix (read-only copy)."""
        return self._matrix.copy()

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def inverse(self) -> SE3:
        """Invert using the orthogonal-rotation shortcut: R_inv = R^T."""
        R_T = self.rotation.T
        t_inv = -R_T @ self.translation
        return SE3.from_rot_trans(R_T, t_inv)

    def compose(self, other: SE3) -> SE3:
        """Return self @ other (i.e. apply *other* first, then *self*)."""
        return SE3(self._matrix @ other._matrix)

    def transform_points(self, points: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply this transform to an (N, 3) point cloud.

        Returns an (N, 3) float32 array.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) points, got shape {pts.shape}")
        # R @ pts.T + t[:, None]  →  (3, N)  then transpose back
        transformed = (self.rotation @ pts.T + self.translation[:, None]).T
        return transformed.astype(np.float32)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __matmul__(self, other: SE3) -> SE3:
        return self.compose(other)

    def __repr__(self) -> str:
        return f"SE3(\n{self._matrix}\n)"
