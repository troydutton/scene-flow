"""
Pure-PyTorch voxelization — replaces mmcv.ops.Voxelization.

Provides both HardVoxelizer and DynamicVoxelizer with the same interface
as the original MMCV-backed implementations but without any C++/CUDA extension
dependencies.
"""
import torch
import torch.nn as nn
from typing import List, Tuple


class _PurePyTorchVoxelization(nn.Module):
    """
    Core voxelisation logic: maps each (x, y, z) point to integer voxel
    coordinates [z_idx, y_idx, x_idx].  Out-of-range points get coordinate -1.

    Args:
        voxel_size:       [vx, vy, vz]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        max_num_points:   max points per voxel (-1 → dynamic/unlimited)
    """

    def __init__(self, voxel_size, point_cloud_range, max_num_points: int = -1,
                 deterministic: bool = True):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (N, C) float tensor; xyz are the first 3 channels.
        Returns:
            voxel_coords: (N, 3) int32 tensor [z, y, x]; -1 for out-of-range pts.
        """
        device = points.device
        pc_range = torch.tensor(self.point_cloud_range, dtype=points.dtype, device=device)
        voxel_size = self.voxel_size.to(device=device, dtype=points.dtype)

        xyz = points[:, :3]
        # Shift by range minimum
        shifted = xyz - pc_range[:3]

        # Compute [x_idx, y_idx, z_idx] then return as [z, y, x] (MMCV convention)
        voxel_idx = (shifted / voxel_size).floor().long()  # (N, 3) → [xi, yi, zi]

        # Grid shape
        grid_size = ((pc_range[3:] - pc_range[:3]) / voxel_size).ceil().long()  # [Gx, Gy, Gz]

        # Mark out-of-range points
        in_range = (
            (voxel_idx[:, 0] >= 0) & (voxel_idx[:, 0] < grid_size[0]) &
            (voxel_idx[:, 1] >= 0) & (voxel_idx[:, 1] < grid_size[1]) &
            (voxel_idx[:, 2] >= 0) & (voxel_idx[:, 2] < grid_size[2])
        )

        # Build output as [z_idx, y_idx, x_idx] (MMCV convention)
        coords = torch.stack([voxel_idx[:, 2], voxel_idx[:, 1], voxel_idx[:, 0]], dim=1)
        coords[~in_range] = -1
        return coords.int()


class HardVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range, max_points_per_voxel: int):
        super().__init__()
        assert max_points_per_voxel > 0, f"max_points_per_voxel must be > 0, got {max_points_per_voxel}"
        self.voxelizer = _PurePyTorchVoxelization(voxel_size, point_cloud_range,
                                                   max_num_points=max_points_per_voxel)

    def forward(self, points: torch.Tensor):
        assert isinstance(points, torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"
        not_nan_mask = ~torch.isnan(points).any(dim=2)
        return {"voxel_coords": self.voxelizer(points[not_nan_mask])}


class DynamicVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxelizer = _PurePyTorchVoxelization(voxel_size, point_cloud_range,
                                                   max_num_points=-1)

    def _get_point_offsets(self, points: torch.Tensor, voxel_coords: torch.Tensor):
        point_cloud_range = torch.tensor(self.point_cloud_range,
                                         dtype=points.dtype, device=points.device)
        min_point = point_cloud_range[:3]
        voxel_size = torch.tensor(self.voxel_size,
                                  dtype=points.dtype, device=points.device)

        # Voxel coords are in the form Z, Y, X :eyeroll:, convert to X, Y, Z
        voxel_coords = voxel_coords[:, [2, 1, 0]]

        # Offsets are computed relative to min point
        voxel_centers = voxel_coords * voxel_size + min_point + voxel_size / 2
        return points - voxel_centers

    def forward(self, points: List[torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        batch_results = []
        for batch_idx in range(len(points)):
            batch_points = points[batch_idx]
            valid_point_idxes = torch.arange(batch_points.shape[0], device=batch_points.device)
            not_nan_mask = ~torch.isnan(batch_points).any(dim=1)
            batch_non_nan_points = batch_points[not_nan_mask]
            valid_point_idxes = valid_point_idxes[not_nan_mask]
            batch_voxel_coords = self.voxelizer(batch_non_nan_points)
            # If any of the coords are -1, then the point is not in the voxel grid
            batch_voxel_coords_mask = (batch_voxel_coords != -1).all(dim=1)

            valid_batch_voxel_coords = batch_voxel_coords[batch_voxel_coords_mask]
            valid_batch_non_nan_points = batch_non_nan_points[batch_voxel_coords_mask]
            valid_point_idxes = valid_point_idxes[batch_voxel_coords_mask]

            point_offsets = self._get_point_offsets(valid_batch_non_nan_points,
                                                    valid_batch_voxel_coords)

            result_dict = {
                "points": valid_batch_non_nan_points,
                "voxel_coords": valid_batch_voxel_coords,
                "point_idxes": valid_point_idxes,
                "point_offsets": point_offsets,
            }
            batch_results.append(result_dict)
        return batch_results