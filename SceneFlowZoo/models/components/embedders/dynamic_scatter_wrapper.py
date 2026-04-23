"""
DynamicScatterWrapper — pure PyTorch replacement for mmcv.ops.DynamicScatter.

The original implementation required MMCV's CUDA extension for the GPU path.
This version uses only torch.unique + scatter_add_ which work on both CPU and GPU
without any external CUDA extensions, while preserving identical output semantics.
"""
from typing import List
import torch
import torch.nn as nn


class DynamicScatterWrapper(nn.Module):
    """
    Voxel-level scatter-reduce for pillar/voxel embeddings.

    Args:
        voxel_size: [vx, vy, vz] — stored but not used directly here (voxelisation
                    happens upstream in the DynamicVoxelizer).
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] — same.
        average_points: if True, average the features of points that fall into the
                        same voxel; if False, sum them.
    """

    def __init__(self, voxel_size: List, point_cloud_range: List, average_points: bool):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward(self, points: torch.Tensor, coors: torch.Tensor):
        """
        Args:
            points: (N, C) float tensor — per-point features
            coors:  (N, 3) int tensor  — voxel coordinates for each point (batch_idx, y, x)

        Returns:
            voxel_feats:  (M, C) float tensor — one feature vector per unique voxel
            unique_coors: (M, 3) int tensor   — unique voxel coordinates
        """
        device = points.device
        voxel_indices = coors.long()

        # Find unique voxels (and the mapping from each point to its voxel)
        unique_voxels, inverse_indices = torch.unique(
            voxel_indices, return_inverse=True, dim=0
        )
        M = unique_voxels.size(0)
        C = points.size(1)

        # Scatter-add all point features into their respective voxels
        aggregated = torch.zeros((M, C), dtype=points.dtype, device=device)
        aggregated.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, C), points)

        if self.average_points:
            counts = torch.zeros(M, dtype=points.dtype, device=device)
            counts.scatter_add_(0, inverse_indices, torch.ones(len(inverse_indices), dtype=points.dtype, device=device))
            counts = counts.clamp(min=1.0)
            aggregated = aggregated / counts.unsqueeze(1)

        return aggregated, unique_voxels
