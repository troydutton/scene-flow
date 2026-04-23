"""Visualization module for comparing raw and corrected rigid scene flow."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from rigid_flow.core.types import SceneFlowPair, FlowResult, BoundingBox


# Class label -> color mapping
_CLASS_COLORS: dict[int, str] = {
    1: "royalblue",   # Vehicle
    2: "limegreen",   # Pedestrian
    3: "gray",        # Sign
    4: "orange",      # Cyclist
}


def _box_corners_2d(box: BoundingBox) -> NDArray[np.float64]:
    """Compute the 4 BEV corners of an oriented bounding box.

    Returns an (4, 2) array of [x, y] corner positions ordered for polygon drawing.
    """
    cx, cy = box.center[0], box.center[1]
    dx, dy = box.dimensions[0], box.dimensions[1]
    cos_h = np.cos(box.heading)
    sin_h = np.sin(box.heading)

    # Half-extents along length and width
    half_l = dx / 2.0
    half_w = dy / 2.0

    # Local corners (front-left, front-right, rear-right, rear-left)
    local = np.array([
        [ half_l,  half_w],
        [ half_l, -half_w],
        [-half_l, -half_w],
        [-half_l,  half_w],
    ])

    rotation = np.array([[cos_h, -sin_h],
                         [sin_h,  cos_h]])
    corners = (rotation @ local.T).T + np.array([cx, cy])
    return corners


def plot_correction_comparison(
    pair: SceneFlowPair,
    result: FlowResult,
    point_to_box: NDArray[np.int32],
    xlim: tuple[float, float] = (-80, 80),
    ylim: tuple[float, float] = (-80, 80),
    point_size: float = 0.5,
    figsize: tuple[float, float] = (20, 8),
) -> Figure:
    """Create a 3-panel BEV comparison of raw vs. corrected rigid flow.

    Parameters
    ----------
    pair : SceneFlowPair
        The input scene flow pair containing points and bounding boxes.
    result : FlowResult
        Flow result with both raw and corrected flow vectors.
    point_to_box : NDArray[np.int32]
        Per-point box assignment. -1 means background.
    xlim, ylim : tuple[float, float]
        Axis limits for the BEV plots.
    point_size : float
        Marker size for scatter plots.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    Figure
        The matplotlib figure with three panels.
    """
    pts = pair.points_t0  # (N, 3)
    x, y = pts[:, 0], pts[:, 1]

    fg_mask = point_to_box >= 0
    bg_mask = ~fg_mask

    # Compute magnitudes
    raw_mag = np.linalg.norm(result.raw_flow, axis=1)
    corrected_mag = np.linalg.norm(result.flow, axis=1)
    correction_mag = np.linalg.norm(result.flow - result.raw_flow, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Panel 1: Raw Flow Magnitude ---
    ax0 = axes[0]
    ax0.scatter(x[bg_mask], y[bg_mask], s=point_size, c="lightgray", edgecolors="none", rasterized=True)
    sc0 = ax0.scatter(
        x[fg_mask], y[fg_mask], s=point_size, c=raw_mag[fg_mask],
        cmap="hot", vmin=0, vmax=3.0, edgecolors="none", rasterized=True,
    )
    fig.colorbar(sc0, ax=ax0, shrink=0.8, label="Flow magnitude (m)")
    ax0.set_title("Raw Flow")

    # --- Panel 2: Corrected Flow Magnitude ---
    ax1 = axes[1]
    ax1.scatter(x[bg_mask], y[bg_mask], s=point_size, c="lightgray", edgecolors="none", rasterized=True)
    sc1 = ax1.scatter(
        x[fg_mask], y[fg_mask], s=point_size, c=corrected_mag[fg_mask],
        cmap="hot", vmin=0, vmax=3.0, edgecolors="none", rasterized=True,
    )
    fig.colorbar(sc1, ax=ax1, shrink=0.8, label="Flow magnitude (m)")
    ax1.set_title("Corrected Flow (median)")

    # --- Panel 3: Correction Magnitude ---
    ax2 = axes[2]
    ax2.scatter(x[bg_mask], y[bg_mask], s=point_size, c="lightgray", edgecolors="none", rasterized=True)
    max_corr = max(float(np.max(correction_mag[fg_mask])), 1e-6) if np.any(fg_mask) else 1.0
    norm2 = Normalize(vmin=-max_corr, vmax=max_corr)
    sc2 = ax2.scatter(
        x[fg_mask], y[fg_mask], s=point_size, c=correction_mag[fg_mask],
        cmap="coolwarm", norm=norm2, edgecolors="none", rasterized=True,
    )
    fig.colorbar(sc2, ax=ax2, shrink=0.8, label="Correction magnitude (m)")
    ax2.set_title("Correction Magnitude")

    # --- Common formatting and box overlays ---
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # Draw bounding box outlines
        for box in pair.boxes_t0:
            corners = _box_corners_2d(box)
            color = _CLASS_COLORS.get(box.class_label, "white")
            poly = Polygon(
                corners, closed=True, fill=False,
                edgecolor=color, linewidth=1.2, linestyle="-",
            )
            ax.add_patch(poly)

    fig.suptitle(
        f"Rigid Correction \u2014 {pair.sequence_id} frame {pair.frame_index}",
        fontsize=14, fontweight="bold",
    )
    fig.subplots_adjust(top=0.90, wspace=0.30)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig
