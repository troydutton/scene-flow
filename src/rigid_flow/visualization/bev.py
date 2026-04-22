"""Bird's-eye view visualization utilities for rigid scene flow."""

from __future__ import annotations

from typing import Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from numpy.typing import NDArray

from rigid_flow.core.types import BoundingBox, SceneFlowPair

# Class label -> (color, display name)
_CLASS_STYLE: dict[int, tuple[str, str]] = {
    1: ("royalblue", "Vehicle"),
    2: ("limegreen", "Pedestrian"),
    3: ("gray", "Sign"),
    4: ("orange", "Cyclist"),
}


def _box_corners_xy(box: BoundingBox) -> NDArray[np.float64]:
    """Compute the 4 oriented corners of a bounding box in the xy plane.

    Returns an (4, 2) array of corners ordered for drawing a closed polygon.
    """
    cx, cy = float(box.center[0]), float(box.center[1])
    dx, dy = float(box.dimensions[0]), float(box.dimensions[1])
    half_l, half_w = dx / 2.0, dy / 2.0

    # Corners in local frame (front-left, front-right, rear-right, rear-left)
    corners_local = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )

    cos_h = np.cos(box.heading)
    sin_h = np.sin(box.heading)
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)

    corners_global = (rot @ corners_local.T).T + np.array([cx, cy])
    return corners_global


def _draw_boxes(
    ax: plt.Axes,
    boxes: list[BoundingBox],
    linewidth: float = 1.5,
    add_legend_entries: bool = True,
) -> None:
    """Draw oriented bounding-box outlines on *ax*.

    Each box is drawn as a closed ``Polygon`` coloured by its ``class_label``.
    When *add_legend_entries* is True a single proxy artist per class is added
    so that ``ax.legend()`` will show the class names.
    """
    seen_classes: set[int] = set()

    for box in boxes:
        color, label = _CLASS_STYLE.get(box.class_label, ("white", "Unknown"))
        corners = _box_corners_xy(box)
        poly = Polygon(
            corners,
            closed=True,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            label=label if box.class_label not in seen_classes and add_legend_entries else None,
        )
        ax.add_patch(poly)
        seen_classes.add(box.class_label)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_bev_with_boxes(
    pair: SceneFlowPair,
    point_to_box: NDArray[np.int32] | None = None,
    xlim: tuple[float, float] = (-80, 80),
    ylim: tuple[float, float] = (-80, 80),
    point_size: float = 0.3,
    figsize: tuple[float, float] = (12, 12),
) -> Figure:
    """Bird's-eye view of *pair.points_t0* with bounding-box overlays.

    Parameters
    ----------
    pair:
        A scene flow pair whose ``points_t0`` and ``boxes_t0`` are visualised.
    point_to_box:
        Optional (N,) array mapping each point to a box index (>= 0 for
        foreground, -1 for background). When provided, foreground points are
        rendered with higher alpha.
    xlim, ylim:
        Axis limits in metres.
    point_size:
        Marker size for the scatter plot.
    figsize:
        Figure size in inches.
    """
    pts = pair.points_t0  # (N, 3)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Determine alpha per point
    if point_to_box is not None:
        fg_mask = point_to_box >= 0
        alpha = np.where(fg_mask, 0.9, 0.25)
    else:
        alpha = 0.6

    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=pts[:, 2],
        cmap="viridis",
        vmin=-3,
        vmax=2,
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Height (m)")

    _draw_boxes(ax, pair.boxes_t0, add_legend_entries=True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"BEV — {pair.sequence_id} frame {pair.frame_index}")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


def plot_flow_magnitude(
    pair: SceneFlowPair,
    flow: NDArray[np.float32],
    xlim: tuple[float, float] = (-80, 80),
    ylim: tuple[float, float] = (-80, 80),
    point_size: float = 0.3,
    figsize: tuple[float, float] = (12, 12),
    vmax: float = 3.0,
) -> Figure:
    """BEV scatter coloured by per-point flow magnitude.

    Parameters
    ----------
    pair:
        Scene flow pair (only ``points_t0`` is used for positions).
    flow:
        (N, 3) flow vectors aligned with ``pair.points_t0``.
    vmax:
        Upper saturation for the colour scale.
    """
    pts = pair.points_t0
    mag = np.linalg.norm(flow, axis=1)  # (N,)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sc = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=mag,
        cmap="hot",
        vmin=0,
        vmax=vmax,
        s=point_size,
        rasterized=True,
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Flow magnitude (m)")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Flow Magnitude")

    fig.tight_layout()
    return fig


def plot_flow_quiver(
    pair: SceneFlowPair,
    flow: NDArray[np.float32],
    point_to_box: NDArray[np.int32],
    max_arrows: int = 2000,
    xlim: tuple[float, float] = (-80, 80),
    ylim: tuple[float, float] = (-80, 80),
    figsize: tuple[float, float] = (12, 12),
) -> Figure:
    """BEV with quiver arrows showing scene flow vectors.

    Foreground and background points are subsampled proportionally so that
    both groups are represented. A light scatter of *all* points is drawn as
    context behind the arrows.

    Parameters
    ----------
    pair:
        Scene flow pair (``points_t0`` and ``boxes_t0`` are used).
    flow:
        (N, 3) flow vectors aligned with ``pair.points_t0``.
    point_to_box:
        (N,) array mapping points to box indices (-1 = background).
    max_arrows:
        Maximum number of quiver arrows to draw.
    """
    pts = pair.points_t0  # (N, 3)
    N = pts.shape[0]

    fg_mask = point_to_box >= 0
    fg_idx = np.where(fg_mask)[0]
    bg_idx = np.where(~fg_mask)[0]

    n_fg = len(fg_idx)
    n_bg = len(bg_idx)

    # Proportional sampling
    if N <= max_arrows:
        sample_idx = np.arange(N)
    else:
        if n_fg == 0:
            sample_idx = np.random.choice(bg_idx, size=max_arrows, replace=False)
        elif n_bg == 0:
            sample_idx = np.random.choice(fg_idx, size=max_arrows, replace=False)
        else:
            fg_fraction = n_fg / N
            n_fg_sample = max(1, int(round(fg_fraction * max_arrows)))
            n_bg_sample = max_arrows - n_fg_sample
            # Clamp to available counts
            n_fg_sample = min(n_fg_sample, n_fg)
            n_bg_sample = min(n_bg_sample, n_bg)
            fg_sample = np.random.choice(fg_idx, size=n_fg_sample, replace=False)
            bg_sample = np.random.choice(bg_idx, size=n_bg_sample, replace=False)
            sample_idx = np.concatenate([fg_sample, bg_sample])

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Background context scatter (all points)
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c="lightgray",
        s=0.1,
        alpha=0.1,
        rasterized=True,
    )

    # Draw quiver arrows split by fg / bg for colouring
    sampled_fg = sample_idx[point_to_box[sample_idx] >= 0]
    sampled_bg = sample_idx[point_to_box[sample_idx] < 0]

    if len(sampled_fg) > 0:
        ax.quiver(
            pts[sampled_fg, 0],
            pts[sampled_fg, 1],
            flow[sampled_fg, 0],
            flow[sampled_fg, 1],
            color="red",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
            headwidth=3,
            label="Foreground",
        )

    if len(sampled_bg) > 0:
        ax.quiver(
            pts[sampled_bg, 0],
            pts[sampled_bg, 1],
            flow[sampled_bg, 0],
            flow[sampled_bg, 1],
            color="steelblue",
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.002,
            headwidth=3,
            label="Background",
        )

    _draw_boxes(ax, pair.boxes_t0, add_legend_entries=False)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Scene Flow Vectors")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig
