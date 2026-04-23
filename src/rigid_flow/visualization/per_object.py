"""Per-object flow distribution histograms for rigid scene flow analysis."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from rigid_flow.core.types import FlowResult, SceneFlowPair

CLASS_NAMES: dict[int, str] = {
    1: "Vehicle",
    2: "Pedestrian",
    3: "Sign",
    4: "Cyclist",
}


def plot_per_object_histograms(
    pair: SceneFlowPair,
    result: FlowResult,
    point_to_box: NDArray[np.int32],
    max_objects: int = 8,
    figsize: tuple[float, float] = (16, 10),
) -> Figure:
    """Grid of histograms showing per-object flow magnitude distributions.

    For each of the top ``max_objects`` objects (ranked by point count), a
    subplot shows overlapping histograms of raw and corrected flow magnitudes.

    Parameters
    ----------
    pair : SceneFlowPair
        The scene flow pair (used for metadata in the title).
    result : FlowResult
        Flow result containing both ``raw_flow`` and ``flow``.
    point_to_box : NDArray[np.int32]
        Per-point box assignment; -1 means background.
    max_objects : int
        Maximum number of objects to plot.
    figsize : tuple[float, float]
        Figure size in inches.

    Returns
    -------
    Figure
        Matplotlib figure with the histogram grid.
    """
    # Identify foreground box indices and sort by point count (descending).
    unique_ids, counts = np.unique(point_to_box[point_to_box >= 0], return_counts=True)
    if len(unique_ids) == 0:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(0.5, 0.5, "No foreground objects", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_axis_off()
        fig.suptitle(
            f"Per-Object Flow Distribution — {pair.sequence_id} frame {pair.frame_index}",
            fontsize=14, fontweight="bold",
        )
        return fig

    order = np.argsort(-counts)
    selected = unique_ids[order[:max_objects]]
    n_selected = len(selected)

    ncols = min(n_selected, 4)
    nrows = math.ceil(n_selected / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, box_idx in enumerate(selected):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        mask = point_to_box == box_idx
        n_pts = int(mask.sum())
        box = pair.boxes_t0[box_idx]

        raw_mag = np.linalg.norm(result.raw_flow[mask], axis=1)
        corrected_mag = np.linalg.norm(result.flow[mask], axis=1)

        # Determine shared bin range.
        bin_max = max(float(raw_mag.max()), float(corrected_mag.max()), 1e-6)
        bins = np.linspace(0, bin_max, 30)

        ax.hist(raw_mag, bins=bins, alpha=0.5, color="blue", label="Raw")
        ax.hist(corrected_mag, bins=bins, alpha=0.5, color="red", label="Corrected")

        # Median vertical lines.
        raw_median = float(np.median(raw_mag))
        corr_median = float(np.median(corrected_mag))
        ax.axvline(raw_median, color="blue", linestyle="--", linewidth=1.2)
        ax.axvline(corr_median, color="red", linestyle="--", linewidth=1.2)

        class_name = CLASS_NAMES.get(box.class_label, f"Class {box.class_label}")
        ax.set_title(f"{class_name} ({n_pts} pts, {box.speed:.1f} m/s)")
        ax.set_xlabel("Flow magnitude (m)")
        ax.set_ylabel("Count")

        # Legend only on first subplot.
        if idx == 0:
            ax.legend(loc="upper right")

    # Hide unused subplots.
    for idx in range(n_selected, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Per-Object Flow Distribution — {pair.sequence_id} frame {pair.frame_index}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    return fig
