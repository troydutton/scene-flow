"""CLI entry point that generates all visualizations for a single frame pair.

Orchestrates: parse data -> compute GT flow -> assign boxes -> aggregate -> visualize.

Usage::

    python -m rigid_flow.visualization.visualize \
        --data-root /path/to/waymo \
        --output-dir ./viz_output \
        --frame-index 0 \
        --method median
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rigid_flow.aggregation.rigid_aggregation import compute_rigid_flow
from rigid_flow.data.waymo_parser import WaymoParser
from rigid_flow.geometry.points_in_boxes import points_in_boxes_cpu
from rigid_flow.pipeline import compute_gt_flow
from rigid_flow.visualization import bev, flow_comparison, per_object

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all rigid scene flow visualizations for a single frame pair.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to directory containing Waymo .tfrecord files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame pair index to visualize (default: 0).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="median",
        choices=["median", "svd"],
        help="Aggregation method (default: median).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() after saving figures.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # 1. Load data and advance to the requested frame index.
    # ------------------------------------------------------------------
    waymo_parser = WaymoParser(args.data_root)
    pair = None
    for idx, p in enumerate(waymo_parser.iterate_pairs()):
        if idx == args.frame_index:
            pair = p
            break

    if pair is None:
        logger.error(
            "Frame index %d not found. Dataset may have fewer frame pairs.",
            args.frame_index,
        )
        return

    logger.info(
        "Loaded pair: seq=%s, frame=%d, N=%d",
        pair.sequence_id,
        pair.frame_index,
        pair.num_points_t0,
    )

    # ------------------------------------------------------------------
    # 2. Compute ground-truth flow.
    # ------------------------------------------------------------------
    pair = compute_gt_flow(pair)

    # ------------------------------------------------------------------
    # 3. Assign points to bounding boxes.
    # ------------------------------------------------------------------
    if pair.boxes_t0:
        boxes_array = np.stack([b.as_7dof for b in pair.boxes_t0], axis=0)
    else:
        boxes_array = np.empty((0, 7), dtype=np.float32)

    point_to_box = points_in_boxes_cpu(pair.points_t0, boxes_array)

    # ------------------------------------------------------------------
    # 4. Rigid aggregation.
    # ------------------------------------------------------------------
    result = compute_rigid_flow(
        pair.points_t0,
        pair.gt_flow,
        point_to_box,
        pair.boxes_t0,
        args.method,
    )

    # ------------------------------------------------------------------
    # 5. Generate and save all plots.
    # ------------------------------------------------------------------
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: list[tuple[str, plt.Figure]] = [
        ("bev_boxes.png", bev.plot_bev_with_boxes(pair, point_to_box)),
        ("flow_magnitude.png", bev.plot_flow_magnitude(pair, pair.gt_flow)),
        ("flow_quiver.png", bev.plot_flow_quiver(pair, pair.gt_flow, point_to_box)),
        (
            "correction_comparison.png",
            flow_comparison.plot_correction_comparison(pair, result, point_to_box),
        ),
        (
            "per_object_histograms.png",
            per_object.plot_per_object_histograms(pair, result, point_to_box),
        ),
    ]

    for filename, fig in plots:
        path = output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s", path)

    # ------------------------------------------------------------------
    # 6. Optionally show interactive window.
    # ------------------------------------------------------------------
    if args.show:
        plt.show()
    else:
        # Close figures to free memory.
        for _, fig in plots:
            plt.close(fig)


if __name__ == "__main__":
    main()
