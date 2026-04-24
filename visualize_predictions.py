#!/usr/bin/env python3
"""Plotly 3D visualization of ZeroFlow predictions for selected frame pairs.

With ``--tfrecord-root``, for each pair index the script writes **two** overlay
HTML files so you can compare side by side:

1. **Ground truth** flow vs **ZeroFlow (no aggregation)** — same points, two colors.
2. **Ground truth** flow vs **geometric median** rigid-pooled prediction — same points, two colors.

Arrow subsampling uses ‖GT‖ among valid points (fixed RNG) so both files show the
same spatial sample of arrows for easier comparison.

Without ``--tfrecord-root``, only a single baseline ZeroFlow HTML is written (no GT).

Defaults target Phase C jelly examples (pair indices 193, 41, 49).

Examples::

    python visualize_predictions.py \\
        --pkl-root results/zeroflow/validation \\
        --feather-root results/zeroflow/sequence_len_002 \\
        --tfrecord-root /path/to/waymo/validation \\
        --segment segment-10203656353524179475_7625_000_7645_000_with_camera_labels \\
        --output-dir figures/zeroflow_plotly \\
        --pair-indices 41 193
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

DEFAULT_PAIR_INDICES = (193, 41, 49)
DEFAULT_SEGMENT = "segment-10203656353524179475_7625_000_7645_000_with_camera_labels"

# Okabe–Ito–style colors
DEFAULT_COLOR_GT = "#009E73"  # bluish green — ground truth
DEFAULT_COLOR_RAW = "#E69F00"  # orange — ZeroFlow no aggregation
DEFAULT_COLOR_GEO = "#0072B2"  # blue — geometric median


def draw_scenes_plotly(
    points: np.ndarray,
    valid_mask: np.ndarray,
    flow_vectors: np.ndarray,
    *,
    filename: str | Path = "visualization.html",
    title: str = "Scene Flow Predictions",
    max_pts: int = 75_000,
    line_render_limit: int = 5000,
) -> Path:
    """Single-method: scatter 3D points + subsampled flow arrows (Plotly HTML)."""
    filename = Path(filename)
    fig = go.Figure()

    if len(points) > max_pts:
        indices = np.random.choice(len(points), max_pts, replace=False)
        points_render = points[indices]
        valid_mask_render = valid_mask[indices]
        flow_render = flow_vectors[indices]
    else:
        points_render = points
        valid_mask_render = valid_mask
        flow_render = flow_vectors

    invalid_points = points_render[~valid_mask_render]
    valid_points = points_render[valid_mask_render]

    fig.add_trace(
        go.Scatter3d(
            x=invalid_points[:, 0],
            y=invalid_points[:, 1],
            z=invalid_points[:, 2],
            mode="markers",
            marker=dict(size=1.0, color="gray", opacity=0.8),
            name="Invalid points",
        )
    )

    valid_flow = flow_render[valid_mask_render]
    flow_magnitudes = np.linalg.norm(valid_flow, axis=1)

    fig.add_trace(
        go.Scatter3d(
            x=valid_points[:, 0],
            y=valid_points[:, 1],
            z=valid_points[:, 2],
            mode="markers",
            marker=dict(
                size=1.0,
                color=flow_magnitudes,
                colorscale="Viridis",
                opacity=0.8,
                colorbar=dict(title="‖flow‖ (m)"),
            ),
            name="Valid points (by ‖flow‖)",
        )
    )

    if len(valid_points) > 0:
        if len(valid_points) > line_render_limit:
            sorted_valid_idx = np.argsort(flow_magnitudes)[::-1]
            valid_idx = sorted_valid_idx[:line_render_limit]
        else:
            valid_idx = np.arange(len(valid_points))

        x_lines: list[float | None] = []
        y_lines: list[float | None] = []
        z_lines: list[float | None] = []
        colors: list[float] = []

        for i in valid_idx:
            x0, y0, z0 = valid_points[i, 0], valid_points[i, 1], valid_points[i, 2]
            dx, dy, dz = valid_flow[i, 0], valid_flow[i, 1], valid_flow[i, 2]
            x_lines.extend([x0, x0 + dx, None])
            y_lines.extend([y0, y0 + dy, None])
            z_lines.extend([z0, z0 + dz, None])
            m = float(flow_magnitudes[i])
            colors.extend([m, m, m])

        fig.add_trace(
            go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z=z_lines,
                mode="lines",
                line=dict(color=colors, colorscale="Jet", width=3, showscale=True),
                name="Scene flow (subsampled)",
            )
        )

    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40),
        title=title,
    )

    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(filename))
    return filename


def draw_overlay_flows_plotly(
    points: np.ndarray,
    valid_mask: np.ndarray,
    flow_a: np.ndarray,
    flow_b: np.ndarray,
    *,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    filename: str | Path,
    title: str,
    max_pts: int = 75_000,
    line_render_limit: int = 5000,
    pick_magnitude_flow: np.ndarray | None = None,
    points_legend_hint: str = "Points (light ‖GT‖)",
) -> Path:
    """Two (N,3) flow fields on the same point cloud; line pick by ‖pick_magnitude_flow‖ (default: flow_a)."""
    filename = Path(filename)
    fig = go.Figure()

    rng = np.random.default_rng(0)
    if len(points) > max_pts:
        indices = rng.choice(len(points), max_pts, replace=False)
        points_r = points[indices]
        mask_r = valid_mask[indices]
        fa = flow_a[indices]
        fb = flow_b[indices]
        pick_ref = (pick_magnitude_flow if pick_magnitude_flow is not None else flow_a)[indices]
    else:
        points_r = points
        mask_r = valid_mask
        fa = flow_a
        fb = flow_b
        pick_ref = pick_magnitude_flow if pick_magnitude_flow is not None else flow_a

    invalid = points_r[~mask_r]
    fig.add_trace(
        go.Scatter3d(
            x=invalid[:, 0],
            y=invalid[:, 1],
            z=invalid[:, 2],
            mode="markers",
            marker=dict(size=1.0, color="#bbbbbb", opacity=0.5),
            name="Invalid / masked",
        )
    )

    valid_pts = points_r[mask_r]
    mag_gt_vis = np.linalg.norm(pick_ref[mask_r], axis=1)
    fig.add_trace(
        go.Scatter3d(
            x=valid_pts[:, 0],
            y=valid_pts[:, 1],
            z=valid_pts[:, 2],
            mode="markers",
            marker=dict(
                size=0.9,
                color=mag_gt_vis,
                colorscale="Greys",
                opacity=0.35,
                showscale=False,
            ),
            name=points_legend_hint,
        )
    )

    va = fa[mask_r]
    vb = fb[mask_r]
    mag_pick = np.linalg.norm(pick_ref[mask_r], axis=1)
    if len(valid_pts) > line_render_limit:
        pick = np.argsort(mag_pick)[::-1][:line_render_limit]
    else:
        pick = np.arange(len(valid_pts))

    def _segments(pts: np.ndarray, flow: np.ndarray, idxs: np.ndarray) -> tuple[list, list, list]:
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        for i in idxs:
            x0, y0, z0 = float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])
            dx, dy, dz = float(flow[i, 0]), float(flow[i, 1]), float(flow[i, 2])
            xs.extend([x0, x0 + dx, None])
            ys.extend([y0, y0 + dy, None])
            zs.extend([z0, z0 + dz, None])
        return xs, ys, zs

    xa, ya, za = _segments(valid_pts, va, pick)
    xb, yb, zb = _segments(valid_pts, vb, pick)

    fig.add_trace(
        go.Scatter3d(
            x=xa,
            y=ya,
            z=za,
            mode="lines",
            line=dict(color=color_a, width=4),
            name=label_a,
            legendgroup="a",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=xb,
            y=yb,
            z=zb,
            mode="lines",
            line=dict(color=color_b, width=4),
            name=label_b,
            legendgroup="b",
        )
    )

    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=48),
        title=title,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(filename))
    return filename


def _load_baseline_from_disk(
    pkl_dir: Path,
    feather_dir: Path,
    pair_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_path = feather_dir / f"{pair_index:010d}.feather"
    pkl_path = pkl_dir / f"{pair_index:06d}.pkl"
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(pkl_path)

    df = pd.read_feather(pred_path)
    with open(pkl_path, "rb") as f:
        gt_data = pickle.load(f)
    pc = np.asarray(gt_data["car_frame_pc"], dtype=np.float32)
    is_valid = df["is_valid"].values.astype(bool)
    flow = df[["flow_tx_m", "flow_ty_m", "flow_tz_m"]].values.astype(np.float32)
    return pc, is_valid, flow


def _load_pair_gt_raw_geo(
    tfrecord_root: Path,
    pkl_root: Path,
    feather_root: Path,
    pair_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return points, valid mask, label GT flow (N,3), raw ZeroFlow (N,3), geo-median (N,3)."""
    from rigid_flow.aggregation.rigid_aggregation import compute_rigid_flow
    from rigid_flow.data.zeroflow_loader import ZeroFlowDataSource
    from rigid_flow.geometry.points_in_boxes import points_in_boxes_cpu
    from rigid_flow.pipeline import compute_gt_flow

    source = ZeroFlowDataSource(pkl_root, feather_root, tfrecord_root)
    for idx, (pair, pred_flow, is_valid) in enumerate(source.iterate_pairs()):
        if idx != pair_index:
            continue
        pair_gt = compute_gt_flow(pair)
        assert pair_gt.gt_flow is not None
        gt = pair_gt.gt_flow.astype(np.float32)

        flow_input = np.zeros_like(pred_flow, dtype=np.float32)
        flow_input[is_valid] = pred_flow[is_valid]
        raw = flow_input.copy()

        if pair_gt.boxes_t0:
            boxes_array = np.stack([b.as_7dof for b in pair_gt.boxes_t0], axis=0)
        else:
            boxes_array = np.empty((0, 7), dtype=np.float32)
        point_to_box = points_in_boxes_cpu(pair_gt.points_t0, boxes_array)
        result = compute_rigid_flow(
            pair_gt.points_t0,
            flow_input,
            point_to_box,
            list(pair_gt.boxes_t0),
            "geometric_median",
        )
        geo = result.flow.astype(np.float32)
        return pair_gt.points_t0, is_valid, gt, raw, geo

    raise LookupError(
        f"No pair at global index {pair_index} in ZeroFlowDataSource "
        f"(check pkl/feather/tfrecord roots)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pkl-root", type=Path, default=Path("results/zeroflow/validation"))
    parser.add_argument("--feather-root", type=Path, default=Path("results/zeroflow/sequence_len_002"))
    parser.add_argument("--segment", type=str, default=DEFAULT_SEGMENT)
    parser.add_argument(
        "--pair-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_PAIR_INDICES),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures/zeroflow_plotly"))
    parser.add_argument(
        "--tfrecord-root",
        type=Path,
        default=None,
        help="Required for GT overlays; writes two HTML files per pair.",
    )
    parser.add_argument("--color-gt", type=str, default=DEFAULT_COLOR_GT)
    parser.add_argument("--color-raw", type=str, default=DEFAULT_COLOR_RAW)
    parser.add_argument("--color-geo", type=str, default=DEFAULT_COLOR_GEO)
    parser.add_argument("--max-pts", type=int, default=75_000)
    parser.add_argument("--line-limit", type=int, default=5000, help="Max arrows per flow field (valid pts).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pkl_dir = args.pkl_root / args.segment
    feather_dir = args.feather_root / args.segment
    if not pkl_dir.is_dir():
        raise FileNotFoundError(f"PKL segment dir not found: {pkl_dir}")
    if not feather_dir.is_dir():
        raise FileNotFoundError(f"Feather segment dir not found: {feather_dir}")

    for pair_idx in args.pair_indices:
        if args.tfrecord_root is not None:
            logger.info("Pair %d — GT vs raw + GT vs geometric median (two HTML files)", pair_idx)
            pc, is_valid, gt, raw, geo = _load_pair_gt_raw_geo(
                args.tfrecord_root,
                args.pkl_root,
                args.feather_root,
                pair_idx,
            )

            out_raw = args.output_dir / f"pair_{pair_idx:04d}_overlay_gt_vs_zeroflow_raw.html"
            draw_overlay_flows_plotly(
                pc,
                is_valid,
                gt,
                raw,
                label_a="Ground truth (labels)",
                label_b="ZeroFlow (no aggregation)",
                color_a=args.color_gt,
                color_b=args.color_raw,
                filename=out_raw,
                title=f"Ground truth vs ZeroFlow (no aggregation) — pair {pair_idx} — {args.segment}",
                max_pts=args.max_pts,
                line_render_limit=args.line_limit,
                pick_magnitude_flow=gt,
                points_legend_hint="Points (light ‖GT‖)",
            )
            logger.info("  wrote %s", out_raw.resolve())

            out_geo = args.output_dir / f"pair_{pair_idx:04d}_overlay_gt_vs_geometric_median.html"
            draw_overlay_flows_plotly(
                pc,
                is_valid,
                gt,
                geo,
                label_a="Ground truth (labels)",
                label_b="ZeroFlow + geometric median (rigid)",
                color_a=args.color_gt,
                color_b=args.color_geo,
                filename=out_geo,
                title=f"Ground truth vs geometric median — pair {pair_idx} — {args.segment}",
                max_pts=args.max_pts,
                line_render_limit=args.line_limit,
                pick_magnitude_flow=gt,
                points_legend_hint="Points (light ‖GT‖)",
            )
            logger.info("  wrote %s", out_geo.resolve())
        else:
            logger.info("Pair %d — baseline only (no tfrecord; no GT overlay)", pair_idx)
            pc, is_valid, flow_raw = _load_baseline_from_disk(pkl_dir, feather_dir, pair_idx)
            out_base = args.output_dir / f"pair_{pair_idx:04d}_baseline_zeroflow.html"
            draw_scenes_plotly(
                pc,
                is_valid,
                flow_raw,
                filename=out_base,
                title=f"ZeroFlow baseline — pair {pair_idx} — {args.segment}",
                max_pts=args.max_pts,
                line_render_limit=args.line_limit,
            )
            logger.info("  wrote %s", out_base.resolve())


if __name__ == "__main__":
    main()
