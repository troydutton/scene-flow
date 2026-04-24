#!/usr/bin/env python3
"""Find and visualize Phase C (ZeroFlow + GT boxes) frames with the largest EPE gain.

Compares **none** (no rigid aggregation) vs **geometric_median** using the same
per-frame metrics produced by ``run_phase_cd.py`` / ``run_all_methods.sh``.

Examples::

    # Print top frames by global mean EPE improvement (none − geometric_median)
    python scripts/visualize_phase_c_jelly.py print-top \\
        --results-root results/full_run

    # Render BEV figures for the single best global-improvement frame (pair index 193)
    python scripts/visualize_phase_c_jelly.py render \\
        --tfrecord-root /path/to/waymo \\
        --pkl-root results/zeroflow/validation \\
        --feather-root results/zeroflow/sequence_len_002 \\
        --pair-index 193 \\
        --output-dir figures/jelly_pair193

    # Same, but address by sequence + frame index (must match ``per_frame_metrics``)
    python scripts/visualize_phase_c_jelly.py render \\
        --tfrecord-root /path/to/waymo \\
        --pkl-root results/zeroflow/validation \\
        --feather-root results/zeroflow/sequence_len_002 \\
        --sequence-id 10203656353524179475_7625_000_7645_000 \\
        --frame-index 193 \\
        --output-dir figures/jelly_frame193
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_per_frame_pair(results_root: Path) -> tuple[list[dict], list[dict]]:
    none_path = results_root / "none" / "phase_c_pred_flow_gt_boxes" / "per_frame_metrics.json"
    geo_path = results_root / "geometric_median" / "phase_c_pred_flow_gt_boxes" / "per_frame_metrics.json"
    if not none_path.is_file():
        raise FileNotFoundError(none_path)
    if not geo_path.is_file():
        raise FileNotFoundError(geo_path)
    none_rows = json.loads(none_path.read_text())
    geo_rows = json.loads(geo_path.read_text())
    if len(none_rows) != len(geo_rows):
        raise ValueError(
            f"Row count mismatch: none={len(none_rows)} geo={len(geo_rows)}"
        )
    for i, (a, b) in enumerate(zip(none_rows, geo_rows)):
        if (a["sequence_id"], a["frame_index"]) != (b["sequence_id"], b["frame_index"]):
            raise ValueError(f"Row {i} sequence/frame mismatch between none and geo JSON")
    return none_rows, geo_rows


def cmd_print_top(args: argparse.Namespace) -> None:
    none_rows, geo_rows = _load_per_frame_pair(Path(args.results_root))

    scored: list[tuple[float, float, float, float, float, int, str, int]] = []
    for i, (a, b) in enumerate(zip(none_rows, geo_rows)):
        en, eg = float(a["epe_mean"]), float(b["epe_mean"])
        if math.isnan(en) or math.isnan(eg):
            continue
        fn = float(a["epe_foreground"])
        fg = float(b["epe_foreground"])
        d_mean = en - eg
        d_fg = fn - fg if not (math.isnan(fn) or math.isnan(fg)) else float("nan")
        fv = float(a["flow_variance_mean"])
        if math.isnan(fv):
            fv = 0.0
        scored.append((d_mean, d_fg, en, eg, fv, i, a["sequence_id"], int(a["frame_index"])))

    by_mean = sorted(scored, reverse=True, key=lambda t: t[0])
    by_fg = sorted(
        [t for t in scored if not math.isnan(t[1])],
        reverse=True,
        key=lambda t: t[1],
    )

    print("=== Top frames by global mean EPE improvement (none − geometric_median) ===\n")
    print(f"{'rank':<5}{'pair_idx':<10}{'Δmean_m':<12}{'none_mean':<12}{'geo_mean':<12}{'Δfg_m':<12}{'flow_var_n':<14}{'frame':<8}{'sequence_id'}")
    for rank, t in enumerate(by_mean[: args.top], start=1):
        d_mean, d_fg, en, eg, fv, i, sid, fi = t
        print(
            f"{rank:<5}{i:<10}{d_mean:<12.5f}{en:<12.5f}{eg:<12.5f}{d_fg:<12.5f}{fv:<14.6f}{fi:<8}{sid}"
        )

    print("\n=== Top frames by foreground EPE improvement (none − geometric_median) ===\n")
    print(f"{'rank':<5}{'pair_idx':<10}{'Δfg_m':<12}{'none_fg':<12}{'geo_fg':<12}{'Δmean_m':<12}{'frame':<8}{'sequence_id'}")
    for rank, t in enumerate(by_fg[: args.top], start=1):
        d_mean, d_fg, en, eg, fv, i, sid, fi = t
        fn = float(none_rows[i]["epe_foreground"])
        fg = float(geo_rows[i]["epe_foreground"])
        print(f"{rank:<5}{i:<10}{d_fg:<12.5f}{fn:<12.5f}{fg:<12.5f}{d_mean:<12.5f}{fi:<8}{sid}")

    best = by_mean[0]
    print("\n--- Recommended single-frame demo (largest global mean improvement) ---")
    print(
        f"  pair_index={best[5]}  sequence_id={best[6]!r}  frame_index={best[7]}\n"
        f"  Δ mean EPE = {best[0]:.4f} m   (none {best[2]:.4f} → geo {best[3]:.4f})\n"
        f"  Δ foreground EPE = {best[1]:.4f} m   (flow_variance_mean none = {best[4]:.6f})"
    )


def _plot_epe_bev(
    pair: Any,
    epe: Any,
    point_to_box: Any,
    title: str,
    vmax: float,
    xlim: tuple[float, float] = (-80, 80),
    ylim: tuple[float, float] = (-80, 80),
):
    import matplotlib.pyplot as plt

    pts = pair.points_t0
    x, y = pts[:, 0], pts[:, 1]
    fg = point_to_box >= 0
    bg = ~fg

    fig, ax = plt.subplots(1, 1, figsize=(11, 10))
    ax.scatter(x[bg], y[bg], s=0.25, c="lightgray", edgecolors="none", rasterized=True)
    sc = ax.scatter(
        x[fg],
        y[fg],
        s=0.6,
        c=epe[fg],
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        edgecolors="none",
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, label="Endpoint error vs GT (m)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _iterate_zeroflow_until(
    tfrecord_root: Path,
    pkl_root: Path,
    feather_root: Path,
    *,
    pair_index: int | None,
    sequence_id: str | None,
    frame_index: int | None,
) -> tuple[Any, Any, Any, int]:
    from rigid_flow.data.zeroflow_loader import ZeroFlowDataSource

    source = ZeroFlowDataSource(pkl_root, feather_root, tfrecord_root)
    for idx, (pair, pred_flow, is_valid) in enumerate(source.iterate_pairs()):
        if pair_index is not None:
            if idx != pair_index:
                continue
        else:
            assert sequence_id is not None and frame_index is not None
            if pair.sequence_id != sequence_id or int(pair.frame_index) != int(frame_index):
                continue
        return pair, pred_flow, is_valid, idx
    raise LookupError(
        "No matching frame pair found. Check roots / pair_index / sequence_id+frame_index."
    )


def cmd_render(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    from rigid_flow.aggregation.rigid_aggregation import compute_rigid_flow
    from rigid_flow.core.types import BoundingBox
    from rigid_flow.geometry.points_in_boxes import points_in_boxes_cpu
    from rigid_flow.pipeline import compute_gt_flow
    from rigid_flow.visualization import bev, flow_comparison

    tf, pk, fe = Path(args.tfrecord_root), Path(args.pkl_root), Path(args.feather_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pair, pred_flow, is_valid, idx = _iterate_zeroflow_until(
        tf,
        pk,
        fe,
        pair_index=args.pair_index,
        sequence_id=args.sequence_id,
        frame_index=args.frame_index,
    )
    logger.info(
        "Matched pair_index=%d seq=%s frame_index=%d N=%d",
        idx,
        pair.sequence_id,
        pair.frame_index,
        pair.num_points_t0,
    )

    pair = compute_gt_flow(pair)
    assert pair.gt_flow is not None
    gt = pair.gt_flow

    flow_input = np.zeros_like(pred_flow)
    flow_input[is_valid] = pred_flow[is_valid]

    if pair.boxes_t0:
        boxes_array = np.stack([b.as_7dof for b in pair.boxes_t0], axis=0)
    else:
        boxes_array = np.empty((0, 7), dtype=np.float32)
    point_to_box = points_in_boxes_cpu(pair.points_t0, boxes_array)
    boxes_list: list[BoundingBox] = list(pair.boxes_t0)

    result_geo = compute_rigid_flow(
        pair.points_t0,
        flow_input,
        point_to_box,
        boxes_list,
        "geometric_median",
    )

    epe_none = np.linalg.norm(flow_input - gt, axis=1).astype(np.float32)
    epe_geo = np.linalg.norm(result_geo.flow - gt, axis=1).astype(np.float32)
    vmax = float(np.percentile(np.concatenate([epe_none, epe_geo]), 99.5))
    vmax = max(vmax, 1e-3)

    # --- Per-point EPE maps ---
    fig_a = _plot_epe_bev(
        pair,
        epe_none,
        point_to_box,
        title=f"EPE vs GT — ZeroFlow baseline (no aggregation)\n{pair.sequence_id}  frame {pair.frame_index}",
        vmax=vmax,
    )
    fig_a.savefig(out / "epe_baseline_none.png", dpi=160, bbox_inches="tight")
    plt.close(fig_a)

    fig_b = _plot_epe_bev(
        pair,
        epe_geo,
        point_to_box,
        title=f"EPE vs GT — after geometric median rigid pooling\n{pair.sequence_id}  frame {pair.frame_index}",
        vmax=vmax,
    )
    fig_b.savefig(out / "epe_after_geometric_median.png", dpi=160, bbox_inches="tight")
    plt.close(fig_b)

    # --- Per-point improvement (none EPE − geo EPE); positive = geo better ---
    delta = epe_none - epe_geo
    dmax = float(np.percentile(np.abs(delta), 99.5))
    dmax = max(dmax, 1e-4)
    pts = pair.points_t0
    x, y = pts[:, 0], pts[:, 1]
    fg = point_to_box >= 0
    bg = ~fg
    fig_d, ax_d = plt.subplots(1, 1, figsize=(11, 10))
    ax_d.scatter(x[bg], y[bg], s=0.25, c="lightgray", edgecolors="none", rasterized=True)
    sc_d = ax_d.scatter(
        x[fg],
        y[fg],
        s=0.6,
        c=delta[fg],
        cmap="RdYlGn",
        norm=Normalize(vmin=-dmax, vmax=dmax),
        edgecolors="none",
        rasterized=True,
    )
    fig_d.colorbar(sc_d, ax=ax_d, shrink=0.7, label="ΔEPE = EPE(none) − EPE(geo)  (m)")
    ax_d.set_xlim(-80, 80)
    ax_d.set_ylim(-80, 80)
    ax_d.set_aspect("equal")
    ax_d.set_title("Where rigid pooling helps (green) vs hurts (red)")
    fig_d.tight_layout()
    fig_d.savefig(out / "epe_improvement_delta.png", dpi=160, bbox_inches="tight")
    plt.close(fig_d)

    # --- Raw / corrected / correction magnitude (geometric median) ---
    fig_c = flow_comparison.plot_correction_comparison(pair, result_geo, point_to_box)
    for ax in fig_c.axes:
        if ax.get_title() == "Corrected Flow (median)":
            ax.set_title("Corrected Flow (geometric median)")
    fig_c.savefig(out / "flow_correction_geometric_median.png", dpi=160, bbox_inches="tight")
    plt.close(fig_c)

    # --- Flow magnitude: raw ZeroFlow vs pooled ---
    fig_m0 = bev.plot_flow_magnitude(
        pair,
        flow_input,
        figsize=(10, 10),
        vmax=args.flow_mag_vmax,
    )
    fig_m0.axes[0].set_title("ZeroFlow predicted flow magnitude (baseline input)")
    fig_m0.savefig(out / "flow_mag_zeroflow_input.png", dpi=160, bbox_inches="tight")
    plt.close(fig_m0)

    fig_m1 = bev.plot_flow_magnitude(
        pair,
        result_geo.flow,
        figsize=(10, 10),
        vmax=args.flow_mag_vmax,
    )
    fig_m1.axes[0].set_title("Flow magnitude after geometric median per box")
    fig_m1.savefig(out / "flow_mag_after_geometric_median.png", dpi=160, bbox_inches="tight")
    plt.close(fig_m1)

    mean_none = float(np.mean(epe_none))
    mean_geo = float(np.mean(epe_geo))
    summary = {
        "pair_index": idx,
        "sequence_id": pair.sequence_id,
        "frame_index": int(pair.frame_index),
        "mean_epe_baseline_none": mean_none,
        "mean_epe_after_geometric_median": mean_geo,
        "mean_epe_improvement_m": mean_none - mean_geo,
        "mean_epe_foreground_baseline": float(np.mean(epe_none[fg])) if np.any(fg) else None,
        "mean_epe_foreground_geo": float(np.mean(epe_geo[fg])) if np.any(fg) else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote figures + summary.json under %s", out.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_top = sub.add_parser("print-top", help="Rank frames by EPE improvement (reads JSON only).")
    p_top.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/full_run"),
        help="Directory containing none/ and geometric_median/ phase_c outputs.",
    )
    p_top.add_argument("--top", type=int, default=15, help="How many rows to print per table.")
    p_top.set_defaults(func=cmd_print_top)

    p_ren = sub.add_parser("render", help="Load one ZeroFlow pair and write PNG figures.")
    p_ren.add_argument("--tfrecord-root", type=Path, required=True)
    p_ren.add_argument("--pkl-root", type=Path, required=True)
    p_ren.add_argument("--feather-root", type=Path, required=True)
    p_ren.add_argument(
        "--pair-index",
        type=int,
        default=None,
        help="0-based index in ZeroFlowDataSource.iterate_pairs() order (matches per_frame_metrics row order).",
    )
    p_ren.add_argument("--sequence-id", type=str, default=None)
    p_ren.add_argument("--frame-index", type=int, default=None)
    p_ren.add_argument("--output-dir", type=Path, required=True)
    p_ren.add_argument(
        "--flow-mag-vmax",
        type=float,
        default=3.0,
        help="Color scale cap for flow magnitude heatmaps (m).",
    )
    p_ren.set_defaults(func=cmd_render)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "render":
        if args.pair_index is None and (
            args.sequence_id is None or args.frame_index is None
        ):
            parser.error("render requires either --pair-index or both --sequence-id and --frame-index")

    args.func(args)


if __name__ == "__main__":
    main()
