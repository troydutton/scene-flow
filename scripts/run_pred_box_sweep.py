"""Run the rigid-flow pipeline with predicted boxes over a score-threshold sweep.

For each threshold in ``--thresholds`` we:

1. Run :func:`rigid_flow.pipeline.run_pipeline` with ``pred_boxes_bin`` set to
   the detection ``.bin`` file and ``score_threshold`` set accordingly.
2. Persist ``per_frame_metrics.json`` + ``aggregate_metrics.json`` under
   ``results/phase_b_gt_flow_pred_boxes/thr_{threshold:.2f}/``.
3. Compute a side-by-side delta against the existing GT-box aggregate in
   ``results/aggregate_metrics.json`` and append it to
   ``results/phase_b_gt_flow_pred_boxes/compare_vs_gt.json``.

Interpretive notes
------------------
- Phase A (GT flow + GT boxes) already yields ``epe_foreground ≈ 1.3e-4`` m
  because GT flow is nearly perfectly rigid per GT box.  Swapping in
  predicted boxes will:
    * Leave some in-GT-box points as background when no predicted box
      overlaps them — they keep raw GT flow (still correct).
    * Put some GT-background points inside false-positive predicted boxes —
      their flow is replaced by the median of that box's points, which for
      static background is close to zero and will produce small but nonzero
      degradation.
    * Split rigid bodies when multiple predicted boxes shadow one GT object;
      ``flow_variance_mean`` and ``dist_preservation_mean`` should capture
      this.
- ``epe_static/slow/fast`` use **per-point GT flow speed** (``||gt_flow||/dt``)
  on predicted-foreground points, not detector ``metadata.speed`` fields.
- ``epe_true_foreground`` (and true-fg speed buckets) score **missed** GT object
  points (pred background) against a **zero-flow baseline** so recall gaps are
  visible; other EPEs still compare the rigidified output to GT.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from rigid_flow.pipeline import run_pipeline


logger = logging.getLogger(__name__)


def _load_gt_aggregate(path: Path) -> dict[str, float]:
    if not path.exists():
        logger.warning("GT aggregate file %s not found; skipping delta comparison", path)
        return {}
    with open(path) as f:
        return json.load(f)


def _delta(pred: dict[str, float], gt: dict[str, float]) -> dict[str, float]:
    """Compute ``pred[k] - gt[k]`` for keys present in both dicts."""
    out: dict[str, float] = {}
    for k, v_pred in pred.items():
        if not isinstance(v_pred, (int, float)):
            continue
        if isinstance(v_pred, float) and math.isnan(v_pred):
            continue
        v_gt = gt.get(k)
        if not isinstance(v_gt, (int, float)):
            continue
        if isinstance(v_gt, float) and math.isnan(v_gt):
            continue
        out[f"delta_{k}"] = float(v_pred) - float(v_gt)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory (searched recursively) containing Waymo .tfrecord files.",
    )
    parser.add_argument(
        "--pred-boxes-bin",
        type=Path,
        default=Path("results/2_stage/detection_pred.bin"),
        help="Path to the Waymo detection-submission .bin file to evaluate.",
    )
    parser.add_argument(
        "--gt-aggregate",
        type=Path,
        default=Path("results/aggregate_metrics.json"),
        help="Aggregate-metrics file from the GT-boxes run, used for delta "
        "comparison.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/phase_b_gt_flow_pred_boxes"),
        help="Root directory for this sweep's outputs.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
        help="Score thresholds to sweep (default: 0.1 0.3 0.5).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="median",
        choices=["none", "mean", "median", "weighted_median", "geometric_median", "svd"],
        help="Rigid aggregation method.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on the number of frame pairs processed.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    gt_aggregate = _load_gt_aggregate(args.gt_aggregate)

    summary: dict[str, dict[str, float]] = {}

    for threshold in args.thresholds:
        thr_dir = args.output_root / f"thr_{threshold:.2f}"
        thr_dir.mkdir(parents=True, exist_ok=True)
        logger.info("=== Running sweep at score_threshold=%.2f -> %s ===", threshold, thr_dir)

        run_pipeline(
            data_root=args.data_root,
            output_dir=thr_dir,
            method=args.method,
            max_pairs=args.max_pairs,
            pred_boxes_bin=args.pred_boxes_bin,
            score_threshold=threshold,
        )

        aggregate_path = thr_dir / "aggregate_metrics.json"
        with open(aggregate_path) as f:
            pred_aggregate = json.load(f)

        summary[f"thr_{threshold:.2f}"] = {
            "score_threshold": float(threshold),
            "aggregate": pred_aggregate,
            "delta_vs_gt_boxes": _delta(pred_aggregate, gt_aggregate),
        }

    compare_path = args.output_root / "compare_vs_gt.json"
    with open(compare_path, "w") as f:
        json.dump(
            {
                "pred_boxes_bin": str(args.pred_boxes_bin),
                "gt_aggregate": str(args.gt_aggregate),
                "results": summary,
            },
            f,
            indent=2,
        )
    logger.info("Wrote sweep comparison to %s", compare_path)


if __name__ == "__main__":
    main()
