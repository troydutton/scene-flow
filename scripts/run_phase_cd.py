"""Run Phase C and Phase D of the rigid-flow evaluation using ZeroFlow predictions.

Phase C — ZeroFlow predicted flow + GT boxes
    Uses the ZeroFlow scene flow predictions as the refinement input with
    ground-truth bounding boxes for point-to-box assignment.  Measures how
    well rigid aggregation corrects a real model's per-point noise.

Phase D — ZeroFlow predicted flow + predicted boxes
    Same ZeroFlow predictions, but boxes come from a detection submission
    ``.bin`` file at a single score threshold.  This is the fully-predicted
    end-to-end configuration.

Outputs are written to:
    results/phase_c_pred_flow_gt_boxes/
    results/phase_d_pred_flow_pred_boxes/thr_{threshold:.2f}/

A ``compare_vs_gt.json`` is written under the Phase D directory with deltas
against the Phase C aggregate (analogous to Phase B's comparison against A).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

from rigid_flow.pipeline import run_zeroflow_pipeline

logger = logging.getLogger(__name__)


def _load_aggregate(path: Path) -> dict[str, float]:
    if not path.exists():
        logger.warning("Aggregate file %s not found; skipping delta comparison", path)
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tfrecord-root",
        type=Path,
        required=True,
        help="Directory (searched recursively) containing Waymo .tfrecord files.",
    )
    parser.add_argument(
        "--pkl-root",
        type=Path,
        required=True,
        help="Root of per-segment pkl directories (e.g. results/zeroflow/validation).",
    )
    parser.add_argument(
        "--feather-root",
        type=Path,
        required=True,
        help="Root of per-segment feather directories (e.g. results/zeroflow/sequence_len_002).",
    )
    parser.add_argument(
        "--pred-boxes-bin",
        type=Path,
        default=Path("results/2_stage/detection_pred.bin"),
        help="Detection submission .bin for Phase D.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Score threshold for Phase D predicted boxes (default: 0.50).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Parent directory for phase_c_* and phase_d_* output folders.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="median",
        choices=["median", "svd"],
        help="Rigid aggregation method.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional cap on the number of frame pairs processed.",
    )
    parser.add_argument(
        "--skip-phase-c",
        action="store_true",
        help="Skip Phase C (e.g. if already computed).",
    )
    parser.add_argument(
        "--skip-phase-d",
        action="store_true",
        help="Skip Phase D.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    phase_c_dir = args.output_root / "phase_c_pred_flow_gt_boxes"
    thr = args.score_threshold
    phase_d_dir = args.output_root / "phase_d_pred_flow_pred_boxes" / f"thr_{thr:.2f}"

    # ---- Phase C: ZeroFlow + GT boxes ----
    if not args.skip_phase_c:
        logger.info("=" * 60)
        logger.info("Phase C: ZeroFlow predicted flow + GT boxes -> %s", phase_c_dir)
        logger.info("=" * 60)

        run_zeroflow_pipeline(
            tfrecord_root=args.tfrecord_root,
            pkl_root=args.pkl_root,
            feather_root=args.feather_root,
            output_dir=phase_c_dir,
            method=args.method,
            max_pairs=args.max_pairs,
        )
    else:
        logger.info("Skipping Phase C (--skip-phase-c)")

    # ---- Phase D: ZeroFlow + predicted boxes ----
    if not args.skip_phase_d:
        logger.info("=" * 60)
        logger.info(
            "Phase D: ZeroFlow predicted flow + predicted boxes (thr=%.2f) -> %s",
            thr,
            phase_d_dir,
        )
        logger.info("=" * 60)

        run_zeroflow_pipeline(
            tfrecord_root=args.tfrecord_root,
            pkl_root=args.pkl_root,
            feather_root=args.feather_root,
            output_dir=phase_d_dir,
            method=args.method,
            max_pairs=args.max_pairs,
            pred_boxes_bin=args.pred_boxes_bin,
            score_threshold=thr,
        )

        # ---- Compare Phase D vs Phase C ----
        phase_c_agg_path = phase_c_dir / "aggregate_metrics.json"
        phase_c_agg = _load_aggregate(phase_c_agg_path)

        phase_d_agg_path = phase_d_dir / "aggregate_metrics.json"
        phase_d_agg = _load_aggregate(phase_d_agg_path)

        compare_dir = args.output_root / "phase_d_pred_flow_pred_boxes"
        compare_path = compare_dir / "compare_vs_gt.json"
        compare_dir.mkdir(parents=True, exist_ok=True)

        with open(compare_path, "w") as f:
            json.dump(
                {
                    "pred_boxes_bin": str(args.pred_boxes_bin),
                    "phase_c_aggregate": str(phase_c_agg_path),
                    "score_threshold": thr,
                    "results": {
                        f"thr_{thr:.2f}": {
                            "score_threshold": thr,
                            "aggregate": phase_d_agg,
                            "delta_vs_phase_c": _delta(phase_d_agg, phase_c_agg),
                        }
                    },
                },
                f,
                indent=2,
            )
        logger.info("Wrote Phase D vs C comparison to %s", compare_path)
    else:
        logger.info("Skipping Phase D (--skip-phase-d)")


if __name__ == "__main__":
    main()
