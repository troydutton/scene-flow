#!/usr/bin/env bash
#
# Run all four phases (A-D) with every aggregation method.
#
# Results are written to:
#   results/{method}/phase_a_gt_flow_gt_boxes/
#   results/{method}/phase_b_gt_flow_pred_boxes/
#   results/{method}/phase_c_pred_flow_gt_boxes/
#   results/{method}/phase_d_pred_flow_pred_boxes/
#
# Usage:
#   bash scripts/run_all_methods.sh \
#     --tfrecord-root /path/to/tfrecords \
#     --pkl-root results/zeroflow/validation \
#     --feather-root results/zeroflow/sequence_len_002
#
# Optional flags:
#   --pred-boxes-bin PATH   (default: results/2_stage/detection_pred.bin)
#   --score-threshold VAL   (default: 0.50)
#   --max-pairs N           (limit frame pairs for debugging)
#   --methods "m1 m2 ..."   (override method list; default: all six)

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
TFRECORD_ROOT=""
PKL_ROOT=""
FEATHER_ROOT=""
PRED_BOXES_BIN="results/2_stage/detection_pred.bin"
SCORE_THRESHOLD="0.50"
MAX_PAIRS=""
METHODS="none mean median weighted_median geometric_median svd"

# ── Parse arguments ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tfrecord-root)  TFRECORD_ROOT="$2";   shift 2 ;;
        --pkl-root)       PKL_ROOT="$2";        shift 2 ;;
        --feather-root)   FEATHER_ROOT="$2";    shift 2 ;;
        --pred-boxes-bin) PRED_BOXES_BIN="$2";  shift 2 ;;
        --score-threshold) SCORE_THRESHOLD="$2"; shift 2 ;;
        --max-pairs)      MAX_PAIRS="$2";       shift 2 ;;
        --methods)        METHODS="$2";         shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$TFRECORD_ROOT" ]]; then
    echo "Error: --tfrecord-root is required" >&2; exit 1
fi
if [[ -z "$PKL_ROOT" ]]; then
    echo "Error: --pkl-root is required" >&2; exit 1
fi
if [[ -z "$FEATHER_ROOT" ]]; then
    echo "Error: --feather-root is required" >&2; exit 1
fi

MAX_PAIRS_FLAG=""
if [[ -n "$MAX_PAIRS" ]]; then
    MAX_PAIRS_FLAG="--max-pairs $MAX_PAIRS"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "======================================================================"
echo "Aggregation method sweep"
echo "  tfrecord-root:  $TFRECORD_ROOT"
echo "  pkl-root:       $PKL_ROOT"
echo "  feather-root:   $FEATHER_ROOT"
echo "  pred-boxes-bin: $PRED_BOXES_BIN"
echo "  threshold:      $SCORE_THRESHOLD"
echo "  methods:        $METHODS"
echo "  max-pairs:      ${MAX_PAIRS:-all}"
echo "======================================================================"

for METHOD in $METHODS; do
    OUTPUT_ROOT="results/${METHOD}"
    echo ""
    echo "======================================================================"
    echo "  METHOD: $METHOD  ->  $OUTPUT_ROOT"
    echo "======================================================================"

    # ── Phase A: GT flow + GT boxes ───────────────────────────────────────
    PHASE_A_DIR="$OUTPUT_ROOT/phase_a_gt_flow_gt_boxes"
    if [[ -f "$PHASE_A_DIR/aggregate_metrics.json" ]]; then
        echo "[Phase A] $METHOD — already exists, skipping"
    else
        echo "[Phase A] $METHOD — GT flow + GT boxes"
        python3 -m rigid_flow.pipeline \
            --data-root "$TFRECORD_ROOT" \
            --output-dir "$PHASE_A_DIR" \
            --method "$METHOD" \
            $MAX_PAIRS_FLAG
    fi

    # ── Phase B: GT flow + predicted boxes ────────────────────────────────
    PHASE_B_DIR="$OUTPUT_ROOT/phase_b_gt_flow_pred_boxes"
    if [[ -f "$PHASE_B_DIR/thr_${SCORE_THRESHOLD}/aggregate_metrics.json" ]]; then
        echo "[Phase B] $METHOD — already exists, skipping"
    else
        echo "[Phase B] $METHOD — GT flow + predicted boxes (thr=$SCORE_THRESHOLD)"
        python3 scripts/run_pred_box_sweep.py \
            --data-root "$TFRECORD_ROOT" \
            --pred-boxes-bin "$PRED_BOXES_BIN" \
            --gt-aggregate "$PHASE_A_DIR/aggregate_metrics.json" \
            --output-root "$PHASE_B_DIR" \
            --thresholds "$SCORE_THRESHOLD" \
            --method "$METHOD" \
            $MAX_PAIRS_FLAG
    fi

    # ── Phase C: ZeroFlow + GT boxes ──────────────────────────────────────
    PHASE_C_DIR="$OUTPUT_ROOT/phase_c_pred_flow_gt_boxes"
    if [[ -f "$PHASE_C_DIR/aggregate_metrics.json" ]]; then
        echo "[Phase C] $METHOD — already exists, skipping"
    else
        echo "[Phase C] $METHOD — ZeroFlow + GT boxes"
        python3 scripts/run_phase_cd.py \
            --tfrecord-root "$TFRECORD_ROOT" \
            --pkl-root "$PKL_ROOT" \
            --feather-root "$FEATHER_ROOT" \
            --pred-boxes-bin "$PRED_BOXES_BIN" \
            --score-threshold "$SCORE_THRESHOLD" \
            --output-root "$OUTPUT_ROOT" \
            --method "$METHOD" \
            --skip-phase-d \
            $MAX_PAIRS_FLAG
    fi

    # ── Phase D: ZeroFlow + predicted boxes ───────────────────────────────
    PHASE_D_DIR="$OUTPUT_ROOT/phase_d_pred_flow_pred_boxes/thr_${SCORE_THRESHOLD}"
    if [[ -f "$PHASE_D_DIR/aggregate_metrics.json" ]]; then
        echo "[Phase D] $METHOD — already exists, skipping"
    else
        echo "[Phase D] $METHOD — ZeroFlow + predicted boxes (thr=$SCORE_THRESHOLD)"
        python3 scripts/run_phase_cd.py \
            --tfrecord-root "$TFRECORD_ROOT" \
            --pkl-root "$PKL_ROOT" \
            --feather-root "$FEATHER_ROOT" \
            --pred-boxes-bin "$PRED_BOXES_BIN" \
            --score-threshold "$SCORE_THRESHOLD" \
            --output-root "$OUTPUT_ROOT" \
            --method "$METHOD" \
            --skip-phase-c \
            $MAX_PAIRS_FLAG
    fi

    echo "[Done] $METHOD — all phases complete"
done

echo ""
echo "======================================================================"
echo "All methods complete. Results in results/{method}/phase_{a,b,c,d}_*/"
echo "======================================================================"
