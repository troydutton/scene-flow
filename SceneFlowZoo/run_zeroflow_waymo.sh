#!/bin/bash
# Run ZeroFlow scene flow inference on all Waymo validation segments.
#
# Outputs: per-point ego-frame scene flow predictions saved as .feather files.
# Output dir: /data/troy/predictions/waymo_zeroflow/
#
# Each .feather file contains columns:
#   - is_valid  (bool)
#   - flow_tx_m (float32) — x-component of ego-frame flow in meters
#   - flow_ty_m (float32) — y-component
#   - flow_tz_m (float32) — z-component
#
# File naming (set by SceneFlowZoo/core_utils/model_saver.py OutputSave):
#   <output_dir>/sequence_len_002/<segment_name>/<frame_idx:010d>.feather
#
# Usage:
#   bash run_zeroflow_waymo.sh
#   bash run_zeroflow_waymo.sh --gpus 2       # multi-GPU
#   bash run_zeroflow_waymo.sh --cpu           # CPU-only mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

CONFIG="configs/fastflow3d/waymo/zeroflow_local.py"
CHECKPOINT="/data/troy/models/waymo/nsfp_distilatation_scaled_50.ckpt"
OUTPUT_DIR="/data/troy/predictions/waymo_zeroflow"

echo "=== ZeroFlow Waymo Inference ==="
echo "Config:      $CONFIG"
echo "Checkpoint:  $CHECKPOINT"
echo "Output dir:  $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Pass through any extra args (e.g. --gpus 2 or --cpu)
conda run -n flow python test_pl.py \
    "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    "$@"

echo ""
echo "=== Inference complete ==="
echo "Predictions saved to: $OUTPUT_DIR"
echo "File count: $(find $OUTPUT_DIR -name '*.feather' | wc -l)"
