#!/bin/bash
# Launch script for SLAM Loop Closure with MegaLoc + MapAnything Double Validation
# This script uses the combined virtual environment

VENV_PATH="megaloc_mapanything_env"

# Check if venv exists
if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please install first:"
    echo "  bash install_combined.sh"
    exit 1
fi

# Activate venv and run script
source "$VENV_PATH/bin/activate"

python slam_loop_closure_megaloc_mapanything.py \
    --images_path /home/ivm/pose_graph/pgSlam/scenario/imgs/ \
    --poses_file /home/ivm/pose_graph/pgSlam/scenario/vertices_stan.txt \
    --fps 10 \
    --start_detection_frame 50 \
    --temporal_distance 50 \
    --megaloc_similarity_threshold 0.6 \
    --mapanything_confidence_threshold 1.2 \
    --temporal_consistency_window 2 \
    --max_frames 1100

# Note: Adjust the paths and parameters above for your dataset
#
# Key parameters:
#   --megaloc_similarity_threshold: MegaLoc detection threshold (default: 0.55)
#   --mapanything_confidence_threshold: MapAnything validation threshold (default: 0.5)
#   --images_path: Path to your image sequence
#   --poses_file: Path to your poses CSV file
