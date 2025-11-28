#!/bin/bash
# Script pour lancer la visualisation des loop closures avec matches visuels

# Usage: bash visualize_matches.sh /path/to/images [output_dir]

if [ $# -lt 1 ]; then
    echo "Usage: bash visualize_matches.sh <images_path> [output_dir]"
    echo ""
    echo "Example:"
    echo "  bash visualize_matches.sh /home/ivm/pose_graph/pgSlam/scenario/imgs/"
    echo "  bash visualize_matches.sh /path/to/images loop_matches_custom"
    exit 1
fi

IMAGES_PATH="$1"
OUTPUT_DIR="${2:-loop_matches}"

echo "Images path: $IMAGES_PATH"
echo "Output dir: $OUTPUT_DIR"
echo ""

python visualize_loop_matches.py \
    --images_path "$IMAGES_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --start_detection_frame 50 \
    --temporal_distance 50 \
    --similarity_threshold 0.55 \
    --temporal_consistency_window 2 \
    --num_patch_matches 20 \
    --device auto

echo ""
echo "✅ Done! Check the results in: $OUTPUT_DIR"
