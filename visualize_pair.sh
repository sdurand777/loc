#!/bin/bash
# Quick script to visualize a loop/query pair with MapAnything

VENV_PATH="megaloc_mapanything_env"

# Check if venv exists
if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please install first:"
    echo "  bash install_combined.sh"
    exit 1
fi

# Activate venv
source "$VENV_PATH/bin/activate"

# Check arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <loop.jpg> <query.jpg> [output.ply]"
    echo ""
    echo "Example:"
    echo "  $0 loop_pairs_validated/double_validated/loop_1.jpg loop_pairs_validated/double_validated/query_1.jpg"
    exit 1
fi

LOOP_IMAGE="$1"
QUERY_IMAGE="$2"
OUTPUT="${3:-scene.ply}"

# Run script
python mapanything_visualize_pair.py \
    --loop "$LOOP_IMAGE" \
    --query "$QUERY_IMAGE" \
    --output "$OUTPUT"

echo ""
echo "To visualize the 3D point cloud:"
echo "  python -c 'import open3d as o3d; pcd = o3d.io.read_point_cloud(\"$OUTPUT\"); o3d.visualization.draw_geometries([pcd])'"
