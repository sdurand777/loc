#!/bin/bash
# Wrapper script for MapAnything with custom intrinsics

VENV_PATH="megaloc_mapanything_env"

# Check if venv exists
if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please install first:"
    echo "  bash install_combined.sh"
    exit 1
fi

# Check arguments
if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <loop.jpg> <query.jpg> <fx> <fy> <cx> <cy> [output.glb]"
    echo ""
    echo "Example:"
    echo "  $0 loop.jpg query.jpg 500.0 500.0 320.0 240.0 scene.glb"
    echo ""
    echo "Intrinsics matrix K:"
    echo "  K = [fx  0  cx]"
    echo "      [0  fy  cy]"
    echo "      [0   0   1]"
    echo ""
    echo "Where:"
    echo "  fx, fy: Focal lengths in pixels"
    echo "  cx, cy: Principal point (optical center) in pixels"
    exit 1
fi

LOOP_IMAGE="$1"
QUERY_IMAGE="$2"
FX="$3"
FY="$4"
CX="$5"
CY="$6"
OUTPUT="${7:-scene.glb}"

# Activate venv
source "$VENV_PATH/bin/activate"

# Run script
python mapanything_pair_with_intrinsics.py \
    --loop "$LOOP_IMAGE" \
    --query "$QUERY_IMAGE" \
    --fx "$FX" --fy "$FY" \
    --cx "$CX" --cy "$CY" \
    --output "$OUTPUT"

echo ""
echo "To view the 3D model:"
echo "  Online: https://gltf-viewer.donmccurdy.com/"
echo "  Blender: blender $OUTPUT"
