#!/bin/bash
# Test script for MapAnything with custom intrinsics

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

# Example with custom intrinsics
# Adjust these values for your camera!
FX=500.0
FY=500.0
CX=320.0
CY=240.0

# Optional: distortion coefficients
K1=0.0
K2=0.0
P1=0.0
P2=0.0
K3=0.0

# Find first validated pair
VALIDATED_DIR="loop_pairs_validated/double_validated"

if [[ ! -d "$VALIDATED_DIR" ]]; then
    echo "❌ No validated loops found in: $VALIDATED_DIR"
    echo ""
    echo "Run slam_megaloc_mapanything.sh first to generate validated pairs"
    exit 1
fi

LOOP_IMG="$VALIDATED_DIR/loop_1.jpg"
QUERY_IMG="$VALIDATED_DIR/query_1.jpg"

if [[ ! -f "$LOOP_IMG" || ! -f "$QUERY_IMG" ]]; then
    echo "❌ Could not find loop_1.jpg and query_1.jpg in $VALIDATED_DIR"
    exit 1
fi

echo "================================"
echo "Testing MapAnything with custom intrinsics"
echo "================================"
echo "Loop:  $LOOP_IMG"
echo "Query: $QUERY_IMG"
echo ""
echo "Intrinsics:"
echo "  fx = $FX, fy = $FY"
echo "  cx = $CX, cy = $CY"
echo ""
echo "Distortion: k1=$K1, k2=$K2, p1=$P1, p2=$P2, k3=$K3"
echo "================================"
echo ""

# Run script
python mapanything_pair_with_intrinsics.py \
    --loop "$LOOP_IMG" \
    --query "$QUERY_IMG" \
    --fx $FX --fy $FY --cx $CX --cy $CY \
    --k1 $K1 --k2 $K2 --p1 $P1 --p2 $P2 --k3 $K3 \
    --output custom_intrinsics_scene.glb

echo ""
echo "================================"
echo "3D model saved: custom_intrinsics_scene.glb"
echo "================================"
echo ""
echo "To view online: https://gltf-viewer.donmccurdy.com/"
echo "Or with Blender: blender custom_intrinsics_scene.glb"
