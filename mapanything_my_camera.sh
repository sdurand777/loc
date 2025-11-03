#!/bin/bash
# MapAnything with your specific camera intrinsics
# Camera calibration parameters from your setup

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
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <loop.jpg> <query.jpg> [output.glb]"
    echo ""
    echo "Example:"
    echo "  $0 loop_pairs_validated/double_validated/loop_1.jpg \\"
    echo "     loop_pairs_validated/double_validated/query_1.jpg \\"
    echo "     my_reconstruction.glb"
    echo ""
    echo "Camera parameters (from your calibration):"
    echo "  fx = 322.580, fy = 322.580"
    echo "  cx = 259.260, cy = 184.882"
    echo "  Distortion: k1=-0.070162, k2=0.075512, p1=0.001229, p2=0.000993, k3=-0.018172"
    exit 1
fi

LOOP_IMAGE="$1"
QUERY_IMAGE="$2"
OUTPUT="${3:-scene_my_camera.glb}"

# Your camera intrinsics
# K_l = np.array([322.580, 0.0, 259.260,
#                 0.0, 322.580, 184.882,
#                 0.0, 0.0, 1.0]).reshape(3,3)
FX=322.580
FY=322.580
CX=259.260
CY=184.882

# Your distortion coefficients
# d_l = np.array([-0.070162237, 0.07551153, 0.0012286149, 0.00099302817, -0.018171599])
K1=-0.070162237
K2=0.07551153
P1=0.0012286149
P2=0.00099302817
K3=-0.018171599

echo "================================"
echo "MapAnything - Your Camera Calibration"
echo "================================"
echo "Loop:  $LOOP_IMAGE"
echo "Query: $QUERY_IMAGE"
echo "Output: $OUTPUT"
echo ""
echo "Camera Intrinsics:"
echo "  fx = $FX"
echo "  fy = $FY"
echo "  cx = $CX"
echo "  cy = $CY"
echo ""
echo "Distortion Coefficients:"
echo "  k1 = $K1"
echo "  k2 = $K2"
echo "  p1 = $P1"
echo "  p2 = $P2"
echo "  k3 = $K3"
echo ""
echo "⚠️  Images will be undistorted before processing"
echo "================================"
echo ""

# Activate venv
source "$VENV_PATH/bin/activate"

# Run with undistortion
python mapanything_pair_with_intrinsics.py \
    --loop "$LOOP_IMAGE" \
    --query "$QUERY_IMAGE" \
    --fx $FX --fy $FY \
    --cx $CX --cy $CY \
    --k1 $K1 --k2 $K2 \
    --p1 $P1 --p2 $P2 \
    --k3 $K3 \
    --undistort \
    --output "$OUTPUT"

echo ""
echo "================================"
echo "✅ Processing complete!"
echo "================================"
echo "3D model saved: $OUTPUT"
echo ""
echo "Undistorted images saved as:"
echo "  - undistorted_$(basename $LOOP_IMAGE)"
echo "  - undistorted_$(basename $QUERY_IMAGE)"
echo ""
echo "To view the 3D model:"
echo "  - Online: https://gltf-viewer.donmccurdy.com/"
echo "  - Blender: blender $OUTPUT"
echo ""
