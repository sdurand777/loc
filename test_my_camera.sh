#!/bin/bash
# Quick test with your camera calibration on first validated pair

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
    echo ""
    echo "Available files:"
    ls "$VALIDATED_DIR"/*.jpg 2>/dev/null | head -5
    exit 1
fi

echo "Testing MapAnything with your camera calibration"
echo "================================================"
echo ""

./mapanything_my_camera.sh "$LOOP_IMG" "$QUERY_IMG" test_my_camera.glb
