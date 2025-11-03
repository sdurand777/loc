#!/bin/bash
# Test script to visualize the first validated loop pair

VENV_PATH="megaloc_mapanything_env"

# Check if venv exists
if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ Virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please install first:"
    echo "  bash install_combined.sh"
    exit 1
fi

# Find first validated pair
VALIDATED_DIR="loop_pairs_validated/double_validated"

if [[ ! -d "$VALIDATED_DIR" ]]; then
    echo "❌ No validated loops found in: $VALIDATED_DIR"
    echo ""
    echo "Run slam_megaloc_mapanything.sh first to generate validated pairs"
    exit 1
fi

LOOP_IMG=$(ls "$VALIDATED_DIR"/loop_1.jpg 2>/dev/null | head -1)
QUERY_IMG=$(ls "$VALIDATED_DIR"/query_1.jpg 2>/dev/null | head -1)

if [[ -z "$LOOP_IMG" || -z "$QUERY_IMG" ]]; then
    echo "❌ Could not find loop_1.jpg and query_1.jpg in $VALIDATED_DIR"
    echo ""
    echo "Available files:"
    ls "$VALIDATED_DIR"/*.jpg 2>/dev/null | head -10
    exit 1
fi

echo "Testing with first validated pair:"
echo "  Loop:  $LOOP_IMG"
echo "  Query: $QUERY_IMG"
echo ""

# Run visualization
./visualize_pair.sh "$LOOP_IMG" "$QUERY_IMG" test_scene.ply
