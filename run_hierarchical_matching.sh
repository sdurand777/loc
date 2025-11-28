#!/bin/bash
# Script to run hierarchical matching pipeline
# MegaLoc → SuperPoint → LightGlue

# Default parameters
IMAGES_PATH="${1:-/path/to/images}"
OUTPUT_DIR="${2:-hierarchical_matches}"
NUM_PATCHES="${3:-30}"
MAX_FRAMES="${4:-}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🎯 HIERARCHICAL MATCHING PIPELINE${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo -e "  Pipeline: ${GREEN}MegaLoc → SuperPoint → LightGlue${NC}"
echo ""
echo -e "  Images path:        ${YELLOW}${IMAGES_PATH}${NC}"
echo -e "  Output directory:   ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  MegaLoc patches:    ${YELLOW}${NUM_PATCHES}${NC}"
if [ -n "$MAX_FRAMES" ]; then
    echo -e "  Max frames:         ${YELLOW}${MAX_FRAMES}${NC}"
fi
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check if images path exists
if [ ! -d "$IMAGES_PATH" ]; then
    echo -e "${YELLOW}⚠️  Warning: Images path does not exist: ${IMAGES_PATH}${NC}"
    echo ""
    echo "Usage: $0 <images_path> [output_dir] [num_patches] [max_frames]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/images hierarchical_matches 10 100"
    echo ""
    exit 1
fi

# Build command
CMD="python hierarchical_matching.py \
    --images_path \"${IMAGES_PATH}\" \
    --output_dir \"${OUTPUT_DIR}\" \
    --num_megaloc_patches ${NUM_PATCHES} \
    --start_detection_frame 50 \
    --temporal_distance 50 \
    --similarity_threshold 0.55 \
    --roi_expansion 1.5 \
    --device auto"

if [ -n "$MAX_FRAMES" ]; then
    CMD="${CMD} --max_frames ${MAX_FRAMES}"
fi

# Run the command
echo -e "${GREEN}Running:${NC} ${CMD}"
echo ""
eval $CMD

echo ""
echo -e "${GREEN}✅ Done!${NC}"
echo -e "Results saved to: ${YELLOW}${OUTPUT_DIR}${NC}"
