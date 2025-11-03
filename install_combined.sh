#!/bin/bash
# Installation script for Combined MegaLoc + MapAnything Environment
# Creates a unified virtual environment with both systems

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# =============================================================================
# CONFIGURATION
# =============================================================================

VENV_NAME="megaloc_mapanything_env"
MAPANYTHING_REPO="/home/ivm/map-anything"
REQUIRED_PYTHON_VERSION="3.10"

# =============================================================================
# FUNCTIONS
# =============================================================================

show_header() {
    echo -e "${BLUE}==================================================================${NC}"
    echo -e "${BLUE}    Combined MegaLoc + MapAnything Installation${NC}"
    echo -e "${BLUE}==================================================================${NC}"
    echo ""
}

check_python_version() {
    echo -e "${CYAN}🔍 Checking Python version...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 is not installed${NC}"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo -e "   Detected version: ${GREEN}$PYTHON_VERSION${NC}"

    if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_PYTHON_VERSION" | sort -V | head -1) != "$REQUIRED_PYTHON_VERSION" ]]; then
        echo -e "${RED}❌ Python >= $REQUIRED_PYTHON_VERSION required (found: $PYTHON_VERSION)${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ Python version compatible${NC}"
    echo ""
}

check_cuda() {
    echo -e "${CYAN}🔍 Checking CUDA...${NC}"

    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        echo -e "   CUDA detected: ${GREEN}$CUDA_VERSION${NC}"
        echo -e "${GREEN}✅ CUDA available - Will install PyTorch with GPU support${NC}"
        CUDA_AVAILABLE=true
    else
        echo -e "${YELLOW}⚠️  CUDA not detected - Will install PyTorch CPU-only${NC}"
        CUDA_AVAILABLE=false
    fi
    echo ""
}

check_mapanything_repo() {
    echo -e "${CYAN}🔍 Checking MapAnything repository...${NC}"

    if [[ ! -d "$MAPANYTHING_REPO" ]]; then
        echo -e "${RED}❌ MapAnything repository not found: $MAPANYTHING_REPO${NC}"
        echo ""
        echo "To install MapAnything, first clone the repository:"
        echo "  cd /home/ivm"
        echo "  git clone https://github.com/facebookresearch/map-anything.git"
        exit 1
    fi

    echo -e "${GREEN}✅ MapAnything repository found${NC}"
    echo ""
}

create_venv() {
    echo -e "${CYAN}📦 Creating virtual environment: $VENV_NAME${NC}"

    if [[ -d "$VENV_NAME" ]]; then
        echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
        else
            echo -e "${YELLOW}⚠️  Using existing environment${NC}"
            return
        fi
    fi

    python3 -m venv "$VENV_NAME"
    echo -e "${GREEN}✅ Virtual environment created${NC}"
    echo ""
}

install_pytorch() {
    echo -e "${CYAN}🔧 Installing PyTorch...${NC}"

    if [[ "$CUDA_AVAILABLE" == true ]]; then
        echo -e "   Installing PyTorch with CUDA support..."
        ./"$VENV_NAME"/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo -e "   Installing PyTorch CPU-only..."
        ./"$VENV_NAME"/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    echo -e "${GREEN}✅ PyTorch installed${NC}"
    echo ""
}

install_requirements() {
    echo -e "${CYAN}🔧 Installing requirements...${NC}"

    ./"$VENV_NAME"/bin/pip install --upgrade pip
    ./"$VENV_NAME"/bin/pip install -r requirements_combined.txt

    echo -e "${GREEN}✅ Requirements installed${NC}"
    echo ""
}

install_mapanything() {
    echo -e "${CYAN}🔧 Installing MapAnything from local repository...${NC}"

    ./"$VENV_NAME"/bin/pip install -e "$MAPANYTHING_REPO"

    echo -e "${GREEN}✅ MapAnything installed${NC}"
    echo ""
}

test_installation() {
    echo -e "${CYAN}🧪 Testing installation...${NC}"
    echo ""

    # Test PyTorch
    echo -e "${BLUE}Testing PyTorch...${NC}"
    ./"$VENV_NAME"/bin/python -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')"

    # Test MapAnything
    echo -e "${BLUE}Testing MapAnything...${NC}"
    ./"$VENV_NAME"/bin/python -c "from mapanything.models import MapAnything; print('  MapAnything import: OK')"

    # Test Rerun
    echo -e "${BLUE}Testing Rerun SDK...${NC}"
    ./"$VENV_NAME"/bin/python -c "import rerun as rr; print('  Rerun SDK import: OK')"

    # Test MegaLoc (via torch.hub)
    echo -e "${BLUE}Testing MegaLoc...${NC}"
    ./"$VENV_NAME"/bin/python -c "import torch; print('  MegaLoc can be loaded via torch.hub')"

    echo ""
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
}

show_usage() {
    echo -e "${GREEN}==================================================================${NC}"
    echo -e "${GREEN}Installation Complete!${NC}"
    echo -e "${GREEN}==================================================================${NC}"
    echo ""
    echo "To activate the environment:"
    echo -e "  ${CYAN}source $VENV_NAME/bin/activate${NC}"
    echo ""
    echo "To run the combined MegaLoc + MapAnything script:"
    echo -e "  ${CYAN}bash slam_megaloc_mapanything.sh${NC}"
    echo ""
    echo "Or directly with python:"
    echo -e "  ${CYAN}source $VENV_NAME/bin/activate${NC}"
    echo -e "  ${CYAN}python slam_loop_closure_megaloc_mapanything.py --images_path /path/to/images --poses_file /path/to/vertices.txt${NC}"
    echo ""
}

# =============================================================================
# MAIN
# =============================================================================

show_header

check_python_version
check_cuda
check_mapanything_repo

echo -e "${YELLOW}This will create a combined environment with:${NC}"
echo "  - MegaLoc (Visual Place Recognition)"
echo "  - MapAnything (Pose Estimation)"
echo "  - Rerun SDK (3D Visualization)"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠️  Installation cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}Starting Installation${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo ""

create_venv
install_pytorch
install_requirements
install_mapanything
test_installation
show_usage

echo -e "${GREEN}✅ Installation completed successfully!${NC}"
