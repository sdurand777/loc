#!/bin/bash
# Installation script for Hierarchical Matching Pipeline
# MegaLoc → SuperPoint → LightGlue

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Installing Hierarchical Matching Pipeline Dependencies${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python version: ${PYTHON_VERSION}"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}✗ pip3 not found. Please install pip first.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} pip3 found"

# Ask for installation method
echo ""
echo -e "${YELLOW}Choose installation method:${NC}"
echo "  1) System-wide (pip install --user)"
echo "  2) Virtual environment (recommended)"
echo ""
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo -e "\n${BLUE}Installing system-wide (--user)...${NC}\n"
        PIP_CMD="pip3 install --user"
        ;;
    2)
        echo -e "\n${BLUE}Creating virtual environment...${NC}\n"
        VENV_NAME="matching_env"

        if [ -d "$VENV_NAME" ]; then
            echo -e "${YELLOW}Virtual environment already exists.${NC}"
            read -p "Delete and recreate? [y/N]: " recreate
            if [[ $recreate == "y" || $recreate == "Y" ]]; then
                rm -rf "$VENV_NAME"
                python3 -m venv "$VENV_NAME"
            fi
        else
            python3 -m venv "$VENV_NAME"
        fi

        source "$VENV_NAME/bin/activate"
        PIP_CMD="pip install"
        echo -e "${GREEN}✓${NC} Virtual environment created: $VENV_NAME"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Step 1: Installing core dependencies${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Install PyTorch (check for CUDA)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} NVIDIA GPU detected"
    echo -e "${BLUE}Installing PyTorch with CUDA support...${NC}"
    $PIP_CMD torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}⚠${NC}  No NVIDIA GPU detected, installing CPU-only PyTorch"
    $PIP_CMD torch torchvision
fi

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Step 2: Installing requirements${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Install other requirements
$PIP_CMD numpy Pillow opencv-python kornia matplotlib scipy tqdm

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Step 3: Installing LightGlue from GitHub${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Install LightGlue from GitHub
echo -e "${BLUE}Cloning and installing LightGlue...${NC}"
$PIP_CMD git+https://github.com/cvg/LightGlue.git

echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  Step 4: Verifying installation${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Test imports
python3 << 'EOF'
import sys
import torch
print(f"✓ PyTorch {torch.__version__} installed")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

try:
    from lightglue import LightGlue, SuperPoint
    print("✓ LightGlue installed successfully")
except ImportError as e:
    print(f"✗ LightGlue import failed: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} installed")
except ImportError:
    print("✗ OpenCV not found")
    sys.exit(1)

try:
    import kornia
    print(f"✓ Kornia installed")
except ImportError:
    print("✗ Kornia not found")
    sys.exit(1)

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__} installed")
except ImportError:
    print("✗ Matplotlib not found")
    sys.exit(1)

print("\n✓ All dependencies installed successfully!")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}  ✓ Installation Complete!${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo ""

    if [[ $choice == 2 ]]; then
        echo -e "${YELLOW}To activate the virtual environment:${NC}"
        echo -e "  ${BLUE}source $VENV_NAME/bin/activate${NC}"
        echo ""
    fi

    echo -e "${YELLOW}To test the pipeline:${NC}"
    echo -e "  ${BLUE}python hierarchical_matching.py --help${NC}"
    echo ""
    echo -e "${YELLOW}Quick start:${NC}"
    echo -e "  ${BLUE}bash run_hierarchical_matching.sh /path/to/images${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}======================================================================${NC}"
    echo -e "${RED}  ✗ Installation Failed${NC}"
    echo -e "${RED}======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Please check the error messages above.${NC}"
    exit 1
fi
