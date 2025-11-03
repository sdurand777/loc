# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MegaLoc is a state-of-the-art image retrieval model for Visual Place Recognition (VPR) tasks. It achieves SOTA performance on most VPR datasets, including both indoor and outdoor scenarios. The model is designed to retrieve visually similar images from large databases for localization purposes.

**Key Resources:**
- [ArXiv Paper](https://arxiv.org/abs/2502.17237)
- [Gradio Demo](https://11fc3a5b420e6672fe.gradio.live/)
- [Model on HuggingFace](https://huggingface.co/gberton/MegaLoc)
- [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) - Tool for comprehensive evaluation

## Model Architecture

The model is composed of three main components:

1. **Backbone (DINOv2)**:
   - Uses `dinov2_vitb14` from Facebook Research
   - Extracts patch tokens (features) and CLS token from images
   - Outputs 768-channel feature maps at 1/14 resolution of input
   - Located in `megaloc_model.py:72-84`

2. **Aggregator (SALAD - Sinkhorn Algorithm for Locally Aggregated Descriptors)**:
   - Converts spatial features into a compact global descriptor
   - Uses optimal transport (Sinkhorn algorithm) for feature aggregation
   - Produces descriptors of dimension: `num_clusters * cluster_dim + token_dim`
   - Default: 64 clusters × 256-dim + 256-dim token = 16,640-dim
   - Projects to final `feat_dim=8448` via linear layer
   - Located in `megaloc_model.py:134-221`

3. **MegaLocModel Pipeline**:
   - Resizes images to nearest multiple of 14 (required by DINOv2)
   - Passes through backbone → aggregator → L2 normalization
   - Returns L2-normalized feature vectors for similarity comparison
   - Located in `megaloc_model.py:13-49`

### Key Implementation Details

- **Image Preprocessing**: Input images must have dimensions divisible by 14. The model automatically resizes to the nearest multiple using antialiasing.
- **Optimal Transport**: The SALAD aggregator uses log-domain Sinkhorn iterations (adapted from OpenGlue, MIT license) to compute soft assignment of features to clusters.
- **Dustbin Parameter**: A learnable parameter `dust_bin` handles outlier features that don't match any cluster well.
- **Feature Normalization**: All features are L2-normalized at multiple stages (cluster features, token, final output) to enable cosine similarity matching.

## Using the Model

### Basic Usage via torch.hub
```python
import torch
model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
```

The `hubconf.py` file defines the torch.hub interface and automatically downloads pretrained weights from GitHub releases (v1.0).

### Dependencies
- torch
- torchvision
- DINOv2 (loaded via torch.hub from facebookresearch/dinov2)

### Model Weights
Pretrained weights are hosted at:
`https://github.com/gmberton/MegaLoc/releases/download/v1.0/megaloc.torch`

## Installation

### Quick Installation (Automated Script)

The easiest way to install all dependencies is using the automated installation script:

```bash
bash install_megaloc.sh
```

The script offers two installation modes:
1. **System-wide installation** - Installs packages with `pip install --user`
2. **Virtual environment** - Creates and uses a Python virtual environment (recommended)

The script will:
- Detect your Python version
- Auto-detect GPU and install appropriate PyTorch version (CUDA or CPU)
- Install all required dependencies (torch, torchvision, Pillow, numpy, rerun-sdk)
- Install optional dependencies (scipy, tqdm)
- Run tests to verify installation

### Manual Installation

If you prefer manual installation, use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Note**: For GPU support with CUDA, install PyTorch separately first:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Combined Installation (MegaLoc + MapAnything)

For the double validation system, use the combined installation:

```bash
bash install_combined.sh
```

This creates a unified environment with:
- MegaLoc (Visual Place Recognition)
- MapAnything (Pose Estimation from Meta)
- Rerun SDK (3D Visualization)
- All required dependencies

**Prerequisites**:
- MapAnything repository must be cloned first:
  ```bash
  cd /home/ivm
  git clone https://github.com/facebookresearch/map-anything.git
  ```

The script will:
- Create a virtual environment `megaloc_mapanything_env`
- Auto-detect GPU and install appropriate PyTorch version
- Install all dependencies from `requirements_combined.txt`
- Install MapAnything from the local repository

### Required Dependencies
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Image transforms and utilities
- `Pillow>=9.0.0` - Image loading and processing
- `numpy>=1.23.0` - Numerical computing
- `rerun-sdk>=0.15.0` - Real-time 3D visualization
- `scipy>=1.9.0` - Scientific computing (optional)
- `tqdm>=4.64.0` - Progress bars (optional)
- `transformers>=4.30.0` - For MapAnything (combined installation only)
- `accelerate>=0.20.0` - For MapAnything (combined installation only)

## Code Attribution

Much of the SALAD aggregator code is adapted from:
- [SALAD](https://github.com/serizba/salad)
- [OpenGlue](https://github.com/ucuapps/OpenGlue) (MIT license) - specifically the optimal transport solver

## Citation

When using this codebase, cite:
```
@InProceedings{Berton_2025_CVPR,
    author    = {Berton, Gabriele and Masone, Carlo},
    title     = {MegaLoc: One Retrieval to Place Them All},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {2861-2867}
}
```

## SLAM Loop Closure Detection

This repository includes custom scripts for SLAM loop closure detection using MegaLoc:

### Available Scripts

1. **`slam_loop_closure.py`** - Basic loop closure detection with static visualizations
   - Processes frames sequentially and detects loop closures
   - Saves visualization images for each detected loop closure
   - Usage: `python slam_loop_closure.py --images_path /path/to/images --fps 10`
   - Located in `slam_loop_closure.py`

2. **`slam_loop_closure_matplotlib.py`** - Matplotlib-based visualization (offline)
   - Optimized for post-processing visualization
   - Generates final 3D trajectory with all loop closures at once
   - No real-time updates - all processing happens first, then visualization
   - Usage: `python slam_loop_closure_matplotlib.py --images_path /path/to/images --poses_file /path/to/vertices.txt`
   - Located in `slam_loop_closure_matplotlib.py`
   - **Note**: Not recommended for real-time interaction due to matplotlib's blocking nature

3. **`slam_loop_closure_realtime.py`** - Real-time 3D visualization with Open3D (threading)
   - Live 3D trajectory visualization during processing
   - Multi-threaded architecture: processing thread + visualization thread
   - Batch updates for improved performance
   - Color-coded loop closures (green=consistent, orange=inconsistent)
   - Spheres mark loop closure locations
   - Usage: `python slam_loop_closure_realtime.py --images_path /path/to/images --poses_file /path/to/vertices.txt`
   - Additional parameters:
     - `--update_every N`: Update visualization every N frames (default: 10)
     - `--invert_poses` / `--no_invert_poses`: Control pose inversion (default: inverts cam-to-world → world-to-cam)
   - Located in `slam_loop_closure_realtime.py`
   - **Note**: Open3D has performance limitations; use Rerun version for better interactivity

4. **`slam_loop_closure_rerun.py`** - ⭐ Real-time visualization with Rerun (RECOMMENDED)
   - **Best option for real-time SLAM visualization**
   - Native viewer application with smooth rendering (launches automatically)
   - True real-time streaming with zero blocking
   - Interactive timeline for frame-by-frame navigation
   - Built-in filtering, multi-view, and inspection tools
   - Native support for robotics/SLAM data types
   - Automatic saving of loop closure image pairs to disk:
     - `loop_pairs/good_loops/`: Temporally consistent loop closures
     - `loop_pairs/bad_loops/`: Inconsistent loop closures
     - Each pair saved as `query_N.jpg` and `loop_N.jpg`
   - Saves recording to `.rrd` file for later playback
   - Usage: `python slam_loop_closure_rerun.py --images_path /path/to/images --poses_file /path/to/vertices.txt`
   - Script: `bash slam_rerun.sh`
   - Located in `slam_loop_closure_rerun.py`
   - **Advantages over Open3D**:
     - No frame drops or interaction lag
     - Timeline scrubbing and playback controls
     - Entity inspection and filtering
     - Multi-threaded by design
     - Native desktop application for best performance

5. **`slam_loop_closure_viz.py`** - Post-processing visualization
   - Generates 3D visualizations from pre-computed loop closures
   - Uses Open3D for rendering
   - Located in `slam_loop_closure_viz.py`

6. **`slam_loop_closure_megaloc_mapanything.py`** - ⭐ Double Validation System (RECOMMENDED FOR HIGH PRECISION)
   - **Combines MegaLoc and MapAnything for maximum reliability**
   - First stage: MegaLoc detects loop closure candidates
   - Second stage: MapAnything verifies each candidate with pose estimation
   - Only displays loops validated by BOTH systems
   - Real-time visualization with Rerun including:
     - 🟢 **Green lines**: Validated loop closures (query ↔ loop)
     - 🟣 **Purple lines**: MapAnything pose prediction (loop → estimated query)
     - 🔴 **Red lines**: Position error (estimated → real query)
     - 🔵 **Cyan spheres**: Estimated query positions by MapAnything
     - 🟢 **Green spheres**: Real query/loop positions
   - Automatic separation of validated/rejected loops:
     - `loop_pairs_validated/megaloc_only/`: All MegaLoc candidates
     - `loop_pairs_validated/double_validated/`: Only double-validated loops
   - Usage: `python slam_loop_closure_megaloc_mapanything.py --images_path /path/to/images --poses_file /path/to/vertices.txt`
   - Script: `bash slam_megaloc_mapanything.sh`
   - Located in `slam_loop_closure_megaloc_mapanything.py`
   - **Advantages**:
     - Eliminates false positives from MegaLoc
     - Provides pose geometry validation
     - Visual feedback on pose estimation accuracy
     - Confidence scores from both systems
     - Best accuracy for critical applications

### Quick Start Scripts

- **`slam_viz.sh`** - Launch Open3D real-time visualization
- **`slam_rerun.sh`** - Launch Rerun real-time visualization (recommended)
- **`slam_megaloc_mapanything.sh`** - Launch double validation system (MegaLoc + MapAnything)
- **`visualize_pair.sh`** - Visualize a single loop/query pair with MapAnything (3D reconstruction)
- **`test_mapanything_pair.sh`** - Quick test with first validated pair

### Key Parameters

All scripts support the following common parameters:
- `--images_path`: Path to directory containing images (format: img*.jpg)
- `--poses_file`: Path to CSV file with pose information (id, tx, ty, tz, qx, qy, qz, qw)
- `--fps`: Processing speed in frames per second (default: 10)
- `--start_detection_frame`: Frame number to begin loop closure detection (default: 50)
- `--temporal_distance`: Minimum temporal gap for loop closure candidates (default: 50 frames)
- `--similarity_threshold`: Minimum similarity score for loop closure (default: 0.55)
- `--temporal_consistency_window`: Window for grouping coherent matches (default: 2, 0=disabled)
- `--top_k`: Number of best matches to consider (default: 3)
- `--device`: Processing device (auto/cuda/cpu, default: auto)
- `--max_frames`: Maximum number of frames to process (default: all)
- `--no_invert_poses`: Skip pose inversion (by default, poses are inverted from cam-to-world to world-to-cam)

#### Additional Parameters for Double Validation System

The `slam_loop_closure_megaloc_mapanything.py` script has additional parameters:
- `--megaloc_similarity_threshold`: MegaLoc detection threshold (default: 0.55)
- `--mapanything_confidence_threshold`: MapAnything validation threshold (default: 0.5)
  - Higher values = fewer but more reliable loops
  - Lower values = more loops but potentially less accurate
  - Recommended range: 0.4-0.7
- `--mapanything_model`: MapAnything model to use (default: "facebook/map-anything")
  - Alternative: "facebook/map-anything-apache" (Apache 2.0 license)

### Loop Closure Detection Algorithm

#### Standard MegaLoc Pipeline

The standard loop closure detection pipeline:

1. **Feature Extraction**: Extract MegaLoc features from each frame sequentially
2. **Similarity Search**: Compute cosine similarity between current frame and database
3. **Temporal Filtering**: Exclude frames within `temporal_distance` to avoid trivial matches
4. **Top-K Selection**: Select K most similar frames
5. **Temporal Consistency Check**: Verify that top matches are temporally grouped (within ±window frames)
6. **Threshold Filtering**: Only accept matches above `similarity_threshold`

Implemented in:
- `find_loop_closure()` - `slam_loop_closure.py:50-80`
- `check_temporal_consistency()` - `slam_loop_closure.py:83-111`

#### Double Validation Pipeline (MegaLoc + MapAnything)

The double validation system adds an extra verification stage:

1. **Stage 1 - MegaLoc Detection** (same as standard pipeline):
   - Feature extraction with MegaLoc
   - Similarity search and temporal filtering
   - Initial loop candidates identification

2. **Stage 2 - MapAnything Verification**:
   - For each MegaLoc candidate:
     - Load query and loop images
     - Run MapAnything pose estimation
     - Extract confidence scores from both views
     - Compute combined confidence score
   - Accept loop only if: `confidence >= mapanything_confidence_threshold`

3. **Visualization**:
   - Only display loops validated by BOTH systems in Rerun
   - Separate storage for MegaLoc-only vs double-validated loops

**Advantages**:
- Eliminates false positives from appearance-based matching
- Provides geometric verification via pose estimation
- Dual confidence scores (appearance + geometry)
- Ideal for safety-critical applications

Implemented in:
- `verify_loop_with_mapanything()` - `slam_loop_closure_megaloc_mapanything.py:182-232`

### Visualization Features

#### Rerun Visualization (`slam_loop_closure_rerun.py`) - RECOMMENDED
- **Trajectory**: Blue line strip showing robot path
- **Current Pose**: Red point showing current processing position
- **Loop Closures**:
  - Green lines for consistent loop closures
  - Orange lines for inconsistent loop closures
- **Loop Poses**: Red spheres at loop closure detection points
- **Timeline**: Interactive scrubbing through frames
- **Controls**: Full mouse/keyboard navigation with smooth rendering in native application
- **Inspection**: Click entities to view details
- **Image Pairs**: Automatically saved to `loop_pairs/` directory for offline analysis
- **Recording**: Full session saved to `.rrd` file, replay with `rerun slam_loop_closures.rrd`

#### Double Validation Visualization (`slam_loop_closure_megaloc_mapanything.py`) - GEOMETRIC VERIFICATION
- **Trajectory**: Blue line strip showing robot path
- **Current Pose**: Red point showing current processing position
- **Validated Loop Closures** (passed both MegaLoc AND MapAnything):
  - 🟢 **Green lines**: Connect real query pose to real loop pose
  - 🟣 **Purple lines**: MapAnything prediction from loop to estimated query position
  - 🔴 **Red lines**: Position error between estimated and real query position
  - 🟢 **Green spheres**: Real query and loop positions (ground truth)
  - 🔵 **Cyan spheres**: Estimated query position by MapAnything
- **Geometric Interpretation**:
  - Short red lines = accurate MapAnything pose estimation
  - Long red lines = large estimation error (potential false positive)
  - Purple lines show the relative transformation predicted by MapAnything
- **Timeline**: Interactive scrubbing through frames
- **Entity Filtering**: Can toggle different visualization layers (errors, predictions, ground truth)
- **Recording**: Full session saved to `slam_megaloc_mapanything.rrd`

This visualization allows you to:
- Assess the geometric consistency of loop closures
- Identify false positives by checking position errors
- Validate that MapAnything predictions align with ground truth
- Debug SLAM failures by inspecting pose estimation accuracy

#### Open3D Visualization (`slam_loop_closure_realtime.py`)
- Blue points: Robot trajectory positions
- Gray lines: Sequential edges (odometry)
- Green lines: Consistent loop closures (matches are grouped)
- Orange lines: Inconsistent loop closures (matches are scattered)
- Red spheres: Loop closure detection points
- Coordinate frame: Shows orientation (X=red, Y=green, Z=blue)

**Controls**:
- Left mouse: Rotate view
- Right mouse: Pan
- Scroll wheel: Zoom

**Performance Notes**:
- Open3D visualization may lag during geometry updates
- Rerun provides significantly smoother interaction

### Input Data Format

**Images**: Sequential images named `img*.jpg` (e.g., img00000.jpg, img00001.jpg, ...)

**Poses CSV** (`vertices.txt`):
```csv
id,tx,ty,tz,qx,qy,qz,qw
0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
1,0.1,0.0,0.0,0.0,0.0,0.0,1.0
...
```

Each row contains:
- `id`: Frame/vertex ID
- `tx,ty,tz`: Translation (position in 3D)
- `qx,qy,qz,qw`: Rotation as quaternion

### Performance Notes

- **Processing Speed**: With CUDA, achieves ~15-20 FPS on modern GPUs
- **Memory Usage**: Features are cached in memory for fast retrieval
- **Real-time Display**:
  - **Rerun**: True non-blocking streaming, smooth 60 FPS interaction
  - **Open3D**: Multi-threaded with batch updates, but may lag during geometry updates
  - **Matplotlib**: Post-processing only, not suitable for real-time

### Pose Coordinate Systems

By default, all scripts assume input poses are in **cam-to-world** format and automatically invert them to **world-to-cam** for correct visualization. This is the standard for most SLAM systems.

- If your poses are already in world-to-cam format, use `--no_invert_poses`
- The inversion is performed using the transformation: T_world_cam = T_cam_world^(-1)
- For rigid transformations: T^(-1) = [R^T | -R^T * t]

### Visualization Libraries Comparison

| Feature | Rerun | Open3D | Matplotlib |
|---------|-------|--------|------------|
| Real-time interaction | ✅ Excellent | ⚠️ Limited | ❌ No |
| Performance | ✅ 60 FPS | ⚠️ 10-30 FPS | ❌ Static |
| Timeline navigation | ✅ Yes | ❌ No | ❌ No |
| Multi-threading | ✅ Native | ⚠️ Manual | ❌ No |
| Setup complexity | ✅ Simple | ⚠️ Medium | ✅ Simple |
| Best for | Real-time SLAM | Static scenes | Final plots |

### Output Files and Directories

The SLAM loop closure scripts generate the following outputs:

1. **Loop Closure Image Pairs** (`loop_pairs/`):
   - `good_loops/` - Temporally consistent loop closures
     - `query_N.jpg` - Query frame that detected the loop
     - `loop_N.jpg` - Matched frame from the past
   - `bad_loops/` - Inconsistent loop closures (matches scattered temporally)
     - Same structure as good_loops

2. **Double Validation Outputs** (`loop_pairs_validated/`) - From `slam_loop_closure_megaloc_mapanything.py`:
   - `megaloc_only/` - All loop candidates detected by MegaLoc
     - `query_N.jpg` - Query frame
     - `loop_N.jpg` - Matched frame
   - `double_validated/` - Only loops validated by BOTH MegaLoc AND MapAnything
     - `query_N.jpg` - Query frame (validated)
     - `loop_N.jpg` - Matched frame (validated)
   - This separation allows easy comparison of MegaLoc candidates vs final validated loops

3. **Rerun Recording** (`slam_loop_closures.rrd` or `slam_megaloc_mapanything.rrd`):
   - Binary recording file containing full 3D visualization session
   - Can be replayed with: `rerun slam_loop_closures.rrd`
   - Allows offline inspection with timeline navigation
   - Compatible with Rerun Viewer application
   - For double validation system: Only contains validated loops in visualization

4. **Visualization Outputs** (legacy scripts):
   - `loop_closure_*.png` - Static visualization images (matplotlib/basic scripts)

5. **MapAnything Pair Visualization** (`scene.ply` or custom name):
   - 3D point cloud reconstruction from loop/query pair
   - Generated by `mapanything_visualize_pair.py`
   - Can be viewed with Open3D, MeshLab, or CloudCompare
   - See `README_MAPANYTHING_PAIR.md` for details

## MapAnything Single Pair Visualization

For detailed analysis of individual loop closures, use the MapAnything pair visualization scripts.

### Quick Test

```bash
bash test_mapanything_pair.sh
```

This automatically finds and visualizes the first validated loop pair.

### Custom Pair

```bash
bash visualize_pair.sh <loop.jpg> <query.jpg> [output.ply]
```

Example:
```bash
bash visualize_pair.sh \
    loop_pairs_validated/double_validated/loop_5.jpg \
    loop_pairs_validated/double_validated/query_5.jpg \
    reconstruction_5.ply
```

### Output

The script generates:
1. **Transformation matrices**:
   - Loop camera pose (cam-to-world)
   - Query camera pose (cam-to-world)
   - Relative transformation (loop → query)

2. **Transformation breakdown**:
   - Translation vector (x, y, z in meters)
   - Rotation (Euler angles in degrees)

3. **Confidence scores**:
   - Per-view mean/min/max confidence

4. **3D point cloud** (PLY format):
   - Colored 3D reconstruction
   - Can be viewed with Open3D, MeshLab, CloudCompare

### View Point Cloud

```bash
# With Open3D
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('scene.ply'); o3d.visualization.draw_geometries([pcd])"

# With MeshLab
meshlab scene.ply

# With CloudCompare
cloudcompare scene.ply
```

### Direct Python Usage

```bash
source megaloc_mapanything_env/bin/activate

python mapanything_visualize_pair.py \
    --loop path/to/loop.jpg \
    --query path/to/query.jpg \
    --output my_scene.ply \
    --model facebook/map-anything
```

**See `README_MAPANYTHING_PAIR.md` for complete documentation.**

## MapAnything with Custom Camera Intrinsics

For accurate 3D reconstruction with known camera calibration parameters.

### When to Use This

Use `mapanything_pair_with_intrinsics.py` when you have:
- Known camera calibration (K matrix: fx, fy, cx, cy)
- Optionally: distortion coefficients (k1, k2, p1, p2, k3)
- Need precise metric reconstruction

### Usage

```bash
source megaloc_mapanything_env/bin/activate

python mapanything_pair_with_intrinsics.py \
    --loop loop.jpg \
    --query query.jpg \
    --fx 500.0 --fy 500.0 --cx 320.0 --cy 240.0 \
    --output scene.glb
```

### With Distortion Correction

If your images have lens distortion (fisheye, wide-angle):

```bash
python mapanything_pair_with_intrinsics.py \
    --loop loop.jpg \
    --query query.jpg \
    --fx 500.0 --fy 500.0 --cx 320.0 --cy 240.0 \
    --k1 -0.1 --k2 0.05 --p1 0.001 --p2 -0.001 --k3 0.0 \
    --undistort \
    --output scene.glb
```

This will:
1. Undistort images using OpenCV
2. Adjust intrinsics automatically
3. Run MapAnything on corrected images
4. Generate accurate 3D reconstruction

### Quick Test

Edit `test_intrinsics.sh` with your camera parameters and run:
```bash
bash test_intrinsics.sh
```

### Pre-configured for Your Camera

If you have specific camera calibration parameters, use the pre-configured scripts:

```bash
# Quick test with your camera
bash test_my_camera.sh

# Custom pair with your camera
bash mapanything_my_camera.sh <loop.jpg> <query.jpg> [output.glb]
```

These scripts are pre-configured with your camera intrinsics:
- fx = fy = 322.580 pixels
- cx = 259.260, cy = 184.882 pixels
- Distortion: k1=-0.070, k2=0.076, p1=0.0012, p2=0.0010, k3=-0.018
- **Automatic undistortion** enabled

See `USAGE_MY_CAMERA.md` for details.

### Camera Intrinsics Format

Intrinsics matrix K (3×3):
```
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
```

Where:
- `fx`, `fy`: Focal lengths in pixels
- `cx`, `cy`: Principal point (optical center) in pixels

Distortion coefficients (optional):
- `k1`, `k2`, `k3`: Radial distortion
- `p1`, `p2`: Tangential distortion

### Output

1. **Relative transformation matrix** (4×4)
2. **Decomposed transformation**:
   - Translation (x, y, z in meters)
   - Rotation (Euler angles in degrees)
3. **Confidence scores** per view
4. **3D model** (GLB file with mesh + textures)

**See `README_CUSTOM_INTRINSICS.md` for complete documentation.**

## License

MIT License - Copyright (c) 2024 Gabriele Berton, Carlo Masone
