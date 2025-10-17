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

2. **`slam_loop_closure_matplotlib.py`** - Interactive matplotlib-based visualization
   - Real-time 2D trajectory visualization with matplotlib
   - Shows pose graph with sequential edges and loop closures
   - Interactive pan/zoom capabilities
   - Usage: `python slam_loop_closure_matplotlib.py --images_path /path/to/images --poses_file /path/to/vertices.txt`
   - Located in `slam_loop_closure_matplotlib.py`

3. **`slam_loop_closure_realtime.py`** - Real-time 3D visualization with Open3D
   - Live 3D trajectory visualization during processing
   - Color-coded loop closures (green=consistent, orange=inconsistent)
   - Interactive 3D view (rotate, pan, zoom)
   - Spheres mark loop closure locations
   - Usage: `python slam_loop_closure_realtime.py --images_path /path/to/images --poses_file /path/to/vertices.txt`
   - Located in `slam_loop_closure_realtime.py:1-410`

4. **`slam_loop_closure_viz.py`** - Post-processing visualization
   - Generates 3D visualizations from pre-computed loop closures
   - Uses Open3D for rendering
   - Located in `slam_loop_closure_viz.py`

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

### Loop Closure Detection Algorithm

The loop closure detection pipeline:

1. **Feature Extraction**: Extract MegaLoc features from each frame sequentially
2. **Similarity Search**: Compute cosine similarity between current frame and database
3. **Temporal Filtering**: Exclude frames within `temporal_distance` to avoid trivial matches
4. **Top-K Selection**: Select K most similar frames
5. **Temporal Consistency Check**: Verify that top matches are temporally grouped (within ±window frames)
6. **Threshold Filtering**: Only accept matches above `similarity_threshold`

Implemented in:
- `find_loop_closure()` - `slam_loop_closure.py:50-80`
- `check_temporal_consistency()` - `slam_loop_closure.py:83-111`

### Visualization Features

**Real-time 3D Visualization** (`slam_loop_closure_realtime.py`):
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
- **Real-time Display**: Open3D visualization updates non-blocking during processing

## License

MIT License - Copyright (c) 2024 Gabriele Berton, Carlo Masone
