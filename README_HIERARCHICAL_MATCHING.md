# Hierarchical Loop Closure Matching Pipeline

**MegaLoc → SuperPoint → LightGlue**

A three-stage hierarchical approach for robust and precise loop closure detection in SLAM applications.

---

## 🎯 Overview

This pipeline combines the strengths of three complementary methods:

```
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 1: MegaLoc (Global Vision)                           │
├─────────────────────────────────────────────────────────────┤
│ • Loop closure detection (global similarity)                │
│ • Coarse patch matching (e.g., 16×16 patches)              │
│ • Output: List of corresponding patch pairs                 │
│   → [(query_patch_1, loop_patch_1, score_1), ...]         │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 2: SuperPoint (Local Keypoints)                      │
├─────────────────────────────────────────────────────────────┤
│ • For each MegaLoc patch pair:                             │
│   - Extract SuperPoint keypoints IN patch regions          │
│   - Spatial masking to focus on relevant areas             │
│ • Output: Dense keypoints and descriptors                  │
│   → Query: (keypoints, descriptors, scores)                │
│   → Loop:  (keypoints, descriptors, scores)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 3: LightGlue (Fine Matching)                         │
├─────────────────────────────────────────────────────────────┤
│ • Contextual matching with attention mechanism              │
│ • Geometric filtering (optional RANSAC)                     │
│ • Output: Pixel-precise correspondences                     │
│   → [(query_kp, loop_kp, confidence), ...]                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Key Advantages

1. **Efficiency**: SuperPoint only runs in relevant regions (not full image)
2. **Robustness**: MegaLoc filters false positives before fine matching
3. **Precision**: LightGlue provides sub-pixel correspondences
4. **Scalability**: Guided matching is faster than exhaustive search

---

## 📦 Installation

### Option 1: Install dependencies

```bash
pip install -r requirements_matching.txt
```

### Option 2: Manual installation

```bash
# Core dependencies
pip install torch torchvision numpy Pillow matplotlib opencv-python

# Feature matching
pip install lightglue kornia

# Or install from GitHub
pip install git+https://github.com/cvg/LightGlue.git
```

### Verify installation

```python
import torch
from lightglue import LightGlue, SuperPoint

print("✓ LightGlue installed successfully!")
```

---

## 🎬 Usage

### Quick Start

```bash
bash run_hierarchical_matching.sh /path/to/images
```

### With custom parameters

```bash
bash run_hierarchical_matching.sh \
    /path/to/images \
    output_directory \
    30 \
    100
```

Parameters:
1. Images path (required)
2. Output directory (default: `hierarchical_matches`)
3. Number of MegaLoc patches to use (default: 30)
4. Max frames to process (default: all)

### Direct Python usage

```bash
python hierarchical_matching.py \
    --images_path /path/to/images \
    --output_dir hierarchical_matches \
    --num_megaloc_patches 30 \
    --roi_expansion 1.5 \
    --similarity_threshold 0.55 \
    --start_detection_frame 50 \
    --temporal_distance 50 \
    --max_frames 500 \
    --device cuda
```

---

## 🎛️ Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--images_path` | Path to images directory | *required* | - |
| `--output_dir` | Output directory | `hierarchical_matches` | - |
| `--num_megaloc_patches` | Number of MegaLoc patches for guidance | 30 | 10-50 |
| `--patch_similarity_threshold` | Minimum patch similarity threshold | 0.40 | 0.0-1.0 |
| `--min_patches` | Minimum patches required for refinement | 20 | 5-50 |
| `--roi_expansion` | ROI expansion factor | 1.5 | 1.0-3.0 |
| `--similarity_threshold` | MegaLoc global similarity threshold | 0.55 | 0.0-1.0 |
| `--start_detection_frame` | Start detection at frame N | 50 | 0-∞ |
| `--temporal_distance` | Minimum temporal gap (frames) | 50 | 0-∞ |
| `--max_frames` | Maximum frames to process | all | 1-∞ |
| `--device` | Processing device | `auto` | `auto`, `cuda`, `cpu` |

### Parameter Tuning Guide

**`num_megaloc_patches`**:
- **Lower (10-20)**: Fewer patches to consider, faster processing
- **Medium (30-40)**: Balanced coverage (recommended)
- **Higher (40-50)**: Maximum coverage, slower but more thorough

**`patch_similarity_threshold`**:
- **0.30-0.40**: Permissive, includes more patches
- **0.40-0.50**: Standard range (recommended)
- **0.50-0.60**: Strict, only high-quality patches

**`min_patches`**:
- **5-10**: Relaxed, proceeds with fewer patches
- **15-25**: Standard requirement (recommended)
- **25-40**: Strict, ensures dense coverage

**`roi_expansion`**:
- **1.0**: ROI = exact patch size
- **1.5** (recommended): 50% expansion for context
- **2.0+**: Large overlap, may include irrelevant features

**`similarity_threshold`**:
- **0.50-0.60**: Standard range for indoor/outdoor scenes
- **0.65+**: Strict, fewer false positives
- **0.40-0.50**: Permissive, more loop candidates

---

## 📊 Output

### Visualization Files

The script generates visualization images in `output_dir/`:

```
hierarchical_matches/
├── hierarchical_001_query00150_loop00035.png
├── hierarchical_002_query00237_loop00098.png
└── ...
```

Each visualization shows:
- **Green/yellow rectangles**: MegaLoc patch correspondences
  - Color intensity = similarity score
- **Cyan circles**: SuperPoint keypoints in relevant regions
- **Green stars**: Matched keypoints by LightGlue
- **Purple lines**: Fine matches (color = confidence)

### Legend

| Visual Element | Meaning |
|---------------|---------|
| 🟩 Green rectangles | High-similarity MegaLoc patches |
| 🟨 Yellow rectangles | Medium-similarity MegaLoc patches |
| 🔵 Cyan circles | SuperPoint keypoints |
| ⭐ Green stars | Matched keypoints (LightGlue) |
| 🟣 Purple lines | Fine correspondences |

---

## 🔧 Pipeline Details

### Stage 1: MegaLoc Guidance

```python
# Extract MegaLoc features
global_feat, spatial_feat = extract_megaloc_features(model, img_tensor, device)

# Find loop closure
top_indices, top_similarities = find_loop_closure(global_feat, database_features)

# Find patch matches
patch_matches, grid_size = find_patch_matches_megaloc(
    query_spatial_feat,
    loop_spatial_feat,
    top_k=num_patches
)
```

**Key point**: MegaLoc provides **coarse but robust** correspondences that guide the next stages.

### Stage 2: SuperPoint Extraction

```python
# Create ROI from MegaLoc patches
query_rois, loop_rois = create_patch_rois(
    patch_matches,
    grid_size,
    img_size,
    expansion_factor=1.5
)

# Extract keypoints in ROI only
query_kpts, query_desc, _ = extract_superpoint_in_rois(
    superpoint,
    query_img,
    query_rois,
    device
)
```

**Key point**: SuperPoint only processes **relevant regions**, making it much faster than full-image extraction.

### Stage 3: LightGlue Matching

```python
# Match with attention-based LightGlue
fine_matches = match_with_lightglue(
    lightglue,
    query_kpts, query_desc,
    loop_kpts, loop_desc,
    img_shape,
    device
)
```

**Key point**: LightGlue uses **graph neural networks** and attention to find accurate correspondences.

---

## 💡 Use Cases

### 1. SLAM Loop Closure

Replace traditional BoW (Bag-of-Words) methods with this pipeline for:
- More robust loop detection (MegaLoc)
- Precise relative pose estimation (SuperPoint + LightGlue)

### 2. Visual Localization

Use for accurate 6-DoF pose estimation:
1. MegaLoc finds similar images
2. SuperPoint+LightGlue extract matches
3. Use matches with PnP/Essential Matrix for pose

### 3. Image Matching

General-purpose image matching with hierarchical approach:
- Fast coarse matching (MegaLoc)
- Accurate fine matching (SuperPoint+LightGlue)

---

## 🧪 Example Results

**Typical performance** (on modern GPU):
- MegaLoc: ~20 ms/image
- SuperPoint (in ROI): ~15 ms/image
- LightGlue: ~30 ms/pair
- **Total**: ~65 ms/loop closure

**Match quality**:
- MegaLoc: 10-20 patch correspondences
- SuperPoint: 100-500 keypoints in ROI
- LightGlue: 50-200 fine matches (after filtering)

---

## 🔮 Future Extensions

The current pipeline stops at **fine matching**. Possible extensions:

### Option 1: Pose Estimation with PnP

If 3D coordinates are known (e.g., from mapping):

```python
# Use solvePnPRansac for 6-DoF pose
rvec, tvec, inliers = cv2.solvePnPRansac(
    points_3d,  # 3D points from map
    points_2d,  # 2D keypoints from LightGlue matches
    camera_matrix,
    dist_coeffs
)
```

### Option 2: Essential Matrix Estimation

If camera intrinsics are known:

```python
# Estimate relative pose
E, mask = cv2.findEssentialMat(
    query_points,
    loop_points,
    camera_matrix,
    method=cv2.RANSAC
)

# Recover pose
_, R, t, mask = cv2.recoverPose(E, query_points, loop_points, camera_matrix)
```

### Option 3: Fundamental Matrix

Without calibration:

```python
# Estimate fundamental matrix
F, mask = cv2.findFundamentalMat(
    query_points,
    loop_points,
    method=cv2.RANSAC
)
```

---

## 📚 References

### MegaLoc
```bibtex
@InProceedings{Berton_2025_CVPR,
    author    = {Berton, Gabriele and Masone, Carlo},
    title     = {MegaLoc: One Retrieval to Place Them All},
    booktitle = {CVPR Workshops},
    year      = {2025}
}
```

### SuperPoint
```bibtex
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={CVPR Workshops},
  year={2018}
}
```

### LightGlue
```bibtex
@inproceedings{lindenberger2023lightglue,
  title={LightGlue: Local Feature Matching at Light Speed},
  author={Lindenberger, Philipp and Sarlin, Paul-Edouard and Pollefeys, Marc},
  booktitle={ICCV},
  year={2023}
}
```

---

## 🐛 Troubleshooting

### LightGlue not found

```bash
pip install lightglue kornia opencv-python
# Or from source
pip install git+https://github.com/cvg/LightGlue.git
```

### CUDA out of memory

Reduce memory usage:
```bash
python hierarchical_matching.py \
    --num_megaloc_patches 15 \
    --roi_expansion 1.2 \
    --max_frames 100
```

### Too few matches

Increase coverage:
```bash
python hierarchical_matching.py \
    --num_megaloc_patches 50 \
    --patch_similarity_threshold 0.35 \
    --min_patches 10 \
    --roi_expansion 2.0 \
    --similarity_threshold 0.50
```

### Too many false positives

Be more strict:
```bash
python hierarchical_matching.py \
    --num_megaloc_patches 30 \
    --patch_similarity_threshold 0.50 \
    --min_patches 25 \
    --similarity_threshold 0.65
```

### Loop candidates rejected (insufficient patches)

If you see many "INSUFFICIENT PATCHES" warnings, adjust:
```bash
python hierarchical_matching.py \
    --num_megaloc_patches 50 \
    --patch_similarity_threshold 0.35 \
    --min_patches 15
```

---

## 📄 License

This pipeline combines:
- **MegaLoc**: MIT License
- **SuperPoint**: Part of LightGlue (Apache 2.0)
- **LightGlue**: Apache 2.0 License

---

## 🙏 Acknowledgments

- MegaLoc team for robust VPR
- CVG Lab (ETH Zurich) for SuperPoint and LightGlue
- PyTorch and OpenCV communities

---

**Happy Matching! 🚀**
