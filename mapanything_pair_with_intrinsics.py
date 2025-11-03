#!/usr/bin/env python3
"""
MapAnything with Custom Intrinsics for Loop/Query Pair

This script processes a pair of images (loop/query) with custom camera intrinsics
and outputs:
1. Relative transformation (pose) between the two cameras
2. 3D reconstruction (GLB file)
3. Optional: Rerun visualization

Usage:
    python mapanything_pair_with_intrinsics.py \\
        --loop loop.jpg \\
        --query query.jpg \\
        --fx 500.0 --fy 500.0 --cx 320.0 --cy 240.0 \\
        --output scene.glb

With distortion coefficients (will undistort images first):
    python mapanything_pair_with_intrinsics.py \\
        --loop loop.jpg \\
        --query query.jpg \\
        --fx 500.0 --fy 500.0 --cx 320.0 --cy 240.0 \\
        --k1 -0.1 --k2 0.05 --p1 0.001 --p2 -0.001 --k3 0.0 \\
        --undistort \\
        --output scene.glb
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import cv2

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.viz import predictions_to_glb
from mapanything.utils.geometry import depthmap_to_world_frame

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


def undistort_image(image_path, K, dist_coeffs, output_path=None):
    """
    Undistort an image using OpenCV.

    Args:
        image_path: Path to input image
        K: Camera intrinsics matrix (3x3)
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
        output_path: Optional path to save undistorted image

    Returns:
        Tuple (undistorted_image_path, new_K)
    """
    # Read image
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    # Get optimal new camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))

    # Undistort
    undistorted = cv2.undistort(img, K, dist_coeffs, None, new_K)

    # Save undistorted image
    if output_path is None:
        output_path = str(Path(image_path).parent / f"undistorted_{Path(image_path).name}")

    cv2.imwrite(output_path, undistorted)

    return output_path, new_K


def resize_to_patch_size(img, patch_size=14):
    """
    Resize image so dimensions are divisible by patch_size.

    Args:
        img: PIL Image
        patch_size: Patch size (default: 14 for DINOv2)

    Returns:
        Resized PIL Image, scale factors (sx, sy)
    """
    W, H = img.size

    # Calculate new dimensions divisible by patch_size
    new_W = (W // patch_size) * patch_size
    new_H = (H // patch_size) * patch_size

    # Avoid zero dimensions
    if new_W == 0:
        new_W = patch_size
    if new_H == 0:
        new_H = patch_size

    # Resize
    img_resized = img.resize((new_W, new_H), Image.LANCZOS)

    # Calculate scale factors
    sx = new_W / W
    sy = new_H / H

    return img_resized, sx, sy


def adjust_intrinsics_for_resize(K, sx, sy):
    """
    Adjust intrinsics matrix after image resize.

    Args:
        K: 3x3 intrinsics matrix
        sx: Scale factor for width
        sy: Scale factor for height

    Returns:
        Adjusted 3x3 intrinsics matrix
    """
    K_adjusted = K.copy()
    K_adjusted[0, 0] *= sx  # fx
    K_adjusted[1, 1] *= sy  # fy
    K_adjusted[0, 2] *= sx  # cx
    K_adjusted[1, 2] *= sy  # cy

    return K_adjusted


def load_image_with_intrinsics(image_path, K_matrix, patch_size=14):
    """
    Load an image and prepare it with custom intrinsics for MapAnything.

    Args:
        image_path: Path to image
        K_matrix: 3x3 intrinsics matrix as numpy array
        patch_size: Patch size for DINOv2 (default: 14)

    Returns:
        Dictionary compatible with MapAnything.infer()
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    W_orig, H_orig = img.size

    # Resize to be divisible by patch_size
    img_resized, sx, sy = resize_to_patch_size(img, patch_size)
    W, H = img_resized.size

    from pathlib import Path
    print(f"  Resized {Path(image_path).name}: {W_orig}x{H_orig} -> {W}x{H}")

    # Adjust intrinsics for the resize
    K_adjusted = adjust_intrinsics_for_resize(K_matrix, sx, sy)
    print(f"    Adjusted intrinsics: fx={K_adjusted[0,0]:.2f}, fy={K_adjusted[1,1]:.2f}, cx={K_adjusted[0,2]:.2f}, cy={K_adjusted[1,2]:.2f}")

    # Convert to tensor and normalize (DINOv2 normalization)
    import torchvision.transforms as tfm
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    img_norm = IMAGE_NORMALIZATION_DICT['dinov2']
    transform = tfm.Compose([
        tfm.ToTensor(),
        tfm.Normalize(mean=img_norm.mean, std=img_norm.std)
    ])

    img_tensor = transform(img_resized).unsqueeze(0)  # (1, 3, H, W)

    # Prepare intrinsics tensor
    K_tensor = torch.from_numpy(K_adjusted).float().unsqueeze(0)  # (1, 3, 3)

    # Create view dictionary
    view = {
        'img': img_tensor,
        'intrinsics': K_tensor,
        'true_shape': np.array([[H, W]]),
        'data_norm_type': ['dinov2'],
        'idx': [0],
        'instance': ['view_0']
    }

    return view


def print_matrix(name, matrix):
    """Pretty print a 4x4 transformation matrix."""
    print(f"\n{name}:")
    print("=" * 60)
    for i in range(4):
        print(f"  [{matrix[i, 0]:9.5f}  {matrix[i, 1]:9.5f}  {matrix[i, 2]:9.5f}  {matrix[i, 3]:9.5f}]")
    print("=" * 60)


def decompose_transform(T):
    """Decompose 4x4 transformation matrix into components."""
    R = T[:3, :3]
    t = T[:3, 3]

    # Extract Euler angles (XYZ convention)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return {
        'rotation_matrix': R,
        'translation': t,
        'euler_xyz_deg': np.degrees([x, y, z]),
        'translation_norm': np.linalg.norm(t)
    }


def main():
    parser = argparse.ArgumentParser(
        description="MapAnything with custom intrinsics for loop/query pair"
    )

    # Image inputs
    parser.add_argument("--loop", type=str, required=True, help="Path to loop image")
    parser.add_argument("--query", type=str, required=True, help="Path to query image")

    # Intrinsics (K matrix)
    parser.add_argument("--fx", type=float, required=True, help="Focal length X")
    parser.add_argument("--fy", type=float, required=True, help="Focal length Y")
    parser.add_argument("--cx", type=float, required=True, help="Principal point X")
    parser.add_argument("--cy", type=float, required=True, help="Principal point Y")

    # Distortion coefficients (optional)
    parser.add_argument("--k1", type=float, default=0.0, help="Radial distortion k1")
    parser.add_argument("--k2", type=float, default=0.0, help="Radial distortion k2")
    parser.add_argument("--p1", type=float, default=0.0, help="Tangential distortion p1")
    parser.add_argument("--p2", type=float, default=0.0, help="Tangential distortion p2")
    parser.add_argument("--k3", type=float, default=0.0, help="Radial distortion k3")
    parser.add_argument("--undistort", action="store_true",
                        help="Undistort images before processing (if distortion coeffs provided)")

    # Model options
    parser.add_argument("--model", type=str, default="facebook/map-anything",
                        help="MapAnything model (default: facebook/map-anything)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    # Output options
    parser.add_argument("--output", type=str, default="scene.glb",
                        help="Output GLB file (default: scene.glb)")
    parser.add_argument("--viz", action="store_true",
                        help="Enable Rerun visualization")

    args = parser.parse_args()

    # Check inputs
    if not Path(args.loop).exists():
        print(f"❌ Loop image not found: {args.loop}")
        sys.exit(1)
    if not Path(args.query).exists():
        print(f"❌ Query image not found: {args.query}")
        sys.exit(1)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 80)
    print("🗺️  MapAnything - Custom Intrinsics Mode")
    print("=" * 80)
    print(f"Loop:    {args.loop}")
    print(f"Query:   {args.query}")
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")
    print("=" * 80)

    # Build intrinsics matrix
    K = np.array([
        [args.fx, 0, args.cx],
        [0, args.fy, args.cy],
        [0, 0, 1]
    ], dtype=np.float32)

    print(f"\n📐 Camera Intrinsics:")
    print(f"  fx = {args.fx:.2f}, fy = {args.fy:.2f}")
    print(f"  cx = {args.cx:.2f}, cy = {args.cy:.2f}")

    # Check for distortion
    dist_coeffs = np.array([args.k1, args.k2, args.p1, args.p2, args.k3], dtype=np.float32)
    has_distortion = np.any(np.abs(dist_coeffs) > 1e-6)

    if has_distortion:
        print(f"\n⚠️  Distortion coefficients detected:")
        print(f"  k1={args.k1:.6f}, k2={args.k2:.6f}, p1={args.p1:.6f}, p2={args.p2:.6f}, k3={args.k3:.6f}")

        if args.undistort:
            print(f"\n🔧 Undistorting images...")
            loop_undistorted, K_loop = undistort_image(args.loop, K, dist_coeffs)
            query_undistorted, K_query = undistort_image(args.query, K, dist_coeffs)

            print(f"  ✓ Undistorted loop:  {loop_undistorted}")
            print(f"  ✓ Undistorted query: {query_undistorted}")

            # Use undistorted images and updated intrinsics
            loop_path = loop_undistorted
            query_path = query_undistorted
            K = K_loop  # Use the optimized K from undistortion

            print(f"\n📐 Updated Intrinsics (after undistortion):")
            print(f"  fx = {K[0,0]:.2f}, fy = {K[1,1]:.2f}")
            print(f"  cx = {K[0,2]:.2f}, cy = {K[1,2]:.2f}")
        else:
            print(f"  ⚠️  WARNING: MapAnything does not handle distortion natively!")
            print(f"  ⚠️  Use --undistort flag to undistort images first.")
            loop_path = args.loop
            query_path = args.query
    else:
        loop_path = args.loop
        query_path = args.query

    # Load model
    print(f"\n🤖 Loading MapAnything model...")
    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()
    print("✅ Model loaded")

    # Load images with custom intrinsics
    print(f"\n📷 Loading images with custom intrinsics...")
    loop_view = load_image_with_intrinsics(loop_path, K)
    query_view = load_image_with_intrinsics(query_path, K)

    # Move to device
    loop_view['img'] = loop_view['img'].to(device)
    loop_view['intrinsics'] = loop_view['intrinsics'].to(device)
    query_view['img'] = query_view['img'].to(device)
    query_view['intrinsics'] = query_view['intrinsics'].to(device)

    views = [loop_view, query_view]
    print("✅ Images loaded with custom intrinsics")

    # Run inference
    print(f"\n🔄 Running inference...")
    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=False,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
        )
    print("✅ Inference complete")

    # Extract poses
    print(f"\n📐 Extracting camera poses...")
    pose_loop = outputs[0]['camera_poses'][0].cpu().numpy()
    pose_query = outputs[1]['camera_poses'][0].cpu().numpy()

    # Compute relative transformation
    pose_loop_inv = np.linalg.inv(pose_loop)
    relative_transform = pose_loop_inv @ pose_query

    # Display results
    print("\n" + "=" * 80)
    print("📊 TRANSFORMATION MATRICES")
    print("=" * 80)

    print_matrix("Loop Camera Pose (cam-to-world)", pose_loop)
    print_matrix("Query Camera Pose (cam-to-world)", pose_query)
    print_matrix("Relative Transformation (loop → query)", relative_transform)

    # Decompose relative transformation
    print("\n" + "=" * 80)
    print("📏 RELATIVE TRANSFORMATION BREAKDOWN")
    print("=" * 80)

    decomp = decompose_transform(relative_transform)
    print(f"\nTranslation:")
    print(f"  x: {decomp['translation'][0]:9.5f} m")
    print(f"  y: {decomp['translation'][1]:9.5f} m")
    print(f"  z: {decomp['translation'][2]:9.5f} m")
    print(f"  norm: {decomp['translation_norm']:9.5f} m")

    print(f"\nRotation (Euler angles XYZ):")
    print(f"  roll  (X): {decomp['euler_xyz_deg'][0]:9.3f}°")
    print(f"  pitch (Y): {decomp['euler_xyz_deg'][1]:9.3f}°")
    print(f"  yaw   (Z): {decomp['euler_xyz_deg'][2]:9.3f}°")

    # Extract confidence scores
    print("\n" + "=" * 80)
    print("🎯 CONFIDENCE SCORES")
    print("=" * 80)

    for idx, pred in enumerate(outputs):
        name = "Loop" if idx == 0 else "Query"
        if 'conf' in pred:
            conf = pred['conf'].cpu().numpy()
            if conf.ndim == 4:
                conf = conf.squeeze(-1)
            mean_conf = float(np.mean(conf))
            min_conf = float(np.min(conf))
            max_conf = float(np.max(conf))
            print(f"\n{name} image:")
            print(f"  Mean confidence: {mean_conf:.4f}")
            print(f"  Min confidence:  {min_conf:.4f}")
            print(f"  Max confidence:  {max_conf:.4f}")

    # Generate 3D reconstruction (GLB)
    print("\n" + "=" * 80)
    print("💾 GENERATING 3D RECONSTRUCTION (GLB)")
    print("=" * 80)

    # Prepare data for GLB export
    world_points_list = []
    images_list = []
    masks_list = []

    for view_idx, pred in enumerate(outputs):
        # Extract depth and convert to world frame
        depthmap_torch = pred["depth_z"][0].squeeze(-1)  # (H, W)
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)

        # Compute 3D points
        pts3d_computed, valid_mask = depthmap_to_world_frame(
            depthmap_torch, intrinsics_torch, camera_pose_torch
        )

        # Convert to numpy
        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()
        pts3d_np = pts3d_computed.cpu().numpy()
        image_np = pred["img_no_norm"][0].cpu().numpy()

        world_points_list.append(pts3d_np)
        images_list.append(image_np)
        masks_list.append(mask)

    # Stack all views
    world_points = np.stack(world_points_list, axis=0)
    images = np.stack(images_list, axis=0)
    final_masks = np.stack(masks_list, axis=0)

    # Create predictions dict for GLB export
    predictions = {
        "world_points": world_points,
        "images": images,
        "final_masks": final_masks,
    }

    # Convert to GLB scene
    print(f"Converting to 3D mesh...")
    scene_3d = predictions_to_glb(predictions, as_mesh=True)

    # Save GLB file
    scene_3d.export(args.output)
    print(f"\n✅ 3D reconstruction saved to: {args.output}")
    print(f"   You can open it with:")
    print(f"   - Blender: blender {args.output}")
    print(f"   - Online: https://gltf-viewer.donmccurdy.com/")

    # Rerun visualization
    if args.viz:
        if not RERUN_AVAILABLE:
            print("\n⚠️  Rerun not available. Install with: pip install rerun-sdk")
        else:
            print("\n📺 Launching Rerun visualization...")
            rr.init("MapAnything_CustomIntrinsics", spawn=True)
            rr.set_time("frame", 0)

            for idx, pred in enumerate(outputs):
                name = "loop" if idx == 0 else "query"
                image_np = pred["img_no_norm"][0].cpu().numpy()
                pose = pred["camera_poses"][0].cpu().numpy()

                rr.log(f"world/{name}", rr.Transform3D(
                    translation=pose[:3, 3],
                    mat3x3=pose[:3, :3],
                ))

                rr.log(f"world/{name}/image", rr.Image(image_np))

    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
