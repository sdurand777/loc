#!/usr/bin/env python3
"""
Simple script to visualize a loop/query image pair with MapAnything.
Generates 3D reconstruction and shows relative transformation.

Usage:
    python mapanything_visualize_pair.py --loop loop.jpg --query query.jpg

Output:
    - Relative transformation matrix
    - 3D point cloud (scene.ply)
    - Optional: mesh reconstruction (scene.glb)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


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


def save_point_cloud(points, colors, output_path):
    """Save point cloud to PLY format."""
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)

    # Filter out invalid points
    valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]

    print(f"\n💾 Saving point cloud with {len(points)} points to {output_path}")

    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for point, color in zip(points, colors):
            r, g, b = (color * 255).astype(np.uint8)
            f.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")

    print(f"✅ Point cloud saved successfully!")


def extract_point_cloud(predictions, views):
    """Extract point cloud from MapAnything predictions."""
    all_points = []
    all_colors = []

    for idx, (pred, view) in enumerate(zip(predictions, views)):
        # Get camera pose
        camera_pose = pred['camera_poses'][0].cpu().numpy()  # (4, 4)

        # Get depth map
        if 'depth' in pred:
            depth = pred['depth'][0].cpu().numpy()  # (H, W) or (H, W, 1)
            if depth.ndim == 3:
                depth = depth.squeeze(-1)
        else:
            print(f"⚠️  No depth map for view {idx}")
            continue

        # Get image for colors
        img = view['img'][0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        # Get intrinsics (assume simple pinhole model)
        H, W = depth.shape
        focal = max(H, W)  # Simple approximation
        cx, cy = W / 2, H / 2

        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Back-project to camera space
        z = depth
        x = (u - cx) * z / focal
        y = (v - cy) * z / focal

        # Stack to (H, W, 3)
        points_cam = np.stack([x, y, z], axis=-1)

        # Transform to world space
        points_cam_homogeneous = np.concatenate([
            points_cam.reshape(-1, 3),
            np.ones((points_cam.reshape(-1, 3).shape[0], 1))
        ], axis=1)  # (N, 4)

        points_world = (camera_pose @ points_cam_homogeneous.T).T[:, :3]  # (N, 3)
        colors = img.reshape(-1, 3)

        all_points.append(points_world)
        all_colors.append(colors)

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
        return points, colors
    else:
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MapAnything reconstruction for an image pair"
    )
    parser.add_argument("--loop", type=str, required=True,
                        help="Path to loop image")
    parser.add_argument("--query", type=str, required=True,
                        help="Path to query image")
    parser.add_argument("--output", type=str, default="scene.ply",
                        help="Output PLY file (default: scene.ply)")
    parser.add_argument("--model", type=str, default="facebook/map-anything",
                        help="MapAnything model (default: facebook/map-anything)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"],
                        help="Device (default: auto)")

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
    print("🗺️  MapAnything - Image Pair Reconstruction")
    print("=" * 80)
    print(f"Loop image:  {args.loop}")
    print(f"Query image: {args.query}")
    print(f"Device:      {device}")
    print(f"Model:       {args.model}")
    print("=" * 80)

    # Load model
    print("\n🤖 Loading MapAnything model...")
    model = MapAnything.from_pretrained(args.model).to(device)
    model.eval()
    print("✅ Model loaded")

    # Load images
    print("\n📷 Loading images...")
    views = load_images([args.loop, args.query])
    print(f"✅ Loaded {len(views)} images")

    # Run inference
    print("\n🔄 Running inference...")
    with torch.no_grad():
        predictions = model.infer(
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
    print("\n📐 Extracting camera poses...")
    pose_loop = predictions[0]['camera_poses'][0].cpu().numpy()
    pose_query = predictions[1]['camera_poses'][0].cpu().numpy()

    # Compute relative transformation
    # relative_transform = loop^-1 @ query
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

    for idx, pred in enumerate(predictions):
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
        else:
            print(f"\n{name} image: No confidence available")

    # Extract and save point cloud
    print("\n" + "=" * 80)
    print("💾 GENERATING 3D RECONSTRUCTION")
    print("=" * 80)

    points, colors = extract_point_cloud(predictions, views)

    if points is not None:
        save_point_cloud(points, colors, args.output)
        print(f"\n✅ 3D reconstruction saved to: {args.output}")
        print(f"   You can open it with:")
        print(f"   - MeshLab: meshlab {args.output}")
        print(f"   - CloudCompare: cloudcompare {args.output}")
        print(f"   - Open3D: python -m open3d.visualization.draw {args.output}")
    else:
        print("\n⚠️  Could not extract 3D point cloud (no depth maps)")

    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
