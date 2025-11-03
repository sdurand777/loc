#!/usr/bin/env python3
"""
SLAM Loop Closure Detection with Double Validation: MegaLoc + MapAnything
Uses MegaLoc for initial loop detection, then MapAnything for pose verification.
Visualizes only loops validated by BOTH systems using Rerun.

Usage:
    python slam_loop_closure_megaloc_mapanything.py \\
        --images_path /path/to/images \\
        --poses_file /path/to/vertices.txt \\
        --mapanything_confidence_threshold 0.5
"""

import argparse
import time
from pathlib import Path
import numpy as np
import csv
import shutil
import tempfile
import os

import torch
from PIL import Image
import torchvision.transforms as tfm
import rerun as rr

# MapAnything imports
from mapanything.models import MapAnything
from mapanything.utils.image import load_images

# Disable some warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_preprocess_transform():
    """Returns preprocessing transformation for MegaLoc images."""
    return tfm.Compose([
        tfm.Resize((224, 224)),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def quaternion_to_rotation_matrix(q):
    """Converts quaternion (qx, qy, qz, qw) to rotation matrix."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def load_poses(poses_file, cam_to_world=True):
    """
    Loads poses from CSV file.

    Args:
        poses_file: Path to CSV file
        cam_to_world: If True, poses are cam-to-world and will be inverted to world-to-cam
    """
    poses = []
    with open(poses_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tx = float(row['tx'])
            ty = float(row['ty'])
            tz = float(row['tz'])
            qx = float(row['qx'])
            qy = float(row['qy'])
            qz = float(row['qz'])
            qw = float(row['qw'])

            if cam_to_world:
                # Invert transformation: T_world_cam = T_cam_world^(-1)
                R_cam_world = quaternion_to_rotation_matrix([qx, qy, qz, qw])
                t_cam_world = np.array([tx, ty, tz])

                # Inversion
                R_world_cam = R_cam_world.T
                t_world_cam = -R_world_cam @ t_cam_world

                position = t_world_cam
            else:
                position = np.array([tx, ty, tz])

            pose = {
                'id': int(row['id']),
                'position': position,
                'quaternion': np.array([qx, qy, qz, qw])
            }
            poses.append(pose)
    return poses


def load_and_preprocess_image(image_path, preprocess):
    """Loads and preprocesses a single image for MegaLoc."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        return img_tensor
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None


def extract_single_feature(model, img_tensor, device):
    """Extracts features from a single image using MegaLoc."""
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        features = model(img_tensor)
        return features.cpu()


def find_loop_closure(query_feature, database_features, current_idx, temporal_distance=50, top_k=3):
    """Searches for loop closures for the current frame using MegaLoc."""
    similarities = torch.matmul(query_feature, database_features.T).squeeze()

    if temporal_distance > 0:
        min_idx = max(0, current_idx - temporal_distance)
        similarities[min_idx:] = float('-inf')

    if similarities.numel() > 0:
        k = min(top_k, (similarities != float('-inf')).sum().item())
        if k > 0:
            top_k_similarities, top_k_indices = torch.topk(similarities, k=k)
            return top_k_indices, top_k_similarities

    return torch.tensor([]), torch.tensor([])


def check_temporal_consistency(match_indices, window):
    """Verifies that matches are temporally consistent."""
    if len(match_indices) <= 1 or window == 0:
        return True, 0

    best_match_idx = match_indices[0]
    min_window = best_match_idx - window
    max_window = best_match_idx + window

    max_spread = 0
    for match_idx in match_indices[1:]:
        spread = abs(match_idx - best_match_idx)
        max_spread = max(max_spread, spread)
        if match_idx < min_window or match_idx > max_window:
            return False, max_spread

    return True, max_spread


def compute_confidence_scores(predictions):
    """
    Computes confidence statistics from MapAnything predictions.

    Args:
        predictions: List of predictions for each view [view1, view2]

    Returns:
        Tuple (conf_mean_view1, conf_mean_view2, conf_mean_combined)
    """
    confidences = []

    for pred in predictions:
        if 'conf' in pred:
            conf = pred['conf'].cpu().numpy()  # Shape: (B, H, W) or (B, H, W, 1)

            # Ensure it's 3D (B, H, W)
            if conf.ndim == 4:
                conf = conf.squeeze(-1)

            # Take only valid zones (mask if available)
            if 'mask' in pred:
                mask = pred['mask'].cpu().numpy().squeeze(-1).astype(bool)
                valid_conf = conf[mask]
            else:
                valid_conf = conf.flatten()

            if len(valid_conf) > 0:
                mean_conf = float(np.mean(valid_conf))
                confidences.append(mean_conf)
            else:
                confidences.append(0.0)
        else:
            confidences.append(None)

    # Return scores
    conf_view1 = confidences[0] if len(confidences) > 0 else None
    conf_view2 = confidences[1] if len(confidences) > 1 else None

    # Combined score (average of two views)
    valid_confs = [c for c in confidences if c is not None]
    conf_combined = float(np.mean(valid_confs)) if valid_confs else None

    return conf_view1, conf_view2, conf_combined


def compose_poses(pose_base, pose_relative):
    """
    Composes two poses: result = pose_base ⊕ pose_relative

    Args:
        pose_base: 4x4 transformation matrix (base frame)
        pose_relative: 4x4 transformation matrix (relative transformation)

    Returns:
        4x4 composed transformation matrix
    """
    return pose_base @ pose_relative


def pose_matrix_to_position(pose_matrix):
    """Extracts 3D position from 4x4 pose matrix."""
    return pose_matrix[:3, 3]


def position_and_quaternion_to_matrix(position, quaternion):
    """
    Converts position and quaternion to 4x4 transformation matrix.

    Args:
        position: [x, y, z]
        quaternion: [qx, qy, qz, qw]

    Returns:
        4x4 transformation matrix
    """
    R = quaternion_to_rotation_matrix(quaternion)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def verify_loop_with_mapanything(query_img_path, loop_img_path, mapanything_model,
                                  device, confidence_threshold):
    """
    Verifies a loop closure candidate using MapAnything pose estimation.

    Args:
        query_img_path: Path to query image
        loop_img_path: Path to loop image
        mapanything_model: MapAnything model instance
        device: torch device
        confidence_threshold: Minimum confidence threshold

    Returns:
        Tuple (is_valid, confidence_score, relative_pose_matrix)
        - relative_pose_matrix: 4x4 matrix representing transformation from loop to query
    """
    try:
        # Load images for MapAnything
        views = load_images([str(query_img_path), str(loop_img_path)])

        if len(views) != 2:
            return False, 0.0, None

        # Run inference
        with torch.no_grad():
            predictions = mapanything_model.infer(
                views,
                memory_efficient_inference=False,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
                apply_confidence_mask=False,
            )

        # Extract confidence scores
        conf_query, conf_loop, conf_mean = compute_confidence_scores(predictions)

        # Check if valid based on confidence
        is_valid = conf_mean is not None and conf_mean >= confidence_threshold

        # Extract relative pose as full 4x4 matrix
        relative_pose_matrix = None
        if len(predictions) >= 2:
            pose_query = predictions[0]['camera_poses'][0].cpu().numpy()
            pose_loop = predictions[1]['camera_poses'][0].cpu().numpy()

            # Compute relative pose: transformation from loop to query
            # relative_pose = loop^-1 @ query
            pose_loop_inv = np.linalg.inv(pose_loop)
            relative_pose_matrix = pose_loop_inv @ pose_query

        return is_valid, conf_mean if conf_mean is not None else 0.0, relative_pose_matrix

    except Exception as e:
        print(f"    ⚠️  MapAnything verification error: {e}")
        return False, 0.0, None


def main():
    parser = argparse.ArgumentParser(
        description="SLAM Loop Closure with MegaLoc + MapAnything Double Validation"
    )
    parser.add_argument("--images_path", type=str, required=True,
                        help="Path to images directory")
    parser.add_argument("--poses_file", type=str, required=True,
                        help="Path to poses CSV file")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Processing FPS (default: 10)")
    parser.add_argument("--start_detection_frame", type=int, default=50,
                        help="Start frame for detection (default: 50)")
    parser.add_argument("--temporal_distance", type=int, default=50,
                        help="Minimum temporal distance (default: 50)")
    parser.add_argument("--megaloc_similarity_threshold", type=float, default=0.55,
                        help="MegaLoc similarity threshold (default: 0.55)")
    parser.add_argument("--mapanything_confidence_threshold", type=float, default=0.5,
                        help="MapAnything confidence threshold (default: 0.5)")
    parser.add_argument("--temporal_consistency_window", type=int, default=2,
                        help="Temporal consistency window (default: 2)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of top matches (default: 3)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device (default: auto)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Max number of frames (default: all)")
    parser.add_argument("--no_invert_poses", action="store_true",
                        help="Don't invert poses")
    parser.add_argument("--mapanything_model", type=str,
                        default="facebook/map-anything",
                        help="MapAnything model name (default: facebook/map-anything)")

    args = parser.parse_args()

    # Configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("🎯 SLAM LOOP CLOSURE - DOUBLE VALIDATION (MegaLoc + MapAnything)")
    print("="*80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"MegaLoc similarity threshold: {args.megaloc_similarity_threshold}")
    print(f"MapAnything confidence threshold: {args.mapanything_confidence_threshold}")
    print("="*80)

    # Initialize Rerun
    output_file = "slam_megaloc_mapanything.rrd"
    rr.init("SLAM MegaLoc+MapAnything", spawn=True)

    print("\n" + "="*80)
    print("📺 RERUN VISUALIZATION (NATIVE VIEWER)")
    print("="*80)
    print("🖥️  The native Rerun viewer opens automatically")
    print("   You can interact with the visualization in real-time!")
    print("")
    print(f"Data will be saved to: {output_file}")
    print("="*80 + "\n")

    # Load poses
    print(f"\n📍 Loading poses...")
    invert = not args.no_invert_poses
    poses = load_poses(args.poses_file, cam_to_world=invert)
    if invert:
        print(f"   ✓ {len(poses)} poses loaded (inverted: cam-to-world → world-to-cam)")
    else:
        print(f"   ✓ {len(poses)} poses loaded (not inverted)")

    # Scan images
    images_path = Path(args.images_path)
    image_files = sorted(images_path.glob("img*.jpg"))

    if len(image_files) == 0:
        print(f"❌ No images found")
        return

    # Limit
    if len(image_files) != len(poses):
        min_count = min(len(image_files), len(poses))
        image_files = image_files[:min_count]
        poses = poses[:min_count]

    if args.max_frames:
        image_files = image_files[:args.max_frames]
        poses = poses[:args.max_frames]

    print(f"📂 {len(image_files)} images to process")

    # Load MegaLoc model
    print(f"\n🤖 Loading MegaLoc model...")
    megaloc_model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    megaloc_model = megaloc_model.to(device)
    megaloc_model.eval()
    print(f"   ✓ MegaLoc model loaded")

    # Load MapAnything model
    print(f"\n🗺️  Loading MapAnything model ({args.mapanything_model})...")
    mapanything_model = MapAnything.from_pretrained(args.mapanything_model).to(device)
    mapanything_model.eval()
    print(f"   ✓ MapAnything model loaded")

    print("\n🎬 STARTING PROCESSING")
    print("="*80)
    print("Rerun visualization opens in a separate window.")
    print("You can interact freely during processing!")
    print("="*80 + "\n")

    # Create directories for image pairs
    loop_pairs_dir = Path("loop_pairs_validated")
    megaloc_only_dir = loop_pairs_dir / "megaloc_only"
    double_validated_dir = loop_pairs_dir / "double_validated"

    megaloc_only_dir.mkdir(parents=True, exist_ok=True)
    double_validated_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Save directories:")
    print(f"   ✓ MegaLoc only: {megaloc_only_dir.absolute()}")
    print(f"   ✓ Double validated: {double_validated_dir.absolute()}\n")

    # Preparation
    preprocess = get_preprocess_transform()
    all_features = []

    # Statistics
    megaloc_candidates = 0
    mapanything_validated = 0
    mapanything_rejected = 0

    # Extract all positions for visualization
    positions = np.array([pose['position'] for pose in poses])

    # Main loop
    for frame_idx, img_path in enumerate(image_files):
        # Log timestamp
        rr.set_time_sequence("frame", frame_idx)

        # Load and extract features with MegaLoc
        img_tensor = load_and_preprocess_image(img_path, preprocess)
        if img_tensor is None:
            continue

        features = extract_single_feature(megaloc_model, img_tensor, device)
        all_features.append(features)

        # Visualize trajectory so far
        trajectory = positions[:frame_idx + 1]
        rr.log("world/trajectory", rr.LineStrips3D([trajectory], colors=[0, 0, 255]))

        # Visualize current position
        rr.log("world/current_pose", rr.Points3D([positions[frame_idx]], colors=[255, 0, 0], radii=0.1))

        # Loop closure detection with MegaLoc
        if frame_idx >= args.start_detection_frame:
            database_features = torch.cat(all_features[:-1], dim=0)

            top_indices, top_similarities = find_loop_closure(
                features,
                database_features,
                frame_idx,
                temporal_distance=args.temporal_distance,
                top_k=args.top_k
            )

            if len(top_indices) > 0:
                best_similarity = top_similarities[0].item()
                match_indices_list = [idx.item() for idx in top_indices]

                is_consistent, max_spread = check_temporal_consistency(
                    match_indices_list,
                    args.temporal_consistency_window
                )

                # Check MegaLoc similarity threshold
                if best_similarity >= args.megaloc_similarity_threshold:
                    megaloc_candidates += 1
                    match_idx = match_indices_list[0]

                    query_img_path = img_path
                    loop_img_path = image_files[match_idx]

                    print(f"\n🔍 Loop candidate {megaloc_candidates}: Frame {frame_idx:05d} ↔ {match_idx:05d}")
                    print(f"    MegaLoc similarity: {best_similarity:.4f}")
                    print(f"    Temporal consistency: {'✓' if is_consistent else '✗'}")

                    # Save to MegaLoc-only directory first
                    query_save_path_megaloc = megaloc_only_dir / f"query_{megaloc_candidates}.jpg"
                    loop_save_path_megaloc = megaloc_only_dir / f"loop_{megaloc_candidates}.jpg"
                    shutil.copy(query_img_path, query_save_path_megaloc)
                    shutil.copy(loop_img_path, loop_save_path_megaloc)

                    # Verify with MapAnything
                    print(f"    🗺️  Verifying with MapAnything...")
                    is_valid_mapanything, mapanything_conf, relative_pose_matrix = verify_loop_with_mapanything(
                        query_img_path,
                        loop_img_path,
                        mapanything_model,
                        device,
                        args.mapanything_confidence_threshold
                    )

                    print(f"    MapAnything confidence: {mapanything_conf:.4f}")

                    if is_valid_mapanything:
                        mapanything_validated += 1

                        # Save to double-validated directory
                        query_save_path_validated = double_validated_dir / f"query_{mapanything_validated}.jpg"
                        loop_save_path_validated = double_validated_dir / f"loop_{mapanything_validated}.jpg"
                        shutil.copy(query_img_path, query_save_path_validated)
                        shutil.copy(loop_img_path, loop_save_path_validated)

                        # Calculate estimated query pose from loop pose + relative transformation
                        if relative_pose_matrix is not None:
                            # Convert loop pose to 4x4 matrix
                            loop_pose_matrix = position_and_quaternion_to_matrix(
                                poses[match_idx]['position'],
                                poses[match_idx]['quaternion']
                            )

                            # Extract rotation and translation from relative pose
                            R_relative = relative_pose_matrix[:3, :3]
                            t_relative_local = relative_pose_matrix[:3, 3]  # In loop's frame

                            # Transform relative translation to world frame
                            R_loop = loop_pose_matrix[:3, :3]
                            t_relative_world = R_loop @ t_relative_local

                            # Compose: estimated_query = loop_pose ⊕ relative_pose
                            estimated_query_matrix = compose_poses(loop_pose_matrix, relative_pose_matrix)
                            estimated_query_position = pose_matrix_to_position(estimated_query_matrix)

                            # Real positions
                            real_query_position = positions[frame_idx]
                            real_loop_position = positions[match_idx]

                            # Calculate errors
                            position_error = np.linalg.norm(estimated_query_position - real_query_position)
                            loop_to_query_distance = np.linalg.norm(real_query_position - real_loop_position)

                            # Verify the computation
                            computed_estimated = real_loop_position + t_relative_world

                            print(f"    Relative translation (loop frame): {t_relative_local}")
                            print(f"    Relative translation (world frame): {t_relative_world}")
                            print(f"    Loop position:        {real_loop_position}")
                            print(f"    → Loop + rel_world:   {computed_estimated}")
                            print(f"    → Estimated query:    {estimated_query_position}")
                            print(f"    Real query:           {real_query_position}")
                            print(f"    Position error:       {position_error:.3f} m")
                            print(f"    Loop-Query distance:  {loop_to_query_distance:.3f} m")

                            # Visualize estimated pose and geometric consistency
                            # 🔵 Cyan point: estimated query position by MapAnything
                            rr.log(f"world/estimated_poses/estimated_{mapanything_validated}",
                                   rr.Points3D([estimated_query_position], colors=[0, 255, 255], radii=0.08))

                            # 🟣 Purple line: from loop to estimated query (MapAnything prediction)
                            purple_line = np.array([positions[match_idx], estimated_query_position])
                            rr.log(f"world/mapanything_prediction/pred_{mapanything_validated}",
                                   rr.LineStrips3D([purple_line], colors=[200, 0, 255]))

                            # 🔴 Red line: from estimated to real query (error visualization)
                            red_line = np.array([estimated_query_position, real_query_position])
                            rr.log(f"world/estimation_error/error_{mapanything_validated}",
                                   rr.LineStrips3D([red_line], colors=[255, 0, 0]))

                        # Visualize validated loop closure (green line)
                        color = [0, 255, 0]  # Green for validated
                        loop_line = np.array([positions[frame_idx], positions[match_idx]])
                        rr.log(f"world/validated_loops/loop_{mapanything_validated}",
                               rr.LineStrips3D([loop_line], colors=[color]))

                        # Mark real poses with green spheres
                        rr.log(f"world/validated_poses/pose_query_{frame_idx}",
                               rr.Points3D([positions[frame_idx]], colors=[0, 255, 0], radii=0.05))
                        rr.log(f"world/validated_poses/pose_loop_{match_idx}",
                               rr.Points3D([positions[match_idx]], colors=[0, 255, 0], radii=0.05))

                        print(f"    ✅ VALIDATED by both systems!")
                    else:
                        mapanything_rejected += 1
                        print(f"    ❌ REJECTED by MapAnything (low confidence)")

        # Progress
        if (frame_idx + 1) % 50 == 0:
            print(f"\n   Progress: {frame_idx + 1}/{len(image_files)} frames")
            print(f"   MegaLoc candidates: {megaloc_candidates}")
            print(f"   Double validated: {mapanything_validated}")
            print(f"   Rejected by MapAnything: {mapanything_rejected}\n")

    # Save Rerun file
    print(f"\n💾 Saving data to {output_file}...")
    rr.save(output_file)

    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE")
    print("="*80)
    print(f"Frames processed: {len(image_files)}")
    print(f"\nLoop Closure Statistics:")
    print(f"  MegaLoc candidates: {megaloc_candidates}")
    print(f"  MapAnything validated: {mapanything_validated} ({mapanything_validated/megaloc_candidates*100:.1f}%)" if megaloc_candidates > 0 else "  MapAnything validated: 0")
    print(f"  MapAnything rejected: {mapanything_rejected} ({mapanything_rejected/megaloc_candidates*100:.1f}%)" if megaloc_candidates > 0 else "  MapAnything rejected: 0")
    print(f"\n📊 Visualization file: {output_file}")
    print(f"   To visualize: rerun {output_file}")
    print(f"\n📁 Saved image pairs:")
    print(f"   MegaLoc candidates: {megaloc_only_dir}/")
    print(f"   Double validated: {double_validated_dir}/")


if __name__ == "__main__":
    main()
