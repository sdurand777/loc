#!/usr/bin/env python3
"""
SLAM Loop Closure Detection avec visualisation Rerun (super fluide!)
Rerun est conçu pour la visualisation robotique temps réel.

Usage:
    python slam_loop_closure_rerun.py --images_path /path/to/images --poses_file /path/to/vertices.txt
"""

import argparse
import time
from pathlib import Path
import numpy as np
import csv

import torch
from PIL import Image
import torchvision.transforms as tfm
import rerun as rr


def get_preprocess_transform():
    """Retourne la transformation de prétraitement pour les images."""
    return tfm.Compose([
        tfm.Resize((224, 224)),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def quaternion_to_rotation_matrix(q):
    """Convertit un quaternion (qx, qy, qz, qw) en matrice de rotation."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def load_poses(poses_file, cam_to_world=True):
    """
    Charge les poses depuis un fichier CSV.

    Args:
        poses_file: Chemin vers le fichier CSV
        cam_to_world: Si True, les poses sont en cam-to-world et seront inversées vers world-to-cam
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
                # Inverser la transformation: T_world_cam = T_cam_world^(-1)
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
    """Charge et prétraite une seule image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        return img_tensor
    except Exception as e:
        print(f"Erreur lors du chargement de {image_path}: {e}")
        return None


def extract_single_feature(model, img_tensor, device):
    """Extrait les features d'une seule image."""
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        features = model(img_tensor)
        return features.cpu()


def find_loop_closure(query_feature, database_features, current_idx, temporal_distance=50, top_k=3):
    """Recherche des loop closures pour la frame actuelle."""
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
    """Vérifie que les matches sont temporellement cohérents."""
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


def main():
    parser = argparse.ArgumentParser(description="SLAM Loop Closure Real-Time Visualization (Rerun)")
    parser.add_argument("--images_path", type=str, required=True,
                        help="Chemin vers le dossier contenant les images")
    parser.add_argument("--poses_file", type=str, required=True,
                        help="Chemin vers le fichier des poses")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="FPS pour le traitement (défaut: 10)")
    parser.add_argument("--start_detection_frame", type=int, default=50,
                        help="Frame de départ pour la détection (défaut: 50)")
    parser.add_argument("--temporal_distance", type=int, default=50,
                        help="Distance temporelle minimale (défaut: 50)")
    parser.add_argument("--similarity_threshold", type=float, default=0.55,
                        help="Seuil de similarité (défaut: 0.55)")
    parser.add_argument("--temporal_consistency_window", type=int, default=2,
                        help="Fenêtre de cohérence temporelle (défaut: 2)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Nombre de matches (défaut: 3)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device (défaut: auto)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Nombre max de frames (défaut: toutes)")
    parser.add_argument("--no_invert_poses", action="store_true",
                        help="Ne pas inverser les poses")

    args = parser.parse_args()

    # Configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("🎯 SLAM LOOP CLOSURE - VISUALISATION RERUN (FLUIDE!)")
    print("="*80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)

    # Initialiser Rerun
    rr.init("SLAM Loop Closures", spawn=True)

    # Charger les poses
    print(f"\n📍 Chargement des poses...")
    invert = not args.no_invert_poses
    poses = load_poses(args.poses_file, cam_to_world=invert)
    if invert:
        print(f"   ✓ {len(poses)} poses chargées (inversées: cam-to-world → world-to-cam)")
    else:
        print(f"   ✓ {len(poses)} poses chargées (non inversées)")

    # Scanner les images
    images_path = Path(args.images_path)
    image_files = sorted(images_path.glob("img*.jpg"))

    if len(image_files) == 0:
        print(f"❌ Aucune image trouvée")
        return

    # Limiter
    if len(image_files) != len(poses):
        min_count = min(len(image_files), len(poses))
        image_files = image_files[:min_count]
        poses = poses[:min_count]

    if args.max_frames:
        image_files = image_files[:args.max_frames]
        poses = poses[:args.max_frames]

    print(f"📂 {len(image_files)} images à traiter")

    # Charger le modèle
    print(f"\n🤖 Chargement du modèle MegaLoc...")
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model = model.to(device)
    model.eval()
    print(f"   ✓ Modèle chargé")

    print("\n🎬 DÉBUT DU TRAITEMENT")
    print("="*80)
    print("La visualisation Rerun s'ouvre dans une fenêtre séparée.")
    print("Vous pouvez interagir librement pendant le traitement!")
    print("="*80 + "\n")

    # Préparation
    preprocess = get_preprocess_transform()
    all_features = []
    total_loops = 0
    consistent_loops = 0

    # Extraire toutes les positions pour la visualisation
    positions = np.array([pose['position'] for pose in poses])

    # Boucle principale
    for frame_idx, img_path in enumerate(image_files):
        # Log du timestamp
        rr.set_time_sequence("frame", frame_idx)

        # Charger et extraire features
        img_tensor = load_and_preprocess_image(img_path, preprocess)
        if img_tensor is None:
            continue

        features = extract_single_feature(model, img_tensor, device)
        all_features.append(features)

        # Visualiser la trajectoire jusqu'à maintenant
        trajectory = positions[:frame_idx + 1]
        rr.log("world/trajectory", rr.LineStrips3D([trajectory], colors=[0, 0, 255]))

        # Visualiser la position actuelle
        rr.log("world/current_pose", rr.Points3D([positions[frame_idx]], colors=[255, 0, 0], radii=0.1))

        # Détection de loop closure
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

                if best_similarity >= args.similarity_threshold:
                    total_loops += 1
                    match_idx = match_indices_list[0]

                    if is_consistent:
                        consistent_loops += 1

                    # Visualiser la loop closure
                    color = [0, 255, 0] if is_consistent else [255, 128, 0]  # Green or Orange
                    loop_line = np.array([positions[frame_idx], positions[match_idx]])
                    rr.log(f"world/loop_closures/loop_{total_loops}",
                           rr.LineStrips3D([loop_line], colors=[color]))

                    # Marquer les poses avec des sphères
                    rr.log(f"world/loop_poses/pose_{frame_idx}",
                           rr.Points3D([positions[frame_idx]], colors=[255, 0, 0], radii=0.05))
                    rr.log(f"world/loop_poses/pose_{match_idx}",
                           rr.Points3D([positions[match_idx]], colors=[255, 0, 0], radii=0.05))

                    status = "✓" if is_consistent else "⚠️"
                    print(f"Loop {total_loops}: Frame {frame_idx:05d} ↔ {match_idx:05d} | "
                          f"Sim: {best_similarity:.4f} {status}")

        # Progression
        if (frame_idx + 1) % 50 == 0:
            print(f"   Progress: {frame_idx + 1}/{len(image_files)} frames")

    print("\n" + "="*80)
    print("✅ TRAITEMENT TERMINÉ")
    print("="*80)
    print(f"Frames traitées: {len(image_files)}")
    print(f"Loop closures: {total_loops} (cohérentes: {consistent_loops})")
    print("\nLa visualisation Rerun reste ouverte.")
    print("Vous pouvez continuer à interagir avec elle!")


if __name__ == "__main__":
    main()
