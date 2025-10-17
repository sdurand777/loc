#!/usr/bin/env python3
"""
SLAM Loop Closure Detection avec visualisation 3D matplotlib en temps réel
La visualisation s'affiche dès le début et se met à jour pendant le traitement.

Usage:
    python slam_loop_closure_matplotlib.py --images_path /path/to/images --poses_file /path/to/vertices.txt
"""

import argparse
import time
from pathlib import Path
import numpy as np
import csv

import torch
from PIL import Image
import torchvision.transforms as tfm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


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
                # Pour une transformation rigide: T^(-1) = [R^T | -R^T * t]
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


class RealtimeMatplotlibVisualizer:
    """Visualiseur matplotlib temps réel pour le graphe de poses."""

    def __init__(self, poses):
        self.poses = poses
        self.positions = np.array([pose['position'] for pose in poses])

        # State
        self.current_frame = 0
        self.loop_closures = []
        self.consistent_closures = set()
        self.pending_loop_closures = []  # Stocke les loop closures sans les afficher

        # Matplotlib
        plt.ion()  # Mode interactif
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Plot objects
        self.trajectory_plot = None
        self.sequential_lines = []
        self.loop_lines = []
        self.loop_spheres = []

        # Configure plot
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('SLAM Real-Time - Loop Closures', fontsize=16, fontweight='bold')

        # Légende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Trajectoire'),
            Patch(facecolor='gray', alpha=0.3, label='Edges séquentiels'),
            Patch(facecolor='green', label='Loop closures ✓'),
            Patch(facecolor='orange', label='Loop closures ⚠️'),
            Patch(facecolor='red', label='Poses avec loops')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        plt.show(block=False)
        plt.pause(0.001)

    def update_trajectory(self, frame_idx, display_frame=None):
        """Update trajectory up to frame_idx."""
        self.current_frame = frame_idx

        # Clear previous trajectory
        if self.trajectory_plot is not None:
            self.trajectory_plot.remove()

        # Plot trajectory
        current_positions = self.positions[:frame_idx + 1]
        self.trajectory_plot = self.ax.scatter(
            current_positions[:, 0],
            current_positions[:, 1],
            current_positions[:, 2],
            c='blue', s=10, alpha=0.6
        )

        # Clear and redraw sequential lines
        for line in self.sequential_lines:
            line.remove()
        self.sequential_lines = []

        if frame_idx > 0:
            for i in range(frame_idx):
                line = self.ax.plot(
                    [self.positions[i, 0], self.positions[i + 1, 0]],
                    [self.positions[i, 1], self.positions[i + 1, 1]],
                    [self.positions[i, 2], self.positions[i + 1, 2]],
                    'gray', alpha=0.3, linewidth=1
                )[0]
                self.sequential_lines.append(line)

        # Adjust limits
        self.ax.set_xlim(current_positions[:, 0].min() - 0.5, current_positions[:, 0].max() + 0.5)
        self.ax.set_ylim(current_positions[:, 1].min() - 0.5, current_positions[:, 1].max() + 0.5)
        self.ax.set_zlim(current_positions[:, 2].min() - 0.5, current_positions[:, 2].max() + 0.5)

        # Update title with current frame info
        if display_frame is not None:
            lag = frame_idx - display_frame
            self.ax.set_title(f'SLAM Real-Time - Frame {frame_idx} (Display: {display_frame}, Lag: {lag})',
                            fontsize=16, fontweight='bold')
        else:
            self.ax.set_title(f'SLAM Real-Time - Frame {frame_idx}',
                            fontsize=16, fontweight='bold')

        plt.draw()
        plt.pause(0.001)

    def add_pending_loop_closure(self, query_idx, match_idx, is_consistent):
        """Stocke une loop closure sans l'afficher (pour optimisation)."""
        self.pending_loop_closures.append((query_idx, match_idx, is_consistent))

    def add_loop_closure(self, query_idx, match_idx, is_consistent):
        """Add a loop closure and draw it immediately."""
        closure_idx = len(self.loop_closures)
        self.loop_closures.append((query_idx, match_idx))

        if is_consistent:
            self.consistent_closures.add(closure_idx)

        # Draw loop closure line
        color = 'green' if is_consistent else 'orange'
        line = self.ax.plot(
            [self.positions[query_idx, 0], self.positions[match_idx, 0]],
            [self.positions[query_idx, 1], self.positions[match_idx, 1]],
            [self.positions[query_idx, 2], self.positions[match_idx, 2]],
            color, linewidth=2, alpha=0.7
        )[0]
        self.loop_lines.append(line)

        # Add spheres at loop closure poses
        for idx in [query_idx, match_idx]:
            sphere = self.ax.scatter(
                [self.positions[idx, 0]],
                [self.positions[idx, 1]],
                [self.positions[idx, 2]],
                c='red', s=50, alpha=0.8, marker='o'
            )
            self.loop_spheres.append(sphere)

        plt.draw()
        plt.pause(0.001)

    def keep_window_open(self):
        """Keep window open at the end."""
        print("\nFenêtre ouverte. Fermez-la pour quitter...")
        plt.ioff()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="SLAM Loop Closure Real-Time Visualization (Matplotlib)")
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
    parser.add_argument("--update_every", type=int, default=5,
                        help="Mettre à jour la visualisation toutes les N frames (défaut: 5)")
    parser.add_argument("--slow_down", type=float, default=0.0,
                        help="Délai artificiel en secondes entre chaque frame (défaut: 0)")
    parser.add_argument("--invert_poses", action="store_true",
                        help="Inverser les poses de cam-to-world vers world-to-cam (défaut: activé)")
    parser.add_argument("--no_invert_poses", action="store_true",
                        help="Ne pas inverser les poses")

    args = parser.parse_args()

    # Configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("🎯 SLAM LOOP CLOSURE - VISUALISATION TEMPS RÉEL (Matplotlib)")
    print("="*80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)

    # Charger les poses
    print(f"\n📍 Chargement des poses...")
    invert = not args.no_invert_poses  # Par défaut on inverse
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

    # Initialiser le visualiseur
    print(f"\n🎨 Initialisation de la visualisation temps réel...")
    visualizer = RealtimeMatplotlibVisualizer(poses)
    print(f"   ✓ Fenêtre ouverte")

    print("\n🎬 DÉBUT DU TRAITEMENT")
    print("="*80)

    # Préparation
    preprocess = get_preprocess_transform()
    all_features = []
    total_loops = 0
    consistent_loops = 0

    # Timers
    total_load_time = 0
    total_feature_time = 0
    total_detection_time = 0

    # Boucle principale
    for frame_idx, img_path in enumerate(image_files):
        frame_start = time.time()
        # Charger et extraire features
        load_start = time.time()
        img_tensor = load_and_preprocess_image(img_path, preprocess)
        if img_tensor is None:
            continue

        features = extract_single_feature(model, img_tensor, device)
        all_features.append(features)
        load_time = time.time() - load_start
        total_load_time += load_time
        total_feature_time += load_time

        # Mise à jour de trajectoire désactivée pour maximiser les performances
        # La visualisation complète sera générée à la fin du traitement

        # Détection de loop closure
        if frame_idx >= args.start_detection_frame:
            detection_start = time.time()

            database_features = torch.cat(all_features[:-1], dim=0)

            top_indices, top_similarities = find_loop_closure(
                features,
                database_features,
                frame_idx,
                temporal_distance=args.temporal_distance,
                top_k=args.top_k
            )

            detection_time = time.time() - detection_start
            total_detection_time += detection_time

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

                    # Stocker pour afficher à la fin (pas d'affichage en temps réel)
                    visualizer.add_pending_loop_closure(frame_idx, match_idx, is_consistent)

                    status = "✓" if is_consistent else "⚠️"
                    print(f"Loop {total_loops}: Frame {frame_idx:05d} ↔ {match_idx:05d} | "
                          f"Sim: {best_similarity:.4f} {status} | "
                          f"Det: {detection_time*1000:.1f}ms")

        # Ralentissement artificiel si demandé
        if args.slow_down > 0:
            time.sleep(args.slow_down)

        frame_total = time.time() - frame_start

        # Progression
        if (frame_idx + 1) % 50 == 0:
            avg_feature = total_feature_time / (frame_idx + 1) * 1000
            avg_detection = total_detection_time / max(1, frame_idx - args.start_detection_frame + 1) * 1000 if frame_idx >= args.start_detection_frame else 0
            print(f"   Progress: {frame_idx + 1}/{len(image_files)} | "
                  f"Avg Feature: {avg_feature:.1f}ms | "
                  f"Avg Detection: {avg_detection:.1f}ms | "
                  f"DB size: {len(all_features)}")

    print("\n" + "="*80)
    print("✅ TRAITEMENT TERMINÉ")
    print("="*80)
    print(f"Frames traitées: {frame_idx + 1}")
    print(f"Loop closures: {total_loops} (cohérentes: {consistent_loops})")
    print(f"\n⏱️  BREAKDOWN DES TEMPS:")
    print(f"   • Feature extraction: {total_feature_time:.2f}s ({total_feature_time/(frame_idx+1)*1000:.1f}ms/frame)")
    print(f"   • Loop detection:     {total_detection_time:.2f}s ({total_detection_time/max(1,frame_idx-args.start_detection_frame+1)*1000:.1f}ms/frame)")

    # Générer la visualisation finale
    print(f"\n🎨 Génération de la visualisation finale...")
    viz_final_start = time.time()

    # Update trajectoire complète
    visualizer.update_trajectory(frame_idx)

    # Ajouter toutes les loop closures PENDANTES (pas les anciennes dans loop_closures)
    print(f"   • Dessin de {len(visualizer.pending_loop_closures)} loop closures...")
    for query_idx, match_idx, is_consistent in visualizer.pending_loop_closures:
        visualizer.add_loop_closure(query_idx, match_idx, is_consistent)

    viz_final_time = time.time() - viz_final_start
    print(f"   ✓ Visualisation générée en {viz_final_time:.2f}s")

    # Garder la fenêtre ouverte
    visualizer.keep_window_open()


if __name__ == "__main__":
    main()
