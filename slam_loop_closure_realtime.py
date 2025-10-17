#!/usr/bin/env python3
"""
SLAM Loop Closure Detection avec visualisation 3D en temps réel
La visualisation s'affiche dès le début et se met à jour pendant le traitement.

Usage:
    python slam_loop_closure_realtime.py --images_path /path/to/images --poses_file /path/to/vertices.txt
"""

import argparse
import time
from pathlib import Path
import numpy as np
import csv
import threading

import torch
from PIL import Image
import torchvision.transforms as tfm
import open3d as o3d


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


class RealtimeVisualizer:
    """Visualiseur Open3D temps réel pour le graphe de poses."""

    def __init__(self, poses):
        self.poses = poses
        self.positions = np.array([pose['position'] for pose in poses])

        # Geometries
        self.pcd = None
        self.sequential_lines = None
        self.loop_lines = None
        self.spheres = []

        # State
        self.current_frame = 0
        self.loop_closures = []
        self.consistent_closures = set()

        # Threading for smooth updates
        self.lock = threading.Lock()
        self.pending_updates = []  # Queue of updates to apply
        self.needs_update = False

        # Visualizer
        self.vis = o3d.visualization.Visualizer()
        self.running = False

    def initialize(self):
        """Initialize the visualizer window."""
        try:
            self.vis.create_window(
                window_name="SLAM Real-Time - Loop Closures",
                width=1280,
                height=720
            )

            # Ajouter les axes
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0]
            )
            self.vis.add_geometry(coordinate_frame)

            # Initialiser le point cloud avec au moins un point
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(self.positions[:1])  # Premier point
            self.pcd.paint_uniform_color([0.1, 0.1, 0.9])
            self.vis.add_geometry(self.pcd)

            # Initialiser les lignes séquentielles
            self.sequential_lines = o3d.geometry.LineSet()
            self.sequential_lines.points = o3d.utility.Vector3dVector(self.positions[:1])
            self.sequential_lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2)))
            self.vis.add_geometry(self.sequential_lines)

            # Initialiser les lignes de loop closures
            self.loop_lines = o3d.geometry.LineSet()
            self.loop_lines.points = o3d.utility.Vector3dVector(self.positions[:1])
            self.loop_lines.lines = o3d.utility.Vector2iVector(np.zeros((0, 2)))
            self.vis.add_geometry(self.loop_lines)

            # Configurer la vue
            ctr = self.vis.get_view_control()
            ctr.set_zoom(0.5)
            ctr.set_front([0.5, -0.5, -0.7])
            ctr.set_lookat(self.positions[0])
            ctr.set_up([0, 0, 1])

            # Premier rendu
            self.vis.poll_events()
            self.vis.update_renderer()

            self.running = True
            return True
        except Exception as e:
            print(f"Erreur lors de l'initialisation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_trajectory(self, frame_idx):
        """Queue trajectory update (non-blocking)."""
        with self.lock:
            self.pending_updates.append(('trajectory', frame_idx))
            self.needs_update = True

    def _apply_trajectory_update(self, frame_idx):
        """Apply trajectory update (called from visualization thread)."""
        self.current_frame = frame_idx

        # Update point cloud
        current_positions = self.positions[:frame_idx + 1]
        self.pcd.points = o3d.utility.Vector3dVector(current_positions)
        self.vis.update_geometry(self.pcd)

        # Update sequential edges
        if frame_idx > 0:
            seq_lines = [[i, i + 1] for i in range(frame_idx)]
            self.sequential_lines.points = o3d.utility.Vector3dVector(current_positions)
            self.sequential_lines.lines = o3d.utility.Vector2iVector(seq_lines)
            colors = [[0.8, 0.8, 0.8] for _ in range(len(seq_lines))]
            self.sequential_lines.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.sequential_lines)

    def add_loop_closure(self, query_idx, match_idx, is_consistent):
        """Queue loop closure addition (non-blocking)."""
        with self.lock:
            self.pending_updates.append(('loop', (query_idx, match_idx, is_consistent)))
            self.needs_update = True

    def _update_all_loop_closures(self):
        """Update all loop closures geometry (optimized batch update)."""
        if not self.loop_closures:
            return

        # Update loop closure lines
        loop_lines_list = []
        loop_colors = []

        for i, (q_idx, m_idx) in enumerate(self.loop_closures):
            loop_lines_list.append([q_idx, m_idx])
            if i in self.consistent_closures:
                loop_colors.append([0.0, 1.0, 0.0])  # Green
            else:
                loop_colors.append([1.0, 0.5, 0.0])  # Orange

        # IMPORTANT: Utiliser TOUTES les positions pour les loop closures
        max_idx = max(max(q_idx, m_idx) for q_idx, m_idx in self.loop_closures)
        positions_for_loops = self.positions[:max_idx + 1]

        self.loop_lines.points = o3d.utility.Vector3dVector(positions_for_loops)
        self.loop_lines.lines = o3d.utility.Vector2iVector(loop_lines_list)
        self.loop_lines.colors = o3d.utility.Vector3dVector(loop_colors)
        self.vis.update_geometry(self.loop_lines)

    def _apply_loop_closure(self, query_idx, match_idx, is_consistent):
        """Apply loop closure addition (called from visualization thread) - deprecated, use batching."""
        closure_idx = len(self.loop_closures)
        self.loop_closures.append((query_idx, match_idx))

        if is_consistent:
            self.consistent_closures.add(closure_idx)

        self._update_all_loop_closures()

        # Add sphere at loop closure poses
        for idx in [query_idx, match_idx]:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.translate(self.positions[idx])
            sphere.paint_uniform_color([1.0, 0.0, 0.0])
            self.vis.add_geometry(sphere)
            self.spheres.append(sphere)

    def process_pending_updates(self):
        """Process all pending updates from the processing thread."""
        if not self.needs_update:
            return

        with self.lock:
            updates = self.pending_updates.copy()
            self.pending_updates.clear()
            self.needs_update = False

        # Apply updates - batching pour optimiser
        last_trajectory_idx = None
        loop_updates = []

        # Grouper les mises à jour
        for update_type, data in updates:
            if update_type == 'trajectory':
                last_trajectory_idx = data  # Garder seulement la dernière trajectoire
            elif update_type == 'loop':
                loop_updates.append(data)

        # Appliquer la dernière trajectoire seulement
        if last_trajectory_idx is not None:
            self._apply_trajectory_update(last_trajectory_idx)

        # Appliquer toutes les loop closures en une seule fois
        if loop_updates:
            for query_idx, match_idx, is_consistent in loop_updates:
                closure_idx = len(self.loop_closures)
                self.loop_closures.append((query_idx, match_idx))
                if is_consistent:
                    self.consistent_closures.add(closure_idx)

            # Une seule mise à jour géométrique pour toutes les loops
            self._update_all_loop_closures()

    def poll_events(self):
        """Poll events and update renderer."""
        if self.running:
            try:
                self.vis.poll_events()
                self.vis.update_renderer()
                return True
            except:
                self.running = False
                return False
        return False

    def run_visualization_loop(self):
        """Run the visualization loop in the main thread."""
        frame_count = 0
        while self.running:
            # Traiter les mises à jour seulement toutes les 5 frames pour plus de fluidité
            if frame_count % 5 == 0:
                self.process_pending_updates()

            if not self.poll_events():
                break

            frame_count += 1
            time.sleep(0.016)  # ~60 FPS for smooth interaction

    def destroy(self):
        """Destroy the visualizer."""
        self.running = False
        self.vis.destroy_window()


def processing_thread(visualizer, model, device, image_files, args):
    """Thread pour le traitement des images (s'exécute en parallèle de la visualisation)."""
    preprocess = get_preprocess_transform()
    all_features = []
    total_loops = 0
    consistent_loops = 0

    for frame_idx, img_path in enumerate(image_files):
        if not visualizer.running:
            break

        # Charger et extraire features
        img_tensor = load_and_preprocess_image(img_path, preprocess)
        if img_tensor is None:
            continue

        features = extract_single_feature(model, img_tensor, device)
        all_features.append(features)

        # Mettre à jour la trajectoire (toutes les N frames)
        if frame_idx % args.update_every == 0 or frame_idx == 0:
            visualizer.update_trajectory(frame_idx)

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

                    # Ajouter au visualiseur
                    visualizer.add_loop_closure(frame_idx, match_idx, is_consistent)

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
    print("\nVous pouvez continuer à interagir avec la visualisation.")
    print("Fermez la fenêtre pour quitter...")


def main():
    parser = argparse.ArgumentParser(description="SLAM Loop Closure Real-Time Visualization")
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
    parser.add_argument("--update_every", type=int, default=10,
                        help="Mettre à jour la visualisation toutes les N frames (défaut: 10)")
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
    print("🎯 SLAM LOOP CLOSURE - VISUALISATION TEMPS RÉEL")
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
    visualizer = RealtimeVisualizer(poses)
    if not visualizer.initialize():
        print("❌ Impossible d'ouvrir la fenêtre de visualisation")
        return
    print(f"   ✓ Fenêtre ouverte")

    print("\n🎬 DÉBUT DU TRAITEMENT")
    print("="*80)
    print("Commandes dans la fenêtre:")
    print("  • Souris gauche: Rotation")
    print("  • Souris droite: Pan")
    print("  • Molette: Zoom")
    print("="*80 + "\n")

    # Lancer le thread de traitement
    thread = threading.Thread(
        target=processing_thread,
        args=(visualizer, model, device, image_files, args),
        daemon=True
    )
    thread.start()

    # Boucle de visualisation (thread principal - interaction fluide)
    visualizer.run_visualization_loop()

    # Attendre la fin du traitement
    thread.join(timeout=1.0)

    visualizer.destroy()


if __name__ == "__main__":
    main()
