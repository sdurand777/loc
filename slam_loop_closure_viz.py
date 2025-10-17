#!/usr/bin/env python3
"""
SLAM Loop Closure Detection avec MegaLoc + Visualisation 3D du graphe de poses
Traite les frames, détecte les loop closures et visualise le graphe avec Open3D.

Usage:
    python slam_loop_closure_viz.py --images_path /path/to/images --poses_file /path/to/vertices.txt
"""

import argparse
import time
from pathlib import Path
import numpy as np
import csv

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


def load_poses(poses_file):
    """
    Charge les poses depuis un fichier CSV.
    Format: id,tx,ty,tz,qx,qy,qz,qw

    Returns:
        poses: Liste de dictionnaires avec 'id', 'position', 'quaternion'
    """
    poses = []
    with open(poses_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pose = {
                'id': int(row['id']),
                'position': np.array([float(row['tx']), float(row['ty']), float(row['tz'])]),
                'quaternion': np.array([float(row['qx']), float(row['qy']),
                                       float(row['qz']), float(row['qw'])])
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


def create_pose_graph_visualization(poses, loop_closures, consistent_closures,
                                   save_path=None, no_display=False):
    """
    Crée une visualisation 3D du graphe de poses avec Open3D.

    Args:
        poses: Liste des poses
        loop_closures: Liste de tuples (query_idx, match_idx, similarity)
        consistent_closures: Set des indices de loop closures cohérentes
        save_path: Chemin pour sauvegarder une image (optionnel)
        no_display: Si True, ne pas afficher la fenêtre interactive
    """
    print("\n🎨 Création de la visualisation 3D du graphe de poses...")

    # Extraire les positions
    positions = np.array([pose['position'] for pose in poses])

    # Créer les points pour la trajectoire
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.paint_uniform_color([0.1, 0.1, 0.9])  # Bleu pour les poses

    # Créer les lignes pour les edges séquentiels (odométrie)
    sequential_lines = []
    for i in range(len(positions) - 1):
        sequential_lines.append([i, i + 1])

    sequential_line_set = o3d.geometry.LineSet()
    sequential_line_set.points = o3d.utility.Vector3dVector(positions)
    sequential_line_set.lines = o3d.utility.Vector2iVector(sequential_lines)
    # Couleur blanche pour les edges séquentiels
    colors_sequential = [[0.8, 0.8, 0.8] for _ in range(len(sequential_lines))]
    sequential_line_set.colors = o3d.utility.Vector3dVector(colors_sequential)

    # Créer les lignes pour les loop closures
    loop_lines = []
    loop_colors = []

    for i, (query_idx, match_idx, similarity) in enumerate(loop_closures):
        loop_lines.append([query_idx, match_idx])

        # Couleur selon cohérence: vert si cohérent, orange sinon
        if i in consistent_closures:
            loop_colors.append([0.0, 1.0, 0.0])  # Vert: cohérent
        else:
            loop_colors.append([1.0, 0.5, 0.0])  # Orange: incohérent

    loop_line_set = o3d.geometry.LineSet()
    loop_line_set.points = o3d.utility.Vector3dVector(positions)
    loop_line_set.lines = o3d.utility.Vector2iVector(loop_lines)
    loop_line_set.colors = o3d.utility.Vector3dVector(loop_colors)

    # Créer les sphères pour les poses avec loop closures
    loop_pose_indices = set()
    for query_idx, match_idx, _ in loop_closures:
        loop_pose_indices.add(query_idx)
        loop_pose_indices.add(match_idx)

    spheres = []
    for idx in loop_pose_indices:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(positions[idx])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Rouge pour les poses avec loops
        spheres.append(sphere)

    # Ajouter les axes au départ
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    geometries = [pcd, sequential_line_set, loop_line_set, coordinate_frame] + spheres

    print("\nLégende:")
    print("  • Points bleus:     Poses de la trajectoire")
    print("  • Lignes blanches:  Edges séquentiels (odométrie)")
    print("  • Lignes vertes:    Loop closures cohérentes ✓")
    print("  • Lignes oranges:   Loop closures incohérentes ⚠️")
    print("  • Sphères rouges:   Poses impliquées dans des loop closures")

    # Sauvegarder une capture ou afficher
    if save_path or no_display:
        # Mode offscreen rendering
        print("\n📸 Génération d'une image de la visualisation...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)

        for geom in geometries:
            vis.add_geometry(geom)

        # Configurer la vue
        ctr = vis.get_view_control()
        ctr.set_zoom(0.5)
        ctr.set_front([0.5, -0.5, -0.7])
        ctr.set_lookat(np.mean(positions, axis=0))
        ctr.set_up([0, 0, 1])

        # Rendre et sauvegarder
        vis.poll_events()
        vis.update_renderer()

        if save_path:
            vis.capture_screen_image(save_path, do_render=True)
            print(f"   ✓ Image sauvegardée: {save_path}")

        vis.destroy_window()

    if not no_display:
        # Mode interactif
        print("\n🖼️  Ouverture de la fenêtre de visualisation...")
        print("\nCommandes:")
        print("  • Souris gauche:    Rotation")
        print("  • Souris droite:    Pan")
        print("  • Molette:          Zoom")
        print("  • Q ou ESC:         Quitter")
        print("\n⏳ Chargement de la visualisation (peut prendre quelques secondes)...\n")

        try:
            # Utiliser draw_geometries qui est bloquant jusqu'à fermeture de la fenêtre
            o3d.visualization.draw_geometries(
                geometries,
                window_name="SLAM Pose Graph - Loop Closures avec MegaLoc",
                width=1920,
                height=1080,
                left=50,
                top=50,
                point_show_normal=False
            )
            print("\n✓ Fenêtre fermée par l'utilisateur")
        except Exception as e:
            print(f"\n⚠️  Erreur lors de l'affichage: {e}")
            import traceback
            traceback.print_exc()
            print("\n   Astuce: Utilisez --save_screenshot output.png pour sauvegarder sans afficher")
            print("   ou --no_display pour générer uniquement l'image")


def main():
    parser = argparse.ArgumentParser(description="SLAM Loop Closure Detection + Visualisation 3D")
    parser.add_argument("--images_path", type=str, required=True,
                        help="Chemin vers le dossier contenant les images")
    parser.add_argument("--poses_file", type=str, required=True,
                        help="Chemin vers le fichier des poses (CSV: id,tx,ty,tz,qx,qy,qz,qw)")
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
    parser.add_argument("--save_screenshot", type=str, default=None,
                        help="Sauvegarder une capture d'écran au lieu d'afficher (ex: output.png)")
    parser.add_argument("--no_display", action="store_true",
                        help="Ne pas ouvrir la fenêtre interactive, seulement sauvegarder")

    args = parser.parse_args()

    # Configuration du device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("🎯 SLAM LOOP CLOSURE DETECTION + VISUALISATION 3D")
    print("="*80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FPS: {args.fps}")
    print(f"Start detection at frame: {args.start_detection_frame}")
    print(f"Temporal distance: {args.temporal_distance} frames")
    print(f"Similarity threshold: {args.similarity_threshold}")
    if args.temporal_consistency_window > 0:
        print(f"Temporal consistency: ±{args.temporal_consistency_window} frames")
    print("="*80)

    # Charger les poses
    print(f"\n📍 Chargement des poses depuis {args.poses_file}...")
    poses = load_poses(args.poses_file)
    print(f"   ✓ {len(poses)} poses chargées")

    # Scanner les images
    images_path = Path(args.images_path)
    image_files = sorted(images_path.glob("img*.jpg"))

    if len(image_files) == 0:
        print(f"❌ Aucune image trouvée dans {images_path}")
        return

    print(f"\n📂 Trouvé {len(image_files)} images")

    # Vérifier la cohérence entre images et poses
    if len(image_files) != len(poses):
        print(f"⚠️  Attention: {len(image_files)} images mais {len(poses)} poses")
        min_count = min(len(image_files), len(poses))
        image_files = image_files[:min_count]
        poses = poses[:min_count]
        print(f"   Limitation à {min_count} frames")

    if args.max_frames:
        image_files = image_files[:args.max_frames]
        poses = poses[:args.max_frames]
        print(f"   Limitation à {len(image_files)} frames")

    # Charger le modèle
    print(f"\n🤖 Chargement du modèle MegaLoc...")
    model_start = time.time()
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model = model.to(device)
    model.eval()
    print(f"   ✓ Modèle chargé en {time.time() - model_start:.2f}s")

    # Préparation
    preprocess = get_preprocess_transform()
    frame_period = 1.0 / args.fps

    # Stockage
    all_features = []
    loop_closures = []  # Liste de (query_idx, match_idx, similarity)
    consistent_closures = set()  # Set des indices de loop closures cohérentes

    # Statistiques
    total_loop_closures = 0
    total_consistent_closures = 0
    total_processing_time = 0

    print(f"\n🎬 DÉBUT DU TRAITEMENT")
    print("="*80)

    start_time = time.time()

    # Boucle principale
    for frame_idx, img_path in enumerate(image_files):
        frame_start = time.time()

        # Charger et prétraiter
        img_tensor = load_and_preprocess_image(img_path, preprocess)
        if img_tensor is None:
            continue

        # Extraire features
        features = extract_single_feature(model, img_tensor, device)
        all_features.append(features)

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

                # Vérifier cohérence
                is_consistent, max_spread = check_temporal_consistency(
                    match_indices_list,
                    args.temporal_consistency_window
                )

                # Si loop closure détectée
                if best_similarity >= args.similarity_threshold:
                    total_loop_closures += 1
                    match_idx = match_indices_list[0]

                    # Stocker la loop closure
                    closure_idx = len(loop_closures)
                    loop_closures.append((frame_idx, match_idx, best_similarity))

                    if is_consistent:
                        total_consistent_closures += 1
                        consistent_closures.add(closure_idx)

                    status = "✓" if is_consistent else "⚠️"
                    print(f"Loop {total_loop_closures}: Frame {frame_idx:05d} ↔ {match_idx:05d} | "
                          f"Sim: {best_similarity:.4f} {status}")

        # Timer
        frame_time = time.time() - frame_start
        total_processing_time += frame_time

        if frame_time < frame_period:
            time.sleep(frame_period - frame_time)

        # Progression
        if (frame_idx + 1) % 50 == 0:
            print(f"   Progress: {frame_idx + 1}/{len(image_files)} frames")

    # Résumé
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("✅ TRAITEMENT TERMINÉ")
    print("="*80)
    print(f"\n📊 STATISTIQUES:")
    print(f"   • Frames traitées:          {len(image_files)}")
    print(f"   • Loop closures détectées:  {total_loop_closures}")
    if args.temporal_consistency_window > 0:
        print(f"     - Cohérentes (✓):         {total_consistent_closures}")
        print(f"     - Incohérentes (⚠️):       {total_loop_closures - total_consistent_closures}")
    print(f"   • Temps total:              {total_time:.2f}s")
    print(f"   • FPS moyen:                {len(image_files)/total_processing_time:.1f}")

    # Visualisation 3D
    if len(loop_closures) > 0:
        print(f"\n🎨 Génération de la visualisation 3D avec {len(loop_closures)} loop closures...")
        create_pose_graph_visualization(
            poses,
            loop_closures,
            consistent_closures,
            save_path=args.save_screenshot,
            no_display=args.no_display
        )
    else:
        print("\n⚠️  Aucune loop closure détectée, pas de visualisation")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
