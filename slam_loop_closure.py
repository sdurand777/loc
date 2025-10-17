#!/usr/bin/env python3
"""
SLAM Loop Closure Detection avec MegaLoc
Traite les frames séquentiellement et détecte les loop closures en temps réel.

Usage:
    python slam_loop_closure.py --images_path /path/to/images --fps 10
    python slam_loop_closure.py --images_path /path/to/images --fps 10 --similarity_threshold 0.6
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as tfm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def get_preprocess_transform():
    """Retourne la transformation de prétraitement pour les images."""
    return tfm.Compose([
        tfm.Resize((224, 224)),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_and_preprocess_image(image_path, preprocess):
    """Charge et prétraite une seule image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        return img_tensor, img
    except Exception as e:
        print(f"Erreur lors du chargement de {image_path}: {e}")
        return None, None


def extract_single_feature(model, img_tensor, device):
    """Extrait les features d'une seule image."""
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)
        features = model(img_tensor)
        return features.cpu()


def find_loop_closure(query_feature, database_features, current_idx, temporal_distance=50, top_k=3):
    """
    Recherche des loop closures pour la frame actuelle.

    Args:
        query_feature: Feature de la frame actuelle [1, feat_dim]
        database_features: Features de toutes les frames précédentes [N, feat_dim]
        current_idx: Index de la frame actuelle
        temporal_distance: Distance temporelle minimale
        top_k: Nombre de matches à retourner

    Returns:
        top_k_indices: Indices des meilleurs matches
        top_k_similarities: Scores de similarité
    """
    # Calculer les similarités
    similarities = torch.matmul(query_feature, database_features.T).squeeze()  # [N]

    # Appliquer le masque temporel
    if temporal_distance > 0:
        min_idx = max(0, current_idx - temporal_distance)
        similarities[min_idx:] = float('-inf')

    # Trouver les top-k
    if similarities.numel() > 0:
        k = min(top_k, (similarities != float('-inf')).sum().item())
        if k > 0:
            top_k_similarities, top_k_indices = torch.topk(similarities, k=k)
            return top_k_indices, top_k_similarities

    return torch.tensor([]), torch.tensor([])


def check_temporal_consistency(match_indices, window):
    """
    Vérifie que les matches sont temporellement cohérents (groupés ensemble).

    Args:
        match_indices: Liste des indices des matches (triés par similarité)
        window: Fenêtre de tolérance (±window frames autour du meilleur match)

    Returns:
        is_consistent: True si tous les matches sont dans la fenêtre
        max_spread: Distance maximale entre le meilleur match et les autres
    """
    if len(match_indices) <= 1 or window == 0:
        return True, 0

    best_match_idx = match_indices[0]
    min_window = best_match_idx - window
    max_window = best_match_idx + window

    # Vérifier que tous les autres matches sont dans la fenêtre
    max_spread = 0
    for match_idx in match_indices[1:]:
        spread = abs(match_idx - best_match_idx)
        max_spread = max(max_spread, spread)

        if match_idx < min_window or match_idx > max_window:
            return False, max_spread

    return True, max_spread


def create_loop_closure_visualization(query_img, query_idx, match_imgs, match_indices,
                                     similarities, output_path, detection_time, is_consistent, max_spread):
    """Crée une visualisation de la loop closure détectée."""
    fig = plt.figure(figsize=(14, 10))

    # Query en haut
    ax_query = plt.subplot(2, 1, 1)
    ax_query.imshow(query_img)

    if is_consistent:
        title = f'🔥 LOOP CLOSURE DETECTED ✓ - Frame {query_idx}\n(Matches groupés: écart max = {max_spread} frames)'
        title_color = 'green'
    else:
        title = f'⚠️  LOOP CLOSURE DETECTED (incohérente) - Frame {query_idx}\n(Matches trop espacés: écart max = {max_spread} frames)'
        title_color = 'orange'

    ax_query.set_title(title, fontsize=14, fontweight='bold', pad=10, color=title_color)
    ax_query.axis('off')

    # Matches en dessous
    num_matches = len(match_imgs)
    for i, (match_img, match_idx, similarity) in enumerate(zip(match_imgs, match_indices, similarities)):
        ax_match = plt.subplot(2, num_matches, num_matches + 1 + i)
        ax_match.imshow(match_img)

        temporal_dist = query_idx - match_idx
        color = 'green' if similarity > 0.7 else 'orange' if similarity > 0.6 else 'red'
        title = f'Match #{i+1} (Frame {match_idx})\nSimilarity: {similarity:.4f}\nΔt: {temporal_dist} frames'
        ax_match.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax_match.axis('off')

    # Ajouter timestamp
    fig.text(0.99, 0.01, f'Detection time: {detection_time:.3f}s',
             ha='right', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="SLAM Loop Closure Detection avec MegaLoc")
    parser.add_argument("--images_path", type=str, required=True,
                        help="Chemin vers le dossier contenant les images (format: img%05d.jpg)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="FPS pour le traitement des frames (défaut: 10)")
    parser.add_argument("--start_detection_frame", type=int, default=50,
                        help="Frame à partir de laquelle commencer la détection de loop closures (défaut: 50)")
    parser.add_argument("--temporal_distance", type=int, default=50,
                        help="Distance temporelle minimale pour exclure les voisins (défaut: 50)")
    parser.add_argument("--similarity_threshold", type=float, default=0.55,
                        help="Seuil de similarité pour détecter une loop closure (défaut: 0.55)")
    parser.add_argument("--temporal_consistency_window", type=int, default=2,
                        help="Fenêtre de cohérence temporelle : les top-k matches doivent être dans ±N frames du meilleur match (défaut: 2, 0=désactivé)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Nombre de meilleurs matches à afficher (défaut: 3)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device à utiliser (défaut: auto)")
    parser.add_argument("--output_dir", type=str, default="results_loop_closures",
                        help="Dossier de sortie pour les visualisations (défaut: results_loop_closures)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Nombre maximum de frames à traiter (défaut: toutes)")

    args = parser.parse_args()

    # Configuration du device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("🎯 SLAM LOOP CLOSURE DETECTION")
    print("="*80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"FPS: {args.fps}")
    print(f"Start detection at frame: {args.start_detection_frame}")
    print(f"Temporal distance: {args.temporal_distance} frames")
    print(f"Similarity threshold: {args.similarity_threshold}")
    if args.temporal_consistency_window > 0:
        print(f"Temporal consistency: ±{args.temporal_consistency_window} frames (matches doivent être groupés)")
    else:
        print(f"Temporal consistency: désactivée")
    print("="*80)

    # Créer le dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scanner les images
    images_path = Path(args.images_path)
    image_files = sorted(images_path.glob("img*.jpg"))

    if len(image_files) == 0:
        print(f"❌ Aucune image trouvée dans {images_path}")
        return

    print(f"\n📂 Trouvé {len(image_files)} images")

    if args.max_frames:
        image_files = image_files[:args.max_frames]
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

    # Stockage des features et images
    all_features = []
    cached_images = []  # Pour garder les images en mémoire pour visualisation

    # Statistiques
    total_loop_closures = 0
    total_consistent_closures = 0
    total_inconsistent_closures = 0
    total_processing_time = 0
    total_elapsed_time = 0
    detection_times = []

    print(f"\n🎬 DÉBUT DU TRAITEMENT (FPS cible={args.fps}, période={frame_period*1000:.1f}ms/frame)")
    print("="*80)

    start_time = time.time()

    # Boucle principale de traitement
    for frame_idx, img_path in enumerate(image_files):
        frame_start = time.time()

        # Charger et prétraiter l'image
        img_tensor, img_pil = load_and_preprocess_image(img_path, preprocess)
        if img_tensor is None:
            continue

        # Extraire les features
        features = extract_single_feature(model, img_tensor, device)
        all_features.append(features)
        cached_images.append(img_pil)

        # Détection de loop closure à partir de la frame spécifiée
        if frame_idx >= args.start_detection_frame:
            # Créer la base de données avec toutes les features précédentes
            database_features = torch.cat(all_features[:-1], dim=0)  # Exclure la frame actuelle

            # Rechercher des loop closures
            detection_start = time.time()
            top_indices, top_similarities = find_loop_closure(
                features,
                database_features,
                frame_idx,
                temporal_distance=args.temporal_distance,
                top_k=args.top_k
            )
            detection_time = time.time() - detection_start
            detection_times.append(detection_time)

            # Afficher les résultats
            if len(top_indices) > 0:
                best_similarity = top_similarities[0].item()
                match_indices_list = [idx.item() for idx in top_indices]

                # Vérifier la cohérence temporelle
                is_consistent, max_spread = check_temporal_consistency(
                    match_indices_list,
                    args.temporal_consistency_window
                )

                print(f"\n📍 Frame {frame_idx:05d}: {img_path.name}")

                for rank, (match_idx, similarity) in enumerate(zip(top_indices, top_similarities), start=1):
                    match_idx = match_idx.item()
                    similarity = similarity.item()
                    temporal_dist = frame_idx - match_idx

                    marker = ""
                    if rank == 1 and similarity >= args.similarity_threshold:
                        if is_consistent:
                            marker = "🔥 LOOP CLOSURE! ✓"
                        else:
                            marker = "⚠️  LOOP CLOSURE (matches espacés)"

                    print(f"   Match {rank}: Frame {match_idx:05d} | Sim: {similarity:.4f} | Δt: {temporal_dist:4d} frames {marker}")

                # Afficher la cohérence temporelle
                if args.temporal_consistency_window > 0:
                    consistency_status = "✓ cohérents" if is_consistent else f"✗ trop espacés (écart max: {max_spread} frames)"
                    print(f"   Cohérence temporelle: {consistency_status}")

                # Sauvegarder la visualisation si loop closure détectée
                if best_similarity >= args.similarity_threshold:
                    total_loop_closures += 1

                    if is_consistent:
                        total_consistent_closures += 1
                    else:
                        total_inconsistent_closures += 1

                    # Préparer les données pour la visualisation
                    match_imgs = [cached_images[idx] for idx in match_indices_list]
                    similarities_list = [sim.item() for sim in top_similarities]

                    output_path = output_dir / f"loop_closure_frame_{frame_idx:05d}.png"
                    create_loop_closure_visualization(
                        img_pil,
                        frame_idx,
                        match_imgs,
                        match_indices_list,
                        similarities_list,
                        output_path,
                        detection_time,
                        is_consistent,
                        max_spread
                    )
                    print(f"   💾 Visualisation sauvegardée: {output_path.name}")

        # Timer pour respecter le FPS
        frame_time = time.time() - frame_start
        total_processing_time += frame_time

        # Simuler le FPS (si le traitement est plus rapide que le FPS cible)
        sleep_time = 0
        if frame_time < frame_period:
            sleep_time = frame_period - frame_time
            time.sleep(sleep_time)

        # Temps total avec sleep
        frame_elapsed = time.time() - frame_start
        total_elapsed_time += frame_elapsed

        # Affichage de progression (toutes les 10 frames)
        if frame_idx % 10 == 0 and frame_idx > 0:
            avg_processing = total_processing_time / (frame_idx + 1)
            avg_elapsed = total_elapsed_time / (frame_idx + 1)
            processing_fps = 1.0 / avg_processing if avg_processing > 0 else 0
            effective_fps = 1.0 / avg_elapsed if avg_elapsed > 0 else 0

            print(f"   Progress: {frame_idx}/{len(image_files)} | "
                  f"Traitement: {avg_processing*1000:.1f}ms ({processing_fps:.1f} FPS) | "
                  f"Effectif: {avg_elapsed*1000:.1f}ms ({effective_fps:.1f} FPS)")

    # Résumé final
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("✅ TRAITEMENT TERMINÉ")
    print("="*80)
    print(f"\n📊 STATISTIQUES:")
    print(f"   • Frames traitées:         {len(image_files)}")
    print(f"   • Loop closures détectées:  {total_loop_closures}")
    if args.temporal_consistency_window > 0:
        print(f"     - Cohérentes (✓):        {total_consistent_closures}")
        print(f"     - Incohérentes (⚠️):      {total_inconsistent_closures}")
    print(f"   • Temps total:             {total_time:.2f}s")
    print(f"\n⚡ PERFORMANCE:")
    print(f"   • FPS cible:               {args.fps:.1f} FPS")
    print(f"   • Temps traitement moyen:  {total_processing_time/len(image_files)*1000:.1f}ms/frame")
    print(f"   • FPS traitement:          {len(image_files)/total_processing_time:.1f} FPS (sans sleep)")
    print(f"   • Temps effectif moyen:    {total_elapsed_time/len(image_files)*1000:.1f}ms/frame")
    print(f"   • FPS effectif:            {len(image_files)/total_elapsed_time:.1f} FPS (avec sleep)")

    if detection_times:
        avg_detection = sum(detection_times) / len(detection_times)
        print(f"   • Temps moyen détection:   {avg_detection*1000:.2f}ms")

    print(f"\n📁 Résultats sauvegardés dans: {output_dir.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()
