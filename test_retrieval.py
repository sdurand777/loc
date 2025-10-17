#!/usr/bin/env python3
"""
Script de test pour MegaLoc.
Permet de tester la récupération d'images similaires à partir d'un dataset.

Usage:
    python test_retrieval.py --images_path /path/to/images --query_indices 0 5 10
    python test_retrieval.py --images_path /path/to/images --query_indices 0 5 10 --top_k 5
"""

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as tfm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_images_from_path(images_path):
    """Charge toutes les images JPG d'un dossier."""
    images_path = Path(images_path)
    image_files = sorted(list(images_path.glob("*.jpg")) + list(images_path.glob("*.JPG")))

    if len(image_files) == 0:
        raise ValueError(f"Aucune image JPG trouvée dans {images_path}")

    print(f"Trouvé {len(image_files)} images dans {images_path}")
    return image_files


def get_preprocess_transform():
    """Retourne la transformation de prétraitement pour les images."""
    return tfm.Compose([
        tfm.Resize((224, 224)),  # Important: resize fixe pour MegaLoc
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_features(model, image_files, device, batch_size=32):
    """Calcule les features pour toutes les images avec batching."""
    features_list = []
    preprocess = get_preprocess_transform()

    print(f"\nCalcul des features pour {len(image_files)} images (batch_size={batch_size})...")
    print(f"Device: {device}")
    model.eval()

    start_time = time.time()
    total_load_time = 0
    total_inference_time = 0

    with torch.no_grad():
        for i in range(0, len(image_files), batch_size):
            batch_paths = image_files[i:i + batch_size]
            batch_images = []

            # Timer: chargement des images
            load_start = time.time()
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = preprocess(img)
                    batch_images.append(img_tensor)
                except Exception as e:
                    print(f"Erreur lors du chargement de {img_path}: {e}")
                    continue
            load_time = time.time() - load_start
            total_load_time += load_time

            if batch_images:
                # Timer: inférence sur le GPU/CPU
                inference_start = time.time()
                batch_tensor = torch.stack(batch_images).to(device)
                features = model(batch_tensor)
                features_list.append(features.cpu())

                # Synchronisation GPU pour timer précis
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                inference_time = time.time() - inference_start
                total_inference_time += inference_time

            # Affichage de progression
            if (i // batch_size) % 10 == 0:
                print(f"  Traitement: {min(i + batch_size, len(image_files))}/{len(image_files)}")

    total_time = time.time() - start_time

    # Concat toutes les features en un seul tensor [N, feat_dim]
    all_features = torch.cat(features_list, dim=0)

    # Affichage des statistiques de temps
    print(f"\nFeatures calculées: {all_features.shape}")
    print(f"\n📊 Statistiques de temps:")
    print(f"  • Temps total:       {total_time:.2f}s")
    print(f"  • Chargement images: {total_load_time:.2f}s ({total_load_time/total_time*100:.1f}%)")
    print(f"  • Inférence modèle:  {total_inference_time:.2f}s ({total_inference_time/total_time*100:.1f}%)")
    print(f"  • Images/seconde:    {len(image_files)/total_time:.1f} img/s")
    if len(image_files) > 0:
        print(f"  • Temps par image:   {total_time/len(image_files)*1000:.1f}ms")

    return all_features


def find_top_matches(query_features, database_features, query_indices, top_k=3, temporal_distance=0):
    """Trouve les top-k matches les plus similaires (par similarité cosine).

    Args:
        query_features: Features des queries [num_queries, feat_dim]
        database_features: Features de la base de données [num_database, feat_dim]
        query_indices: Indices des queries dans la base de données
        top_k: Nombre de matches à retourner
        temporal_distance: Distance temporelle minimale (en frames) pour exclure les voisins temporels
    """
    start_time = time.time()

    # Calcul des similarités cosine (les features sont déjà L2-normalisées)
    similarities = torch.matmul(query_features, database_features.T)  # [num_queries, num_database]

    # Appliquer le masque de distance temporelle si nécessaire
    if temporal_distance > 0:
        for i, query_idx in enumerate(query_indices):
            # Créer un masque pour exclure les frames dans la fenêtre temporelle
            min_idx = max(0, query_idx - temporal_distance)
            max_idx = min(database_features.shape[0] - 1, query_idx + temporal_distance)

            # Mettre à -inf les similarités dans la fenêtre temporelle
            similarities[i, min_idx:max_idx + 1] = float('-inf')

    # Trouve les top-k indices pour chaque query
    top_k_values, top_k_indices = torch.topk(similarities, k=top_k, dim=1)

    elapsed = time.time() - start_time
    print(f"  ⏱️  Temps de recherche des matches: {elapsed:.3f}s")

    return top_k_indices, top_k_values


def create_visualization(query_img_path, match_img_paths, similarities, output_path, query_idx, match_indices):
    """Crée une visualisation avec la query en haut et les matches en dessous."""
    # Charger les images
    query_img = Image.open(query_img_path).convert('RGB')
    match_imgs = [Image.open(path).convert('RGB') for path in match_img_paths]

    # Redimensionner les images pour avoir une taille uniforme
    target_width = 400

    def resize_maintain_aspect(img, target_w):
        aspect = img.height / img.width
        target_h = int(target_w * aspect)
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    query_img = resize_maintain_aspect(query_img, target_width)
    match_imgs = [resize_maintain_aspect(img, target_width) for img in match_imgs]

    # Créer la figure avec matplotlib
    fig = plt.figure(figsize=(14, 10))

    # Afficher la query en haut (grande)
    ax_query = plt.subplot(2, 1, 1)
    ax_query.imshow(query_img)
    ax_query.set_title(f'QUERY (Index: {query_idx})', fontsize=16, fontweight='bold', pad=10)
    ax_query.axis('off')

    # Afficher les matches en dessous (3 images côte à côte)
    for i, (match_img, similarity, match_idx) in enumerate(zip(match_imgs, similarities, match_indices)):
        ax_match = plt.subplot(2, 3, 4 + i)
        ax_match.imshow(match_img)

        # Titre avec le rang, l'indice, et la similarité
        temporal_dist = abs(match_idx - query_idx)
        title = f'Match #{i+1} (Index: {match_idx})\nSimilarity: {similarity:.4f}\nΔt: {temporal_dist} frames'
        ax_match.set_title(title, fontsize=11, fontweight='bold')
        ax_match.axis('off')

    plt.tight_layout()

    # Sauvegarder l'image
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Visualisation sauvegardée: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test de récupération d'images avec MegaLoc")
    parser.add_argument("--images_path", type=str, required=True,
                        help="Chemin vers le dossier contenant les images JPG")
    parser.add_argument("--query_indices", type=int, nargs='+', required=True,
                        help="Indices des images à utiliser comme queries (ex: 0 5 10)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Nombre de meilleurs matches à retourner (défaut: 3)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device à utiliser: auto (défaut, utilise cuda si disponible), cuda, ou cpu")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Dossier de sortie pour les visualisations (défaut: results)")
    parser.add_argument("--temporal_distance", type=int, default=0,
                        help="Distance temporelle minimale (en frames) pour exclure les voisins temporels (défaut: 0, pas de filtrage)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Taille du batch pour l'extraction de features (défaut: 32)")

    args = parser.parse_args()

    # Timer global
    script_start_time = time.time()

    # Déterminer le device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device auto-détecté: {device}")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA demandé mais non disponible, utilisation du CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        print(f"🔧 Device sélectionné: {device}")

    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Créer le dossier de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement des images
    print("\n" + "="*80)
    print("📂 CHARGEMENT DES IMAGES")
    print("="*80)
    load_images_start = time.time()
    image_files = load_images_from_path(args.images_path)
    load_images_time = time.time() - load_images_start
    print(f"⏱️  Temps de scan des images: {load_images_time:.3f}s")

    # Validation des indices de query
    for idx in args.query_indices:
        if idx < 0 or idx >= len(image_files):
            raise ValueError(f"Index de query {idx} hors limites (0-{len(image_files)-1})")

    # Chargement du modèle
    print("\n" + "="*80)
    print("🤖 CHARGEMENT DU MODÈLE")
    print("="*80)
    model_load_start = time.time()
    print(f"Chargement de MegaLoc sur {device}...")
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model = model.to(device)
    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"⏱️  Temps de chargement du modèle: {model_load_time:.2f}s")

    # Calcul des features pour toutes les images
    print("\n" + "="*80)
    print("🔍 EXTRACTION DES FEATURES")
    print("="*80)
    all_features = compute_features(model, image_files, device, args.batch_size)

    # Extraction des features des queries
    query_features = all_features[args.query_indices]

    # Recherche des meilleurs matches
    print("\n" + "="*80)
    print("🎯 RECHERCHE DES MATCHES")
    print("="*80)
    if args.temporal_distance > 0:
        print(f"Top-{args.top_k} matches par query (exclusion: ±{args.temporal_distance} frames)")
    else:
        print(f"Top-{args.top_k} matches par query")

    top_k_indices, top_k_similarities = find_top_matches(
        query_features,
        all_features,
        query_indices=args.query_indices,
        top_k=args.top_k,
        temporal_distance=args.temporal_distance
    )

    # Affichage des résultats et génération des visualisations
    print("\n" + "="*80)
    print("📊 RÉSULTATS")
    print("="*80)

    vis_start_time = time.time()

    for i, query_idx in enumerate(args.query_indices):
        print(f"\nQuery #{query_idx}: {image_files[query_idx].name}")
        print("-" * 80)

        match_paths = []
        similarities = []
        match_indices_list = []

        for rank, (match_idx, similarity) in enumerate(
            zip(top_k_indices[i], top_k_similarities[i]), start=1
        ):
            match_idx = match_idx.item()
            similarity = similarity.item()

            temporal_dist = abs(match_idx - query_idx)
            marker = " (SELF)" if match_idx == query_idx else ""
            print(f"  Rank {rank}: Index {match_idx:4d} | "
                  f"Similarity: {similarity:.4f} | "
                  f"Δt: {temporal_dist:4d} frames | "
                  f"{image_files[match_idx].name}{marker}")

            match_paths.append(image_files[match_idx])
            similarities.append(similarity)
            match_indices_list.append(match_idx)

        # Créer la visualisation
        output_path = output_dir / f"query_{query_idx}_matches.png"
        create_visualization(
            image_files[query_idx],
            match_paths,
            similarities,
            output_path,
            query_idx,
            match_indices_list
        )

    vis_time = time.time() - vis_start_time

    # Résumé final avec tous les timers
    total_time = time.time() - script_start_time

    print("\n" + "="*80)
    print("✅ TERMINÉ")
    print("="*80)
    print(f"Visualisations sauvegardées dans: {output_dir.absolute()}")
    print("\n⏱️  RÉSUMÉ DES TEMPS:")
    print(f"  • Scan des images:        {load_images_time:.2f}s")
    print(f"  • Chargement du modèle:   {model_load_time:.2f}s")
    print(f"  • Extraction des features: (voir détails ci-dessus)")
    print(f"  • Création des visu.:      {vis_time:.2f}s")
    print(f"  • TEMPS TOTAL:            {total_time:.2f}s")
    print("="*80)


if __name__ == "__main__":
    main()
