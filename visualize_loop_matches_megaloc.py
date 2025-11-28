#!/usr/bin/env python3
"""
Script pour détecter les loop closures avec MegaLoc et visualiser les matches MegaLoc.
Utilise les features spatiales MegaLoc (cluster_features de SALAD) pour les correspondances.

Usage:
    python visualize_loop_matches_megaloc.py --images_path /path/to/images --output_dir loop_matches
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch

import torch
from PIL import Image
import torchvision.transforms as tfm


def get_preprocess_transform():
    """Transformation pour MegaLoc."""
    return tfm.Compose([
        # Resize supprimé car l'image est déjà redimensionnée à 224x224 dans load_and_preprocess_image
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_and_preprocess_image(image_path, preprocess):
    """Charge et prétraite une image."""
    try:
        img = Image.open(image_path).convert('RGB')
        # Redimensionner à 224x224 pour que les coordonnées correspondent aux patches
        img_resized = img.resize((224, 224), Image.LANCZOS)
        # Créer le tensor à partir de l'image redimensionnée (pas de l'originale)
        # Le preprocess va appliquer ToTensor et Normalize, mais PAS un second resize car l'image est déjà 224x224
        img_tensor = preprocess(img_resized)
        return img_resized, img_tensor
    except Exception as e:
        print(f"Erreur lors du chargement de {image_path}: {e}")
        return None, None


def extract_megaloc_features(model, img_tensor, device):
    """
    Extrait les features globales MegaLoc ET les features spatiales MegaLoc.

    Returns:
        global_features: Features MegaLoc normalisées [1, 8448]
        spatial_features: Cluster features SALAD [1, 256, H//14, W//14] (features MegaLoc spatiales)
    """
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Resize to multiple of 14 if needed
        b, c, h, w = img_tensor.shape
        if h % 14 != 0 or w % 14 != 0:
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            img_tensor = tfm.functional.resize(img_tensor, [h, w], antialias=True)

        # Extract DINOv2 features first
        backbone_features, cls_token = model.backbone(img_tensor)

        # Extract MegaLoc spatial features (cluster_features from SALAD)
        # These are the features transformed by MegaLoc's learned MLP
        salad_module = model.aggregator.agg
        spatial_features = salad_module.cluster_features(backbone_features)

        # Extract global features (full MegaLoc pipeline)
        global_features = model(img_tensor)

        return global_features.cpu(), spatial_features.cpu()


def find_loop_closure(query_feature, database_features, current_idx, temporal_distance=50, top_k=3):
    """Recherche des loop closures pour la frame actuelle."""
    similarities = torch.matmul(query_feature, database_features.T).squeeze()

    # Filtrage temporel
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
    """Vérifie la cohérence temporelle des matches."""
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


def find_patch_matches_megaloc(query_features, loop_features, top_k=20):
    """
    Trouve les meilleurs matches entre les patches MegaLoc de deux images.

    Args:
        query_features: [1, 256, H, W] - Cluster features MegaLoc
        loop_features: [1, 256, H, W] - Cluster features MegaLoc
        top_k: Nombre de meilleurs matches à retourner

    Returns:
        matches: Liste de tuples (query_pos, loop_pos, similarity)
        grid_size: (H, W) dimensions de la grille
    """
    # Flatten spatial dimensions
    C, H, W = query_features.shape[1], query_features.shape[2], query_features.shape[3]
    query_flat = query_features.squeeze(0).reshape(C, -1).T  # [H*W, 256]
    loop_flat = loop_features.squeeze(0).reshape(C, -1).T    # [H*W, 256]

    # Normalize (MegaLoc features are L2-normalized)
    query_flat = torch.nn.functional.normalize(query_flat, p=2, dim=1)
    loop_flat = torch.nn.functional.normalize(loop_flat, p=2, dim=1)

    # Compute similarity matrix using cosine similarity
    similarity_matrix = torch.matmul(query_flat, loop_flat.T)  # [H*W, H*W]

    # For each query patch, find best match in loop image
    best_similarities, best_indices = torch.max(similarity_matrix, dim=1)

    # Get top-k matches
    top_k_sims, top_k_query_indices = torch.topk(best_similarities, min(top_k, len(best_similarities)))

    matches = []
    for i in range(len(top_k_query_indices)):
        query_idx = top_k_query_indices[i].item()
        loop_idx = best_indices[query_idx].item()
        sim = top_k_sims[i].item()

        # Convert flat indices to 2D positions
        query_y, query_x = query_idx // W, query_idx % W
        loop_y, loop_x = loop_idx // W, loop_idx % W

        matches.append(((query_x, query_y), (loop_x, loop_y), sim))

    return matches, (H, W)


def visualize_matches(query_img, loop_img, matches, grid_size, similarity_score, output_path):
    """
    Crée une visualisation des matches MegaLoc entre deux images.

    Args:
        query_img: Image PIL de la query
        loop_img: Image PIL du loop
        matches: Liste de (query_pos, loop_pos, similarity)
        grid_size: (H, W) taille de la grille de patches MegaLoc
        similarity_score: Score de similarité global MegaLoc
        output_path: Chemin de sauvegarde
    """
    # Create figure with two images side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Loop Closure Match (MegaLoc Features) - Similarity: {similarity_score:.4f}',
                 fontsize=16, fontweight='bold')

    # Display images
    ax1.imshow(query_img)
    ax1.set_title('Query Image', fontsize=14)
    ax1.axis('off')

    ax2.imshow(loop_img)
    ax2.set_title('Loop Image', fontsize=14)
    ax2.axis('off')

    # Get image dimensions
    # PIL Image.size returns (width, height) !
    img_w, img_h = query_img.size  # width, height
    grid_h, grid_w = grid_size

    # Patch size in pixels
    patch_h = img_h / grid_h
    patch_w = img_w / grid_w

    # Debug info
    print(f"   Image size: {img_w}x{img_h}, Grid: {grid_h}x{grid_w}, Patch size: {patch_w:.1f}x{patch_h:.1f}")

    # Set exact axis limits to avoid matplotlib auto-scaling issues
    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)  # Inverted Y for image coordinates
    ax2.set_xlim(0, img_w)
    ax2.set_ylim(img_h, 0)  # Inverted Y for image coordinates

    # Force aspect ratio to be equal (no distortion)
    # Using 'datalim' to avoid adding margins that would shift coordinates
    ax1.set_aspect('equal', adjustable='datalim')
    ax2.set_aspect('equal', adjustable='datalim')

    # CRITICAL: Call tight_layout BEFORE drawing patches/lines
    # This finalizes the axes positions so transformations are correct
    plt.tight_layout()

    # Force matplotlib to compute final transformations
    # This ensures transData transformations are accurate
    fig.canvas.draw()

    # Normalize similarities for coloring
    similarities = [m[2] for m in matches]
    min_sim, max_sim = min(similarities), max(similarities)

    # Draw matches
    for (qx, qy), (lx, ly), sim in matches:
        # Normalize color (green = high similarity, red = low)
        if max_sim > min_sim:
            norm_sim = (sim - min_sim) / (max_sim - min_sim)
        else:
            norm_sim = 1.0

        color = plt.cm.RdYlGn(norm_sim)

        # Calculate CENTER of each patch (barycentre)
        query_center_x = qx * patch_w + patch_w / 2.0
        query_center_y = qy * patch_h + patch_h / 2.0
        loop_center_x = lx * patch_w + patch_w / 2.0
        loop_center_y = ly * patch_h + patch_h / 2.0

        # Draw semi-transparent rectangles to show patch boundaries (for debugging)
        # Uncomment to visualize patch locations
        # query_rect = patches.Rectangle((qx * patch_w, qy * patch_h), patch_w, patch_h,
        #                                linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
        # loop_rect = patches.Rectangle((lx * patch_w, ly * patch_h), patch_w, patch_h,
        #                               linewidth=1, edgecolor=color, facecolor=color, alpha=0.2)
        # ax1.add_patch(query_rect)
        # ax2.add_patch(loop_rect)

        # Draw smaller circles at patch centers
        circle_radius = min(patch_w, patch_h) / 3.5  # Adaptive radius based on patch size
        query_circle = patches.Circle((query_center_x, query_center_y), radius=circle_radius,
                                     color=color, alpha=0.9, zorder=3, linewidth=1.5,
                                     edgecolor='white')
        loop_circle = patches.Circle((loop_center_x, loop_center_y), radius=circle_radius,
                                    color=color, alpha=0.9, zorder=3, linewidth=1.5,
                                    edgecolor='white')
        ax1.add_patch(query_circle)
        ax2.add_patch(loop_circle)

        # Draw connecting line from center to center using ConnectionPatch
        # ConnectionPatch automatically handles coordinate transformations between axes
        con = ConnectionPatch(
            xyA=(loop_center_x, loop_center_y),    # Point in loop image (ax2)
            xyB=(query_center_x, query_center_y),  # Point in query image (ax1)
            coordsA="data", coordsB="data",
            axesA=ax2, axesB=ax1,
            color=color, alpha=0.7, linewidth=1.5, zorder=2
        )
        ax2.add_artist(con)

    # Add legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                               norm=plt.Normalize(vmin=min_sim, vmax=max_sim))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal',
                       pad=0.05, fraction=0.046)
    cbar.set_label('MegaLoc Patch Similarity (Cosine)', fontsize=12)

    # Add info text
    info_text = f'Grid: {grid_h}×{grid_w} patches | Feature dim: 256 (MegaLoc cluster features)'
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic', color='gray')

    # CRITICAL: Don't use bbox_inches='tight' - it invalidates all coordinate transformations!
    # Save with normal bbox to preserve the transformations
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"   Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Loop Closure Detection with MegaLoc Visual Matches")
    parser.add_argument("--images_path", type=str, required=True,
                       help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="loop_matches_megaloc",
                       help="Output directory for match visualizations")
    parser.add_argument("--start_detection_frame", type=int, default=50,
                       help="Start detection at frame (default: 50)")
    parser.add_argument("--temporal_distance", type=int, default=50,
                       help="Minimum temporal distance (default: 50)")
    parser.add_argument("--similarity_threshold", type=float, default=0.55,
                       help="Similarity threshold (default: 0.55)")
    parser.add_argument("--temporal_consistency_window", type=int, default=2,
                       help="Temporal consistency window (default: 2)")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Number of top matches to consider (default: 3)")
    parser.add_argument("--num_patch_matches", type=int, default=20,
                       help="Number of patch matches to visualize (default: 20)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use (default: auto)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process (default: all)")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("🎯 LOOP CLOSURE DETECTION WITH MEGALOC VISUAL MATCHES")
    print("="*80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("")
    print("ℹ️  Using MegaLoc cluster features (256-dim) for patch matching")
    print("   These are spatial features transformed by MegaLoc's learned network")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    good_loops_dir = output_dir / "good_loops"
    bad_loops_dir = output_dir / "bad_loops"
    good_loops_dir.mkdir(parents=True, exist_ok=True)
    bad_loops_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directories:")
    print(f"   Good loops: {good_loops_dir.absolute()}")
    print(f"   Bad loops:  {bad_loops_dir.absolute()}")

    # Load images
    images_path = Path(args.images_path)
    image_files = sorted(images_path.glob("img*.jpg"))

    if len(image_files) == 0:
        print(f"❌ No images found in {images_path}")
        return

    if args.max_frames:
        image_files = image_files[:args.max_frames]

    print(f"\n📂 Found {len(image_files)} images")

    # Load model
    print(f"\n🤖 Loading MegaLoc model...")
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model = model.to(device)
    model.eval()
    print(f"   ✓ Model loaded")

    print("\n🎬 STARTING PROCESSING")
    print("="*80)

    # Preprocessing
    preprocess = get_preprocess_transform()
    all_global_features = []
    all_spatial_features = []
    all_images = []

    total_loops = 0
    consistent_loops = 0

    # Main loop
    for frame_idx, img_path in enumerate(image_files):
        # Load image
        img_pil, img_tensor = load_and_preprocess_image(img_path, preprocess)
        if img_tensor is None:
            continue

        # Extract MegaLoc features (global + spatial)
        global_feat, spatial_feat = extract_megaloc_features(model, img_tensor, device)
        all_global_features.append(global_feat)
        all_spatial_features.append(spatial_feat)
        all_images.append(img_pil)

        # Loop closure detection
        if frame_idx >= args.start_detection_frame:
            database_features = torch.cat(all_global_features[:-1], dim=0)

            top_indices, top_similarities = find_loop_closure(
                global_feat,
                database_features,
                frame_idx,
                temporal_distance=args.temporal_distance,
                top_k=args.top_k
            )

            if len(top_indices) > 0:
                best_similarity = top_similarities[0].item()
                match_indices_list = [idx.item() for idx in top_indices]

                # Check temporal consistency
                is_consistent, max_spread = check_temporal_consistency(
                    match_indices_list,
                    args.temporal_consistency_window
                )

                if best_similarity >= args.similarity_threshold:
                    total_loops += 1
                    match_idx = match_indices_list[0]

                    if is_consistent:
                        consistent_loops += 1

                    print(f"\n{'='*80}")
                    status = "✓ CONSISTENT" if is_consistent else "⚠️  INCONSISTENT"
                    print(f"Loop {total_loops}: Frame {frame_idx:05d} ↔ {match_idx:05d} | "
                          f"Sim: {best_similarity:.4f} {status}")

                    # Find patch matches using MegaLoc spatial features
                    print(f"   Finding {args.num_patch_matches} best MegaLoc patch matches...")
                    matches, grid_size = find_patch_matches_megaloc(
                        all_spatial_features[frame_idx],
                        all_spatial_features[match_idx],
                        top_k=args.num_patch_matches
                    )

                    # Visualize
                    target_dir = good_loops_dir if is_consistent else bad_loops_dir
                    output_path = target_dir / f"match_{total_loops:03d}_query{frame_idx:05d}_loop{match_idx:05d}.png"

                    visualize_matches(
                        all_images[frame_idx],
                        all_images[match_idx],
                        matches,
                        grid_size,
                        best_similarity,
                        output_path
                    )

        # Progress
        if (frame_idx + 1) % 50 == 0:
            print(f"\n   Progress: {frame_idx + 1}/{len(image_files)} frames processed")

    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE")
    print("="*80)
    print(f"Frames processed: {len(image_files)}")
    print(f"Loop closures detected: {total_loops}")
    print(f"  - Consistent: {consistent_loops}")
    print(f"  - Inconsistent: {total_loops - consistent_loops}")
    print(f"\n📊 Visualizations saved to: {output_dir.absolute()}")
    print("\nℹ️  These matches use MegaLoc's learned cluster features (256-dim),")
    print("   NOT raw DINOv2 features. They represent MegaLoc's internal representation.")
    print("="*80)


if __name__ == "__main__":
    main()
