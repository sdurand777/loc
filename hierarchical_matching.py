#!/usr/bin/env python3
"""
Hierarchical Loop Closure Matching Pipeline:
MegaLoc (global) → SuperPoint (local keypoints) → LightGlue (fine matching)

This script combines:
1. MegaLoc for robust loop closure detection and coarse patch matching
2. SuperPoint for dense keypoint extraction in relevant regions
3. LightGlue for accurate feature matching with attention

Usage:
    python hierarchical_matching.py --images_path /path/to/images --output_dir hierarchical_matches
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch

import torch
from PIL import Image
import torchvision.transforms as tfm
import cv2

# LightGlue imports
try:
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import load_image, rbd
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    print("WARNING: LightGlue not installed. Install with: pip install lightglue")


def get_preprocess_transform():
    """Transformation pour MegaLoc."""
    return tfm.Compose([
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_and_preprocess_image(image_path, preprocess, target_size=224):
    """
    Charge et prétraite une image.

    Returns:
        img_original: Image PIL originale (taille native)
        img_resized: Image PIL redimensionnée pour affichage
        img_tensor: Tensor pour MegaLoc
    """
    try:
        img_original = Image.open(image_path).convert('RGB')
        img_resized = img_original.resize((target_size, target_size), Image.LANCZOS)
        img_tensor = preprocess(img_resized)
        return img_original, img_resized, img_tensor
    except Exception as e:
        print(f"Erreur lors du chargement de {image_path}: {e}")
        return None, None, None


def extract_megaloc_features(model, img_tensor, device):
    """
    Extrait les features globales MegaLoc ET les features spatiales MegaLoc.

    Returns:
        global_features: Features MegaLoc normalisées [1, 8448]
        spatial_features: Cluster features SALAD [1, 256, H//14, W//14]
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


def find_patch_matches_megaloc(query_features, loop_features, top_k=20, patch_threshold=0.40):
    """
    Trouve les meilleurs matches entre les patches MegaLoc de deux images.

    Args:
        query_features: [1, 256, H, W] - Cluster features MegaLoc
        loop_features: [1, 256, H, W] - Cluster features MegaLoc
        top_k: Nombre de meilleurs matches à considérer
        patch_threshold: Seuil de similarité minimum pour les patches (default: 0.40)

    Returns:
        matches: Liste de tuples (query_pos, loop_pos, similarity) filtrés par seuil
        grid_size: (H, W) dimensions de la grille
    """
    # Flatten spatial dimensions
    C, H, W = query_features.shape[1], query_features.shape[2], query_features.shape[3]
    query_flat = query_features.squeeze(0).reshape(C, -1).T  # [H*W, 256]
    loop_flat = loop_features.squeeze(0).reshape(C, -1).T    # [H*W, 256]

    # Normalize
    query_flat = torch.nn.functional.normalize(query_flat, p=2, dim=1)
    loop_flat = torch.nn.functional.normalize(loop_flat, p=2, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(query_flat, loop_flat.T)  # [H*W, H*W]

    # For each query patch, find best match in loop image
    best_similarities, best_indices = torch.max(similarity_matrix, dim=1)

    # Get top-k matches
    top_k_sims, top_k_query_indices = torch.topk(best_similarities, min(top_k, len(best_similarities)))

    # # modif pour tout prendre pas top 30
    # top_k_sims, top_k_query_indices = torch.topk(best_similarities, len(best_similarities))

    # Filter matches by threshold
    matches = []
    for i in range(len(top_k_query_indices)):
        query_idx = top_k_query_indices[i].item()
        loop_idx = best_indices[query_idx].item()
        sim = top_k_sims[i].item()

        # Only keep patches above threshold
        if sim >= patch_threshold:
            # Convert flat indices to 2D positions
            query_y, query_x = query_idx // W, query_idx % W
            loop_y, loop_x = loop_idx // W, loop_idx % W

            matches.append(((query_x, query_y), (loop_x, loop_y), sim))

    return matches, (H, W)


def create_patch_rois(patch_matches, grid_size, img_size, expansion_factor=1.5):
    """
    Crée des ROI (regions of interest) à partir des matches MegaLoc.

    Args:
        patch_matches: Liste de ((qx, qy), (lx, ly), sim)
        grid_size: (H, W) taille de la grille MegaLoc
        img_size: (img_w, img_h) taille de l'image
        expansion_factor: Facteur d'expansion des ROI (1.0 = taille du patch)

    Returns:
        query_rois: Liste de (x1, y1, x2, y2) pour query
        loop_rois: Liste de (x1, y1, x2, y2) pour loop
    """
    img_w, img_h = img_size
    grid_h, grid_w = grid_size

    patch_w = img_w / grid_w
    patch_h = img_h / grid_h

    query_rois = []
    loop_rois = []

    for (qx, qy), (lx, ly), sim in patch_matches:
        # Calculer le centre et la taille étendue
        expansion = expansion_factor

        # Query ROI
        qcx = qx * patch_w + patch_w / 2
        qcy = qy * patch_h + patch_h / 2
        qw = patch_w * expansion
        qh = patch_h * expansion
        qx1 = max(0, int(qcx - qw / 2))
        qy1 = max(0, int(qcy - qh / 2))
        qx2 = min(img_w, int(qcx + qw / 2))
        qy2 = min(img_h, int(qcy + qh / 2))
        query_rois.append((qx1, qy1, qx2, qy2))

        # Loop ROI
        lcx = lx * patch_w + patch_w / 2
        lcy = ly * patch_h + patch_h / 2
        lw = patch_w * expansion
        lh = patch_h * expansion
        lx1 = max(0, int(lcx - lw / 2))
        ly1 = max(0, int(lcy - lh / 2))
        lx2 = min(img_w, int(lcx + lw / 2))
        ly2 = min(img_h, int(lcy + lh / 2))
        loop_rois.append((lx1, ly1, lx2, ly2))

    return query_rois, loop_rois


def extract_superpoint_in_rois(extractor, img_pil, rois, device):
    """
    Extrait les keypoints SuperPoint dans des ROI spécifiques.

    Args:
        extractor: Modèle SuperPoint
        img_pil: Image PIL
        rois: Liste de (x1, y1, x2, y2)
        device: torch device

    Returns:
        all_keypoints: Liste de keypoints [(x, y), ...]
        all_descriptors: Tensor de descripteurs [N, 256]
        all_scores: Liste de scores [N]
    """
    # Convertir PIL en tensor pour SuperPoint
    img_cv = np.array(img_pil)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Convertir en tensor [1, 1, H, W]
    img_tensor = torch.from_numpy(img_gray).float()[None, None] / 255.0
    img_tensor = img_tensor.to(device)

    # Extraire sur l'image entière
    with torch.no_grad():
        feats = extractor.extract(img_tensor)

    # Filtrer les keypoints dans les ROI
    all_keypoints = []
    all_descriptors = []
    all_scores = []

    kpts = feats['keypoints'][0].cpu().numpy()  # [N, 2]
    desc = feats['descriptors'][0].cpu()  # [256, M] where M <= N
    scores = feats['keypoint_scores'][0].cpu().numpy()  # [N]

    # IMPORTANT: desc peut avoir moins de colonnes que kpts n'a de lignes
    # (SuperPoint limite parfois le nombre de descripteurs)
    num_kpts = len(kpts)
    num_desc = desc.shape[1] if desc.ndim > 1 else 0

    if num_desc < num_kpts:
        print(f"      ⚠️  Warning: {num_kpts} keypoints but only {num_desc} descriptors. Using first {num_desc} keypoints.")
        kpts = kpts[:num_desc]
        scores = scores[:num_desc]

    # Pour chaque keypoint, vérifier s'il est dans une ROI
    for i, (x, y) in enumerate(kpts):
        if i >= num_desc:  # Sécurité supplémentaire
            break
        for x1, y1, x2, y2 in rois:
            if x1 <= x <= x2 and y1 <= y <= y2:
                all_keypoints.append((x, y))
                all_descriptors.append(desc[:, i])
                all_scores.append(scores[i])
                break  # Un keypoint peut être dans plusieurs ROI, on ne le garde qu'une fois

    if len(all_keypoints) > 0:
        all_descriptors = torch.stack(all_descriptors, dim=1)  # [256, N]
    else:
        all_descriptors = torch.empty((256, 0))

    return all_keypoints, all_descriptors, all_scores


def match_with_lightglue(extractor, matcher, img0_pil, img1_pil, query_rois, loop_rois, device):
    """
    Matche les keypoints avec SuperPoint + LightGlue.

    Stratégie simplifiée:
    1. Extraire TOUS les keypoints de chaque image avec SuperPoint
    2. Matcher avec LightGlue
    3. Filtrer les matches pour ne garder que ceux dans les ROI

    Args:
        extractor: Modèle SuperPoint
        matcher: Modèle LightGlue
        img0_pil, img1_pil: Images PIL (query et loop)
        query_rois: Liste de (x1, y1, x2, y2) pour query
        loop_rois: Liste de (x1, y1, x2, y2) pour loop
        device: torch device

    Returns:
        matches: Liste de (idx0, idx1, confidence)
        kpts0_list: Liste de keypoints query pour visualisation
        kpts1_list: Liste de keypoints loop pour visualisation
    """
    # Convertir images PIL en grayscale tensors
    img0_gray = cv2.cvtColor(np.array(img0_pil), cv2.COLOR_RGB2GRAY)
    img1_gray = cv2.cvtColor(np.array(img1_pil), cv2.COLOR_RGB2GRAY)

    img0_tensor = torch.from_numpy(img0_gray).float()[None, None] / 255.0  # [1, 1, H, W]
    img1_tensor = torch.from_numpy(img1_gray).float()[None, None] / 255.0  # [1, 1, H, W]

    img0_tensor = img0_tensor.to(device)
    img1_tensor = img1_tensor.to(device)

    # Extraire les features avec SuperPoint sur les images complètes
    with torch.no_grad():
        feats0 = extractor.extract(img0_tensor)
        feats1 = extractor.extract(img1_tensor)

    # Matcher avec LightGlue
    with torch.no_grad():
        matches_dict = matcher({'image0': feats0, 'image1': feats1})

    # Récupérer les keypoints et matches
    kpts0 = feats0['keypoints'][0].cpu().numpy()  # [N, 2]
    kpts1 = feats1['keypoints'][0].cpu().numpy()  # [M, 2]

    matches_indices = matches_dict['matches0'][0].cpu().numpy()  # [N]
    match_confidence = matches_dict['matching_scores0'][0].cpu().numpy()  # [N]

    # Filtrer les matches pour ne garder que ceux dans les ROI
    filtered_matches = []
    kpts0_in_roi = []
    kpts1_in_roi = []

    for i, j in enumerate(matches_indices):
        if j >= 0:  # Match trouvé
            kpt0 = kpts0[i]
            kpt1 = kpts1[j]

            # Vérifier si les deux keypoints sont dans les ROI
            kpt0_in_roi = False
            kpt1_in_roi = False

            for x1, y1, x2, y2 in query_rois:
                if x1 <= kpt0[0] <= x2 and y1 <= kpt0[1] <= y2:
                    kpt0_in_roi = True
                    break

            for x1, y1, x2, y2 in loop_rois:
                if x1 <= kpt1[0] <= x2 and y1 <= kpt1[1] <= y2:
                    kpt1_in_roi = True
                    break

            # Garder le match seulement si les deux keypoints sont dans les ROI
            if kpt0_in_roi and kpt1_in_roi:
                filtered_matches.append((len(kpts0_in_roi), len(kpts1_in_roi), float(match_confidence[i])))
                kpts0_in_roi.append(tuple(kpt0))
                kpts1_in_roi.append(tuple(kpt1))

    return filtered_matches, kpts0_in_roi, kpts1_in_roi


def visualize_hierarchical_matches(query_img, loop_img,
                                   patch_matches, grid_size,
                                   query_kpts, loop_kpts,
                                   fine_matches,
                                   similarity_score, output_path):
    """
    Visualise les matches hiérarchiques : patches MegaLoc + keypoints SuperPoint + matches LightGlue.

    Args:
        query_img: Image PIL de la query
        loop_img: Image PIL du loop
        patch_matches: Liste de ((qx, qy), (lx, ly), sim) - patches MegaLoc
        grid_size: (H, W) taille de la grille MegaLoc
        query_kpts: Liste de (x, y) keypoints query
        loop_kpts: Liste de (x, y) keypoints loop
        fine_matches: Liste de (idx0, idx1, confidence) - matches LightGlue
        similarity_score: Score de similarité global MegaLoc
        output_path: Chemin de sauvegarde
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Hierarchical Matching: MegaLoc → SuperPoint → LightGlue | Similarity: {similarity_score:.4f}',
                 fontsize=16, fontweight='bold')

    # Display images
    ax1.imshow(query_img)
    ax1.set_title('Query Image', fontsize=14)
    ax1.axis('off')

    ax2.imshow(loop_img)
    ax2.set_title('Loop Image', fontsize=14)
    ax2.axis('off')

    # Get image dimensions
    img_w, img_h = query_img.size
    grid_h, grid_w = grid_size

    # Set exact axis limits
    ax1.set_xlim(0, img_w)
    ax1.set_ylim(img_h, 0)
    ax2.set_xlim(0, img_w)
    ax2.set_ylim(img_h, 0)

    # Force aspect ratio
    ax1.set_aspect('equal', adjustable='datalim')
    ax2.set_aspect('equal', adjustable='datalim')

    plt.tight_layout()
    fig.canvas.draw()

    # Patch size in pixels
    patch_h = img_h / grid_h
    patch_w = img_w / grid_w

    # 1. Draw MegaLoc patches (semi-transparent rectangles)
    for (qx, qy), (lx, ly), sim in patch_matches:
        # Color based on similarity
        color = plt.cm.RdYlGn((sim - 0.5) / 0.5)  # Normalize to [0.5, 1.0] range

        # Draw rectangles
        query_rect = mpatches.Rectangle((qx * patch_w, qy * patch_h), patch_w, patch_h,
                                        linewidth=2, edgecolor=color, facecolor=color, alpha=0.15)
        loop_rect = mpatches.Rectangle((lx * patch_w, ly * patch_h), patch_w, patch_h,
                                       linewidth=2, edgecolor=color, facecolor=color, alpha=0.15)
        ax1.add_patch(query_rect)
        ax2.add_patch(loop_rect)

    # 2. Draw SuperPoint keypoints
    if len(query_kpts) > 0:
        qkpts = np.array(query_kpts)
        ax1.scatter(qkpts[:, 0], qkpts[:, 1], c='cyan', s=20, marker='o',
                   edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)

    if len(loop_kpts) > 0:
        lkpts = np.array(loop_kpts)
        ax2.scatter(lkpts[:, 0], lkpts[:, 1], c='cyan', s=20, marker='o',
                   edgecolors='black', linewidths=0.5, alpha=0.8, zorder=3)

    # 3. Draw LightGlue matches
    for idx0, idx1, conf in fine_matches:
        if idx0 < len(query_kpts) and idx1 < len(loop_kpts):
            qx, qy = query_kpts[idx0]
            lx, ly = loop_kpts[idx1]

            # Color based on confidence
            color = plt.cm.viridis(conf)

            # Highlight matched keypoints
            ax1.scatter([qx], [qy], c='lime', s=40, marker='*',
                       edgecolors='white', linewidths=1, zorder=4)
            ax2.scatter([lx], [ly], c='lime', s=40, marker='*',
                       edgecolors='white', linewidths=1, zorder=4)

            # Draw connection
            con = ConnectionPatch(
                xyA=(lx, ly), xyB=(qx, qy),
                coordsA="data", coordsB="data",
                axesA=ax2, axesB=ax1,
                color=color, alpha=0.6, linewidth=1.5, zorder=2
            )
            ax2.add_artist(con)

    # Add info text
    n_patches = len(patch_matches)
    n_query_kpts = len(query_kpts)
    n_loop_kpts = len(loop_kpts)
    n_matches = len(fine_matches)

    info_text = (f'MegaLoc: {n_patches} patch pairs | '
                f'SuperPoint: {n_query_kpts} + {n_loop_kpts} keypoints | '
                f'LightGlue: {n_matches} matches')
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11,
            style='italic', color='darkblue', fontweight='bold')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.15, edgecolor='green',
                      label='MegaLoc Patches (high similarity)'),
        mpatches.Patch(facecolor='yellow', alpha=0.15, edgecolor='yellow',
                      label='MegaLoc Patches (medium similarity)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
                  markeredgecolor='black', markersize=8, label='SuperPoint Keypoints'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='lime',
                  markeredgecolor='white', markersize=12, label='Matched Keypoints'),
        plt.Line2D([0], [0], color='purple', linewidth=2, label='LightGlue Matches')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Matching: MegaLoc → SuperPoint → LightGlue")
    parser.add_argument("--images_path", type=str, required=True,
                       help="Path to images directory")
    parser.add_argument("--output_dir", type=str, default="hierarchical_matches",
                       help="Output directory for visualizations")
    parser.add_argument("--start_detection_frame", type=int, default=50,
                       help="Start detection at frame (default: 50)")
    parser.add_argument("--temporal_distance", type=int, default=50,
                       help="Minimum temporal distance (default: 50)")
    parser.add_argument("--similarity_threshold", type=float, default=0.55,
                       help="MegaLoc global similarity threshold for loop closure detection (default: 0.55)")
    parser.add_argument("--patch_similarity_threshold", type=float, default=0.40,
                       help="MegaLoc patch similarity threshold (default: 0.40)")
    parser.add_argument("--num_megaloc_patches", type=int, default=30,
                       help="Number of top MegaLoc patches to consider (default: 30)")
    parser.add_argument("--min_patches", type=int, default=20,
                       help="Minimum number of patches required to proceed to SuperPoint+LightGlue (default: 20)")
    parser.add_argument("--roi_expansion", type=float, default=1.5,
                       help="ROI expansion factor (default: 1.5)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use (default: auto)")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process (default: all)")

    args = parser.parse_args()

    # Check LightGlue availability
    if not LIGHTGLUE_AVAILABLE:
        print("\n❌ ERROR: LightGlue not installed!")
        print("Install with: pip install lightglue opencv-python kornia")
        print("Or: pip install git+https://github.com/cvg/LightGlue.git")
        return

    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("="*80)
    print("🎯 HIERARCHICAL MATCHING PIPELINE")
    print("="*80)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\n📊 Pipeline:")
    print("  1️⃣  MegaLoc → Global loop closure detection + coarse patch matching")
    print("  2️⃣  SuperPoint → Dense keypoint extraction in relevant regions")
    print("  3️⃣  LightGlue → Fine-grained matching with attention")
    print("="*80)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir.absolute()}")

    # Load images
    images_path = Path(args.images_path)
    image_files = sorted(images_path.glob("img*.jpg"))

    if len(image_files) == 0:
        print(f"❌ No images found in {images_path}")
        return

    if args.max_frames:
        image_files = image_files[:args.max_frames]

    print(f"📂 Found {len(image_files)} images")

    # Load models
    print(f"\n🤖 Loading models...")

    # 1. MegaLoc
    print(f"  Loading MegaLoc...")
    megaloc_model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    megaloc_model = megaloc_model.to(device)
    megaloc_model.eval()
    print(f"    ✓ MegaLoc loaded")

    # 2. SuperPoint
    print(f"  Loading SuperPoint...")
    # Increase max_num_keypoints to avoid descriptor truncation
    superpoint = SuperPoint(max_num_keypoints=4096).eval().to(device)
    print(f"    ✓ SuperPoint loaded (max 4096 keypoints)")

    # 3. LightGlue
    print(f"  Loading LightGlue...")
    lightglue = LightGlue(features='superpoint').eval().to(device)
    print(f"    ✓ LightGlue loaded")

    print("\n🎬 STARTING PROCESSING")
    print("="*80)

    # Preprocessing
    preprocess = get_preprocess_transform()
    all_global_features = []
    all_spatial_features = []
    all_images_original = []
    all_images_resized = []

    total_loops = 0

    # Main loop
    for frame_idx, img_path in enumerate(image_files):
        # Load image
        img_orig, img_resized, img_tensor = load_and_preprocess_image(img_path, preprocess)
        if img_tensor is None:
            continue

        # Extract MegaLoc features
        global_feat, spatial_feat = extract_megaloc_features(megaloc_model, img_tensor, device)
        all_global_features.append(global_feat)
        all_spatial_features.append(spatial_feat)
        all_images_original.append(img_orig)
        all_images_resized.append(img_resized)

        # Loop closure detection
        if frame_idx >= args.start_detection_frame:
            database_features = torch.cat(all_global_features[:-1], dim=0)

            top_indices, top_similarities = find_loop_closure(
                global_feat,
                database_features,
                frame_idx,
                temporal_distance=args.temporal_distance,
                top_k=1  # Only process best match
            )

            if len(top_indices) > 0:
                best_similarity = top_similarities[0].item()

                if best_similarity >= args.similarity_threshold:
                    total_loops += 1
                    match_idx = top_indices[0].item()

                    print(f"\n{'='*80}")
                    print(f"🔄 Loop Candidate #{total_loops}: Frame {frame_idx:05d} ↔ {match_idx:05d}")
                    print(f"{'='*80}")

                    # LEVEL 1: Loop closure detection with MegaLoc
                    print(f"  ✅ STEP 1: MegaLoc Loop Closure Detection")
                    print(f"      Global similarity: {best_similarity:.4f} (threshold: {args.similarity_threshold})")
                    print(f"      → Loop closure DETECTED")

                    # LEVEL 2: Find high-quality MegaLoc patch matches
                    print(f"\n  🔍 STEP 2: MegaLoc Patch Matching")
                    print(f"      Looking for patches with similarity > {args.patch_similarity_threshold}...")
                    patch_matches, grid_size = find_patch_matches_megaloc(
                        all_spatial_features[frame_idx],
                        all_spatial_features[match_idx],
                        top_k=args.num_megaloc_patches,
                        patch_threshold=args.patch_similarity_threshold
                    )

                    if len(patch_matches) > 0:
                        avg_patch_sim = sum([m[2] for m in patch_matches]) / len(patch_matches)
                        print(f"      ✓ Found {len(patch_matches)} patches above threshold")
                        print(f"      Average patch similarity: {avg_patch_sim:.4f}")
                    else:
                        print(f"      ✗ No patches found above threshold {args.patch_similarity_threshold}")

                    # Check minimum patches requirement
                    if len(patch_matches) < args.min_patches:
                        print(f"\n  ⚠️  INSUFFICIENT PATCHES: {len(patch_matches)} < {args.min_patches} (minimum required)")
                        print(f"      → Skipping SuperPoint + LightGlue refinement")
                        print(f"      → Loop candidate REJECTED\n")
                        total_loops -= 1  # Don't count this loop
                        continue

                    # LEVEL 3: SuperPoint + LightGlue fine matching
                    print(f"\n  ✓ SUFFICIENT PATCHES: {len(patch_matches)} >= {args.min_patches}")
                    print(f"  🎯 STEP 3: SuperPoint + LightGlue Fine Matching")

                    img_size = all_images_resized[frame_idx].size  # (width, height)
                    query_rois, loop_rois = create_patch_rois(
                        patch_matches, grid_size, img_size,
                        expansion_factor=args.roi_expansion
                    )

                    print(f"      Creating {len(query_rois)} ROI pairs from patches...")
                    fine_matches, query_kpts, loop_kpts = match_with_lightglue(
                        superpoint,  # extractor
                        lightglue,   # matcher
                        all_images_resized[frame_idx],  # query image PIL
                        all_images_resized[match_idx],  # loop image PIL
                        query_rois,
                        loop_rois,
                        device
                    )
                    print(f"      ✓ SuperPoint: {len(query_kpts)} (query) + {len(loop_kpts)} (loop) keypoints")
                    print(f"      ✓ LightGlue: {len(fine_matches)} fine matches in ROI")

                    # Visualize
                    print(f"\n  📊 Creating visualization...")
                    output_path = output_dir / f"hierarchical_{total_loops:03d}_query{frame_idx:05d}_loop{match_idx:05d}.png"

                    visualize_hierarchical_matches(
                        all_images_resized[frame_idx],
                        all_images_resized[match_idx],
                        patch_matches,
                        grid_size,
                        query_kpts,
                        loop_kpts,
                        fine_matches,
                        best_similarity,
                        output_path
                    )
                    print(f"  ✅ Loop #{total_loops} VALIDATED and saved\n")

        # Progress
        if (frame_idx + 1) % 50 == 0:
            print(f"\n   Progress: {frame_idx + 1}/{len(image_files)} frames processed")

    print("\n" + "="*80)
    print("✅ PROCESSING COMPLETE")
    print("="*80)
    print(f"Frames processed: {len(image_files)}")
    print(f"Loop closures detected: {total_loops}")
    print(f"\n📊 Visualizations saved to: {output_dir.absolute()}")
    print("\n💡 The visualizations show:")
    print("   • Green/yellow rectangles: MegaLoc patch correspondences")
    print("   • Cyan circles: SuperPoint keypoints in relevant regions")
    print("   • Green stars: Matched keypoints by LightGlue")
    print("   • Purple lines: Fine matches with confidence-based coloring")
    print("="*80)


if __name__ == "__main__":
    main()
