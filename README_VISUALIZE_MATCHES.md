# Loop Closure Visualization with Visual Matches

Ce script détecte les loop closures avec MegaLoc et génère des **visualisations composites** montrant les **correspondances visuelles** (matches) entre les paires d'images query-loop.

## 🔥 NOUVEAU : Version MegaLoc (RECOMMANDÉE)

**Deux versions disponibles** :

### 1. `visualize_matches_megaloc.sh` ⭐ (RECOMMANDÉ)
- Utilise les **features MegaLoc spatiales** (cluster features de SALAD, 256-dim)
- Features transformées par le réseau MegaLoc appris
- Correspondances basées sur la représentation interne de MegaLoc
- **Plus cohérent** avec la détection globale

### 2. `visualize_matches.sh` (Original)
- Utilise les **features DINOv2 brutes** (768-dim)
- Features avant transformation MegaLoc
- Pour comparer avec la baseline DINOv2

## 🎯 Fonctionnalités

- ✅ Détection de loop closures avec **MegaLoc** (même processus que `slam_rerun.sh`)
- ✅ Extraction des **features spatiales MegaLoc** pour trouver les correspondances locales
- ✅ Visualisation des **20 meilleurs matches** entre les paires
- ✅ Images composites côte-à-côte avec **lignes colorées** reliant les patches correspondants
- ✅ **Code couleur** : vert = haute similarité, rouge = basse similarité
- ✅ Séparation automatique : bons loops (cohérents) vs mauvais loops (incohérents)

## 🚀 Utilisation

### Version MegaLoc (RECOMMANDÉE) ⭐

```bash
bash visualize_matches_megaloc.sh /path/to/images [output_dir]
```

**Exemple** :
```bash
# Utilisation basique
bash visualize_matches_megaloc.sh /home/ivm/pose_graph/pgSlam/scenario/imgs/

# Avec dossier de sortie personnalisé
bash visualize_matches_megaloc.sh /path/to/images my_results_megaloc
```

### Version DINOv2 (baseline)

```bash
bash visualize_matches.sh /path/to/images [output_dir]
```

### Script Python direct

**Version MegaLoc** :
```bash
python visualize_loop_matches_megaloc.py \
    --images_path /path/to/images \
    --output_dir loop_matches_megaloc \
    --similarity_threshold 0.55 \
    --num_patch_matches 20
```

**Version DINOv2** :
```bash
python visualize_loop_matches.py \
    --images_path /path/to/images \
    --output_dir loop_matches \
    --similarity_threshold 0.55 \
    --num_patch_matches 20
```

## 📊 Paramètres disponibles

| Paramètre | Description | Défaut |
|-----------|-------------|--------|
| `--images_path` | Dossier contenant les images (img*.jpg) | **Requis** |
| `--output_dir` | Dossier de sortie pour les visualisations | `loop_matches` |
| `--start_detection_frame` | Frame de départ pour la détection | `50` |
| `--temporal_distance` | Distance temporelle minimale (frames) | `50` |
| `--similarity_threshold` | Seuil de similarité MegaLoc (0-1) | `0.55` |
| `--temporal_consistency_window` | Fenêtre de cohérence temporelle | `2` |
| `--num_patch_matches` | Nombre de matches visuels à afficher | `20` |
| `--device` | Device (auto/cuda/cpu) | `auto` |
| `--max_frames` | Nombre max de frames à traiter | Toutes |

## 📁 Structure de sortie

```
loop_matches/
├── good_loops/     # Loop closures temporellement cohérentes
│   ├── match_001_query00123_loop00045.png
│   ├── match_002_query00234_loop00078.png
│   └── ...
└── bad_loops/      # Loop closures incohérentes
    ├── match_003_query00156_loop00023.png
    └── ...
```

Chaque image générée contient :
- **Gauche** : Image query (frame actuelle)
- **Droite** : Image loop (frame correspondante du passé)
- **Cercles colorés** : Patches correspondants
- **Lignes colorées** : Connexions entre patches (vert = haute similarité, rouge = basse)
- **Barre de couleur** : Échelle de similarité des patches

## 🎨 Interprétation des visualisations

### Code couleur des matches

- 🟢 **Vert** : Patch matches avec haute similarité (> 0.8)
- 🟡 **Jaune** : Similarité moyenne (0.6 - 0.8)
- 🔴 **Rouge** : Faible similarité (< 0.6)

### Qualité du loop closure

- **Good loops** : Les matches sont cohérents (regroupés dans le temps)
- **Bad loops** : Les matches sont dispersés temporellement

## 🔬 Comment ça fonctionne ?

### Pipeline commun

1. **Détection globale** : MegaLoc calcule un descripteur global (8448-dim) pour chaque image
2. **Recherche de similarité** : Comparaison cosinus entre query et base de données
3. **Filtrage temporel** : Exclusion des frames trop proches (< temporal_distance)

### Matching local (différence entre versions)

**Version MegaLoc** ⭐ :
4. **Extraction MegaLoc** : SALAD cluster_features transforme DINOv2 → MegaLoc spatial (256-dim par patch)
5. **Matching MegaLoc** : Similarité cosinus entre cluster features MegaLoc
6. **Visualisation** : Affichage des N meilleurs matches MegaLoc avec code couleur

**Version DINOv2** :
4. **Extraction DINOv2** : Extrait directement les patch features DINOv2 (768-dim par patch)
5. **Matching DINOv2** : Similarité cosinus entre features DINOv2 brutes
6. **Visualisation** : Affichage des N meilleurs matches DINOv2 avec code couleur

## 📐 Architecture technique

### Version MegaLoc (Recommandée)

```
Input Image (224x224)
    ↓
DINOv2 Backbone
    ↓ [1, 768, 16, 16]
SALAD cluster_features (Conv2d + MLP)
    ↓ [1, 256, 16, 16] ← UTILISÉ POUR LES MATCHES
├─→ MegaLoc Spatial Features → Matching local (cosine similarity)
│
└─→ SALAD Aggregation (Sinkhorn + pooling) → Global Features [1, 8448] → Loop detection
```

- **Patches** : 16×16 grille de patches (14×14 pixels chacun)
- **Features** : **256 dimensions par patch (MegaLoc cluster features)**
- **Transformation** : Features transformées par un MLP appris avec MegaLoc
- **Matches** : Similarité cosinus entre features MegaLoc

### Version DINOv2 (Baseline)

```
Input Image (224x224)
    ↓
DINOv2 Backbone
    ↓ [1, 768, 16, 16] ← UTILISÉ POUR LES MATCHES
├─→ DINOv2 Patch Features → Matching local (cosine similarity)
│
└─→ SALAD Aggregation → Global Features [1, 8448] → Loop detection
```

- **Features** : 768 dimensions par patch (DINOv2 brut)
- **Transformation** : Aucune (features DINOv2 pures)
- **Matches** : Similarité cosinus entre features DINOv2

## 🔬 Différence entre les versions

| Aspect | Version MegaLoc ⭐ | Version DINOv2 |
|--------|-------------------|----------------|
| **Features utilisées** | Cluster features SALAD (256-dim) | Patch features DINOv2 (768-dim) |
| **Transformation** | MLP appris avec MegaLoc | Aucune |
| **Cohérence** | Aligné avec détection globale | Baseline pré-entraînée |
| **Dimensionnalité** | 256-dim (compacte) | 768-dim (dense) |
| **Recommandé pour** | Production, cohérence maximale | Comparaison, analyse |

**Pourquoi utiliser la version MegaLoc ?**
- Les cluster features ont été **apprises** avec le modèle MegaLoc
- Elles représentent la **transformation interne** de MegaLoc
- Plus **cohérent** : utilise la même représentation pour détection globale et matches locaux
- Plus **compact** : 256-dim vs 768-dim

## 💡 Exemples d'utilisation

### Traitement rapide (100 premières frames)

```bash
python visualize_loop_matches_megaloc.py \
    --images_path /path/to/images \
    --max_frames 100 \
    --output_dir quick_test
```

### Haute précision (seuil élevé)

```bash
python visualize_loop_matches_megaloc.py \
    --images_path /path/to/images \
    --similarity_threshold 0.70 \
    --num_patch_matches 30
```

### Debugging (afficher plus de matches)

```bash
python visualize_loop_matches_megaloc.py \
    --images_path /path/to/images \
    --num_patch_matches 50 \
    --temporal_distance 30
```

### Comparaison MegaLoc vs DINOv2

```bash
# Version MegaLoc
bash visualize_matches_megaloc.sh /path/to/images output_megaloc

# Version DINOv2
bash visualize_matches.sh /path/to/images output_dinov2

# Comparer les résultats visuellement
```

## 🔍 Différences avec les autres scripts

| Script | Sortie | Visualisation | Features | Validation |
|--------|--------|---------------|----------|------------|
| `slam_rerun.sh` | Fichier .rrd 3D | Timeline interactive | MegaLoc global | MegaLoc seul |
| `slam_megaloc_mapanything.sh` | Fichier .rrd + images | 3D + géométrie | MegaLoc global | MegaLoc + MapAnything |
| **`visualize_matches_megaloc.sh`** ⭐ | **Images composites** | **Matches MegaLoc** | **MegaLoc spatial (256-dim)** | **MegaLoc seul** |
| `visualize_matches.sh` | Images composites | Matches DINOv2 | DINOv2 brut (768-dim) | MegaLoc seul |

## ⚙️ Prérequis

Les mêmes que pour MegaLoc :
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `Pillow >= 9.0.0`
- `matplotlib >= 3.5.0`
- `numpy >= 1.23.0`

Installer avec :
```bash
bash install_megaloc.sh
```

## 🐛 Troubleshooting

### Erreur : "No images found"
- Vérifiez que les images sont nommées `img*.jpg` (img00000.jpg, img00001.jpg, ...)

### Erreur : "CUDA out of memory"
- Utilisez `--device cpu` ou traitez moins de frames à la fois avec `--max_frames`

### Pas de loop détecté
- Diminuez `--similarity_threshold` (ex: 0.45)
- Diminuez `--temporal_distance` (ex: 30)
- Vérifiez que vous avez au moins `start_detection_frame + temporal_distance` images

## 📝 Notes

- Les visualisations sont sauvegardées en PNG haute résolution (150 DPI)
- Le traitement prend ~0.5-1 seconde par frame sur GPU
- Les patches font 14×14 pixels dans l'image originale (224×224)
- La grille de patches est de 16×16 (224 ÷ 14 = 16)

## 🎓 Citation

Si vous utilisez ce script, citez MegaLoc :

```bibtex
@InProceedings{Berton_2025_CVPR,
    author    = {Berton, Gabriele and Masone, Carlo},
    title     = {MegaLoc: One Retrieval to Place Them All},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {2861-2867}
}
```
