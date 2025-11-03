# MapAnything Image Pair Visualization

Scripts simples pour visualiser une paire d'images (loop/query) avec MapAnything et générer la reconstruction 3D.

## 📋 Prérequis

Installation de l'environnement combiné :
```bash
bash install_combined.sh
```

## 🚀 Utilisation

### Option 1 : Test rapide avec une paire validée

```bash
bash test_mapanything_pair.sh
```

Ce script :
- Trouve automatiquement la première paire dans `loop_pairs_validated/double_validated/`
- Lance la reconstruction 3D
- Génère `test_scene.ply`

### Option 2 : Paire personnalisée

```bash
bash visualize_pair.sh <loop.jpg> <query.jpg> [output.ply]
```

Exemple :
```bash
bash visualize_pair.sh \
    loop_pairs_validated/double_validated/loop_5.jpg \
    loop_pairs_validated/double_validated/query_5.jpg \
    my_reconstruction.ply
```

### Option 3 : Appel direct Python

```bash
source megaloc_mapanything_env/bin/activate

python mapanything_visualize_pair.py \
    --loop path/to/loop.jpg \
    --query path/to/query.jpg \
    --output scene.ply
```

## 📊 Sorties du script

### 1. Matrices de transformation

Le script affiche :
- **Loop Camera Pose** : Pose de la caméra loop (cam-to-world)
- **Query Camera Pose** : Pose de la caméra query (cam-to-world)
- **Relative Transformation** : Transformation de loop vers query

### 2. Décomposition de la transformation relative

```
Translation:
  x: 1.86908 m
  y: 0.22209 m
  z: -0.08841 m
  norm: 1.88456 m

Rotation (Euler angles XYZ):
  roll  (X): 5.234°
  pitch (Y): -12.456°
  yaw   (Z): 178.901°
```

### 3. Scores de confiance

Pour chaque image :
- Mean confidence : Score moyen
- Min/Max confidence : Plage des scores

### 4. Reconstruction 3D (fichier .ply)

Point cloud au format PLY contenant :
- Position 3D de chaque point
- Couleur RGB

## 🎨 Visualiser le point cloud

### Avec Open3D (Python)

```bash
python -c "import open3d as o3d; pcd = o3d.io.read_point_cloud('scene.ply'); o3d.visualization.draw_geometries([pcd])"
```

Ou créez un petit script `view_ply.py` :
```python
import open3d as o3d
import sys

pcd = o3d.io.read_point_cloud(sys.argv[1])
o3d.visualization.draw_geometries([pcd])
```

Puis :
```bash
python view_ply.py scene.ply
```

### Avec MeshLab

```bash
meshlab scene.ply
```

### Avec CloudCompare

```bash
cloudcompare scene.ply
```

## 📐 Interprétation de la transformation relative

La **matrice de transformation relative** `T_relative` transforme un point du référentiel de la **loop** vers le référentiel de la **query** :

```
p_query = T_relative @ p_loop
```

Composants :
```
T_relative = [R | t]  (4x4 matrix)
             [0 | 1]

R = rotation matrix (3x3)
t = translation vector (3x1)
```

**Translation** : Distance et direction de la caméra loop vers la caméra query
**Rotation** : Changement d'orientation entre les deux caméras

## 🔍 Exemple de sortie complète

```
================================================================================
🗺️  MapAnything - Image Pair Reconstruction
================================================================================
Loop image:  loop_pairs_validated/double_validated/loop_1.jpg
Query image: loop_pairs_validated/double_validated/query_1.jpg
Device:      cuda
Model:       facebook/map-anything
================================================================================

🤖 Loading MapAnything model...
✅ Model loaded

📷 Loading images...
✅ Loaded 2 images

🔄 Running inference...
✅ Inference complete

📐 Extracting camera poses...

================================================================================
📊 TRANSFORMATION MATRICES
================================================================================

Loop Camera Pose (cam-to-world):
============================================================
  [ 0.98234   0.12345  -0.13456   1.03240]
  [-0.11223   0.98765   0.10987  -0.72021]
  [ 0.14567  -0.09876   0.98432   0.64758]
  [ 0.00000   0.00000   0.00000   1.00000]
============================================================

Relative Transformation (loop → query):
============================================================
  [ 0.99234   0.02345  -0.01234   1.86908]
  [-0.02123   0.99876   0.04321   0.22209]
  [ 0.01456  -0.04234   0.99901  -0.08841]
  [ 0.00000   0.00000   0.00000   1.00000]
============================================================

================================================================================
📏 RELATIVE TRANSFORMATION BREAKDOWN
================================================================================

Translation:
  x: 1.86908 m
  y: 0.22209 m
  z: -0.08841 m
  norm: 1.88456 m

Rotation (Euler angles XYZ):
  roll  (X): 1.234°
  pitch (Y): -2.456°
  yaw   (Z): 0.901°

================================================================================
🎯 CONFIDENCE SCORES
================================================================================

Loop image:
  Mean confidence: 0.7845
  Min confidence:  0.2341
  Max confidence:  0.9876

Query image:
  Mean confidence: 0.8123
  Min confidence:  0.3456
  Max confidence:  0.9654

================================================================================
💾 GENERATING 3D RECONSTRUCTION
================================================================================

💾 Saving point cloud with 245678 points to scene.ply
✅ Point cloud saved successfully!

✅ 3D reconstruction saved to: scene.ply
   You can open it with:
   - MeshLab: meshlab scene.ply
   - Open3D: python -m open3d.visualization.draw scene.ply

================================================================================
✅ PROCESSING COMPLETE
================================================================================
```

## 🛠️ Options avancées

```bash
python mapanything_visualize_pair.py --help
```

Options disponibles :
- `--loop` : Chemin vers l'image loop (requis)
- `--query` : Chemin vers l'image query (requis)
- `--output` : Fichier de sortie PLY (défaut: scene.ply)
- `--model` : Modèle MapAnything à utiliser (défaut: facebook/map-anything)
- `--device` : Device (auto/cuda/cpu, défaut: auto)

### Utiliser le modèle Apache

```bash
python mapanything_visualize_pair.py \
    --loop loop.jpg \
    --query query.jpg \
    --model facebook/map-anything-apache
```

## 📝 Notes

- Le point cloud généré contient les points 3D reconstruits des deux vues
- Les couleurs proviennent des images originales
- La qualité dépend de la texture et de l'overlap entre les images
- Les scores de confiance indiquent la fiabilité de la reconstruction
