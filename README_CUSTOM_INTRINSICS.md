# MapAnything avec Intrinsics Personnalisés

Script pour utiliser MapAnything avec **vos propres paramètres de calibration caméra**.

## 🎯 Cas d'usage

Ce script est utile quand vous avez :
- ✅ Calibration caméra connue (matrice K : fx, fy, cx, cy)
- ✅ Optionnellement : coefficients de distortion (k1, k2, p1, p2, k3)
- ✅ Une paire d'images loop/query

## 📐 Format des Intrinsics

### Matrice K (3×3)
```
K = [fx  0  cx]
    [0  fy  cy]
    [0   0   1]
```

Où :
- `fx`, `fy` : focales en pixels (horizontal et vertical)
- `cx`, `cy` : point principal en pixels (centre optique)

### Coefficients de distortion (optionnel)
```
dist_coeffs = [k1, k2, p1, p2, k3]
```

Où :
- `k1`, `k2`, `k3` : distortion radiale
- `p1`, `p2` : distortion tangentielle

⚠️ **Important** : MapAnything ne gère **pas** la distortion nativement. Utilisez `--undistort` pour corriger les images avant traitement.

## 🚀 Utilisation

### Option 1 : Test rapide

Éditez `test_intrinsics.sh` avec vos paramètres de caméra :
```bash
FX=500.0  # Votre focale X
FY=500.0  # Votre focale Y
CX=320.0  # Votre point principal X
CY=240.0  # Votre point principal Y
```

Puis lancez :
```bash
bash test_intrinsics.sh
```

### Option 2 : Commande directe

```bash
source megaloc_mapanything_env/bin/activate

python mapanything_pair_with_intrinsics.py \
    --loop path/to/loop.jpg \
    --query path/to/query.jpg \
    --fx 500.0 --fy 500.0 \
    --cx 320.0 --cy 240.0 \
    --output my_scene.glb
```

### Option 3 : Avec correction de distortion

Si vos images ont de la distortion (fisheye, grand angle, etc.) :

```bash
python mapanything_pair_with_intrinsics.py \
    --loop loop.jpg \
    --query query.jpg \
    --fx 500.0 --fy 500.0 --cx 320.0 --cy 240.0 \
    --k1 -0.1 --k2 0.05 --p1 0.001 --p2 -0.001 --k3 0.0 \
    --undistort \
    --output scene_undistorted.glb
```

Le script va :
1. Détecter la distortion
2. **Undistort** les images avec OpenCV
3. Ajuster automatiquement les intrinsics
4. Lancer MapAnything sur les images corrigées

## 📊 Sorties

### 1. Matrices de transformation

```
Loop Camera Pose (cam-to-world):
============================================================
  [ 1.00000  -0.00010   0.00008   0.00030]
  [ 0.00010   1.00000   0.00005   0.00014]
  [-0.00008  -0.00005   1.00000   0.00691]
  [ 0.00000   0.00000   0.00000   1.00000]
============================================================

Relative Transformation (loop → query):
============================================================
  [ 0.89082   0.19718  -0.40934   1.97626]
  [-0.28907   0.94103  -0.17579   0.27710]
  [ 0.35054   0.27492   0.89529   0.03100]
  [ 0.00000   0.00000   0.00000   1.00000]
============================================================
```

### 2. Décomposition de la transformation

```
Translation:
  x:   1.97626 m
  y:   0.27710 m
  z:   0.03100 m
  norm:   1.99584 m

Rotation (Euler angles XYZ):
  roll  (X):    17.070°
  pitch (Y):   -20.520°
  yaw   (Z):   -17.978°
```

### 3. Scores de confiance

```
Loop image:
  Mean confidence: 1.2907
  Min confidence:  1.0000
  Max confidence:  3.1154

Query image:
  Mean confidence: 1.1785
  Min confidence:  1.0000
  Max confidence:  2.1936
```

### 4. Modèle 3D (GLB)

Fichier `scene.glb` contenant :
- Mesh 3D reconstruit
- Textures des deux vues
- Peut être ouvert avec Blender, viewers en ligne, etc.

## 🎨 Visualiser le modèle 3D

### En ligne (le plus simple)

1. Ouvrez https://gltf-viewer.donmccurdy.com/
2. Glissez-déposez votre fichier `.glb`

### Avec Blender

```bash
blender custom_intrinsics_scene.glb
```

### Avec Python + Trimesh

```python
import trimesh

scene = trimesh.load('scene.glb')
scene.show()
```

## 📝 Exemple complet

Supposons que vous avez une caméra avec :
- Résolution : 640×480 pixels
- Focale : 500 pixels (fx = fy = 500)
- Point principal : centre de l'image (cx=320, cy=240)
- Distortion radiale : k1 = -0.15

```bash
source megaloc_mapanything_env/bin/activate

# Avec correction de distortion
python mapanything_pair_with_intrinsics.py \
    --loop loop_pairs_validated/double_validated/loop_1.jpg \
    --query loop_pairs_validated/double_validated/query_1.jpg \
    --fx 500.0 --fy 500.0 \
    --cx 320.0 --cy 240.0 \
    --k1 -0.15 --k2 0.0 --p1 0.0 --p2 0.0 --k3 0.0 \
    --undistort \
    --output my_reconstruction.glb
```

Résultat :
1. Images `undistorted_loop_1.jpg` et `undistorted_query_1.jpg` créées
2. Intrinsics ajustés automatiquement
3. Transformation relative calculée
4. Modèle 3D `my_reconstruction.glb` généré

## 🔍 Comment obtenir vos intrinsics ?

### Option 1 : Calibration OpenCV

```python
import cv2
import numpy as np

# Utiliser un pattern de calibration (échiquier)
# ... (procédure de calibration standard OpenCV)

# Résultat :
# K = matrice 3×3
# dist_coeffs = [k1, k2, p1, p2, k3]
```

### Option 2 : Depuis fichier COLMAP

Si vous avez un fichier `cameras.txt` de COLMAP :
```
# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 640 480 500.0 500.0 320.0 240.0
```

Extraire : fx=500, fy=500, cx=320, cy=240

### Option 3 : Approximation simple

Si vous n'avez pas de calibration précise :
```python
# Approximation basique
W, H = image_width, image_height
fx = fy = max(W, H)  # Souvent une bonne première estimation
cx = W / 2
cy = H / 2
```

## ⚠️ Limitations

1. **Distortion** : MapAnything ne gère pas la distortion en interne
   - Solution : Utilisez `--undistort` pour corriger les images avant

2. **Intrinsics variables** : Le script suppose les mêmes intrinsics pour loop et query
   - Si différents : lancez deux fois avec images différentes

3. **Format d'image** : Supporte JPG, PNG
   - RAW/DNG non supporté directement

## 🆘 Dépannage

### Erreur "No depth maps"

```
⚠️  No depth map for view 0
⚠️  Could not extract 3D point cloud (no depth maps)
```

C'est normal ! MapAnything génère quand même :
- ✅ Les poses (transformation relative)
- ✅ Les scores de confiance
- ⚠️ Mais pas toujours les depth maps

### Images undistorted non créées

Vérifiez que :
- OpenCV est installé : `pip install opencv-python`
- Les coefficients de distortion sont non-nuls
- Le flag `--undistort` est présent

### Mauvaise reconstruction

Vérifiez que :
- Les intrinsics correspondent bien à vos images
- Les images ont suffisamment d'overlap
- Les images ne sont pas trop floues
- L'échelle est cohérente (focales en pixels, pas en mm!)

## 📚 Références

- MapAnything : https://github.com/facebookresearch/map-anything
- Calibration OpenCV : https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
- Format GLB : https://www.khronos.org/gltf/
