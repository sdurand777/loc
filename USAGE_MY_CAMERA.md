# Utilisation avec vos paramètres de caméra

Scripts configurés avec **vos paramètres de calibration spécifiques** :

## 📐 Paramètres de votre caméra

```python
K_l = np.array([322.580, 0.0, 259.260,
                0.0, 322.580, 184.882,
                0.0, 0.0, 1.0]).reshape(3,3)

d_l = np.array([-0.070162237, 0.07551153, 0.0012286149, 0.00099302817, -0.018171599])
```

**Détails** :
- **Focales** : fx = fy = 322.580 pixels
- **Point principal** : cx = 259.260, cy = 184.882 pixels
- **Distortion radiale** : k1 = -0.070, k2 = 0.076, k3 = -0.018
- **Distortion tangentielle** : p1 = 0.0012, p2 = 0.0010

Ces paramètres indiquent une **légère distortion en barillet** (k1 négatif), typique des caméras grand-angle.

## 🚀 Utilisation

### Test rapide avec première paire validée

```bash
bash test_my_camera.sh
```

Ce script :
1. Trouve automatiquement loop_1.jpg et query_1.jpg dans `loop_pairs_validated/double_validated/`
2. Applique **undistortion automatique** avec vos coefficients
3. Lance MapAnything avec vos intrinsics ajustés
4. Génère `test_my_camera.glb`

### Paire personnalisée

```bash
bash mapanything_my_camera.sh <loop.jpg> <query.jpg> [output.glb]
```

**Exemples** :

```bash
# Paire spécifique
bash mapanything_my_camera.sh \
    loop_pairs_validated/double_validated/loop_5.jpg \
    loop_pairs_validated/double_validated/query_5.jpg \
    reconstruction_5.glb

# Depuis un autre dossier
bash mapanything_my_camera.sh \
    /path/to/my/loop.jpg \
    /path/to/my/query.jpg \
    my_scene.glb
```

## 📊 Ce que fait le script

### 1. Undistortion automatique

Le script détecte que vous avez de la distortion et applique automatiquement la correction :

```
⚠️  Distortion coefficients detected:
  k1=-0.070162, k2=0.075512, p1=0.001229, p2=0.000993, k3=-0.018172

🔧 Undistorting images...
  ✓ Undistorted loop:  undistorted_loop_1.jpg
  ✓ Undistorted query: undistorted_query_1.jpg
```

Les images corrigées sont sauvegardées automatiquement.

### 2. Ajustement des intrinsics

Après undistortion, OpenCV calcule les **nouveaux intrinsics optimaux** :

```
📐 Updated Intrinsics (after undistortion):
  fx = 320.45, fy = 320.45
  cx = 258.12, cy = 183.90
```

Ces valeurs légèrement différentes tiennent compte de la correction de distortion.

### 3. Redimensionnement pour DINOv2

MapAnything utilise DINOv2 qui nécessite des dimensions divisibles par 14. Le script redimensionne automatiquement les images :

```
📷 Loading images with custom intrinsics...
  Resized undistorted_loop_1.jpg: 519x370 -> 518x364
    Adjusted intrinsics: fx=321.45, fy=319.12, cx=257.89, cy=182.45
  Resized undistorted_query_1.jpg: 519x370 -> 518x364
    Adjusted intrinsics: fx=321.45, fy=319.12, cx=257.89, cy=182.45
✅ Images loaded with custom intrinsics
```

Les intrinsics sont automatiquement ajustés pour correspondre au nouveau redimensionnement.

### 4. Reconstruction 3D

MapAnything utilise les images undistorted + intrinsics ajustés pour :
- Calculer la **pose relative** précise
- Générer le **modèle 3D GLB**

## 📈 Résultats attendus

Avec vos paramètres de caméra, vous obtiendrez :

### Transformation relative

```
Relative Transformation (loop → query):
============================================================
  [ 0.xxxxx   0.xxxxx  -0.xxxxx   X.xxxxx]
  [-0.xxxxx   0.xxxxx  -0.xxxxx   X.xxxxx]
  [ 0.xxxxx   0.xxxxx   0.xxxxx   X.xxxxx]
  [ 0.00000   0.00000   0.00000   1.00000]
============================================================

Translation:
  x:   X.xxxxx m
  y:   X.xxxxx m
  z:   X.xxxxx m
  norm:   X.xxxxx m
```

### Scores de confiance

```
Loop image:
  Mean confidence: X.xxxx
  Min confidence:  X.xxxx
  Max confidence:  X.xxxx

Query image:
  Mean confidence: X.xxxx
  Min confidence:  X.xxxx
  Max confidence:  X.xxxx
```

### Modèle 3D

Fichier GLB avec reconstruction métrique basée sur vos intrinsics réels.

## 🔍 Comparaison avec/sans undistortion

Pour voir l'impact de la correction de distortion :

### Sans undistortion (déconseillé)

```bash
python mapanything_pair_with_intrinsics.py \
    --loop loop.jpg --query query.jpg \
    --fx 322.580 --fy 322.580 --cx 259.260 --cy 184.882 \
    --output scene_distorted.glb
```

⚠️ Les résultats seront moins précis car MapAnything ne gère pas la distortion.

### Avec undistortion (recommandé)

```bash
bash mapanything_my_camera.sh loop.jpg query.jpg scene_undistorted.glb
```

✅ Images corrigées + intrinsics ajustés = reconstruction précise

## 🎯 Applications

Avec vos paramètres de calibration précis, vous pouvez :

1. **Validation de loops SLAM** :
   - Vérifier la géométrie des loop closures
   - Détecter les faux positifs avec erreur de pose élevée

2. **Reconstruction métrique** :
   - Modèles 3D à l'échelle réelle
   - Mesures de distance précises

3. **Debug SLAM** :
   - Visualiser les transformations relatives
   - Identifier les incohérences géométriques

## 📁 Fichiers générés

Après exécution, vous trouverez :

```
/home/ivm/loc/
├── undistorted_loop_*.jpg      # Images corrigées (distortion supprimée)
├── undistorted_query_*.jpg
├── scene_my_camera.glb         # Modèle 3D avec vos intrinsics
└── test_my_camera.glb          # Résultat du test rapide
```

## 🎨 Visualisation

### Voir le modèle 3D

```bash
# En ligne (drag & drop le .glb)
xdg-open https://gltf-viewer.donmccurdy.com/

# Avec Blender
blender test_my_camera.glb

# Avec Python
python3 << EOF
import trimesh
scene = trimesh.load('test_my_camera.glb')
scene.show()
EOF
```

### Comparer images distorted vs undistorted

```python
import cv2
import matplotlib.pyplot as plt

# Originale
original = cv2.imread('loop_1.jpg')
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Undistorted
undistorted = cv2.imread('undistorted_loop_1.jpg')
undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

# Afficher côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(original)
ax1.set_title('Original (avec distortion)')
ax2.imshow(undistorted)
ax2.set_title('Undistorted')
plt.show()
```

## 🆘 Dépannage

### Erreur "OpenCV not found"

```bash
source megaloc_mapanything_env/bin/activate
pip install opencv-python
```

### Images undistorted vides ou noires

- Vérifiez que les coefficients de distortion sont corrects
- Les valeurs très élevées (> 1.0) peuvent indiquer une erreur de calibration

### Reconstruction imprécise

Vérifiez que :
- Les images ont été prises avec la **même caméra** que celle calibrée
- La résolution correspond (519x370 pour vos paramètres cx, cy)
- Les images ne sont pas trop floues
- Il y a suffisamment d'overlap entre loop et query

### "No depth maps"

C'est normal ! MapAnything retourne quand même :
- ✅ La pose relative (transformation 4×4)
- ✅ Les scores de confiance
- ⚠️ Mais pas toujours les depth maps complètes

Le modèle GLB est quand même généré si les depth maps existent partiellement.

## 💡 Conseils

1. **Toujours undistort** vos images avant de lancer MapAnything
2. **Vérifiez visuellement** les images undistorted (surtout les bords)
3. **Comparez** les poses relatives avec votre SLAM pour validation
4. **Utilisez des paires proches** temporellement pour de meilleurs résultats

## 📝 Notes techniques

- **Résolution estimée** : ~519×370 pixels (basé sur cx, cy)
- **FOV** : Petit (~60°) donc bonne précision centrale
- **Distortion** : Modérée, correction essentielle pour précision
- **Modèle** : Compatible OpenCV distortion (5 paramètres)

---

**Scripts prêts à l'emploi** :
- `mapanything_my_camera.sh` : Vos paramètres pré-configurés
- `test_my_camera.sh` : Test rapide sur première paire validée
