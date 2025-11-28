#!/usr/bin/env python3
"""
Script de debug pour visualiser les transformations de coordonnées
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# Créer une image de test
img = Image.new('RGB', (224, 224), color='lightgray')

# Paramètres
grid_h, grid_w = 16, 16
patch_h = 224 / 16
patch_w = 224 / 16

print(f"Image size: 224x224")
print(f"Grid: {grid_h}x{grid_w}")
print(f"Patch size: {patch_w}x{patch_h}")

# Créer la figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Afficher les images
ax1.imshow(img)
ax1.set_title('Query', fontsize=14)
ax1.axis('off')

ax2.imshow(img)
ax2.set_title('Loop', fontsize=14)
ax2.axis('off')

# Set limits
ax1.set_xlim(0, 224)
ax1.set_ylim(224, 0)
ax2.set_xlim(0, 224)
ax2.set_ylim(224, 0)

# Aspect ratio
ax1.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')

# CRITICAL: tight_layout BEFORE drawing
plt.tight_layout()

# Dessiner une grille pour visualiser les patches
for i in range(grid_h + 1):
    y = i * patch_h
    ax1.axhline(y=y, color='blue', linewidth=0.5, alpha=0.5)
    ax2.axhline(y=y, color='blue', linewidth=0.5, alpha=0.5)

for j in range(grid_w + 1):
    x = j * patch_w
    ax1.axvline(x=x, color='blue', linewidth=0.5, alpha=0.5)
    ax2.axvline(x=x, color='blue', linewidth=0.5, alpha=0.5)

# Test: dessiner quelques matches
test_matches = [
    (5, 5, 10, 10),  # query patch (5,5) -> loop patch (10,10)
    (8, 3, 8, 12),   # query patch (8,3) -> loop patch (8,12)
    (12, 14, 3, 2),  # query patch (12,14) -> loop patch (3,2)
]

for qx, qy, lx, ly in test_matches:
    # Calculate centers
    query_center_x = qx * patch_w + patch_w / 2.0
    query_center_y = qy * patch_h + patch_h / 2.0
    loop_center_x = lx * patch_w + patch_w / 2.0
    loop_center_y = ly * patch_h + patch_h / 2.0

    print(f"\nMatch: Query({qx},{qy}) -> Loop({lx},{ly})")
    print(f"  Query center in pixels: ({query_center_x:.1f}, {query_center_y:.1f})")
    print(f"  Loop center in pixels: ({loop_center_x:.1f}, {loop_center_y:.1f})")

    # Draw circles at centers
    circle_radius = 4
    query_circle = patches.Circle((query_center_x, query_center_y), radius=circle_radius,
                                 color='red', alpha=0.9, zorder=3, linewidth=2,
                                 edgecolor='white')
    loop_circle = patches.Circle((loop_center_x, loop_center_y), radius=circle_radius,
                                color='green', alpha=0.9, zorder=3, linewidth=2,
                                edgecolor='white')
    ax1.add_patch(query_circle)
    ax2.add_patch(loop_circle)

    # Draw connecting line - METHOD 1: Figure coordinates
    trans_ax1 = ax1.transData.transform((query_center_x, query_center_y))
    trans_ax2 = ax2.transData.transform((loop_center_x, loop_center_y))
    trans_fig = fig.transFigure.inverted()
    coord1 = trans_fig.transform(trans_ax1)
    coord2 = trans_fig.transform(trans_ax2)

    print(f"  Transformed coords: fig({coord1[0]:.3f}, {coord1[1]:.3f}) -> fig({coord2[0]:.3f}, {coord2[1]:.3f})")

    line = plt.Line2D([coord1[0], coord2[0]], [coord1[1], coord2[1]],
                     transform=fig.transFigure, color='purple', alpha=0.7,
                     linewidth=2, zorder=2)
    fig.add_artist(line)

print("\n" + "="*80)
print("Saving with bbox_inches='tight'...")
plt.savefig('/tmp/debug_tight.png', dpi=150, bbox_inches='tight')
print("Saved: /tmp/debug_tight.png")

print("\nSaving WITHOUT bbox_inches='tight'...")
plt.savefig('/tmp/debug_normal.png', dpi=150)
print("Saved: /tmp/debug_normal.png")

plt.close()

print("\n" + "="*80)
print("✅ Debug images created!")
print("Compare /tmp/debug_tight.png vs /tmp/debug_normal.png")
print("="*80)
