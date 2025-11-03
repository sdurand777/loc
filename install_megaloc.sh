#!/bin/bash
# Script d'installation pour MegaLoc SLAM Loop Closure
# Installation system-wide avec pip ou dans un venv

set -e  # Arrêter en cas d'erreur

echo "======================================================================"
echo "   Installation de MegaLoc SLAM Loop Closure Dependencies"
echo "======================================================================"
echo ""

# Vérifier que Python 3 est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé!"
    echo "   Installez Python 3.8+ avant de continuer."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python détecté: $PYTHON_VERSION"
echo ""

# Demander le mode d'installation
echo "Choisissez le mode d'installation:"
echo "  1) System-wide (pip install --user)"
echo "  2) Virtual environment (venv)"
read -p "Votre choix [1/2]: " install_choice

case $install_choice in
    1)
        echo ""
        echo "📦 Installation SYSTEM-WIDE (pip --user)"
        echo "======================================================================"
        PIP_CMD="python3 -m pip install --user"
        ;;
    2)
        echo ""
        echo "📦 Installation dans VIRTUAL ENVIRONMENT"
        echo "======================================================================"

        # Créer le venv s'il n'existe pas
        if [ ! -d "venv" ]; then
            echo "Création du virtual environment..."
            python3 -m venv venv
            echo "✓ Virtual environment créé: ./venv"
        else
            echo "✓ Virtual environment existant détecté: ./venv"
        fi

        # Activer le venv
        source venv/bin/activate
        PIP_CMD="pip install"
        echo "✓ Virtual environment activé"
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac

echo ""
echo "🔧 Mise à jour de pip..."
$PIP_CMD --upgrade pip

echo ""
echo "📚 Installation des dépendances principales..."
echo "======================================================================"

# PyTorch (installation avec CUDA si disponible)
echo ""
echo "1/5 - Installation de PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU NVIDIA détecté, installation avec CUDA support"
    $PIP_CMD torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   Pas de GPU détecté, installation CPU-only"
    $PIP_CMD torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Pillow
echo ""
echo "2/5 - Installation de Pillow..."
$PIP_CMD Pillow

# NumPy
echo ""
echo "3/5 - Installation de NumPy..."
$PIP_CMD numpy

# Rerun SDK
echo ""
echo "4/5 - Installation de Rerun SDK..."
$PIP_CMD rerun-sdk

# Dépendances supplémentaires (optionnelles mais utiles)
echo ""
echo "5/5 - Installation de dépendances supplémentaires..."
$PIP_CMD scipy tqdm

echo ""
echo "======================================================================"
echo "✅ INSTALLATION TERMINÉE!"
echo "======================================================================"

# Instructions finales
if [ "$install_choice" == "2" ]; then
    echo ""
    echo "📝 Pour utiliser MegaLoc, activez d'abord le virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "Puis lancez le script:"
    echo "   bash slam_rerun.sh"
    echo ""
    echo "Pour désactiver le venv plus tard:"
    echo "   deactivate"
else
    echo ""
    echo "📝 Vous pouvez maintenant lancer le script:"
    echo "   bash slam_rerun.sh"
fi

echo ""
echo "======================================================================"

# Test de l'installation
echo ""
echo "🧪 Test de l'installation..."
python3 << 'EOF'
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    import torchvision
    print(f"✓ TorchVision {torchvision.__version__}")

    import PIL
    print(f"✓ Pillow {PIL.__version__}")

    import numpy as np
    print(f"✓ NumPy {np.__version__}")

    import rerun as rr
    print(f"✓ Rerun SDK {rr.__version__}")

    print("\n✅ Toutes les dépendances sont correctement installées!")

except ImportError as e:
    print(f"\n❌ Erreur d'importation: {e}")
    exit(1)
EOF

echo ""
echo "======================================================================"
echo "🚀 Prêt à lancer MegaLoc SLAM Loop Closure!"
echo "======================================================================"
