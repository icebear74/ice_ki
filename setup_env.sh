#!/bin/bash

# ============================================================
# VSR+++ Setup Script - Option B Training Environment
# ============================================================

set -e  # Exit on error

# Colors for output
GREEN='\033[92m'
CYAN='\033[96m'
RED='\033[91m'
YELLOW='\033[93m'
BOLD='\033[1m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║    VSR+++ Option B - Environment Setup Script             ║"
echo "║    Video Super Resolution Training Environment            ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ============================================================
# 1. System Requirements Check
# ============================================================
echo -e "${CYAN}📋 Schritt 1: System-Anforderungen prüfen...${RESET}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 ist nicht installiert!${RESET}"
    echo "Bitte installieren Sie Python 3.8 oder höher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} gefunden${RESET}"

# Check if we have pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}✗ pip3 ist nicht installiert!${RESET}"
    exit 1
fi
echo -e "${GREEN}✓ pip3 gefunden${RESET}"

# Check NVIDIA GPU (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU erkannt:${RESET}"
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠ Keine NVIDIA GPU erkannt - Training wird auf CPU laufen (nicht empfohlen)${RESET}"
fi

# ============================================================
# 2. Virtual Environment Setup
# ============================================================
echo -e "\n${CYAN}📦 Schritt 2: Virtuelle Umgebung erstellen...${RESET}"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}⚠ Virtuelle Umgebung existiert bereits in '${VENV_DIR}'${RESET}"
    read -p "Möchten Sie sie löschen und neu erstellen? (j/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[JjYy]$ ]]; then
        echo -e "${CYAN}Lösche alte Umgebung...${RESET}"
        rm -rf "$VENV_DIR"
    else
        echo -e "${YELLOW}Verwende existierende Umgebung...${RESET}"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${CYAN}Erstelle neue virtuelle Umgebung in '${VENV_DIR}'...${RESET}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtuelle Umgebung erstellt${RESET}"
else
    echo -e "${GREEN}✓ Verwende existierende Umgebung${RESET}"
fi

# ============================================================
# 3. Activate Virtual Environment
# ============================================================
echo -e "\n${CYAN}🔌 Schritt 3: Aktiviere virtuelle Umgebung...${RESET}"

source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtuelle Umgebung aktiviert${RESET}"
echo -e "   Python: $(which python)"

# ============================================================
# 4. Upgrade pip, setuptools, wheel
# ============================================================
echo -e "\n${CYAN}⬆️  Schritt 4: Upgrade pip und Build-Tools...${RESET}"

pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓ pip, setuptools und wheel aktualisiert${RESET}"

# ============================================================
# 5. Install PyTorch (with CUDA support if available)
# ============================================================
echo -e "\n${CYAN}🔥 Schritt 5: PyTorch installieren...${RESET}"

if command -v nvidia-smi &> /dev/null; then
    # Check CUDA version
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo -e "${CYAN}Erkannte CUDA Version: ${CUDA_VERSION}${RESET}"
    
    # Install PyTorch with CUDA support
    echo -e "${CYAN}Installiere PyTorch mit CUDA-Unterstützung...${RESET}"
    echo -e "${YELLOW}Dies kann einige Minuten dauern...${RESET}"
    
    # For CUDA 11.8 or higher, use the latest stable PyTorch
    if [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l) -eq 1 ]] 2>/dev/null || true; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
    else
        # Fallback to CPU version if CUDA version is too old
        echo -e "${YELLOW}⚠ CUDA Version zu alt, installiere CPU-Version${RESET}"
        pip install torch torchvision --quiet
    fi
else
    echo -e "${YELLOW}Installiere PyTorch CPU-Version...${RESET}"
    pip install torch torchvision --quiet
fi

echo -e "${GREEN}✓ PyTorch installiert${RESET}"

# Verify PyTorch installation
python -c "import torch; print(f'   PyTorch Version: {torch.__version__}')"
python -c "import torch; print(f'   CUDA verfügbar: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'   CUDA Version: {torch.version.cuda}')"
    python -c "import torch; print(f'   GPU: {torch.cuda.get_device_name(0)}')"
fi

# ============================================================
# 6. Install Other Dependencies
# ============================================================
echo -e "\n${CYAN}📚 Schritt 6: Weitere Abhängigkeiten installieren...${RESET}"

if [ -f "requirements.txt" ]; then
    # Install requirements except torch/torchvision (already installed)
    grep -v "^torch" requirements.txt > /tmp/requirements_filtered.txt || true
    
    if [ -s /tmp/requirements_filtered.txt ]; then
        echo -e "${CYAN}Installiere Pakete aus requirements.txt...${RESET}"
        pip install -r /tmp/requirements_filtered.txt --quiet
    fi
    rm /tmp/requirements_filtered.txt
    
    echo -e "${GREEN}✓ Alle Abhängigkeiten installiert${RESET}"
else
    echo -e "${YELLOW}⚠ requirements.txt nicht gefunden${RESET}"
    echo -e "${CYAN}Installiere Standard-Pakete...${RESET}"
    pip install opencv-python tensorboard numpy tqdm --quiet
    echo -e "${GREEN}✓ Standard-Pakete installiert${RESET}"
fi

# ============================================================
# 7. Verify Installation
# ============================================================
echo -e "\n${CYAN}🔍 Schritt 7: Installation überprüfen...${RESET}"

PACKAGES=("torch" "torchvision" "cv2" "tensorboard" "numpy" "tqdm")
ALL_OK=true

for pkg in "${PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'N/A'))" 2>/dev/null || echo "N/A")
        echo -e "${GREEN}✓ $pkg ($VERSION)${RESET}"
    else
        echo -e "${RED}✗ $pkg nicht gefunden!${RESET}"
        ALL_OK=false
    fi
done

# ============================================================
# 8. Create Directory Structure
# ============================================================
echo -e "\n${CYAN}📁 Schritt 8: Verzeichnisstruktur erstellen...${RESET}"

# Note: We don't create /mnt/data as it's user-specific
# Just show what's needed

echo -e "${YELLOW}"
echo "Hinweis: Die folgenden Verzeichnisse werden beim ersten Training erstellt:"
echo "  - /mnt/data/training/Universal/Mastermodell/Learn/checkpoints"
echo "  - /mnt/data/training/Universal/Mastermodell/Learn/logs"
echo ""
echo "Stellen Sie sicher, dass Ihr Dataset hier liegt:"
echo "  - /mnt/data/training/Dataset/Universal/Mastermodell/Patches/GT"
echo "  - /mnt/data/training/Dataset/Universal/Mastermodell/Patches/LR"
echo "  - /mnt/data/training/Dataset/Universal/Mastermodell/Val/GT"
echo "  - /mnt/data/training/Dataset/Universal/Mastermodell/Val/LR"
echo -e "${RESET}"

# ============================================================
# 9. Test Model Import
# ============================================================
echo -e "\n${CYAN}🧪 Schritt 9: Modell-Import testen...${RESET}"

if python -c "from model_vsrppp_v2 import VSRTriplePlus_3x; print('✓ Modell erfolgreich importiert')" 2>/dev/null; then
    echo -e "${GREEN}✓ model_vsrppp_v2.py funktioniert${RESET}"
else
    echo -e "${RED}✗ Fehler beim Import von model_vsrppp_v2.py${RESET}"
    ALL_OK=false
fi

# ============================================================
# 10. Create Activation Script
# ============================================================
echo -e "\n${CYAN}📝 Schritt 10: Aktivierungs-Script erstellen...${RESET}"

cat > activate.sh << 'EOF'
#!/bin/bash
# VSR+++ Environment Activation Script

source venv/bin/activate
echo -e "\033[92m✓ VSR+++ Umgebung aktiviert\033[0m"
echo ""
echo "Verfügbare Befehle:"
echo "  python train.py          - Training starten"
echo "  tensorboard --logdir ... - TensorBoard starten"
echo "  deactivate               - Umgebung verlassen"
echo ""
EOF

chmod +x activate.sh
echo -e "${GREEN}✓ activate.sh erstellt${RESET}"

# ============================================================
# Summary
# ============================================================
echo -e "\n${BOLD}${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              ✓ Setup erfolgreich abgeschlossen!           ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}Alle Komponenten wurden erfolgreich installiert!${RESET}"
    echo ""
    echo -e "${BOLD}Nächste Schritte:${RESET}"
    echo ""
    echo -e "${CYAN}1. Umgebung aktivieren:${RESET}"
    echo -e "   ${YELLOW}source venv/bin/activate${RESET}"
    echo -e "   oder: ${YELLOW}source activate.sh${RESET}"
    echo ""
    echo -e "${CYAN}2. Training starten:${RESET}"
    echo -e "   ${YELLOW}python train.py${RESET}"
    echo ""
    echo -e "${CYAN}3. TensorBoard starten (in neuem Terminal):${RESET}"
    echo -e "   ${YELLOW}tensorboard --logdir /mnt/data/training/Universal/Mastermodell/Learn/logs${RESET}"
    echo ""
    echo -e "${CYAN}4. Umgebung verlassen:${RESET}"
    echo -e "   ${YELLOW}deactivate${RESET}"
    echo ""
    echo -e "${BOLD}Dokumentation:${RESET} Siehe README.md für weitere Details"
    echo ""
else
    echo -e "${RED}⚠ Es gab einige Probleme bei der Installation.${RESET}"
    echo -e "${YELLOW}Bitte überprüfen Sie die obigen Fehlermeldungen.${RESET}"
    exit 1
fi
