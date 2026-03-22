#!/usr/bin/env bash

# ============================================================
# VSR+++ Setup Script - Option B Training Environment
# ============================================================
# Usage:
#   ./setup_env.sh
#   ./setup_env.sh --python /usr/bin/python3.11
#   PYTHON_EXECUTABLE=/usr/bin/python3.11 ./setup_env.sh
#   ./setup_env.sh --no-torch        (skip torch/torchvision installation)
#   ./setup_env.sh --install-torch   (explicit opt-in, now the default)
# ============================================================

set -euo pipefail

# Colors for output
GREEN='\033[92m'
CYAN='\033[96m'
RED='\033[91m'
YELLOW='\033[93m'
BOLD='\033[1m'
RESET='\033[0m'

# ---- Defaults ----
VENV_DIR="venv"
INSTALL_TORCH=true   # torch is installed by default; use --no-torch to skip
# Allow PYTHON_EXECUTABLE env variable as override; CLI --python overrides that
PY_BIN="${PYTHON_EXECUTABLE:-}"

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PY_BIN="$2"; shift 2;;
    --install-torch) INSTALL_TORCH=true; shift;;   # explicit opt-in (already the default)
    --no-torch) INSTALL_TORCH=false; shift;;        # opt-out: skip torch installation
    -h|--help)
      echo "Usage: $0 [--python /path/to/python3.11] [--no-torch]"
      exit 0;;
    *) echo -e "${RED}Unbekannte Option: $1${RESET}"; exit 2;;
  esac
done

echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║    VSR+++ Option B - Environment Setup Script             ║"
echo "║    Video Super Resolution Training Environment            ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ============================================================
# 1. Resolve Python executable
# ============================================================
echo -e "${CYAN}📋 Schritt 1: Python-Executable bestimmen...${RESET}"

if [ -n "${PY_BIN}" ]; then
  if ! command -v "$PY_BIN" &>/dev/null; then
    echo -e "${RED}✗ Angegebene Python-Executable '$PY_BIN' nicht gefunden!${RESET}"
    exit 1
  fi
else
  # Prefer python3.11 over generic python3 (avoids accidentally using 3.12+)
  if command -v python3.11 &>/dev/null; then
    PY_BIN="python3.11"
  elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
  else
    echo -e "${RED}✗ Python 3 ist nicht installiert!${RESET}"
    echo "Bitte installiere Python 3.11 oder nutze --python /pfad/zu/python3.11"
    exit 1
  fi
fi

PYTHON_VERSION=$("$PY_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} (${PY_BIN})${RESET}"

# Warn if version is newer than 3.11 (some packages may not yet support it)
PY_MINOR=$("$PY_BIN" -c 'import sys; print(sys.version_info[1])')
PY_MAJOR=$("$PY_BIN" -c 'import sys; print(sys.version_info[0])')
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -gt 11 ]; then
  echo -e "${YELLOW}⚠ Python ${PYTHON_VERSION} > 3.11 erkannt. Einige Pakete (z.B. ältere torch-Wheels)${RESET}"
  echo -e "${YELLOW}  unterstützen möglicherweise diese Version noch nicht vollständig.${RESET}"
  echo -e "${YELLOW}  Empfehlung: --python /usr/bin/python3.11 für maximale Kompatibilität.${RESET}"
fi

# Check if we have pip (use the selected Python's pip module — avoids issues when
# pip3 binary is absent but python -m pip works, e.g. deadsnakes PPA installs)
if ! "$PY_BIN" -m pip --version &>/dev/null; then
    echo -e "${RED}✗ pip ist nicht verfügbar für '$PY_BIN'!${RESET}"
    echo -e "${YELLOW}  Tipp (Debian/Ubuntu): sudo apt install python3.11-venv python3-pip${RESET}"
    echo -e "${YELLOW}  Tipp (macOS):         brew install python@3.11${RESET}"
    exit 1
fi
echo -e "${GREEN}✓ pip verfügbar ($(\"$PY_BIN\" -m pip --version 2>&1 | head -1))${RESET}"

# ============================================================
# 2. CUDA detection (improved)
# ============================================================
echo -e "\n${CYAN}🖥️  Schritt 2: GPU / CUDA erkennen...${RESET}"

CUDA_DETECTED=false
CUDA_VERSION=""
TORCH_INDEX_URL=""

if command -v nvidia-smi &>/dev/null; then
  echo -e "${GREEN}✓ nvidia-smi gefunden${RESET}"
  nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader 2>/dev/null | \
    awk '{print "  GPU: " $0}' || true

  # Parse CUDA version robustly (handles different nvidia-smi output layouts)
  CUDA_VERSION=$(nvidia-smi 2>/dev/null \
    | grep -iE "CUDA Version" \
    | grep -oE '[0-9]+\.[0-9]+' \
    | head -1 || true)

  if [ -n "$CUDA_VERSION" ]; then
    CUDA_DETECTED=true
    echo -e "${GREEN}✓ CUDA Version erkannt: ${CUDA_VERSION}${RESET}"

    # Determine best PyTorch wheel index
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d'.' -f2)

    if [ "$CUDA_MAJOR" -gt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; }; then
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
      echo -e "${CYAN}  → Wheel-Index: cu121 (CUDA ≥ 12.1)${RESET}"
    elif [ "$CUDA_MAJOR" -gt 11 ] || { [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; }; then
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
      echo -e "${CYAN}  → Wheel-Index: cu118 (CUDA ≥ 11.8)${RESET}"
    else
      echo -e "${YELLOW}⚠ CUDA ${CUDA_VERSION} < 11.8 — CPU-Fallback wird verwendet${RESET}"
    fi
  else
    echo -e "${YELLOW}⚠ nvidia-smi vorhanden, aber CUDA-Version nicht parsebar — CPU-Fallback${RESET}"
  fi
else
  echo -e "${YELLOW}⚠ Keine NVIDIA GPU erkannt (nvidia-smi fehlt) — Training läuft auf CPU${RESET}"
fi

# ============================================================
# 3. Virtual Environment Setup
# ============================================================
echo -e "\n${CYAN}📦 Schritt 3: Virtuelle Umgebung erstellen...${RESET}"

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
    echo -e "${CYAN}Erstelle neue virtuelle Umgebung mit ${PY_BIN} in '${VENV_DIR}'...${RESET}"
    "$PY_BIN" -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtuelle Umgebung erstellt${RESET}"
else
    echo -e "${GREEN}✓ Verwende existierende Umgebung${RESET}"
fi

# ============================================================
# 4. Activate Virtual Environment
# ============================================================
echo -e "\n${CYAN}🔌 Schritt 4: Aktiviere virtuelle Umgebung...${RESET}"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtuelle Umgebung aktiviert${RESET}"
echo -e "   Python: $(which python)"

# ============================================================
# 5. Upgrade pip, setuptools, wheel
# ============================================================
echo -e "\n${CYAN}⬆️  Schritt 5: Upgrade pip und Build-Tools...${RESET}"

pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓ pip, setuptools und wheel aktualisiert${RESET}"

# ============================================================
# 6. Install dependencies from requirements.txt (without torch)
# ============================================================
echo -e "\n${CYAN}📚 Schritt 6: Abhängigkeiten installieren (ohne torch/torchvision)...${RESET}"

if [ -f "requirements.txt" ]; then
    TMP_REQ=$(mktemp)
    grep -v -E '^(torch|torchvision)([=<>!~[:space:]]|$)' requirements.txt > "$TMP_REQ" || true
    if [ -s "$TMP_REQ" ]; then
        echo -e "${CYAN}Installiere Pakete aus requirements.txt...${RESET}"
        pip install -r "$TMP_REQ" --quiet
        echo -e "${GREEN}✓ requirements.txt (ohne torch) installiert${RESET}"
    else
        echo -e "${YELLOW}⚠ Keine weiteren Pakete in requirements.txt nach torch-Filter${RESET}"
    fi
    rm -f "$TMP_REQ"
else
    echo -e "${YELLOW}⚠ requirements.txt nicht gefunden${RESET}"
    echo -e "${CYAN}Installiere Mindest-Pakete...${RESET}"
    pip install opencv-python tensorboard numpy tqdm psutil rich Pillow --quiet
    echo -e "${GREEN}✓ Mindest-Pakete installiert${RESET}"
fi

# ============================================================
# 7. Optional: Install PyTorch
# ============================================================
if [ "$INSTALL_TORCH" = true ]; then
  echo -e "\n${CYAN}🔥 Schritt 7: PyTorch installieren...${RESET}"
  echo -e "${YELLOW}Dies kann einige Minuten dauern...${RESET}"

  if [ -n "$TORCH_INDEX_URL" ]; then
    echo -e "${CYAN}Installiere torch + torchvision (${TORCH_INDEX_URL})...${RESET}"
    pip install torch torchvision --index-url "$TORCH_INDEX_URL" --quiet
  else
    echo -e "${YELLOW}Kein passendes CUDA-Wheel — installiere CPU-Version...${RESET}"
    pip install torch torchvision --quiet
  fi

  echo -e "${GREEN}✓ PyTorch installiert${RESET}"
  python -c "import torch; print(f'   PyTorch Version: {torch.__version__}')"
  python -c "import torch; print(f'   CUDA verfügbar:  {torch.cuda.is_available()}')"
  if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'   CUDA Version:    {torch.version.cuda}')"
    python -c "import torch; print(f'   GPU:             {torch.cuda.get_device_name(0)}')"
  fi
else
  echo -e "\n${YELLOW}ℹ PyTorch wird übersprungen (--no-torch angegeben).${RESET}"
  echo -e "${YELLOW}  Manuell (GPU cu118):  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118${RESET}"
  echo -e "${YELLOW}  Manuell (CPU only):   pip install torch torchvision${RESET}"
fi

# ============================================================
# 8. Verify core package imports
# ============================================================
echo -e "\n${CYAN}🔍 Schritt 8: Installation überprüfen...${RESET}"

ALL_OK=true
CORE_PACKAGES=("numpy" "cv2" "tensorboard" "tqdm" "psutil" "PIL")
# torch/torchvision only checked if --install-torch was used
if [ "$INSTALL_TORCH" = true ]; then
  CORE_PACKAGES=("torch" "torchvision" "${CORE_PACKAGES[@]}")
fi

for pkg in "${CORE_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'N/A'))" 2>/dev/null || echo "N/A")
        echo -e "${GREEN}✓ $pkg ($VERSION)${RESET}"
    else
        echo -e "${RED}✗ $pkg nicht gefunden!${RESET}"
        ALL_OK=false
    fi
done

# ============================================================
# 9. Import-Analyse und automatische Installation fehlender Pakete
# ============================================================
echo -e "\n${CYAN}🔎 Schritt 9: Import-Analyse der Python-Quelldateien...${RESET}"
echo -e "${CYAN}   (fehlende Pakete werden automatisch installiert)${RESET}"

STDLIB_PACKAGES="os sys re json math time argparse subprocess threading logging \
  pathlib typing collections itertools functools hashlib tempfile signal shutil \
  glob atexit errno queue random select traceback socket datetime tty termios \
  curses http concurrent atexit io abc copy struct weakref dataclasses enum \
  contextlib warnings gc platform uuid"

# Mapping: Python import name → pip package name (when they differ)
declare -A PIP_NAME_MAP=(
  ["cv2"]="opencv-python"
  ["PIL"]="Pillow"
  ["sklearn"]="scikit-learn"
  ["skimage"]="scikit-image"
  ["yaml"]="PyYAML"
  ["attr"]="attrs"
  ["bs4"]="beautifulsoup4"
  ["gi"]="PyGObject"
  ["wx"]="wxPython"
  ["usb"]="pyusb"
)

# Packages that require special/manual installation (warn only, do not pip install blindly)
SKIP_AUTO_INSTALL=("torch2trt" "tensorrt" "pycuda" "onnxruntime_gpu" "onnxruntime-gpu")

MISSING_PKGS=()
FOUND_IMPORTS=$(find . -name "*.py" 2>/dev/null \
  | xargs grep -h -E "^(import|from) [a-zA-Z_][a-zA-Z0-9_]*" 2>/dev/null \
  | grep -oE "^(import|from) [a-zA-Z_][a-zA-Z0-9_]*" \
  | awk '{print $2}' \
  | sort -u || true)

echo -e "  Gefundene Top-Level-Importe:"
while IFS= read -r pkg; do
  [[ -z "$pkg" ]] && continue
  # Skip stdlib
  is_std=false
  for std in $STDLIB_PACKAGES; do
    if [[ "$pkg" == "$std" ]]; then is_std=true; break; fi
  done
  $is_std && continue
  # Skip known local packages
  [[ "$pkg" =~ ^(vsr_plusplus_NEU|dataset_generator_v2|config|category_utils|generation_plan|interactive_selector|streaming_extractor|video_manager|utils|core)$ ]] && continue

  if python -c "import $pkg" 2>/dev/null; then
    VERSION=$(python -c "import $pkg; print(getattr($pkg, '__version__', 'ok'))" 2>/dev/null || echo "ok")
    echo -e "    ${GREEN}✓ $pkg ($VERSION)${RESET}"
  else
    echo -e "    ${YELLOW}⚠ $pkg — nicht installiert${RESET}"
    MISSING_PKGS+=("$pkg")
  fi
done <<< "$FOUND_IMPORTS"

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
  echo -e "\n${CYAN}📦 Installiere fehlende Pakete automatisch...${RESET}"
  for mp in "${MISSING_PKGS[@]}"; do
    # Check skip list (platform-specific / needs special setup)
    is_skip=false
    for skip_pkg in "${SKIP_AUTO_INSTALL[@]}"; do
      if [[ "$mp" == "$skip_pkg" ]]; then is_skip=true; break; fi
    done
    if $is_skip; then
      echo -e "    ${YELLOW}⚠ $mp — übersprungen (manuelle Installation erforderlich)${RESET}"
      continue
    fi
    # Resolve pip package name
    pip_pkg="${PIP_NAME_MAP[$mp]:-$mp}"
    echo -e "    ${CYAN}→ pip install $pip_pkg${RESET}"
    if pip install "$pip_pkg" --quiet; then
      echo -e "    ${GREEN}✓ $pip_pkg installiert${RESET}"
    else
      echo -e "    ${RED}✗ $pip_pkg konnte nicht installiert werden!${RESET}"
      ALL_OK=false
    fi
  done
fi

# ============================================================
# 10. ffmpeg check
# ============================================================
echo -e "\n${CYAN}🎬 Schritt 10: ffmpeg prüfen...${RESET}"
if command -v ffmpeg &>/dev/null; then
  FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | grep -oE 'version [^ ]+' || echo "gefunden")
  echo -e "${GREEN}✓ ffmpeg ${FFMPEG_VER}${RESET}"
else
  echo -e "${RED}✗ ffmpeg nicht gefunden!${RESET}"
  echo -e "${YELLOW}  Das Dataset-Generator-Modul (streaming_extractor.py) benötigt ffmpeg.${RESET}"
  echo -e "${YELLOW}  Installation: sudo apt install ffmpeg   |   brew install ffmpeg${RESET}"
fi

# ============================================================
# 11. Optional: run tools/check_imports.sh
# ============================================================
if [ -f "tools/check_imports.sh" ]; then
  echo -e "\n${CYAN}🧪 Schritt 11: Import-Check (tools/check_imports.sh)...${RESET}"
  bash tools/check_imports.sh || true
fi

# ============================================================
# 12. Create Directory Structure hint
# ============================================================
echo -e "\n${CYAN}📁 Verzeichnisstruktur-Hinweis:${RESET}"
echo -e "${YELLOW}"
echo "Stellen Sie sicher, dass Ihr Dataset hier liegt:"
echo "  - /mnt/data/training/Dataset/Universal/Mastermodell/Patches/GT"
echo "  - /mnt/data/training/Dataset/Universal/Mastermodell/Patches/LR"
echo "  - /mnt/data/training/Dataset/Universal/Mastermodell/Val/GT"
echo "  - /mnt/data/training/Dataset/Universal/Mastermodell/Val/LR"
echo -e "${RESET}"

# ============================================================
# 13. Create Activation Script
# ============================================================
echo -e "${CYAN}📝 Aktivierungs-Script erstellen...${RESET}"

cat > activate.sh << 'EOF'
#!/bin/bash
# VSR+++ Environment Activation Script

source venv/bin/activate
echo -e "\033[92m✓ VSR+++ Umgebung aktiviert\033[0m"
echo ""
echo "Verfügbare Befehle:"
echo "  python vsr_plusplus_NEU/train.py  - Training starten"
echo "  tensorboard --logdir ...          - TensorBoard starten"
echo "  deactivate                        - Umgebung verlassen"
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
    echo -e "${GREEN}Alle geprüften Komponenten wurden erfolgreich installiert!${RESET}"
    echo ""
    echo -e "${BOLD}Nächste Schritte:${RESET}"
    echo ""
    echo -e "${CYAN}1. Umgebung aktivieren:${RESET}"
    echo -e "   ${YELLOW}source venv/bin/activate${RESET}"
    echo -e "   oder: ${YELLOW}source activate.sh${RESET}"
    echo ""
    echo -e "${CYAN}2. Training starten:${RESET}"
    echo -e "   ${YELLOW}python vsr_plusplus_NEU/train.py${RESET}"
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
