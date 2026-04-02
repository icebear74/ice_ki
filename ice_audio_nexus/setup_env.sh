#!/bin/bash
# =============================================================================
# ice_audio_nexus – Setup Script
# =============================================================================
# Stable production setup for Tesla P100 (SM 6.0) / P4 (SM 6.1) Pascal GPUs.
#
# Key version pins (derived from debugging session):
#   Python:          3.12 (preferred) or 3.11
#   torch:           2.4.1+cu118  (Pascal support; cu130 breaks CUBLAS)
#   torchaudio:      2.4.1+cu118
#   numpy:           <2.0.0       (avoid AttributeError: np.NaN)
#   huggingface_hub: <0.25.0      (keep use_auth_token param)
#   pyannote.audio:  ==3.1.1      (stable for numpy<2 + old PyTorch)
#
# Usage:
#   cd ice_audio_nexus
#   bash setup_env.sh
# =============================================================================

set -e

BOLD='\033[1m'
GREEN='\033[92m'
CYAN='\033[96m'
YELLOW='\033[93m'
RED='\033[91m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   ice_audio_nexus – Production Setup (Pascal P100/P4)       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ---------------------------------------------------------------------------
# 1. Resolve Python binary (3.12 preferred, 3.11 fallback)
# ---------------------------------------------------------------------------
echo -e "${CYAN}🐍 Schritt 1: Python-Executable ermitteln...${RESET}"
PY_BIN=""
for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$MAJOR" = "3" ] && [ "$MINOR" -ge 11 ]; then
            PY_BIN="$candidate"
            echo -e "${GREEN}✓ Verwende $candidate (Version $PY_VER)${RESET}"
            break
        fi
    fi
done

if [ -z "$PY_BIN" ]; then
    echo -e "${RED}❌ Python 3.11+ nicht gefunden. Bitte installieren.${RESET}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create / activate venv
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}📦 Schritt 2: Virtuelle Umgebung erstellen...${RESET}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠ Vorhandene venv wird gelöscht und neu erstellt...${RESET}"
    rm -rf venv
fi
"$PY_BIN" -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓ venv bereit: $(python --version)${RESET}"

# ---------------------------------------------------------------------------
# 3. Install all high-level AI packages FIRST
#    (they will pull in a wrong/recent torch – we fix that in step 5)
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}🤖 Schritt 3: KI-Pakete installieren (torch kommt danach)...${RESET}"

# numpy<2 must be pinned before anything else pulls in 2.x
pip install "numpy<2.0.0" --quiet

# huggingface_hub<0.25 keeps use_auth_token support for pyannote 3.1.1
pip install "huggingface_hub<0.25.0" --quiet

# pyannote.audio 3.1.1 – stable on numpy<2 + old torch
pip install "pyannote.audio==3.1.1" --quiet

# faster-whisper + audio helpers
pip install faster-whisper librosa soundfile audioread --quiet

# Remove torchcodec – it requires CUDA 12.x+ and breaks Pascal cards
pip uninstall -y torchcodec 2>/dev/null || true

echo -e "${GREEN}✓ KI-Pakete installiert${RESET}"

# ---------------------------------------------------------------------------
# 4. Web-UI & DB packages
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}🌐 Schritt 4: Web-UI-Pakete installieren...${RESET}"
pip install \
    "fastapi[standard]" \
    "uvicorn[standard]" \
    jinja2 \
    python-multipart \
    aiofiles \
    "python-dotenv" \
    mariadb \
    --quiet
echo -e "${GREEN}✓ Web-UI-Pakete installiert${RESET}"

# ---------------------------------------------------------------------------
# 5. Force-install compatible torch (CUDA 11.8 – Pascal SM 6.0/6.1 support)
#    This MUST come last to override whatever pyannote/whisper pulled in.
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}🔥 Schritt 5: PyTorch 2.4.1+cu118 (Pascal-kompatibel) erzwingen...${RESET}"
pip uninstall -y torch torchaudio torchvision 2>/dev/null || true
pip install --no-cache-dir \
    "torch==2.4.1+cu118" \
    "torchaudio==2.4.1+cu118" \
    --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓ PyTorch 2.4.1+cu118 installiert${RESET}"

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
echo -e "\n${CYAN}🔍 Schritt 6: Hardware-Kompatibilitätstest...${RESET}"
python3 << 'PYCHECK'
import sys
import torch

print(f"Python:  {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

# Ensure we actually got cu118
if "cu118" not in torch.__version__:
    print(f"❌ FEHLER: Falsche Torch-Version! Erwartet cu118, gefunden: {torch.__version__}")
    sys.exit(1)

import numpy as np
print(f"NumPy:   {np.__version__}")
if tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 0):
    print("❌ FEHLER: NumPy >= 2.0 – bitte 'pip install numpy<2.0.0' ausführen")
    sys.exit(1)

import importlib.metadata
hf_ver = importlib.metadata.version("huggingface_hub")
print(f"huggingface_hub: {hf_ver}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        mem_mb = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)
        print(f"GPU {i}: {name}  (SM {major}.{minor}, {mem_mb} MB)")

    # Real matrix-multiply test (this was the CUBLAS crash point)
    try:
        a = torch.randn(64, 64, device="cuda")
        b = torch.randn(64, 64, device="cuda")
        _ = a @ b
        torch.cuda.synchronize()
        print("✅ GPU-MatMul erfolgreich – Pascal-Karten funktionieren!")
    except Exception as e:
        print(f"❌ GPU-MatMul fehlgeschlagen: {e}")
        sys.exit(1)
else:
    print("⚠  CUDA nicht verfügbar – läuft auf CPU")

print("\n✅ Alle Checks bestanden. Scanner kann gestartet werden.")
PYCHECK

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════════╗"
echo "║   Setup abgeschlossen!                                      ║"
echo "║   Aktivieren:  source venv/bin/activate                    ║"
echo "║   Scanner:     python -m processor.scanner --help          ║"
echo "║   Web-UI:      uvicorn web_ui.api:app --host 0.0.0.0 ...   ║"
echo -e "╚══════════════════════════════════════════════════════════════╝${RESET}"
