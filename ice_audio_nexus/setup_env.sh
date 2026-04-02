#!/usr/bin/env bash
# ============================================================
# ice_audio_nexus – Environment Setup Script
# Erstellt eine Python 3.12 venv mit allen notwendigen Bibliotheken
# ============================================================
# Verwendung:
#   ./setup_env.sh
#   ./setup_env.sh --python /usr/bin/python3.12
# ============================================================

set -euo pipefail

GREEN='\033[92m'
CYAN='\033[96m'
RED='\033[91m'
YELLOW='\033[93m'
BOLD='\033[1m'
RESET='\033[0m'

VENV_DIR="venv"
PY_BIN="${PYTHON_EXECUTABLE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PY_BIN="$2"; shift 2;;
    -h|--help)
      echo "Verwendung: $0 [--python /pfad/zu/python3.12]"
      exit 0;;
    *) echo -e "${RED}Unbekannte Option: $1${RESET}"; exit 2;;
  esac
done

echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║    ice_audio_nexus – Environment Setup                    ║"
echo "║    KI-basierte Video-Audio-Analyse & Personenidentifikation║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ============================================================
# 1. Python-Executable bestimmen (bevorzuge 3.12)
# ============================================================
echo -e "${CYAN}📋 Schritt 1: Python-Executable bestimmen...${RESET}"

if [ -n "${PY_BIN}" ]; then
  if ! command -v "$PY_BIN" &>/dev/null; then
    echo -e "${RED}✗ Angegebene Python-Executable '$PY_BIN' nicht gefunden!${RESET}"
    exit 1
  fi
else
  if command -v python3.12 &>/dev/null; then
    PY_BIN="python3.12"
  elif command -v python3 &>/dev/null; then
    PY_BIN="python3"
  else
    echo -e "${RED}✗ Python 3 ist nicht installiert!${RESET}"
    echo "Bitte installiere Python 3.12: sudo apt install python3.12 python3.12-venv"
    exit 1
  fi
fi

PYTHON_VERSION=$("$PY_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} (${PY_BIN})${RESET}"

if ! "$PY_BIN" -m pip --version &>/dev/null; then
  echo -e "${RED}✗ pip nicht verfügbar für '$PY_BIN'!${RESET}"
  echo -e "${YELLOW}  Tipp: sudo apt install python3.12-venv python3-pip${RESET}"
  exit 1
fi

# ============================================================
# 2. CUDA-Erkennung
# ============================================================
echo -e "\n${CYAN}🖥️  Schritt 2: GPU / CUDA erkennen...${RESET}"

TORCH_INDEX_URL=""
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader 2>/dev/null | \
    awk '{print "  GPU: " $0}' || true
  CUDA_VERSION=$(nvidia-smi 2>/dev/null \
    | grep -iE "CUDA Version" \
    | grep -oE '[0-9]+\.[0-9]+' \
    | head -1 || true)
  if [ -n "$CUDA_VERSION" ]; then
    echo -e "${GREEN}✓ CUDA Version erkannt: ${CUDA_VERSION}${RESET}"
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d'.' -f2)
    if [ "$CUDA_MAJOR" -gt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; }; then
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
      echo -e "${CYAN}  → Wheel-Index: cu121 (CUDA ≥ 12.1)${RESET}"
    elif [ "$CUDA_MAJOR" -gt 11 ] || { [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; }; then
      TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
      echo -e "${CYAN}  → Wheel-Index: cu118 (CUDA ≥ 11.8)${RESET}"
    else
      echo -e "${YELLOW}⚠ CUDA ${CUDA_VERSION} < 11.8 – CPU-Fallback${RESET}"
    fi
  else
    echo -e "${YELLOW}⚠ nvidia-smi vorhanden, aber CUDA-Version nicht parsebar – CPU-Fallback${RESET}"
  fi
else
  echo -e "${YELLOW}⚠ Keine NVIDIA GPU erkannt – CPU-Fallback${RESET}"
fi

# ============================================================
# 3. Virtuelle Umgebung erstellen
# ============================================================
echo -e "\n${CYAN}📦 Schritt 3: Virtuelle Umgebung erstellen...${RESET}"

if [ -d "$VENV_DIR" ]; then
  echo -e "${YELLOW}⚠ Virtuelle Umgebung existiert bereits in '${VENV_DIR}'${RESET}"
  read -p "Möchten Sie sie löschen und neu erstellen? (j/n): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[JjYy]$ ]]; then
    rm -rf "$VENV_DIR"
  else
    echo -e "${YELLOW}Verwende existierende Umgebung...${RESET}"
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PY_BIN" -m venv "$VENV_DIR"
  echo -e "${GREEN}✓ Virtuelle Umgebung erstellt${RESET}"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtuelle Umgebung aktiviert: $(which python)${RESET}"

# ============================================================
# 4. Build-Tools upgraden
# ============================================================
echo -e "\n${CYAN}⬆️  Schritt 4: pip und Build-Tools upgraden...${RESET}"
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓ pip, setuptools, wheel aktualisiert${RESET}"

# ============================================================
# 5. PyTorch installieren
# ============================================================
echo -e "\n${CYAN}🔥 Schritt 5: PyTorch installieren...${RESET}"
if [ -n "$TORCH_INDEX_URL" ]; then
  pip install torch torchaudio --index-url "$TORCH_INDEX_URL" --quiet
  echo -e "${GREEN}✓ PyTorch + torchaudio (CUDA) installiert${RESET}"
else
  pip install torch torchaudio --quiet
  echo -e "${GREEN}✓ PyTorch + torchaudio (CPU) installiert${RESET}"
fi

# ============================================================
# 6. KI-Bibliotheken installieren
# ============================================================
echo -e "\n${CYAN}🤖 Schritt 6: KI-Bibliotheken installieren...${RESET}"

echo -e "${CYAN}  → pyannote.audio (Speaker Diarization)...${RESET}"
pip install pyannote.audio --quiet
echo -e "${GREEN}  ✓ pyannote.audio${RESET}"

echo -e "${CYAN}  → faster-whisper (Transkription)...${RESET}"
pip install faster-whisper --quiet
echo -e "${GREEN}  ✓ faster-whisper${RESET}"

# ============================================================
# 7. Datenbank-Connector installieren
# ============================================================
echo -e "\n${CYAN}🗄️  Schritt 7: MariaDB-Connector installieren...${RESET}"
pip install mariadb --quiet
echo -e "${GREEN}✓ mariadb${RESET}"

# ============================================================
# 8. Web-Framework installieren
# ============================================================
echo -e "\n${CYAN}🌐 Schritt 8: Web-Framework installieren...${RESET}"
pip install fastapi uvicorn[standard] python-multipart jinja2 websockets python-dotenv aiofiles --quiet
echo -e "${GREEN}✓ fastapi, uvicorn, jinja2, websockets, python-dotenv, aiofiles${RESET}"

# ============================================================
# 9. Hilfsbibliotheken
# ============================================================
echo -e "\n${CYAN}🔧 Schritt 9: Hilfsbibliotheken installieren...${RESET}"
pip install numpy scipy --quiet
echo -e "${GREEN}✓ numpy, scipy${RESET}"

# ============================================================
# 10. .env-Datei vorbereiten
# ============================================================
echo -e "\n${CYAN}📝 Schritt 10: .env-Konfiguration prüfen...${RESET}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/.env" ]; then
  if [ -f "${SCRIPT_DIR}/.env.example" ]; then
    cp "${SCRIPT_DIR}/.env.example" "${SCRIPT_DIR}/.env"
    echo -e "${YELLOW}⚠ .env aus .env.example erstellt – bitte Zugangsdaten eintragen!${RESET}"
    echo -e "${YELLOW}  Datei: ${SCRIPT_DIR}/.env${RESET}"
  fi
else
  echo -e "${GREEN}✓ .env bereits vorhanden${RESET}"
fi

# ============================================================
# 11. ffmpeg prüfen
# ============================================================
echo -e "\n${CYAN}🎬 Schritt 11: FFmpeg prüfen...${RESET}"
if command -v ffmpeg &>/dev/null; then
  FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1 | grep -oE 'version [^ ]+' || echo "gefunden")
  echo -e "${GREEN}✓ ffmpeg ${FFMPEG_VER}${RESET}"
  # CUDA-Support prüfen
  if ffmpeg -hwaccels 2>/dev/null | grep -q cuda; then
    echo -e "${GREEN}✓ FFmpeg unterstützt CUDA-Hardwarebeschleunigung${RESET}"
  else
    echo -e "${YELLOW}⚠ FFmpeg ohne CUDA-Support gefunden. Für maximale Performance FFmpeg mit CUDA empfohlen.${RESET}"
  fi
else
  echo -e "${RED}✗ ffmpeg nicht gefunden!${RESET}"
  echo -e "${YELLOW}  Installation: sudo apt install ffmpeg${RESET}"
fi

# ============================================================
# Summary
# ============================================================
echo -e "\n${BOLD}${GREEN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║    ✓ ice_audio_nexus Setup abgeschlossen!                 ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "${BOLD}Nächste Schritte:${RESET}"
echo ""
echo -e "  ${CYAN}1. Zugangsdaten in .env eintragen:${RESET}"
echo -e "     ${YELLOW}nano .env${RESET}"
echo ""
echo -e "  ${CYAN}2. Umgebung aktivieren:${RESET}"
echo -e "     ${YELLOW}source venv/bin/activate${RESET}"
echo ""
echo -e "  ${CYAN}3. Scanner starten (Folge analysieren):${RESET}"
echo -e "     ${YELLOW}python processor/scanner.py --video /pfad/zur/episode.mkv --source \"The Walking Dead\" --episode \"S01E01\"${RESET}"
echo ""
echo -e "  ${CYAN}4. Web-Interface starten:${RESET}"
echo -e "     ${YELLOW}python web_ui/api.py${RESET}"
echo -e "     Browser: ${YELLOW}http://localhost:8000${RESET}"
echo ""
