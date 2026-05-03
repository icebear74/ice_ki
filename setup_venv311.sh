#!/usr/bin/env bash
# ============================================================
# setup_venv311.sh
# Erstellt eine saubere Python-3.11-venv in ./venv311/
# und testet die Lauffähigkeit für diese Hardware:
#   GPU  : Tesla P100-PCIE-16GB (CUDA Capability 6.0)
#   nvcc : 12.0 (Toolkit)
#   Treiber: 580.x (meldet CUDA 13.0 in nvidia-smi — das ist
#            die vom Treiber maximal unterstützte Version,
#            nicht die des installierten Toolkits!)
#
# Warum Python 3.11 (nicht 3.12)?
#   Für Python 3.12 waren keine PyTorch-Wheels verfügbar, die auf
#   der Tesla P100 (CC 6.0, CUDA 12.0) korrekt liefen.
#   Python 3.11 + cu121-Wheel ist die getestete Kombination.
#
# Wichtige P100/CC-6.0-Einschränkungen:
#   ✗ torch.compile / Triton  → erfordert CC ≥ 7.0
#   ✗ Flash Attention          → erfordert CC ≥ 7.5
#   ✗ bfloat16 (nativ)        → erfordert CC ≥ 8.0 (nur float16/fp32)
#   ✓ AMP (float16 / fp32)    → läuft
#   ✓ TensorRT PyPI-Paket      → läuft (CC 6.0 unterstützt)
# ============================================================
set -euo pipefail

# ---------- Farben ----------
GREEN='\033[92m'; CYAN='\033[96m'; RED='\033[91m'
YELLOW='\033[93m'; BOLD='\033[1m'; RESET='\033[0m'

VENV_DIR="venv311"
FORCE=false
SKIP_TORCH=false

# ---------- Argumente ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-f)   FORCE=true;       shift ;;
    --no-torch)   SKIP_TORCH=true;  shift ;;
    -h|--help)
      echo "Usage: $0 [--force] [--no-torch]"
      echo "  --force      venv311/ ohne Rückfrage löschen und neu erstellen"
      echo "  --no-torch   torch/torchvision überspringen (nur requirements.txt)"
      exit 0 ;;
    *) echo -e "${RED}Unbekannte Option: $1${RESET}"; exit 2 ;;
  esac
done

# ---------- Banner ----------
echo -e "${BOLD}${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║   setup_venv311.sh — Python 3.11 venv für P100/CC-6.0    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ============================================================
# 1. Python 3.11 prüfen  (3.12 ist NICHT kompatibel!)
# ============================================================
# Warum explizit 3.11?
#   Für Python 3.12 gab es zum Zeitpunkt der Entwicklung keine
#   PyTorch-Wheels, die auf der Tesla P100 (CC 6.0, CUDA 12.0)
#   korrekt liefen. Python 3.11 + cu121-Wheel ist die einzige
#   getestete, funktionierende Kombination für diesen Server.
# ============================================================
echo -e "${CYAN}[1/9] Python 3.11 suchen (3.12+ wird abgelehnt)...${RESET}"
PY_BIN=""
for candidate in python3.11 python3 python; do
  if command -v "$candidate" &>/dev/null; then
    ver=$("$candidate" -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))' 2>/dev/null || echo "?")
    minor=$("$candidate" -c 'import sys; print(sys.version_info[1])' 2>/dev/null || echo "0")
    if [[ "$ver" == "3.11" ]]; then
      PY_BIN="$candidate"
      break
    elif [[ "$minor" -ge 12 ]]; then
      echo -e "${YELLOW}  Überspringe $candidate ($ver) — für Python ≥ 3.12 fehlen kompatible${RESET}"
      echo -e "${YELLOW}  PyTorch-Wheels für Tesla P100 (CC 6.0, CUDA 12.0).${RESET}"
    fi
  fi
done

if [[ -z "$PY_BIN" ]]; then
  echo -e "${RED}✗ Python 3.11 nicht gefunden!${RESET}"
  echo -e "${YELLOW}  Tipp (Debian/Ubuntu): sudo apt install python3.11 python3.11-venv${RESET}"
  echo -e "${YELLOW}  Tipp (deadsnakes PPA): sudo add-apt-repository ppa:deadsnakes/ppa${RESET}"
  echo -e "${RED}  WICHTIG: Python 3.12+ NICHT verwenden — fehlende Wheel-Kompatibilität mit P100.${RESET}"
  exit 1
fi
echo -e "${GREEN}✓ Python 3.11 gefunden: $PY_BIN ($("$PY_BIN" -V 2>&1))${RESET}"

# venv-Modul prüfen
if ! "$PY_BIN" -m venv --help &>/dev/null; then
  echo -e "${RED}✗ python3.11-venv fehlt!${RESET}"
  echo -e "${YELLOW}  Tipp: sudo apt install python3.11-venv${RESET}"
  exit 1
fi

# ============================================================
# 2. CUDA-Version ermitteln (nvcc bevorzugt, Fallback nvidia-smi)
# ============================================================
echo -e "\n${CYAN}[2/9] CUDA-Toolkit-Version ermitteln...${RESET}"

NVCC_BIN=""
NVCC_CUDA=""
DRIVER_CUDA=""
CUDA_MAJOR=0
CUDA_MINOR=0

# nvcc in PATH und in Standard-Installationspfaden suchen
# nvcc suchen — /usr/local/cuda* zuerst (NVIDIA-Installer), /usr/bin/nvcc ist
# oft die alte apt-Version (nvidia-cuda-toolkit, z.B. 10.x/11.x) und wird
# absichtlich ZULETZT geprüft.
NVCC_SEARCH_PATHS=(
  /usr/local/cuda/bin/nvcc
  /usr/local/cuda-12.0/bin/nvcc
  /usr/local/cuda-12.1/bin/nvcc
  /usr/local/cuda-12.2/bin/nvcc
  /usr/local/cuda-12.3/bin/nvcc
  /usr/local/cuda-12.4/bin/nvcc
  /usr/local/cuda-11.8/bin/nvcc
  /usr/local/cuda-11/bin/nvcc
)
# dynamisch alle /usr/local/cuda-*/bin/nvcc einbeziehen
for d in /usr/local/cuda-*/bin/nvcc; do
  NVCC_SEARCH_PATHS+=("$d")
done
# /usr/bin/nvcc (apt nvidia-cuda-toolkit) als letzten Fallback
NVCC_SEARCH_PATHS+=("$(command -v nvcc 2>/dev/null || true)")

echo -e "  Suche nvcc..."
for candidate in "${NVCC_SEARCH_PATHS[@]}"; do
  [[ -z "$candidate" ]] && continue
  [[ -x "$candidate" ]] || continue
  NVCC_BIN="$candidate"
  NVCC_RAW=$("$NVCC_BIN" --version 2>&1 | head -5)
  echo -e "  ${GREEN}✓ nvcc gefunden: ${NVCC_BIN}${RESET}"
  echo -e "  ${CYAN}  Version-Output:${RESET}"
  echo "$NVCC_RAW" | sed 's/^/    /'
  break
done

if [[ -n "$NVCC_BIN" ]]; then
  NVCC_CUDA=$("$NVCC_BIN" --version 2>/dev/null \
    | grep -oE 'release [0-9]+\.[0-9]+' | head -1 | awk '{print $2}' || true)
  if [[ -n "$NVCC_CUDA" ]]; then
    CUDA_MAJOR=$(echo "$NVCC_CUDA" | cut -d'.' -f1)
    CUDA_MINOR=$(echo "$NVCC_CUDA" | cut -d'.' -f2)
    echo -e "  ${GREEN}✓ CUDA Toolkit (nvcc): ${NVCC_CUDA}${RESET}"
  else
    echo -e "  ${YELLOW}⚠ nvcc gefunden, aber Version nicht parsebar — Ausgabe:${RESET}"
    "$NVCC_BIN" --version 2>&1 | head -5 | sed 's/^/    /'
  fi
else
  echo -e "  ${YELLOW}⚠ nvcc nicht gefunden (weder in PATH noch in Standard-Pfaden)${RESET}"
  echo -e "  ${YELLOW}  Gesucht in: /usr/local/cuda*/bin/nvcc, /usr/bin/nvcc${RESET}"
fi

# nvidia-smi: GPU-Info + Treiber-CUDA anzeigen
if command -v nvidia-smi &>/dev/null; then
  DRIVER_CUDA=$(nvidia-smi 2>/dev/null \
    | grep -iE 'CUDA Version' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null \
    | head -1 || echo "unbekannt")
  GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
    | head -1 || echo "?")
  echo -e "  ${GREEN}✓ GPU: ${GPU_NAME} (Compute Capability: ${GPU_CC})${RESET}"
  if [[ -n "$DRIVER_CUDA" ]]; then
    echo -e "  ${CYAN}  nvidia-smi CUDA (Treiber-Max): ${DRIVER_CUDA}${RESET}"
    echo -e "  ${YELLOW}  Hinweis: nvidia-smi zeigt die vom Treiber max. unterstützte CUDA-Version,${RESET}"
    echo -e "  ${YELLOW}  NICHT die des Toolkits (nvcc). Für Wheels ist nvcc maßgeblich.${RESET}"
  fi
  # Falls nvcc nicht gefunden: nvidia-smi-Wert als Fallback für Wheel-Wahl verwenden
  if [[ $CUDA_MAJOR -eq 0 && -n "$DRIVER_CUDA" ]]; then
    echo -e "  ${YELLOW}  nvcc nicht gefunden — verwende nvidia-smi-Wert (${DRIVER_CUDA}) für Wheel-Auswahl${RESET}"
    echo -e "  ${YELLOW}  (nvidia-smi zeigt Treiber-Max, echtes Toolkit könnte niedriger sein)${RESET}"
    CUDA_MAJOR=$(echo "$DRIVER_CUDA" | cut -d'.' -f1)
    CUDA_MINOR=$(echo "$DRIVER_CUDA" | cut -d'.' -f2)
  fi
else
  echo -e "  ${YELLOW}⚠ nvidia-smi nicht gefunden — kein NVIDIA-Treiber aktiv?${RESET}"
fi

# Wheel-Index nach CUDA-Version wählen und Entscheidung erklären
WHEEL_INDEX=""
WHEEL_FALLBACK=""
WHEEL_SRC="${NVCC_CUDA:-nvidia-smi:${DRIVER_CUDA:-?}}"
if [[ $CUDA_MAJOR -ge 12 && $CUDA_MINOR -ge 8 ]]; then
  WHEEL_INDEX="https://download.pytorch.org/whl/cu128"
  WHEEL_FALLBACK="https://download.pytorch.org/whl/cu124"
  echo -e "  ${CYAN}→ PyTorch-Wheel: cu128 (CUDA ≥ 12.8, Quelle: ${WHEEL_SRC})${RESET}"
elif [[ $CUDA_MAJOR -ge 12 && $CUDA_MINOR -ge 4 ]]; then
  WHEEL_INDEX="https://download.pytorch.org/whl/cu124"
  WHEEL_FALLBACK="https://download.pytorch.org/whl/cu121"
  echo -e "  ${CYAN}→ PyTorch-Wheel: cu124 (CUDA ≥ 12.4, Quelle: ${WHEEL_SRC})${RESET}"
elif [[ $CUDA_MAJOR -ge 12 && $CUDA_MINOR -ge 0 ]]; then
  WHEEL_INDEX="https://download.pytorch.org/whl/cu121"
  echo -e "  ${CYAN}→ PyTorch-Wheel: cu121 (CUDA 12.0–12.3, Quelle: ${WHEEL_SRC})${RESET}"
elif [[ $CUDA_MAJOR -eq 11 && $CUDA_MINOR -ge 8 ]]; then
  WHEEL_INDEX="https://download.pytorch.org/whl/cu118"
  echo -e "  ${CYAN}→ PyTorch-Wheel: cu118 (CUDA 11.8–11.x, Quelle: ${WHEEL_SRC})${RESET}"
elif [[ $CUDA_MAJOR -gt 0 ]]; then
  echo -e "  ${YELLOW}→ CUDA ${CUDA_MAJOR}.${CUDA_MINOR} < 11.8 — CPU-Wheel als Fallback${RESET}"
else
  echo -e "  ${RED}→ CUDA nicht erkannt (nvcc und nvidia-smi beide fehlgeschlagen) — CPU-Wheel${RESET}"
  echo -e "  ${YELLOW}  Prüfe: which nvcc | nvcc --version | nvidia-smi${RESET}"
  echo -e "  ${YELLOW}  Falls CUDA installiert: export PATH=\$PATH:/usr/local/cuda/bin${RESET}"
fi

# ============================================================
# 3. venv anlegen
# ============================================================
echo -e "\n${CYAN}[3/9] venv311/ anlegen...${RESET}"

if [[ -d "$VENV_DIR" ]]; then
  if [[ "$FORCE" == "true" ]]; then
    echo -e "${YELLOW}  --force: lösche vorhandenes ${VENV_DIR}...${RESET}"
    rm -rf "$VENV_DIR"
  else
    read -r -p "  '${VENV_DIR}' existiert bereits. Löschen und neu erstellen? (j/N): " REPLY
    echo
    if [[ "$REPLY" =~ ^[JjYy]$ ]]; then
      rm -rf "$VENV_DIR"
    else
      echo -e "${CYAN}  Verwende vorhandene Umgebung.${RESET}"
    fi
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PY_BIN" -m venv "$VENV_DIR"
  echo -e "${GREEN}✓ venv311/ erstellt${RESET}"
else
  echo -e "${GREEN}✓ venv311/ vorhanden${RESET}"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Aktiviert: $(which python) ($(python -V 2>&1))${RESET}"

# ============================================================
# 4. pip / Build-Tools aktualisieren
# ============================================================
echo -e "\n${CYAN}[4/9] pip / setuptools / wheel aktualisieren...${RESET}"
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓ pip $(pip --version | awk '{print $2}')${RESET}"

# ============================================================
# 5. requirements.txt installieren (ohne torch/torchvision)
# ============================================================
echo -e "\n${CYAN}[5/9] requirements.txt installieren (ohne torch/torchvision)...${RESET}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [[ -f "$REQ_FILE" ]]; then
  TMP_REQ=$(mktemp)
  grep -v -E '^(torch|torchvision)([=<>!~[:space:]]|$)' "$REQ_FILE" > "$TMP_REQ" || true
  if [[ -s "$TMP_REQ" ]]; then
    pip install -r "$TMP_REQ"
    echo -e "${GREEN}✓ requirements.txt installiert${RESET}"
  else
    echo -e "${YELLOW}  Keine weiteren Pakete nach torch-Filter${RESET}"
  fi
  rm -f "$TMP_REQ"
else
  echo -e "${YELLOW}  requirements.txt nicht gefunden — übersprungen${RESET}"
fi

# ============================================================
# 6. PyTorch installieren
# ============================================================
if [[ "$SKIP_TORCH" == "false" ]]; then
  echo -e "\n${CYAN}[6/9] PyTorch + torchvision installieren...${RESET}"
  echo -e "${YELLOW}  (kann einige Minuten dauern)${RESET}"

  TORCH_OK=false
  if [[ -n "$WHEEL_INDEX" ]]; then
    echo -e "  ${CYAN}→ pip install torch torchvision --index-url ${WHEEL_INDEX}${RESET}"
    if pip install torch torchvision --index-url "$WHEEL_INDEX"; then
      TORCH_OK=true
    elif [[ -n "${WHEEL_FALLBACK:-}" ]]; then
      echo -e "  ${YELLOW}  Primärer Index fehlgeschlagen — Fallback: ${WHEEL_FALLBACK}${RESET}"
      if pip install torch torchvision --index-url "$WHEEL_FALLBACK"; then
        TORCH_OK=true
      fi
    fi
  else
    echo -e "  ${YELLOW}  Kein CUDA-Wheel — CPU-Version${RESET}"
    if pip install torch torchvision; then
      TORCH_OK=true
    fi
  fi

  if [[ "$TORCH_OK" == "true" ]]; then
    echo -e "${GREEN}✓ PyTorch installiert${RESET}"
    python -c "import torch; print(f'  Version: {torch.__version__}')"
    python -c "import torch; print(f'  CUDA verfügbar: {torch.cuda.is_available()}')"
  else
    echo -e "${RED}✗ PyTorch-Installation fehlgeschlagen!${RESET}"
  fi
else
  echo -e "\n${YELLOW}[6/9] PyTorch übersprungen (--no-torch)${RESET}"
fi

# ============================================================
# 7. Kompatibilitäts-Test P100 / CC 6.0
# ============================================================
echo -e "\n${CYAN}[7/9] Kompatibilitäts-Test für P100 (CC 6.0)...${RESET}"

COMPAT_PASS=0
COMPAT_WARN=0
COMPAT_FAIL=0

run_test() {
  local label="$1"
  local code="$2"
  local severity="${3:-fail}"   # fail | warn | info
  if python -c "$code" 2>/dev/null; then
    echo -e "  ${GREEN}✓ ${label}${RESET}"
    COMPAT_PASS=$((COMPAT_PASS + 1))
  else
    if [[ "$severity" == "warn" ]]; then
      echo -e "  ${YELLOW}⚠ ${label} (erwartet auf P100/CC 6.0)${RESET}"
      COMPAT_WARN=$((COMPAT_WARN + 1))
    else
      echo -e "  ${RED}✗ ${label}${RESET}"
      COMPAT_FAIL=$((COMPAT_FAIL + 1))
    fi
  fi
}

# Basis-Imports
run_test "import torch" "import torch"
run_test "import torchvision" "import torchvision"
run_test "import numpy" "import numpy"
run_test "import cv2" "import cv2"
run_test "import PIL" "import PIL"
run_test "import tensorboard" "import tensorboard"
run_test "import tqdm" "import tqdm"
run_test "import psutil" "import psutil"
run_test "import rich" "import rich"

# CUDA-Basis
run_test "torch.cuda.is_available()" \
  "import torch; assert torch.cuda.is_available(), 'CUDA nicht verfügbar'" fail

# Compute Capability prüfen und dokumentieren
python -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit(0)
cc = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f'  GPU erkannt: {name}  (CC {cc[0]}.{cc[1]})')
if cc < (7, 0):
    print(f'  → CC {cc[0]}.{cc[1]} < 7.0: torch.compile/Triton NICHT verfügbar (erwartet für P100)')
if cc < (7, 5):
    print(f'  → CC {cc[0]}.{cc[1]} < 7.5: Flash Attention NICHT verfügbar (erwartet für P100)')
if cc < (8, 0):
    print(f'  → CC {cc[0]}.{cc[1]} < 8.0: bfloat16 NICHT nativ verfügbar (float16/fp32 laufen)')
" 2>/dev/null || true

# FP16 (float16) — läuft auf P100
run_test "torch FP16 Tensor auf CUDA" "
import torch
if not torch.cuda.is_available(): raise SystemExit(0)
t = torch.randn(4, 4, dtype=torch.float16).cuda()
assert t.dtype == torch.float16
"

# bfloat16 — läuft NICHT nativ auf P100 (CC 6.0)
run_test "torch bfloat16 auf CUDA (CC ≥ 8.0 nötig)" "
import torch
if not torch.cuda.is_available(): raise SystemExit(0)
cc = torch.cuda.get_device_capability(0)
if cc < (8, 0): raise RuntimeError('CC < 8.0')
t = torch.randn(4, 4, dtype=torch.bfloat16).cuda()
" warn

# torch.compile — braucht CC ≥ 7.0
run_test "torch.compile (CC ≥ 7.0 nötig)" "
import torch
if not torch.cuda.is_available(): raise SystemExit(0)
cc = torch.cuda.get_device_capability(0)
if cc < (7, 0): raise RuntimeError('CC < 7.0')
m = torch.nn.Linear(4, 4).cuda()
cm = torch.compile(m)
_ = cm(torch.randn(2, 4).cuda())
" warn

# AMP autocast (FP16) — sollte auf P100 funktionieren
run_test "AMP autocast (float16)" "
import torch
if not torch.cuda.is_available(): raise SystemExit(0)
m = torch.nn.Linear(16, 16).cuda()
x = torch.randn(4, 16).cuda()
with torch.amp.autocast('cuda', dtype=torch.float16):
    y = m(x)
assert y.dtype == torch.float16
"

# GradScaler — AMP-Training-Kernkomponente
run_test "AMP GradScaler" "
import torch
scaler = torch.cuda.amp.GradScaler()
assert scaler is not None
"

# Simple Conv2d forward pass (Kern des VSR-Modells)
run_test "torch.nn.Conv2d forward auf CUDA" "
import torch
if not torch.cuda.is_available(): raise SystemExit(0)
conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda().half()
x = torch.randn(1, 3, 64, 64).cuda().half()
y = conv(x)
assert y.shape == (1, 16, 64, 64)
"

# Gradient Checkpointing
run_test "torch.utils.checkpoint" "
import torch
import torch.utils.checkpoint as cp
lin = torch.nn.Linear(8, 8).cuda()
x = torch.randn(2, 8, requires_grad=True).cuda()
out = cp.checkpoint(lin, x, use_reentrant=False)
out.sum().backward()
"

# ffmpeg
echo ""
if command -v ffmpeg &>/dev/null; then
  FFVER=$(ffmpeg -version 2>&1 | head -1 | grep -oE 'version [^ ]+' || echo "gefunden")
  echo -e "  ${GREEN}✓ ffmpeg ${FFVER}${RESET}"
  COMPAT_PASS=$((COMPAT_PASS + 1))
else
  echo -e "  ${RED}✗ ffmpeg fehlt — dataset_generator_v2 benötigt es${RESET}"
  echo -e "  ${YELLOW}   sudo apt install ffmpeg${RESET}"
  COMPAT_FAIL=$((COMPAT_FAIL + 1))
fi

# ============================================================
# 8. Import-Check aller .py-Dateien im Repo
# ============================================================
echo -e "\n${CYAN}[8/9] Import-Check der Projekt-.py-Dateien...${RESET}"

STDLIB="os sys re json math time argparse subprocess threading logging \
  pathlib typing collections itertools functools hashlib tempfile signal \
  shutil glob atexit errno queue random select traceback socket datetime \
  tty termios curses http concurrent io abc copy struct weakref dataclasses \
  enum contextlib warnings gc platform uuid multiprocessing"

declare -A PIP_MAP=(
  ["cv2"]="opencv-python"
  ["PIL"]="Pillow"
  ["sklearn"]="scikit-learn"
  ["skimage"]="scikit-image"
  ["yaml"]="PyYAML"
)
SKIP_AUTO=("torch2trt" "tensorrt" "pycuda" "onnxruntime_gpu")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPORTS=$(find "$REPO_ROOT" -name "*.py" 2>/dev/null \
  | xargs grep -h -E "^(import|from) [a-zA-Z_][a-zA-Z0-9_]*" 2>/dev/null \
  | grep -oE "^(import|from) [a-zA-Z_][a-zA-Z0-9_]*" \
  | awk '{print $2}' | sort -u || true)

LOCAL_PKGS="vsr_plusplus_NEU dataset_generator_v2 config category_utils \
  generation_plan interactive_selector streaming_extractor video_manager \
  utils core ice_audio_nexus ice_brain tools"

IMPORT_FAIL=0
while IFS= read -r pkg; do
  [[ -z "$pkg" ]] && continue
  # stdlib überspringen
  is_std=false
  for s in $STDLIB; do [[ "$pkg" == "$s" ]] && is_std=true && break; done
  $is_std && continue
  # lokale Module überspringen
  is_local=false
  for l in $LOCAL_PKGS; do [[ "$pkg" == "$l" ]] && is_local=true && break; done
  $is_local && continue

  if python -c "import $pkg" 2>/dev/null; then
    ver=$(python -c "import $pkg; print(getattr($pkg,'__version__','ok'))" 2>/dev/null || echo "ok")
    echo -e "  ${GREEN}✓ $pkg ($ver)${RESET}"
  else
    # skip-Liste?
    is_skip=false
    for sk in "${SKIP_AUTO[@]}"; do [[ "$pkg" == "$sk" ]] && is_skip=true && break; done
    if $is_skip; then
      echo -e "  ${YELLOW}⚠ $pkg — manuelle Installation erforderlich (übersprungen)${RESET}"
      COMPAT_WARN=$((COMPAT_WARN + 1))
      continue
    fi
    # versuche automatisch zu installieren
    pip_pkg="${PIP_MAP[$pkg]:-$pkg}"
    echo -e "  ${YELLOW}⚠ $pkg fehlt — versuche: pip install ${pip_pkg}${RESET}"
    if pip install "$pip_pkg" --quiet 2>/dev/null; then
      echo -e "  ${GREEN}  ✓ ${pip_pkg} nachinstalliert${RESET}"
    else
      echo -e "  ${RED}  ✗ ${pip_pkg} konnte nicht installiert werden${RESET}"
      IMPORT_FAIL=$((IMPORT_FAIL + 1))
    fi
  fi
done <<< "$IMPORTS"

# ============================================================
# 9. Zusammenfassung
# ============================================================
echo -e "\n${BOLD}${CYAN}[9/9] Zusammenfassung${RESET}"
echo -e "─────────────────────────────────────────────────────"
echo -e "  ${GREEN}✓ Tests bestanden : ${COMPAT_PASS}${RESET}"
echo -e "  ${YELLOW}⚠ Warnungen       : ${COMPAT_WARN} (P100/CC-6.0-Einschränkungen, erwartet)${RESET}"
echo -e "  ${RED}✗ Fehler          : $((COMPAT_FAIL + IMPORT_FAIL))${RESET}"
echo -e "─────────────────────────────────────────────────────"

if [[ $COMPAT_FAIL -eq 0 && $IMPORT_FAIL -eq 0 ]]; then
  echo -e "\n${BOLD}${GREEN}✓ venv311/ ist lauffähig!${RESET}"
  echo -e "${YELLOW}  Warnungen betreffen nur Features, die CC ≥ 7.0 benötigen (P100 = CC 6.0):${RESET}"
  echo -e "${YELLOW}  • torch.compile / Triton → deaktiviert (USE_COMPILE = False in config.py)${RESET}"
  echo -e "${YELLOW}  • bfloat16 → AMP mit float16 stattdessen nutzen (bereits so konfiguriert)${RESET}"
  echo -e "${YELLOW}  • Flash Attention → wird im Modell nicht verwendet${RESET}"
  echo ""
  echo -e "${BOLD}Aktivieren:${RESET}"
  echo -e "  ${CYAN}source ${VENV_DIR}/bin/activate${RESET}"
  echo ""
  echo -e "${BOLD}Training starten:${RESET}"
  echo -e "  ${CYAN}python vsr_plusplus_NEU/train.py${RESET}"
  echo ""
  deactivate 2>/dev/null || true
  exit 0
else
  echo -e "\n${BOLD}${RED}✗ Einige Tests fehlgeschlagen — siehe Fehlermeldungen oben.${RESET}"
  deactivate 2>/dev/null || true
  exit 1
fi
