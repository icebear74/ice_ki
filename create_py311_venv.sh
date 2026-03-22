#!/usr/bin/env bash
set -euo pipefail

# create_py311_venv.sh
# Erzeugt eine virtuelle Umgebung mit einer wählbaren Python-Executable.
# Usage:
#   ./create_py311_venv.sh --python /usr/bin/python3.11 --venv venv --install-requirements --install-torch
#
VENV_DIR="venv-py311"
PY_BIN=""
INSTALL_REQ=false
INSTALL_TORCH=false
FORCE=false
ASSUME_YES=false
QUIET=false

print_help() {
  cat <<EOF
Usage: $0 [options]
Options:
  --venv DIR              venv-Verzeichnis (default: venv-py311)
  --python PATH           Python-Executable to use (required)
  --install-requirements  Install requirements.txt (skips torch/torchvision lines)
  --install-torch         Try to install torch + torchvision (detects CUDA via nvidia-smi)
  --force, -f             Lösche vorhandene venv ohne Nachfrage
  -y, --yes               Automatisch mit "ja" antworten
  --quiet                 Weniger Ausgabe
  -h, --help              Zeige diese Hilfe
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV_DIR="$2"; shift 2;;
    --python) PY_BIN="$2"; shift 2;;
    --install-requirements) INSTALL_REQ=true; shift;;
    --install-torch) INSTALL_TORCH=true; shift;;
    --force|-f) FORCE=true; shift;;
    -y|--yes) ASSUME_YES=true; shift;;
    --quiet) QUIET=true; shift;;
    -h|--help) print_help; exit 0;;
    *) echo "Unknown option: $1"; print_help; exit 2;;
  esac
done

echolog() {
  if [ "$QUIET" = false ]; then
    echo -e "$@"
  fi
}

if [ -z "${PY_BIN}" ]; then
  echo "Bitte --python /pfad/zu/python3.11 angeben."
  exit 3
fi

if ! command -v "$PY_BIN" &>/dev/null; then
  echo "Die angegebene Python-Executable '$PY_BIN' wurde nicht gefunden."
  exit 4
fi

PY_VER=$($PY_BIN -c 'import sys; print("{}.{}".format(*sys.version_info[:2]))' 2>/dev/null || echo "unbekannt")
echolog "Verwende Python: $PY_BIN (Version: $PY_VER)"

# optional: enforce 3.11 if desired (nicht erzwungen, nur Hinweis)
if ! $PY_BIN -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
  echolog "Hinweis: ausgewählte Python-Version ist nicht 3.11."
  echolog "Wenn du strikt Python 3.11 möchtest, starte mit einer passenden --python-Executable."
fi

# venv handling
if [ -d "$VENV_DIR" ]; then
  if [ "$FORCE" = true ] || [ "$ASSUME_YES" = true ]; then
    echolog "Lösche vorhandene virtuelle Umgebung $VENV_DIR..."
    rm -rf "$VENV_DIR"
  else
    read -p "Verzeichnis '$VENV_DIR' existiert. Löschen und neu erstellen? (j/N): " -r
    echo
    if [[ $REPLY =~ ^[JjYy]$ ]]; then
      rm -rf "$VENV_DIR"
    else
      echolog "Verwende existierende Umgebung $VENV_DIR."
    fi
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  echolog "Erstelle venv in: $VENV_DIR"
  "$PY_BIN" -m venv "$VENV_DIR"
  echolog "✓ venv erstellt"
fi

# activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echolog "Aktiviert: $(which python) ($(python -V 2>&1))"

# upgrade pip
echolog "Upgrade pip/setuptools/wheel..."
pip install --upgrade pip setuptools wheel

# install requirements.txt (ohne torch/torchvision) falls angefragt
if [ "$INSTALL_REQ" = true ]; then
  if [ -f "requirements.txt" ]; then
    echolog "Installiere requirements.txt (ohne torch/torchvision)..."
    TMP_REQ=$(mktemp)
    grep -v -E '^(torch|torchvision)([=<>!~]|$)' requirements.txt > "$TMP_REQ" || true
    if [ -s "$TMP_REQ" ]; then
      pip install -r "$TMP_REQ"
      echolog "✓ requirements installiert"
    else
      echolog "Keine weiteren requirements nach Filter."
    fi
    rm -f "$TMP_REQ"
  else
    echolog "requirements.txt nicht gefunden."
  fi
fi

# install torch if requested (heuristisch)
if [ "$INSTALL_TORCH" = true ]; then
  echolog "Versuche PyTorch-Installation (heuristisch)..."
  if command -v nvidia-smi &>/dev/null; then
    CUDA_LINE=$(nvidia-smi | grep -i "CUDA Version" || true)
    CUDA_VER=$(echo "$CUDA_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    echolog "Gefundene CUDA-Version: ${CUDA_VER:-none}"
    if [ -n "$CUDA_VER" ]; then
      # fallback: use cu118 index for >=11.8, otherwise CPU
      major_minor=$(echo "$CUDA_VER" | awk -F. '{printf "%.1f", $1+$2/10}')
      if awk "BEGIN{exit !($major_minor >= 11.8)}"; then
        echolog "Installiere torch + torchvision (cu118 wheel index)..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
      else
        echolog "CUDA < 11.8 erkannt — installiere CPU-Version als Fallback."
        pip install torch torchvision
      fi
    else
      echolog "CUDA-Version nicht erkennbar — installiere CPU-Version."
      pip install torch torchvision
    fi
  else
    echolog "Keine NVIDIA GPU (nvidia-smi nicht gefunden) — installiere CPU-Version von PyTorch."
    pip install torch torchvision
  fi
  echolog "✓ PyTorch-Installation versucht"
fi

echolog ""
echolog "Setup abgeschlossen. Aktivieren: source $VENV_DIR/bin/activate"
echolog "Python in venv: $(python -V 2>&1)"
echolog "pip: $(pip -V 2>&1)"

deactivate 2>/dev/null || true
