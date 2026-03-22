#!/usr/bin/env bash
# tools/check_imports.sh
# Versucht, alle Core-Pakete des VSR++-Projekts zu importieren und gibt
# Version und Erfolg/Fehler aus.  Muss innerhalb der aktivierten venv laufen.
#
# Aufruf:
#   source venv/bin/activate && bash tools/check_imports.sh

set -euo pipefail

GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
CYAN='\033[96m'
RESET='\033[0m'

echo -e "${CYAN}========================================${RESET}"
echo -e "${CYAN}  VSR++ Import-Check${RESET}"
echo -e "${CYAN}========================================${RESET}"
echo -e "  Python: $(python --version 2>&1)"
echo -e "  pip:    $(pip --version 2>&1)"
echo -e "${CYAN}----------------------------------------${RESET}"

ALL_OK=true

check_pkg() {
  local pkg="$1"
  local import_name="${2:-$pkg}"
  if python - <<EOF 2>/dev/null
import $import_name
ver = getattr($import_name, '__version__', None)
print(ver if ver else 'ok')
EOF
  then
    local ver
    ver=$(python -c "import $import_name; print(getattr($import_name, '__version__', 'ok'))" 2>/dev/null || echo "ok")
    echo -e "  ${GREEN}✓ ${pkg} (${ver})${RESET}"
  else
    echo -e "  ${RED}✗ ${pkg} — nicht installiert oder Importfehler${RESET}"
    ALL_OK=false
  fi
}

# Core ML
check_pkg "torch"
check_pkg "torchvision"

# Image / numerics
check_pkg "numpy"
check_pkg "cv2" "cv2"
check_pkg "Pillow" "PIL"

# Training utilities
check_pkg "tensorboard"
check_pkg "tqdm"

# System / monitoring
check_pkg "psutil"

echo -e "${CYAN}----------------------------------------${RESET}"
if [ "$ALL_OK" = true ]; then
  echo -e "${GREEN}✓ Alle Pakete erfolgreich importiert.${RESET}"
else
  echo -e "${RED}⚠ Einige Pakete fehlen. Bitte setup_env.sh erneut ausführen oder manuell installieren.${RESET}"
  echo -e "${YELLOW}  Tipp: pip install <paketname>${RESET}"
fi
echo -e "${CYAN}========================================${RESET}"
