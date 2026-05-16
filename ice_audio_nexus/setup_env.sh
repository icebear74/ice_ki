#!/bin/bash
set -euo pipefail

echo "== ice_audio_nexus visual Step-1 setup =="

PY_BIN=""
for candidate in python3.12 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PY_BIN="$candidate"
    break
  fi
done

if [ -z "$PY_BIN" ]; then
  echo "Python 3.11+ not found"
  exit 1
fi

if [ -d "venv" ]; then
  rm -rf venv
fi

"$PY_BIN" -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install \
  "fastapi[standard]" \
  "uvicorn[standard]" \
  jinja2 \
  python-dotenv \
  mariadb \
  numpy \
  opencv-python-headless

echo "Setup complete."
echo "Next:"
echo "  source venv/bin/activate"
echo "  uvicorn web_ui.api:app --host 0.0.0.0 --port 8765"
