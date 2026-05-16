#!/bin/bash
set -euo pipefail

echo "== ice_audio_nexus visual Step-1 setup (Torch inference stack) =="

PY_BIN=""
for candidate in python3.11 python3.12 python3; do
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

TORCH_INDEX_URL=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -iE "CUDA Version" | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  CUDA_MAJOR=$(echo "${CUDA_VERSION:-0.0}" | cut -d'.' -f1)
  CUDA_MINOR=$(echo "${CUDA_VERSION:-0.0}" | cut -d'.' -f2)
  if [ "$CUDA_MAJOR" -ge 13 ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
  elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
  elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
  elif [ "$CUDA_MAJOR" -gt 11 ] || { [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; }; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
  fi
fi

if [ -n "$TORCH_INDEX_URL" ]; then
  echo "Installing torch/torchvision from $TORCH_INDEX_URL"
  pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
  echo "Installing CPU torch/torchvision wheels"
  pip install torch torchvision
fi

pip install \
  "fastapi[standard]" \
  "uvicorn[standard]" \
  jinja2 \
  python-dotenv \
  mariadb \
  numpy \
  opencv-python-headless \
  facenet-pytorch \
  huggingface_hub

echo
echo "== Torch CUDA diagnostics =="
python - <<'PY'
import os
import sys
from pathlib import Path

try:
    import torch
    import cv2
    import facenet_pytorch
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] Dependency import failed: {exc}")
    sys.exit(1)

print(f"Python: {sys.version.split()[0]}")
print(f"Torch version: {torch.__version__}")
print(f"Torch CUDA available: {torch.cuda.is_available()}")
print(f"Torch CUDA version: {torch.version.cuda}")
count = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Torch visible CUDA devices: {count}")
for idx in range(count):
    try:
        print(f"  - GPU[{idx}]: {torch.cuda.get_device_name(idx)}")
    except Exception:  # noqa: BLE001
        print(f"  - GPU[{idx}]: unknown")

selected = "cpu"
preferred_id = int(os.getenv("FACE_GPU_DEVICE_ID", "0"))
if torch.cuda.is_available() and count > 0:
    selected_idx = preferred_id if 0 <= preferred_id < count else 0
    selected = f"cuda:{selected_idx}"
print(f"Selected scanner device (FACE_GPU_DEVICE_ID={preferred_id}): {selected}")

print(f"OpenCV version (I/O only): {cv2.__version__}")
print(f"facenet-pytorch version: {getattr(facenet_pytorch, '__version__', 'unknown')}")

face_data_dir = Path(os.getenv("FACE_DATA_DIR", "data/faces")).resolve()
models_dir = Path(os.getenv("FACE_MODELS_DIR", str(face_data_dir / "models"))).resolve()
print(f"Torch model cache root: {models_dir}")
print(f"  TORCH_HOME: {models_dir / 'torch_home'}")
print(f"  HF_HOME:    {models_dir / 'huggingface'}")
PY

echo
echo "Setup complete."
echo "Next:"
echo "  source venv/bin/activate"
echo "  uvicorn web_ui.api:app --host 0.0.0.0 --port 8765"
echo "  python -m processor.scanner --diagnose-torch"
