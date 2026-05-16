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

# ---------------------------------------------------------------------------
# Detect GPU compute capability to pick a compatible PyTorch wheel.
#
# Pascal GPUs (CC 6.0 = Tesla P100, CC 6.1 = Tesla P4) are NOT supported by
# PyTorch wheels built for CUDA 12.4+ (cu124/cu128).  The last wheel index
# that carries sm_60/sm_61 kernels is cu121, pinned to torch 2.4.x which is
# the last upstream release shipping Pascal PTX/cubin in the cu121 build.
#
# Modern GPUs (CC >= 7.0) can use the latest cu128/cu124 wheels without
# restriction.
# ---------------------------------------------------------------------------

TORCH_INDEX_URL=""
TORCH_VERSION_PIN=""   # optional "==x.y.z" version pin (for Pascal compat)
HAS_PASCAL=false       # any GPU with CC < 7.0 detected?

if command -v nvidia-smi >/dev/null 2>&1; then
  # Read driver-reported CUDA version (upper bound supported by driver)
  CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -iE "CUDA Version" | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
  CUDA_MAJOR=$(echo "${CUDA_VERSION:-0.0}" | cut -d'.' -f1)
  CUDA_MINOR=$(echo "${CUDA_VERSION:-0.0}" | cut -d'.' -f2)

  echo "Detected CUDA driver version: ${CUDA_VERSION:-unknown}"
  echo "Scanning GPU compute capabilities ..."

  # Probe per-GPU CC via nvidia-smi (available on all modern drivers)
  while IFS= read -r cc_line; do
    [[ -z "$cc_line" ]] && continue
    CC_MAJOR=$(echo "$cc_line" | cut -d'.' -f1)
    echo "  GPU compute capability: $cc_line"
    if [ "$CC_MAJOR" -lt 7 ]; then
      HAS_PASCAL=true
      echo "  → Pascal/Maxwell GPU detected (CC $cc_line, requires cu121 wheel)"
    fi
  done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)

  if [ "$HAS_PASCAL" = true ]; then
    # Force cu121 + pin to last torch version that shipped sm_60/sm_61 kernels.
    # torch 2.4.1+cu121 is the last release with confirmed Pascal SM support.
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    TORCH_VERSION_PIN="==2.4.1"
    echo "Pascal GPU present → using torch${TORCH_VERSION_PIN}+cu121 (supports sm_60 / sm_61)"
  elif [ "$CUDA_MAJOR" -ge 13 ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    echo "CUDA $CUDA_VERSION → using torch cu128"
  elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    echo "CUDA $CUDA_VERSION → using torch cu124"
  elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    echo "CUDA $CUDA_VERSION → using torch cu121"
  elif [ "$CUDA_MAJOR" -gt 11 ] || { [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; }; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    echo "CUDA $CUDA_VERSION → using torch cu118"
  fi
fi

if [ -n "$TORCH_INDEX_URL" ]; then
  echo "Installing torch${TORCH_VERSION_PIN} + torchvision from $TORCH_INDEX_URL"
  pip install "torch${TORCH_VERSION_PIN}" torchvision --index-url "$TORCH_INDEX_URL"
else
  echo "No NVIDIA GPU detected — installing CPU torch/torchvision wheels"
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
compiled_ccs: set[int] = set()
try:
    compiled_ccs = {int(a) for a in torch.cuda.get_arch_list()}
except Exception:  # noqa: BLE001
    pass
if compiled_ccs:
    print(f"Wheel compiled for: {', '.join(f'sm_{c}' for c in sorted(compiled_ccs))}")

for idx in range(count):
    try:
        name = torch.cuda.get_device_name(idx)
        cc = torch.cuda.get_device_capability(idx)
        cc_int = cc[0] * 10 + cc[1]
        if compiled_ccs:
            compat = "✓ compatible" if any(wcc <= cc_int for wcc in compiled_ccs) else "✗ INCOMPATIBLE with this torch wheel"
        else:
            compat = "(compatibility unknown)"
        print(f"  - GPU[{idx}]: {name}  CC {cc[0]}.{cc[1]}  {compat}")
    except Exception:  # noqa: BLE001
        print(f"  - GPU[{idx}]: unknown")

selected = "cpu"
preferred_id = int(os.getenv("FACE_GPU_DEVICE_ID", "0"))
if torch.cuda.is_available() and count > 0:
    selected_idx = preferred_id if 0 <= preferred_id < count else 0
    cc = torch.cuda.get_device_capability(selected_idx)
    cc_int = cc[0] * 10 + cc[1]
    if compiled_ccs and not any(wcc <= cc_int for wcc in compiled_ccs):
        selected = f"cpu (GPU[{selected_idx}] CC {cc[0]}.{cc[1]} incompatible with wheel)"
    else:
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
