#!/bin/bash
set -euo pipefail

echo "== ice_audio_nexus visual Step-1 setup =="
OPENCV_PYTHON_PACKAGE="${OPENCV_PYTHON_PACKAGE:-opencv-python-headless}"

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
  "$OPENCV_PYTHON_PACKAGE"

echo
echo "== OpenCV CUDA diagnostics =="
python - <<'PY'
import re
import sys

try:
    import cv2
except Exception as exc:  # noqa: BLE001
    print(f"[ERROR] OpenCV import failed: {exc}")
    sys.exit(1)

print(f"OpenCV version: {cv2.__version__}")
build = cv2.getBuildInformation()
cuda_build = bool(re.search(r"NVIDIA CUDA:\s+YES", build))
cudnn_build = bool(re.search(r"cuDNN:\s+YES", build))
print(f"CUDA build enabled: {cuda_build}")
print(f"cuDNN build enabled: {cudnn_build}")

cuda_mod = hasattr(cv2, "cuda")
print(f"cv2.cuda module available: {cuda_mod}")
count = 0
if cuda_mod:
    try:
        count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as exc:  # noqa: BLE001
        print(f"CUDA probe error: {exc}")
print(f"CUDA devices visible to OpenCV: {count}")
if count > 0 and cuda_mod:
    for idx in range(count):
        try:
            name = cv2.cuda.DeviceInfo(idx).name()
        except Exception:  # noqa: BLE001
            name = "unknown"
        print(f"  - GPU[{idx}]: {name}")

if not cuda_build:
    print(
        "[WARN] This OpenCV build has no CUDA DNN support. "
        "Scanner falls back to CPU. Install a CUDA-enabled OpenCV build "
        "and set OPENCV_PYTHON_PACKAGE accordingly."
    )
PY

echo "Setup complete."
echo "Next:"
echo "  source venv/bin/activate"
echo "  uvicorn web_ui.api:app --host 0.0.0.0 --port 8765"
echo "  python -m processor.scanner --diagnose-opencv"
