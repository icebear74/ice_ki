#!/usr/bin/env bash
# Verify that the node and K3s can actually run CUDA workloads on the
# Tesla P100 BEFORE deploying the stack.
#
#   textgenerator/scripts/verify-gpu.sh
set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-stack}"
# Keep this in sync with the CUDA generation used by the Oobabooga image.
CUDA_TEST_IMAGE="${CUDA_TEST_IMAGE:-nvidia/cuda:11.8.0-base-ubuntu22.04}"

echo "== Host driver =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found - install the NVIDIA driver and container toolkit first." >&2
  exit 1
fi

echo
echo "== K3s RuntimeClass 'nvidia' =="
kubectl get runtimeclass nvidia

echo
echo "== Advertised GPU capacity =="
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.capacity.nvidia\.com/gpu}{"\n"}{end}'

echo
echo "== In-cluster CUDA smoke test =="
kubectl -n "${NAMESPACE}" delete pod gpu-smoke-test --ignore-not-found >/dev/null
kubectl -n "${NAMESPACE}" run gpu-smoke-test \
  --image="${CUDA_TEST_IMAGE}" \
  --restart=Never \
  --overrides='{"spec":{"runtimeClassName":"nvidia","containers":[{"name":"gpu-smoke-test","image":"'"${CUDA_TEST_IMAGE}"'","command":["nvidia-smi"],"resources":{"limits":{"nvidia.com/gpu":1}}}]}}'
kubectl -n "${NAMESPACE}" wait --for=condition=Ready pod/gpu-smoke-test --timeout=300s || true
kubectl -n "${NAMESPACE}" logs pod/gpu-smoke-test || true
kubectl -n "${NAMESPACE}" delete pod gpu-smoke-test --ignore-not-found

cat <<'EOF'

Reminder: the Tesla P100 is a Pascal card (compute capability 6.0). If the
Oobabooga pod later logs "no kernel image is available for execution on the
device" or "CUDA error: no kernel image", the container image was built
without sm_60 support - pin an older CUDA 11.x based image tag in
textgenerator/k8s/02-oobabooga.yaml.
EOF
