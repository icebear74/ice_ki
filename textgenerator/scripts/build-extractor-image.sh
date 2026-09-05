#!/usr/bin/env bash
# Build the character-extractor image and import it into the K3s image store.
#
# The image is local-only (it exists in no registry), which is why the
# deployment uses `imagePullPolicy: Never`. Without this step the pod fails
# with ErrImagePull / ErrImageNeverPull.
#
#   sudo textgenerator/scripts/build-extractor-image.sh [TAG]
set -euo pipefail

TAG="${1:-ice-ki/character-extractor:0.1.0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="$(cd -- "${SCRIPT_DIR}/../extractor" && pwd)"
NAMESPACE="${NAMESPACE:-ai-stack}"

if command -v docker >/dev/null 2>&1; then
  echo "Building ${TAG} with docker ..."
  docker build -t "${TAG}" "${CONTEXT_DIR}"
  echo "Importing into the K3s image store ..."
  docker save "${TAG}" | k3s ctr images import -
elif command -v nerdctl >/dev/null 2>&1; then
  # nerdctl can build straight into the k8s.io containerd namespace.
  echo "Building ${TAG} with nerdctl ..."
  nerdctl --namespace k8s.io build -t "${TAG}" "${CONTEXT_DIR}"
else
  echo "Neither docker nor nerdctl found - install one of them to build the image." >&2
  exit 1
fi

echo
echo "Imported images matching character-extractor:"
k3s ctr images ls -q | grep -F "character-extractor" || true

echo
# On a fresh cluster the deployment does not exist yet - the image simply has
# to be in the image store before `kubectl apply -k` runs. Only restart an
# already running deployment so it picks up the new image.
if kubectl -n "${NAMESPACE}" get deploy/character-extractor >/dev/null 2>&1; then
  echo "Restarting the deployment ..."
  kubectl -n "${NAMESPACE}" rollout restart deploy/character-extractor
  kubectl -n "${NAMESPACE}" rollout status deploy/character-extractor --timeout=180s
else
  echo "deploy/character-extractor does not exist yet - skipping the restart."
  echo "Deploy the stack with: kubectl apply -k textgenerator/k8s"
fi
