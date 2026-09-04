#!/usr/bin/env bash
# Prepare the host directories used by the hostPath PersistentVolumes of the
# textgenerator stack. Run once on the K3s node before `kubectl apply -k`.
#
#   sudo textgenerator/scripts/prepare-host-dirs.sh [STORAGE_ROOT]
set -euo pipefail

STORAGE_ROOT="${1:-/var/lib/k3s-ai-stack}"
# uid/gid used by the SillyTavern ("node") and extractor containers.
OWNER_UID="${OWNER_UID:-1000}"
OWNER_GID="${OWNER_GID:-1000}"

DIRS=(
  "sillytavern/config"
  "sillytavern/data"
  # SillyTavern creates its default user directory on first start. The
  # characters/ directory below is a mount point for the shared character
  # PVC - it must exist beforehand, otherwise the kubelet creates the parent
  # directories as root and SillyTavern (uid 1000) cannot write into them.
  "sillytavern/data/default-user"
  "sillytavern/data/default-user/characters"
  # Writable application directories relative to /home/node/app.
  "sillytavern/backups"
  "sillytavern/plugins"
  "sillytavern/extensions"
  "oobabooga/models"
  "oobabooga/character/characters"
  "oobabooga/character/loras"
  "shared/characters"
  "extractor/profiles"
  "extractor/raw"
  "comfyui/models"
  "comfyui/input"
  "comfyui/output"
  "comfyui/user"
  "comfyui/workflows"
)

if [[ "$(id -u)" -ne 0 ]]; then
  echo "This script must run as root (it creates directories under ${STORAGE_ROOT})." >&2
  exit 1
fi

for dir in "${DIRS[@]}"; do
  target="${STORAGE_ROOT}/${dir}"
  mkdir -p "${target}"
  chown "${OWNER_UID}:${OWNER_GID}" "${target}"
  chmod 0775 "${target}"
  echo "prepared ${target}"
done

echo
echo "Storage root ready: ${STORAGE_ROOT}"
echo "If you changed the root, update the hostPath values in textgenerator/k8s/01-storage.yaml."
