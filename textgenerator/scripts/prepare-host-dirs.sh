#!/usr/bin/env bash
# Prepare the host directories used by the hostPath PersistentVolumes of the
# textgenerator stack. Run once on the K3s node before `kubectl apply -k`
# (scripts/build.sh calls this automatically).
#
#   sudo textgenerator/scripts/prepare-host-dirs.sh [STORAGE_ROOT]
#
# Kubernetes does NOT apply fsGroup to hostPath volumes, so the ownership of
# these directories is what decides whether the containers can write. They all
# run as uid/gid 1000 (SillyTavern via PUID/PGID, the extractor via its
# securityContext).
#
# If you change STORAGE_ROOT you must also change the hostPath values in
# textgenerator/k8s/01-storage.yaml - they are not templated.
set -euo pipefail

STORAGE_ROOT="${1:-/mnt/aistack}"
# uid/gid used by the SillyTavern ("node") and extractor containers.
OWNER_UID="${OWNER_UID:-1000}"
OWNER_GID="${OWNER_GID:-1000}"

# Every directory is listed explicitly, including the intermediate ones:
# `mkdir -p a/b` creates "a" as root, and only the paths named here get
# chowned afterwards.
DIRS=(
  # --- SillyTavern -------------------------------------------------------
  "sillytavern"
  "sillytavern/config"
  "sillytavern/data"
  # SillyTavern creates its default user directory on first start. The
  # characters/ directory below is the mount point of the shared character
  # PVC - it must exist beforehand, otherwise the kubelet creates the parent
  # directories as root and SillyTavern (uid 1000) cannot write its user data.
  "sillytavern/data/default-user"
  "sillytavern/data/default-user/characters"
  # Writable application directories relative to /home/node/app.
  "sillytavern/backups"
  "sillytavern/plugins"
  "sillytavern/extensions"
  # --- Oobabooga ---------------------------------------------------------
  "oobabooga"
  "oobabooga/models"
  "oobabooga/character"
  "oobabooga/character/characters"
  "oobabooga/character/loras"
  # --- Shared between SillyTavern and the extractor ----------------------
  "shared"
  "shared/characters"
  # --- Character extractor ----------------------------------------------
  "extractor"
  "extractor/profiles"
  "extractor/raw"
  # --- Optional ComfyUI --------------------------------------------------
  "comfyui"
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

if [[ ! -d "${STORAGE_ROOT}" ]]; then
  echo "Storage root ${STORAGE_ROOT} does not exist." >&2
  echo "Create and mount the data volume there first, then re-run this script." >&2
  exit 1
fi

# Warn if the root is on the OS filesystem rather than a dedicated mount - the
# stack can fill it up with model files otherwise.
if ! mountpoint -q "${STORAGE_ROOT}" 2>/dev/null; then
  echo "note: ${STORAGE_ROOT} is not a separate mount point - make sure the"
  echo "      underlying filesystem has room for the model files."
fi

mkdir -p "${STORAGE_ROOT}"
chown "${OWNER_UID}:${OWNER_GID}" "${STORAGE_ROOT}"
chmod 0775 "${STORAGE_ROOT}"

for dir in "${DIRS[@]}"; do
  target="${STORAGE_ROOT}/${dir}"
  mkdir -p "${target}"
  chown "${OWNER_UID}:${OWNER_GID}" "${target}"
  chmod 0775 "${target}"
  echo "prepared ${target}"
done

echo
echo "Storage root ready: ${STORAGE_ROOT}"
df -h "${STORAGE_ROOT}" | tail -1 | awk '{print "Free space: " $4 " of " $2}'
echo
echo "If you changed the root, update the hostPath values in"
echo "textgenerator/k8s/01-storage.yaml as well."
