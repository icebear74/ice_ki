#!/usr/bin/env bash
# Back up the textgenerator stack data from the host storage root.
#
#   sudo textgenerator/scripts/backup.sh [DESTINATION_DIR] [STORAGE_ROOT]
#
# Because all volumes are hostPath directories, this reads straight from disk -
# no running pod is required. Model weights and generated images are excluded:
# they are large and re-downloadable/re-creatable. Set INCLUDE_MODELS=1 to
# include them anyway.
set -euo pipefail

DEST="${1:-/var/backups/textgenerator}"
STORAGE_ROOT="${2:-/mnt/aistack}"
INCLUDE_MODELS="${INCLUDE_MODELS:-0}"
KEEP="${KEEP:-7}"

if [[ ! -d "${STORAGE_ROOT}" ]]; then
  echo "Storage root ${STORAGE_ROOT} does not exist." >&2
  exit 1
fi

mkdir -p "${DEST}"
stamp="$(date +%Y%m%d-%H%M%S)"
archive="${DEST}/textgenerator-${stamp}.tar.gz"

excludes=()
if [[ "${INCLUDE_MODELS}" != "1" ]]; then
  excludes+=(
    --exclude="./oobabooga/models"
    --exclude="./comfyui/models"
    --exclude="./comfyui/output"
  )
  echo "Excluding model weights and ComfyUI output (INCLUDE_MODELS=1 to keep them)."
fi

echo "Archiving ${STORAGE_ROOT} -> ${archive}"
# Backing up a live directory tree can catch a file mid-write. Stop the
# workloads first for a fully consistent snapshot:
#   kubectl -n ai-stack scale deploy --all --replicas=0
tar -czf "${archive}" -C "${STORAGE_ROOT}" "${excludes[@]}" .
chmod 0600 "${archive}"

echo "Wrote $(du -h "${archive}" | cut -f1) to ${archive}"

# Rotate: keep the newest ${KEEP} archives.
mapfile -t old < <(ls -1t "${DEST}"/textgenerator-*.tar.gz 2>/dev/null | tail -n "+$((KEEP + 1))")
for f in "${old[@]:-}"; do
  [[ -n "${f}" ]] || continue
  echo "Removing old backup ${f}"
  rm -f "${f}"
done

echo
echo "Copy the archive off this machine - hostPath storage has no redundancy."
echo "Restore with:"
echo "  kubectl -n ai-stack scale deploy --all --replicas=0"
echo "  sudo tar -xzf ${archive} -C ${STORAGE_ROOT}"
echo "  sudo textgenerator/scripts/prepare-host-dirs.sh ${STORAGE_ROOT}"
echo "  kubectl -n ai-stack scale deploy --all --replicas=1"
