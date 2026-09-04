#!/usr/bin/env bash
# Back up the small, valuable parts of the stack: SillyTavern config/chats,
# character cards and extracted person profiles.
#
# Model files, checkpoints and generated images are EXCLUDED on purpose -
# they are large and reproducible.
#
#   sudo textgenerator/scripts/backup.sh [TARGET_DIR] [STORAGE_ROOT]
set -euo pipefail

TARGET_DIR="${1:-/var/backups/k3s-ai-stack}"
STORAGE_ROOT="${2:-/var/lib/k3s-ai-stack}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="${TARGET_DIR}/textgenerator-${STAMP}.tar.gz"

if [[ ! -d "${STORAGE_ROOT}" ]]; then
  echo "Storage root ${STORAGE_ROOT} does not exist." >&2
  exit 1
fi

mkdir -p "${TARGET_DIR}"

tar czf "${ARCHIVE}" \
  --exclude="oobabooga/models" \
  --exclude="comfyui/models" \
  --exclude="comfyui/output" \
  -C "${STORAGE_ROOT}" \
  sillytavern shared extractor \
  $( [[ -d "${STORAGE_ROOT}/oobabooga/character" ]] && echo "oobabooga/character" ) \
  $( [[ -d "${STORAGE_ROOT}/comfyui/workflows" ]] && echo "comfyui/workflows" )

echo "Backup written: ${ARCHIVE}"
echo "Restore with: tar xzf ${ARCHIVE} -C ${STORAGE_ROOT}"
echo "Note: local storage does not protect against disk failure - copy the"
echo "archive to a different machine or external disk."
