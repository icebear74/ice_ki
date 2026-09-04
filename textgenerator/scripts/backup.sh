#!/usr/bin/env bash
# Back up the small, valuable parts of the stack: SillyTavern config/chats,
# shared character cards and extracted person profiles.
#
# The data lives in Longhorn volumes, so it is not reachable through a host
# directory any more. This script streams a tar archive out of the running
# pods that already have the volumes mounted.
#
# Model files, checkpoints and generated images are EXCLUDED on purpose -
# they are large and reproducible.
#
#   textgenerator/scripts/backup.sh [TARGET_DIR]
#
# For a scheduled, application consistent backup of the whole volume set,
# configure a Longhorn backup target (S3 or NFS) and recurring snapshots in
# the Longhorn UI instead - see textgenerator/README.md.
set -euo pipefail

TARGET_DIR="${1:-/var/backups/k3s-ai-stack}"
NAMESPACE="${NAMESPACE:-ai-stack}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${TARGET_DIR}/textgenerator-${STAMP}"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl not found." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# dump <name> <deployment> <container> <parent-dir> <entry> [entry...]
dump() {
  local name="$1" deploy="$2" container="$3" parent="$4"
  shift 4
  local archive="${OUT_DIR}/${name}.tar.gz"

  if ! kubectl -n "${NAMESPACE}" get "deployment/${deploy}" >/dev/null 2>&1; then
    echo "skip ${name}: deployment/${deploy} not found"
    return 0
  fi

  echo "backing up ${name} from deployment/${deploy} ..."
  if kubectl -n "${NAMESPACE}" exec "deployment/${deploy}" -c "${container}" -- \
      tar czf - -C "${parent}" "$@" > "${archive}"; then
    echo "  -> ${archive}"
  else
    echo "  !! failed for ${name}" >&2
    rm -f "${archive}"
    return 1
  fi
}

dump sillytavern sillytavern sillytavern /home/node/app config data
dump shared-characters sillytavern sillytavern /home/node/app/data/default-user characters
dump extractor character-extractor character-extractor /data extractor

echo
echo "Backup written: ${OUT_DIR}"
echo "Restore example (SillyTavern):"
echo "  kubectl -n ${NAMESPACE} exec -i deployment/sillytavern -c sillytavern -- \\"
echo "    tar xzf - -C /home/node/app < ${OUT_DIR}/sillytavern.tar.gz"
echo
echo "Longhorn replicas protect against a single disk failing, not against"
echo "deletion or host loss - copy this directory to another machine."
