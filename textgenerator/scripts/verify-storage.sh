#!/usr/bin/env bash
# Check the host storage prerequisites for the hostPath PersistentVolumes of
# the textgenerator stack.
#
#   textgenerator/scripts/verify-storage.sh [STORAGE_ROOT]
#
# Exits non-zero if a hard requirement is missing.
set -uo pipefail

STORAGE_ROOT="${1:-/mnt/aistack}"
NAMESPACE="${NAMESPACE:-ai-stack}"
OWNER_UID="${OWNER_UID:-1000}"
OWNER_GID="${OWNER_GID:-1000}"
# Minimum free space to consider the volume usable at all (GiB). Model files
# alone are easily tens of GiB.
MIN_FREE_GIB="${MIN_FREE_GIB:-50}"
rc=0

note() { printf '  %s\n' "$*"; }
ok()   { printf '[ ok ] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
fail() { printf '[fail] %s\n' "$*"; rc=1; }

# Keep in sync with scripts/prepare-host-dirs.sh and the hostPath values in
# k8s/01-storage.yaml.
DIRS=(
  "sillytavern/config"
  "sillytavern/data"
  "sillytavern/data/default-user/characters"
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

echo "== Storage root ${STORAGE_ROOT} =="
if [[ ! -d "${STORAGE_ROOT}" ]]; then
  fail "${STORAGE_ROOT} does not exist - run: sudo scripts/prepare-host-dirs.sh"
  exit "${rc}"
fi
ok "${STORAGE_ROOT} exists"

if mountpoint -q "${STORAGE_ROOT}" 2>/dev/null; then
  ok "${STORAGE_ROOT} is a dedicated mount point"
else
  note "${STORAGE_ROOT} is not a separate mount - it shares the parent filesystem."
fi

free_gib="$(df -BG --output=avail "${STORAGE_ROOT}" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ -n "${free_gib}" ]]; then
  if [[ "${free_gib}" -lt "${MIN_FREE_GIB}" ]]; then
    warn "only ${free_gib}Gi free - model files need considerably more."
  else
    ok "${free_gib}Gi free"
  fi
fi

echo
echo "== Directories and ownership =="
# hostPath volumes ignore fsGroup, so the ownership on disk is what decides
# whether the containers (uid/gid 1000) can write.
missing=0
wrong_owner=0
for dir in "${DIRS[@]}"; do
  target="${STORAGE_ROOT}/${dir}"
  if [[ ! -d "${target}" ]]; then
    echo "  missing: ${target}"
    missing=$((missing + 1))
    continue
  fi
  owner="$(stat -c '%u:%g' "${target}" 2>/dev/null)"
  if [[ "${owner}" != "${OWNER_UID}:${OWNER_GID}" ]]; then
    echo "  wrong owner (${owner}, expected ${OWNER_UID}:${OWNER_GID}): ${target}"
    wrong_owner=$((wrong_owner + 1))
  fi
done

if [[ "${missing}" -eq 0 && "${wrong_owner}" -eq 0 ]]; then
  ok "all ${#DIRS[@]} directories present and owned by ${OWNER_UID}:${OWNER_GID}"
else
  fail "${missing} missing, ${wrong_owner} with wrong ownership"
  note "Fix with: sudo textgenerator/scripts/prepare-host-dirs.sh ${STORAGE_ROOT}"
fi

echo
echo "== Cluster =="
if ! command -v kubectl >/dev/null 2>&1 || ! kubectl version >/dev/null 2>&1; then
  warn "cannot reach the cluster - skipping cluster checks"
  exit "${rc}"
fi

if kubectl get storageclass textgen-hostpath >/dev/null 2>&1; then
  ok "StorageClass textgen-hostpath exists"
else
  note "StorageClass textgen-hostpath not applied yet (kubectl apply -k textgenerator/k8s)"
fi

echo
echo "== PersistentVolumes =="
kubectl get pv -l app.kubernetes.io/part-of=textgenerator 2>/dev/null || true

echo
echo "== PersistentVolumeClaims in ${NAMESPACE} =="
if kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl -n "${NAMESPACE}" get pvc 2>/dev/null || true
  # jsonpath rather than parsing the table: no header/column surprises.
  pending="$(kubectl -n "${NAMESPACE}" get pvc \
    -o jsonpath='{range .items[?(@.status.phase!="Bound")]}{.metadata.name}{" "}{end}' 2>/dev/null)"
  if [[ -n "${pending// /}" ]]; then
    warn "PVC(s) not Bound: ${pending}"
    note "Inspect with: textgenerator/scripts/diagnose.sh"
  fi
else
  note "namespace ${NAMESPACE} does not exist yet"
fi

echo
echo "Local storage does not protect against disk failure - copy the backups"
echo "produced by textgenerator/scripts/backup.sh to another machine."
exit "${rc}"
