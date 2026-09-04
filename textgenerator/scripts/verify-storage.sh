#!/usr/bin/env bash
# Check the host and cluster prerequisites for the Longhorn backed PVCs of the
# textgenerator stack. Replaces the old prepare-host-dirs.sh - no host
# directories have to be created any more, Longhorn provisions the volumes.
#
#   textgenerator/scripts/verify-storage.sh
#
# Exits non-zero if a hard requirement is missing.
set -uo pipefail

NAMESPACE="${NAMESPACE:-ai-stack}"
STORAGE_CLASS="${STORAGE_CLASS:-longhorn}"
rc=0

note() { printf '  %s\n' "$*"; }
ok()   { printf '[ ok ] %s\n' "$*"; }
warn() { printf '[warn] %s\n' "$*"; }
fail() { printf '[fail] %s\n' "$*"; rc=1; }

echo "== Host packages =="
# open-iscsi is required for every Longhorn volume, nfs-common only for
# ReadWriteMany volumes (shared-characters-pvc uses RWX).
if command -v iscsiadm >/dev/null 2>&1; then
  ok "open-iscsi present ($(command -v iscsiadm))"
else
  fail "iscsiadm not found - install with: sudo apt-get install -y open-iscsi"
fi

if command -v mount.nfs >/dev/null 2>&1 || command -v mount.nfs4 >/dev/null 2>&1; then
  ok "NFS client present (needed for the ReadWriteMany shared volume)"
else
  fail "mount.nfs not found - install with: sudo apt-get install -y nfs-common"
fi

if lsmod 2>/dev/null | grep -q '^iscsi_tcp'; then
  ok "iscsi_tcp kernel module loaded"
else
  warn "iscsi_tcp module not loaded - run: sudo modprobe iscsi_tcp"
fi

echo
echo "== Cluster =="
if ! command -v kubectl >/dev/null 2>&1; then
  warn "kubectl not found - skipping cluster checks"
  exit "${rc}"
fi

if ! kubectl version >/dev/null 2>&1; then
  warn "cannot reach the cluster - skipping cluster checks"
  exit "${rc}"
fi

if kubectl get storageclass "${STORAGE_CLASS}" >/dev/null 2>&1; then
  ok "StorageClass ${STORAGE_CLASS} installed"
  replicas="$(kubectl get storageclass "${STORAGE_CLASS}" \
    -o jsonpath='{.parameters.numberOfReplicas}' 2>/dev/null)"
  reclaim="$(kubectl get storageclass "${STORAGE_CLASS}" \
    -o jsonpath='{.reclaimPolicy}' 2>/dev/null)"
  note "numberOfReplicas=${replicas:-<unset>} reclaimPolicy=${reclaim:-<unset>}"

  nodes="$(kubectl get nodes --no-headers 2>/dev/null | wc -l)"
  if [[ "${nodes}" -le 1 && -n "${replicas}" && "${replicas}" -gt 1 ]]; then
    warn "single node cluster but numberOfReplicas=${replicas} - volumes will stay Degraded."
    warn "Set the replica count of the ${STORAGE_CLASS} class to 1."
  fi
  if [[ "${reclaim}" == "Delete" ]]; then
    note "reclaimPolicy is Delete: deleting a PVC also deletes its data."
    note "build.sh --clean therefore keeps the PVCs unless --purge-data is given."
  fi
else
  fail "StorageClass '${STORAGE_CLASS}' not found - is Longhorn installed?"
fi

echo
echo "== PersistentVolumeClaims in ${NAMESPACE} =="
if kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  kubectl -n "${NAMESPACE}" get pvc 2>/dev/null || true
  pending="$(kubectl -n "${NAMESPACE}" get pvc --no-headers 2>/dev/null | awk '$2!="Bound"' | wc -l)"
  if [[ "${pending}" -gt 0 ]]; then
    warn "${pending} PVC(s) not Bound - check: kubectl -n ${NAMESPACE} describe pvc"
  fi
else
  note "namespace ${NAMESPACE} does not exist yet"
fi

echo
echo "Longhorn replicates inside the cluster - it is NOT a backup."
echo "Use textgenerator/scripts/backup.sh and/or a Longhorn backup target."
exit "${rc}"
