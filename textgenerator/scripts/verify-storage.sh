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
LONGHORN_NAMESPACE="${LONGHORN_NAMESPACE:-longhorn-system}"
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

# The module is often loaded on demand when the first volume is attached, so
# a missing module is only a hint, not an error.
if [[ -d /sys/module/iscsi_tcp ]] || lsmod 2>/dev/null | grep -q '^iscsi_tcp'; then
  ok "iscsi_tcp kernel module loaded"
elif systemctl is-active --quiet iscsid 2>/dev/null; then
  ok "iscsid running (iscsi_tcp is loaded on demand)"
else
  note "iscsi_tcp not loaded yet - normally loaded automatically on first attach."
  note "Load it up front with: sudo modprobe iscsi_tcp"
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
  note "numberOfReplicas=${replicas:-<unset, global default applies>} reclaimPolicy=${reclaim:-<unset>}"

  # The Longhorn global setting "Default Replica Count" only applies when the
  # StorageClass carries NO numberOfReplicas parameter: the volume mutating
  # webhook fills it in solely when spec.numberOfReplicas == 0. A value set on
  # the class therefore always wins. Show both so the difference is obvious.
  global_replicas="$(kubectl -n "${LONGHORN_NAMESPACE}" get settings.longhorn.io \
    default-replica-count -o jsonpath='{.value}' 2>/dev/null)"
  if [[ -n "${global_replicas}" ]]; then
    note "Longhorn global default-replica-count=${global_replicas} (used ONLY when the class sets no parameter)"
  fi

  nodes="$(kubectl get nodes --no-headers 2>/dev/null | wc -l)"
  if [[ "${nodes}" -le 1 && -n "${replicas}" && "${replicas}" -gt 1 ]]; then
    warn "Single node cluster, but StorageClass ${STORAGE_CLASS} requests ${replicas} replicas."
    warn "New volumes will be created Degraded (only 1 of ${replicas} replicas schedulable)."
    if [[ -n "${global_replicas}" && "${global_replicas}" -eq 1 ]]; then
      warn "The global setting of ${global_replicas} does NOT override this - the class parameter wins."
    fi
    warn "Fix with:"
    warn "  kubectl patch storageclass ${STORAGE_CLASS} --type=merge \\"
    warn "    -p '{\"parameters\":{\"numberOfReplicas\":\"1\"}}'"
    warn "Existing volumes keep the count they were created with (see below)."
  fi
  if [[ "${reclaim}" == "Delete" ]]; then
    note "reclaimPolicy is Delete: deleting a PVC also deletes its data."
    note "build.sh --clean therefore keeps the PVCs unless --purge-data is given."
  fi
else
  fail "StorageClass '${STORAGE_CLASS}' not found - is Longhorn installed?"
fi

# Ground truth: what the existing volumes were actually created with. A
# StorageClass change is never propagated back to volumes that already exist.
if kubectl get crd volumes.longhorn.io >/dev/null 2>&1; then
  echo
  echo "== Longhorn volumes (desired replicas vs. health) =="
  kubectl -n "${LONGHORN_NAMESPACE}" get volumes.longhorn.io \
    -o custom-columns=NAME:.metadata.name,REPLICAS:.spec.numberOfReplicas,STATE:.status.state,ROBUSTNESS:.status.robustness,PVC:.status.kubernetesStatus.pvcName \
    2>/dev/null || note "no Longhorn volumes yet"
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
