#!/usr/bin/env bash
# Collect everything needed to diagnose a stuck textgenerator pod - especially
# pods stuck in Pending/ContainerCreating because a hostPath volume cannot be
# bound or written to.
#
#   textgenerator/scripts/diagnose.sh
#
# Read the output top down: the first section that shows an error is normally
# the cause. Paste the whole output into a bug report / chat if unsure.
#
# Environment:
#   NAMESPACE     Stack namespace (default: ai-stack)
#   STORAGE_ROOT  Host storage root (default: /mnt/aistack)
set -uo pipefail

NAMESPACE="${NAMESPACE:-ai-stack}"

hdr() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }

command -v kubectl >/dev/null 2>&1 || { echo "kubectl not found." >&2; exit 1; }
kubectl version >/dev/null 2>&1 || { echo "Cannot reach the cluster." >&2; exit 1; }

hdr "Pods"
kubectl -n "${NAMESPACE}" get pods -o wide

hdr "Pod events (the actual reason a pod is stuck)"
# Sorted by time so the newest complaint is last. This is the section that was
# missing when a pod "just says Pending".
kubectl -n "${NAMESPACE}" get events --sort-by=.lastTimestamp \
  --field-selector involvedObject.kind=Pod 2>/dev/null | tail -40

hdr "Waiting / non-running containers"
for pod in $(kubectl -n "${NAMESPACE}" get pods -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
  phase="$(kubectl -n "${NAMESPACE}" get pod "${pod}" -o jsonpath='{.status.phase}' 2>/dev/null)"
  [[ "${phase}" == "Running" || "${phase}" == "Succeeded" ]] && continue
  echo "--- ${pod} (${phase})"
  kubectl -n "${NAMESPACE}" get pod "${pod}" -o jsonpath=\
'{range .status.initContainerStatuses[*]}init/{.name}: {.state}{"\n"}{end}{range .status.containerStatuses[*]}{.name}: {.state}{"\n"}{end}' 2>/dev/null
  echo
  kubectl -n "${NAMESPACE}" describe pod "${pod}" 2>/dev/null | sed -n '/^Events:/,$p'
done

hdr "PersistentVolumeClaims"
kubectl -n "${NAMESPACE}" get pvc -o wide
for pvc in $(kubectl -n "${NAMESPACE}" get pvc -o jsonpath='{.items[?(@.status.phase!="Bound")].metadata.name}' 2>/dev/null); do
  echo "--- not Bound: ${pvc}"
  kubectl -n "${NAMESPACE}" describe pvc "${pvc}" | sed -n '/^Events:/,$p'
done


hdr "PersistentVolumes"
# hostPath PVs are static: each one is pre-bound to its PVC via claimRef. A PV
# stuck in "Available" while its PVC is Pending means the claim does not match
# (size, accessModes, storageClassName or volumeName).
kubectl get pv -l app.kubernetes.io/part-of=textgenerator -o custom-columns=\
NAME:.metadata.name,\
CAPACITY:.spec.capacity.storage,\
ACCESS:.spec.accessModes,\
RECLAIM:.spec.persistentVolumeReclaimPolicy,\
STATUS:.status.phase,\
CLAIM:.spec.claimRef.name,\
PATH:.spec.hostPath.path 2>/dev/null

hdr "StorageClass"
kubectl get storageclass textgen-hostpath 2>/dev/null \
  || note "StorageClass textgen-hostpath not found - apply k8s/01-storage.yaml"

hdr "Host directories"
# hostPath volumes ignore fsGroup, so ownership on disk decides whether the
# containers (uid/gid 1000) can write. This only inspects the local node - run
# it on the node that runs the pods.
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/aistack}"
if [[ -d "${STORAGE_ROOT}" ]]; then
  df -h "${STORAGE_ROOT}" | tail -2
  echo
  bad=0
  while IFS= read -r dir; do
    owner="$(stat -c '%u:%g' "${dir}" 2>/dev/null)"
    if [[ "${owner}" != "1000:1000" ]]; then
      echo "  wrong owner (${owner}): ${dir}"
      bad=$((bad + 1))
    fi
  done < <(find "${STORAGE_ROOT}" -maxdepth 2 -type d 2>/dev/null)
  if [[ "${bad}" -eq 0 ]]; then
    echo "  all directories under ${STORAGE_ROOT} owned by 1000:1000"
  else
    note "Fix with: sudo textgenerator/scripts/prepare-host-dirs.sh ${STORAGE_ROOT}"
  fi
else
  note "${STORAGE_ROOT} not found on this machine."
  note "Either you are not on the K3s node, or prepare-host-dirs.sh has not run."
fi

hdr "Node"
kubectl get nodes -o wide 2>/dev/null

hdr "Next steps"
note "* PVC Pending: the PV it is pinned to via volumeName must exist and match"
note "  in size, accessModes and storageClassName (textgen-hostpath)."
note "* PVC stuck Terminating / immutable field errors after a storage change:"
note "  delete the old PVCs first, the data under ${STORAGE_ROOT} survives."
note "* Pod cannot write (EACCES): run prepare-host-dirs.sh on the node."
note "* Multi-node cluster: hostPath data only exists on one node. Pin the"
note "  pods with a nodeSelector, or the pod may start on a node with no data."
note "* See the troubleshooting section in textgenerator/README.md."
