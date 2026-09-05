#!/usr/bin/env bash
# Collect everything needed to diagnose a stuck textgenerator pod - especially
# pods stuck in Pending/ContainerCreating because a Longhorn volume cannot be
# scheduled or attached.
#
#   textgenerator/scripts/diagnose.sh
#
# Read the output top down: the first section that shows an error is normally
# the cause. Paste the whole output into a bug report / chat if unsure.
#
# Environment:
#   NAMESPACE           Stack namespace (default: ai-stack)
#   LONGHORN_NAMESPACE  Longhorn namespace (default: longhorn-system)
set -uo pipefail

NAMESPACE="${NAMESPACE:-ai-stack}"
LONGHORN_NAMESPACE="${LONGHORN_NAMESPACE:-longhorn-system}"

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

if ! kubectl get crd volumes.longhorn.io >/dev/null 2>&1; then
  hdr "Longhorn"
  note "Longhorn CRDs not found - is Longhorn installed?"
  exit 0
fi

hdr "Longhorn volumes"
# spec.numberOfReplicas is the DESIRED count the volume was created with. It is
# NOT updated when the StorageClass changes afterwards.
kubectl -n "${LONGHORN_NAMESPACE}" get volumes.longhorn.io -o custom-columns=\
NAME:.metadata.name,\
REPLICAS:.spec.numberOfReplicas,\
SIZE:.spec.size,\
STATE:.status.state,\
ROBUSTNESS:.status.robustness,\
NODE:.status.currentNodeID,\
PVC:.status.kubernetesStatus.pvcName 2>/dev/null

hdr "Longhorn volumes that are not schedulable"
# A volume whose replicas cannot be scheduled never attaches, and the pod stays
# in Pending/ContainerCreating with an attach/mount error.
found_unschedulable=0
for vol in $(kubectl -n "${LONGHORN_NAMESPACE}" get volumes.longhorn.io -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
  sched="$(kubectl -n "${LONGHORN_NAMESPACE}" get volumes.longhorn.io "${vol}" \
    -o jsonpath='{.status.conditions[?(@.type=="Scheduled")].status}' 2>/dev/null)"
  if [[ "${sched}" == "False" ]]; then
    found_unschedulable=1
    reason="$(kubectl -n "${LONGHORN_NAMESPACE}" get volumes.longhorn.io "${vol}" \
      -o jsonpath='{.status.conditions[?(@.type=="Scheduled")].reason}: {.status.conditions[?(@.type=="Scheduled")].message}' 2>/dev/null)"
    echo "${vol}: ${reason}"
  fi
done
[[ "${found_unschedulable}" -eq 0 ]] && note "all volumes schedulable"

hdr "Longhorn replicas"
kubectl -n "${LONGHORN_NAMESPACE}" get replicas.longhorn.io -o custom-columns=\
NAME:.metadata.name,\
VOLUME:.spec.volumeName,\
NODE:.spec.nodeID,\
DESIRED:.spec.desireState,\
CURRENT:.status.currentState 2>/dev/null

hdr "Longhorn node disk capacity vs. scheduled storage"
# "storageScheduled" exceeding "storageMaximum" x over-provisioning percentage
# is the classic reason a large PVC (the 200Gi model volume) never schedules.
# The JSON is passed through an environment variable: piping it into python
# would clash with the heredoc, which also occupies stdin.
if command -v python3 >/dev/null 2>&1; then
  LH_NODES_JSON="$(kubectl -n "${LONGHORN_NAMESPACE}" get nodes.longhorn.io -o json 2>/dev/null)" \
  python3 <<'PY'
import json, os
def gib(v):
    try:
        return f"{int(v)/2**30:.1f}Gi"
    except (TypeError, ValueError):
        return "?"
raw = os.environ.get("LH_NODES_JSON") or ""
try:
    data = json.loads(raw)
except ValueError:
    print("    could not read Longhorn node status")
    raise SystemExit(0)
for node in data.get("items", []):
    name = node["metadata"]["name"]
    for disk, st in ((node.get("status") or {}).get("diskStatus") or {}).items():
        print(f"{name}/{disk}: available={gib(st.get('storageAvailable'))} "
              f"maximum={gib(st.get('storageMaximum'))} "
              f"scheduled={gib(st.get('storageScheduled'))}")
        for cond in st.get("conditions") or []:
            if cond.get("status") != "True":
                print(f"    {cond.get('type')}={cond.get('status')}: "
                      f"{cond.get('reason')} {cond.get('message')}")
PY
else
  note "python3 not available - inspect manually:"
  note "  kubectl -n ${LONGHORN_NAMESPACE} get nodes.longhorn.io -o yaml"
fi

hdr "Requested PVC sizes"
kubectl -n "${NAMESPACE}" get pvc -o jsonpath=\
'{range .items[*]}{.metadata.name}{"\t"}{.spec.resources.requests.storage}{"\n"}{end}' 2>/dev/null

hdr "Relevant Longhorn settings"
for setting in default-replica-count storage-over-provisioning-percentage \
               storage-minimal-available-percentage \
               allow-volume-creation-with-degraded-availability; do
  value="$(kubectl -n "${LONGHORN_NAMESPACE}" get settings.longhorn.io "${setting}" \
    -o jsonpath='{.value}' 2>/dev/null)"
  echo "${setting}=${value:-<default>}"
done

hdr "Next steps"
note "* Volume not schedulable / not enough space: shrink the PVC requests in"
note "  k8s/01-storage.yaml (oobabooga-models-pvc 200Gi, comfyui-data-pvc 100Gi)"
note "  or raise Longhorn's storage-over-provisioning-percentage."
note "* Volume attached but the pod cannot mount it: check the pod events above."
note "* See the troubleshooting section in textgenerator/README.md."
