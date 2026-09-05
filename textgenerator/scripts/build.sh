#!/usr/bin/env bash
# One-stop build & deploy for the textgenerator stack.
#
#   textgenerator/scripts/build.sh [OPTIONS]
#
# Without options it runs the full happy path:
#   1. prerequisite checks (kubectl, cluster, host storage, container tooling)
#   1b. create the host directories under /mnt/aistack
#   2. build + import the character-extractor image
#   3. kubectl apply -k textgenerator/k8s
#   4. wait for the deployments to become available
#   5. print the NodePort endpoints
#
# Options:
#   --clean         Remove the stack before deploying (keeps the PVCs, so no
#                   data is lost). Combine with --purge-data to delete them.
#   --purge-data    Only together with --clean: also delete the PVCs and PVs.
#                   The hostPath PVs use reclaimPolicy: Retain, so the files
#                   under STORAGE_ROOT survive - delete them by hand if you
#                   really want to start over. Needed when a PVC has to change
#                   an immutable field (storageClassName, volumeName, size).
#   --clean-only    Clean up and exit without deploying.
#   --skip-build    Do not rebuild the extractor image.
#   --skip-verify   Do not run verify-gpu.sh / verify-storage.sh.
#   --with-comfyui  Scale the optional ComfyUI deployment to 1 after applying.
#                   Off by default - it competes for the single GPU.
#   --yes, -y       Do not ask for confirmation on destructive actions.
#   --help, -h      Show this help.
#
# Environment:
#   NAMESPACE       Target namespace (default: ai-stack)
#   EXTRACTOR_TAG   Extractor image tag (default: ice-ki/character-extractor:0.1.0)
#   ROLLOUT_TIMEOUT Per-deployment wait (default: 900s - model images are huge)
#   STORAGE_ROOT    Host storage root (default: /mnt/aistack). Must match the
#                   hostPath values in k8s/01-storage.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(cd -- "${SCRIPT_DIR}/../k8s" && pwd)"

NAMESPACE="${NAMESPACE:-ai-stack}"
EXTRACTOR_TAG="${EXTRACTOR_TAG:-ice-ki/character-extractor:0.1.0}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-900s}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/aistack}"

DO_CLEAN=0
DO_PURGE_DATA=0
DO_CLEAN_ONLY=0
DO_BUILD=1
DO_VERIFY=1
WITH_COMFYUI=0
ASSUME_YES=0

# Deployments that must become available. ComfyUI is intentionally absent -
# it ships with replicas: 0.
CORE_DEPLOYMENTS=(text-generation-webui sillytavern character-extractor)

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# Print the leading comment block of this file as the help text.
usage() {
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)        DO_CLEAN=1 ;;
    --purge-data)   DO_PURGE_DATA=1 ;;
    --clean-only)   DO_CLEAN=1; DO_CLEAN_ONLY=1 ;;
    --skip-build)   DO_BUILD=0 ;;
    --skip-verify)  DO_VERIFY=0 ;;
    --with-comfyui) WITH_COMFYUI=1 ;;
    -y|--yes)       ASSUME_YES=1 ;;
    -h|--help)      usage; exit 0 ;;
    *)              die "Unknown option: $1 (try --help)" ;;
  esac
  shift
done

if [[ "${DO_PURGE_DATA}" -eq 1 && "${DO_CLEAN}" -eq 0 ]]; then
  die "--purge-data only makes sense together with --clean or --clean-only."
fi

confirm() {
  [[ "${ASSUME_YES}" -eq 1 ]] && return 0
  local answer
  read -r -p "$1 [yes/NO] " answer
  [[ "${answer}" == "yes" ]]
}

# --------------------------------------------------------------------------
step "Checking prerequisites"
# --------------------------------------------------------------------------
command -v kubectl >/dev/null 2>&1 || die "kubectl not found."
kubectl version >/dev/null 2>&1 || die "Cannot reach the cluster (check KUBECONFIG)."
info "cluster reachable"

if [[ -d "${STORAGE_ROOT}" ]]; then
  info "storage root ${STORAGE_ROOT} present"
else
  die "Storage root ${STORAGE_ROOT} not found. Create/mount it, then re-run.
    The stack stores everything in hostPath volumes below that directory.
    If you use a different path, change STORAGE_ROOT here AND the hostPath
    values in k8s/01-storage.yaml."
fi

if ! kubectl kustomize "${K8S_DIR}" >/dev/null 2>&1; then
  die "kubectl cannot render ${K8S_DIR} - check the manifests."
fi
info "manifests render cleanly"

# --------------------------------------------------------------------------
if [[ "${DO_CLEAN}" -eq 1 ]]; then
  step "Cleaning up the existing stack"

  if ! kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
    info "namespace ${NAMESPACE} does not exist - nothing to clean"
  else
    # Delete workloads and services but keep the PVCs, unless explicitly
    # asked to purge them.
    info "deleting Deployments, Services and ConfigMaps ..."
    kubectl -n "${NAMESPACE}" delete deployment,service,configmap \
      -l app.kubernetes.io/part-of=textgenerator --ignore-not-found --wait=true

    info "waiting for pods to disappear ..."
    kubectl -n "${NAMESPACE}" wait --for=delete pod \
      -l app.kubernetes.io/part-of=textgenerator --timeout=300s 2>/dev/null || true

    if [[ "${DO_PURGE_DATA}" -eq 1 ]]; then
      # The PVs use reclaimPolicy: Retain, so this only removes the Kubernetes
      # objects - the files under STORAGE_ROOT stay. Both PVC and PV have to
      # go: a Retain-ed PV stays "Released" and refuses to bind again.
      warn "Deleting the PVCs and PVs. The data in ${STORAGE_ROOT} is NOT touched."
      kubectl -n "${NAMESPACE}" get pvc -l app.kubernetes.io/part-of=textgenerator 2>/dev/null || true
      if confirm "Really delete all textgenerator PVCs/PVs in namespace ${NAMESPACE}?"; then
        kubectl -n "${NAMESPACE}" delete pvc \
          -l app.kubernetes.io/part-of=textgenerator --ignore-not-found --wait=true
        kubectl delete pv \
          -l app.kubernetes.io/part-of=textgenerator --ignore-not-found --wait=true
        info "PVCs and PVs deleted - ${STORAGE_ROOT} untouched"
        info "To really erase the data: sudo rm -rf ${STORAGE_ROOT}/*"
      else
        info "aborted - PVCs kept"
      fi
    else
      info "PVCs kept (pass --purge-data to delete them as well)"
    fi
  fi

  if [[ "${DO_CLEAN_ONLY}" -eq 1 ]]; then
    step "Done (--clean-only)"
    exit 0
  fi
fi

# --------------------------------------------------------------------------
step "Ensuring namespace ${NAMESPACE} exists"
# Created before the verification and build steps on purpose: the in-cluster
# CUDA smoke test and the rollout restart of the extractor both need it, and
# on a fresh cluster (or right after --clean --purge-data) it does not exist.
if kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1; then
  info "namespace ${NAMESPACE} already exists"
elif [[ "${NAMESPACE}" == "ai-stack" ]]; then
  kubectl apply -f "${K8S_DIR}/00-namespace.yaml"
else
  # kustomization.yaml pins the manifests to ai-stack, so a custom namespace
  # only works together with a kustomize overlay - create it anyway.
  kubectl create namespace "${NAMESPACE}"
fi

# --------------------------------------------------------------------------
step "Preparing host directories in ${STORAGE_ROOT}"
# hostPath volumes ignore fsGroup, so the directories must exist and be owned
# by uid/gid 1000 before the pods start. Needs root - skipped with a hint if
# the script is not run as root and the directories already look right.
if [[ "$(id -u)" -eq 0 ]]; then
  "${SCRIPT_DIR}/prepare-host-dirs.sh" "${STORAGE_ROOT}" >/dev/null \
    && info "host directories ready" \
    || warn "prepare-host-dirs.sh failed - the pods may hit permission errors."
elif command -v sudo >/dev/null 2>&1; then
  sudo "${SCRIPT_DIR}/prepare-host-dirs.sh" "${STORAGE_ROOT}" >/dev/null \
    && info "host directories ready (via sudo)" \
    || warn "prepare-host-dirs.sh failed - run it manually as root."
else
  warn "not root and no sudo - run this first:"
  warn "  sudo ${SCRIPT_DIR}/prepare-host-dirs.sh ${STORAGE_ROOT}"
fi

# --------------------------------------------------------------------------
if [[ "${DO_VERIFY}" -eq 1 ]]; then
  step "Verifying GPU and storage prerequisites"
  # Informational: a failure here is reported but does not abort the deploy,
  # because the pods simply stay Pending until the host is fixed.
  "${SCRIPT_DIR}/verify-storage.sh" "${STORAGE_ROOT}" || warn "verify-storage.sh reported problems (see above)."
  "${SCRIPT_DIR}/verify-gpu.sh"     || warn "verify-gpu.sh reported problems (see above)."
fi

# --------------------------------------------------------------------------
if [[ "${DO_BUILD}" -eq 1 ]]; then
  step "Building the character-extractor image"
  # The image exists in no registry and the deployment uses
  # imagePullPolicy: Never, so it has to be in the K3s image store.
  NAMESPACE="${NAMESPACE}" "${SCRIPT_DIR}/build-extractor-image.sh" "${EXTRACTOR_TAG}"
else
  step "Skipping the extractor image build (--skip-build)"
fi

# --------------------------------------------------------------------------
step "Applying the manifests"
kubectl apply -k "${K8S_DIR}"

if [[ "${WITH_COMFYUI}" -eq 1 ]]; then
  warn "ComfyUI and text-generation-webui share one 16 GB GPU - expect VRAM"
  warn "pressure. Scale one of them to 0 if generation fails."
  kubectl -n "${NAMESPACE}" scale deploy/comfyui --replicas=1
fi

# --------------------------------------------------------------------------
step "Waiting for the deployments to become available"
info "timeout ${ROLLOUT_TIMEOUT} per deployment (the model image is several GB)"
rollout_rc=0
for deploy in "${CORE_DEPLOYMENTS[@]}"; do
  if ! kubectl -n "${NAMESPACE}" rollout status "deploy/${deploy}" --timeout="${ROLLOUT_TIMEOUT}"; then
    warn "deploy/${deploy} is not ready yet"
    rollout_rc=1
  fi
done

# --------------------------------------------------------------------------
step "Stack status"
kubectl -n "${NAMESPACE}" get pods,svc,pvc

step "Endpoints"
NODE_IP="$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}' 2>/dev/null)"
NODE_IP="${NODE_IP:-<node-ip>}"
while read -r name port; do
  [[ -z "${name}" ]] && continue
  info "http://${NODE_IP}:${port}    (${name})"
done < <(kubectl -n "${NAMESPACE}" get svc -o \
  jsonpath='{range .items[?(@.spec.type=="NodePort")]}{.metadata.name}{" "}{.spec.ports[0].nodePort}{"\n"}{end}' 2>/dev/null)

info "character-extractor is ClusterIP only:"
info "  kubectl -n ${NAMESPACE} port-forward svc/character-extractor 8080:8080"

if [[ "${rollout_rc}" -ne 0 ]]; then
  echo
  warn "Not everything is ready. Collect the details with:"
  warn "  ${SCRIPT_DIR}/diagnose.sh"
  warn "or look directly at:"
  warn "  kubectl -n ${NAMESPACE} logs -f deploy/text-generation-webui"
  warn "See the troubleshooting section in textgenerator/README.md."
fi
exit "${rollout_rc}"
