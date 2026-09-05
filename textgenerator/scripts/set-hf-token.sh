#!/usr/bin/env bash
# Create or update the HuggingFace access token Secret used by
# text-generation-webui.
#
#   textgenerator/scripts/set-hf-token.sh hf_xxxxxxxxxxxxxxxx
#   textgenerator/scripts/set-hf-token.sh                # prompts, no echo
#   textgenerator/scripts/set-hf-token.sh --from-env     # reads $HF_TOKEN
#   textgenerator/scripts/set-hf-token.sh --delete       # removes the Secret
#
# The token is needed for gated repositories (Llama, Gemma, ...) and raises
# the anonymous download rate limit. Public models work without one.
#
# Get a token at https://huggingface.co/settings/tokens - a "read" token is
# enough. The Secret is deliberately NOT part of the manifests, so the token
# never ends up in git.
set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-stack}"
SECRET_NAME="${SECRET_NAME:-huggingface-token}"
DEPLOYMENT="text-generation-webui"

die() { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '    %s\n' "$*"; }

command -v kubectl >/dev/null 2>&1 || die "kubectl not found."
kubectl version >/dev/null 2>&1 || die "Cannot reach the cluster (check KUBECONFIG)."

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
  exit 0
fi

kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 \
  || die "Namespace ${NAMESPACE} does not exist - deploy the stack first."

if [[ "${1:-}" == "--delete" ]]; then
  kubectl -n "${NAMESPACE}" delete secret "${SECRET_NAME}" --ignore-not-found
  info "Secret removed. Restart the backend to drop the token from the pod:"
  info "  kubectl -n ${NAMESPACE} rollout restart deploy/${DEPLOYMENT}"
  exit 0
fi

case "${1:-}" in
  --from-env)
    token="${HF_TOKEN:-}"
    [[ -n "${token}" ]] || die "HF_TOKEN is empty in this shell."
    ;;
  "")
    # Read without echoing so the token does not end up in the shell history
    # or in the terminal scrollback.
    read -rsp "HuggingFace token (input hidden): " token
    echo
    ;;
  *)
    token="$1"
    echo "note: passing the token as an argument leaves it in your shell" >&2
    echo "      history. Run the script without arguments to be prompted." >&2
    ;;
esac

[[ -n "${token}" ]] || die "No token given."

# HuggingFace tokens start with "hf_". Warn rather than refuse - the prefix is
# a convention, not a guarantee.
if [[ "${token}" != hf_* ]]; then
  echo "note: the token does not start with 'hf_' - is it the right value?" >&2
fi

# --dry-run + apply so an existing Secret is updated instead of erroring.
kubectl -n "${NAMESPACE}" create secret generic "${SECRET_NAME}" \
  --from-literal=HF_TOKEN="${token}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "${NAMESPACE}" label secret "${SECRET_NAME}" \
  app.kubernetes.io/part-of=textgenerator --overwrite >/dev/null

unset token

echo
info "Secret ${SECRET_NAME} stored in namespace ${NAMESPACE}."
info "The pod only picks it up on restart:"
info "  kubectl -n ${NAMESPACE} rollout restart deploy/${DEPLOYMENT}"
info "Verify afterwards with:"
info "  kubectl -n ${NAMESPACE} exec deploy/${DEPLOYMENT} -- printenv HF_TOKEN"
