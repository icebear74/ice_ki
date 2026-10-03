#!/usr/bin/env bash
set +x
set -euo pipefail

if (( $# != 1 )) || [[ ! -f "$1" ]]; then
  echo "Usage: $0 RENDERED_MANIFEST" >&2
  exit 2
fi
manifest=$1
encoded=$(kubectl -n comfyui get secret comfyui-model-api --ignore-not-found -o 'jsonpath={.data.token}')
if [[ -n "$encoded" ]]; then
  token=$(printf '%s' "$encoded" | base64 --decode)
  if [[ -z "${token//[[:space:]]/}" ]]; then
    echo "Existing comfyui-model-api token is empty; refusing to overwrite it." >&2
    exit 1
  fi
  unset token encoded
  echo "Existing model API token preserved."
else
  # Distinguish an absent Secret from an existing Secret with a missing key.
  name=$(kubectl -n comfyui get secret comfyui-model-api --ignore-not-found -o name)
  if [[ -n "$name" ]]; then
    echo "Existing comfyui-model-api has no token key; refusing to overwrite it." >&2
    exit 1
  fi
  kubectl create namespace comfyui --dry-run=client -o yaml | kubectl apply -f -
  umask 077
  token_dir=$(mktemp -d)
  trap 'rm -rf -- "$token_dir"' EXIT
  openssl rand -hex 32 > "$token_dir/token"
  kubectl -n comfyui create secret generic comfyui-model-api --from-file="token=$token_dir/token"
  rm -rf -- "$token_dir"
  trap - EXIT
  echo "Model API token generated and stored in Kubernetes Secret."
fi

version=$(kubectl -n comfyui get secret comfyui-model-api -o 'jsonpath={.metadata.resourceVersion}')
if [[ ! "$version" =~ ^[0-9]+$ ]]; then
  echo "Cannot determine model API Secret version; refusing deployment." >&2
  exit 1
fi
kubectl apply -f "$manifest"
# The same annotation is a no-op; retries and Secret changes reload pod environments.
kubectl -n comfyui patch deployment comfyui webui --type=merge \
  -p "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"comfyui.ice-ki/model-api-secret-version\":\"${version}\"}}}}}"
echo "Show the token locally with: $(dirname -- "${BASH_SOURCE[0]}")/show-api-token.sh"
