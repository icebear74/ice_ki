#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
  echo "Usage: $0 [REGISTRY/comfyui-ollama:TAG]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=${1:-}
if [[ -z "$image" ]]; then
  image=$(kubectl -n comfyui get deployment ollama \
    -o 'jsonpath={.spec.template.spec.containers[?(@.name=="ollama")].image}')
fi
if [[ -z "$image" || "$image" == *registry.example.invalid* ||
      ! "$image" =~ ^[a-zA-Z0-9][a-zA-Z0-9._/:@-]*$ ]]; then
  echo "Specify a valid, already-pushed Ollama image accessible from the cluster nodes." >&2
  exit 2
fi

echo "Deploying only Ollama with image: $image"
sed "s#registry.example.invalid:5000/comfyui-ollama:1#${image}#g" \
  "${script_dir}/k8s/deploy.yaml" | kubectl apply -l app=ollama -f -
if ! kubectl -n comfyui rollout status deployment/ollama --timeout=25m; then
  echo "Ollama rollout failed. Check image addresses, node registry access and init-container events:" >&2
  kubectl -n comfyui describe pods -l app=ollama >&2 || true
  exit 1
fi
kubectl -n comfyui exec deployment/ollama -c ollama -- ollama list
