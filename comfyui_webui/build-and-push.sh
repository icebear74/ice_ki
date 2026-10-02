#!/usr/bin/env bash
set -euo pipefail

if (( $# > 2 )); then
  echo "Usage: $0 [K3S_REGISTRY_HOST:PORT] [OUTPUT_MANIFEST]" >&2
  exit 2
fi

push_registry=127.0.0.1:5000
pull_registry=${1:-$push_registry}
tag=${IMAGE_TAG:-1}
ollama_image=${OLLAMA_IMAGE:-ollama/ollama:0.35.0}
case "$pull_registry" in
  ""|*[!a-zA-Z0-9.:-]*) echo "Registry must be a hostname or IP with optional port (no URL scheme or path)." >&2; exit 2 ;;
esac
if [[ ! "$tag" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]]; then
  echo "Invalid IMAGE_TAG." >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
template="${script_dir}/k8s/deploy.yaml"
output=${2:-/tmp/deploy_comfyui.yaml}
if [[ "$(realpath -m -- "$output")" == "$(realpath -- "$template")" ]]; then
  echo "Choose an output path other than the source manifest." >&2
  exit 2
fi

curl --fail --silent --show-error --max-time 5 "http://${push_registry}/v2/" > /dev/null
docker build -f "${script_dir}/Dockerfile" -t "${push_registry}/comfyui-webui:${tag}" "$script_dir"
docker build -f "${script_dir}/Dockerfile.comfyui" -t "${push_registry}/comfyui:${tag}" "$script_dir"
docker pull "$ollama_image"
docker tag "$ollama_image" "${push_registry}/comfyui-ollama:${tag}"
for name in comfyui-webui comfyui comfyui-ollama; do
  docker push "${push_registry}/${name}:${tag}"
done

sed -E "s#registry.example.invalid:5000/(comfyui-webui|comfyui|comfyui-ollama):1#${pull_registry}/\\1:${tag}#g" \
  "$template" > "$output"
echo "Deployment manifest: $output"
echo "Apply with: kubectl apply -f $output"
echo "Configure HTTP registry access on ALL nodes; use the registry's LAN address for a multi-node cluster."
echo "WebUI: http://<node-ip>:30080 | ComfyUI: http://<node-ip>:30188"
