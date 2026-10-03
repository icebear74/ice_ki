#!/usr/bin/env bash
set -euo pipefail

deploy=false
if [[ "${1:-}" == "--deploy" ]]; then
  deploy=true
  shift
fi
if (( $# > 2 )); then
  echo "Usage: $0 [--deploy] [K3S_REGISTRY_HOST:PORT] [OUTPUT_MANIFEST]" >&2
  exit 2
fi

push_registry=127.0.0.1:5000
pull_registry=${1:-$push_registry}
tag=${IMAGE_TAG:-1}
ollama_image=${OLLAMA_IMAGE:-ollama/ollama:0.35.0}
webui_nodeport=${WEBUI_NODEPORT:-30080}
comfyui_nodeport=${COMFYUI_NODEPORT:-30188}
for port in "$webui_nodeport" "$comfyui_nodeport"; do
  if [[ ! "$port" =~ ^[0-9]{5}$ ]] || (( 10#$port < 30000 || 10#$port > 32767 )); then
    echo "NodePorts must be integers in the default Kubernetes range 30000-32767." >&2
    exit 2
  fi
done
if [[ "$webui_nodeport" == "$comfyui_nodeport" ]]; then
  echo "WebUI and ComfyUI require distinct NodePorts." >&2
  exit 2
fi
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

sed -E \
  -e "s#registry.example.invalid:5000/(comfyui-webui|comfyui|comfyui-ollama):1#${pull_registry}/\\1:${tag}#g" \
  -e 's#nodePort: 30080#nodePort: __WEBUI_NODEPORT__#' \
  -e 's#nodePort: 30188#nodePort: __COMFYUI_NODEPORT__#' \
  -e "s#__WEBUI_NODEPORT__#${webui_nodeport}#" \
  -e "s#__COMFYUI_NODEPORT__#${comfyui_nodeport}#" \
  "$template" > "$output"
echo "Deployment manifest: $output"
if [[ "$deploy" == true ]]; then
  bash "${script_dir}/deploy.sh" "$output"
else
  echo "Deploy with automatic API token provisioning: ${script_dir}/deploy.sh $output"
fi
echo "Configure HTTP registry access on ALL nodes; use the registry's LAN address for a multi-node cluster."
echo "WebUI: http://<node-ip>:${webui_nodeport} | ComfyUI: http://<node-ip>:${comfyui_nodeport}"
echo "Services are included in this manifest; use --deploy to apply it and provision the API token."
echo "Check kubectl apply errors (especially occupied NodePorts), then: kubectl -n comfyui get services"
