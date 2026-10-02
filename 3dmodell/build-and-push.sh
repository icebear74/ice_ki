#!/usr/bin/env bash
set -euo pipefail

if (( $# > 2 )); then
  echo "Usage: $0 [K3S_REGISTRY_HOST:PORT] [OUTPUT_MANIFEST]" >&2
  echo "Build and push to the unauthenticated local registry at 127.0.0.1:5000." >&2
  exit 2
fi

push_registry=127.0.0.1:5000
pull_registry=${1:-$push_registry}
tag=${IMAGE_TAG:-2.1-shape-1}
case "$pull_registry" in
  ""|*[!a-zA-Z0-9.:-]*) echo "K3s registry must be a hostname or IP with optional port (no URL scheme or path)." >&2; exit 2 ;;
esac
case "$tag" in
  ""|*[!a-zA-Z0-9._-]*) echo "Invalid IMAGE_TAG." >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
template="${script_dir}/deploy_hunyuan3d.yaml"
output=${2:-/tmp/deploy_hunyuan3d.yaml}
if [[ "$output" == "$template" ]]; then
  echo "Choose an output path other than the source manifest." >&2
  exit 2
fi

if ! curl --fail --silent --show-error --max-time 5 "http://${push_registry}/v2/" > /dev/null; then
  echo "Local registry not reachable without authentication at http://${push_registry}/v2/" >&2
  exit 1
fi

image="${push_registry}/hunyuan3d:${tag}"
docker build -f "${script_dir}/Dockerfile" -t "$image" "$script_dir"
docker push "$image"

if ! grep -q 'registry.example.invalid:5000/hunyuan3d:2.1-shape-1' "$template"; then
  echo "Image placeholder not found in $template" >&2
  exit 1
fi
sed "s|registry.example.invalid:5000/hunyuan3d:2.1-shape-1|${pull_registry}/hunyuan3d:${tag}|" "$template" > "$output"
echo "Pushed $image"
echo "Deployment manifest: $output"
echo "Apply with: kubectl apply -f $output"
echo "Ensure all k3s nodes can reach $pull_registry over HTTP (configure k3s registries.yaml) and have enough image storage (~8 GB)."
echo "If the registry is not on every k3s node, pass its host LAN address as the first argument instead of using 127.0.0.1."
echo "Model weights and generated data are downloaded to the 30Gi hunyuan3d-hf-cache-pvc on first start."
