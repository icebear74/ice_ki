#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 REGISTRY_HOST:PORT [OUTPUT_MANIFEST]" >&2
  echo "Use an address reachable by every k3s node; authenticate with docker login first if needed." >&2
  exit 2
fi

registry=$1
tag=${IMAGE_TAG:-2.1-shape-1}
case "$registry" in
  ""|*[!a-zA-Z0-9.:-]*) echo "Registry must be a hostname or IP with optional port (no URL scheme or path)." >&2; exit 2 ;;
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

image="${registry}/hunyuan3d:${tag}"
docker build -f "${script_dir}/Dockerfile" -t "$image" "$script_dir"
docker push "$image"

if ! grep -q 'registry.example.invalid:5000/hunyuan3d:2.1-shape-1' "$template"; then
  echo "Image placeholder not found in $template" >&2
  exit 1
fi
sed "s|registry.example.invalid:5000/hunyuan3d:2.1-shape-1|${image}|" "$template" > "$output"
echo "Pushed $image"
echo "Deployment manifest: $output"
echo "Apply with: kubectl apply -f $output"
echo "Ensure all k3s nodes can pull from $registry and have enough image storage (~8 GB)."
echo "Model weights and generated data are downloaded to the 30Gi hunyuan3d-hf-cache-pvc on first start."
