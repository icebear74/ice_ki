#!/usr/bin/env bash
set +x
set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi

encoded=$(kubectl -n comfyui get secret comfyui-model-api -o 'jsonpath={.data.token}')
if [[ -z "$encoded" ]]; then
  echo "Secret comfyui-model-api has no token. Run deploy.sh to provision it if absent." >&2
  exit 1
fi
token=$(printf '%s' "$encoded" | base64 --decode)
if [[ -z "${token//[[:space:]]/}" ]]; then
  echo "Secret comfyui-model-api contains an empty token." >&2
  exit 1
fi
printf '%s\n' "$token"
