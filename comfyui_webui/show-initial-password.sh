#!/usr/bin/env bash
set +x
set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi

if ! credentials=$(kubectl -n comfyui exec deployment/webui -c webui -- cat /data/bootstrap_credentials.txt); then
  echo "Initial WebUI credentials could not be read. Check cluster access and pod readiness; the bootstrap file may already have been deleted." >&2
  exit 1
fi
if [[ -z "${credentials//[[:space:]]/}" ]]; then
  echo "The WebUI bootstrap credentials file is empty." >&2
  exit 1
fi
printf '%s\n' "$credentials"
