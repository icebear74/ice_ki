#!/usr/bin/env bash
set +x
set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi

command -v kubectl >/dev/null || { echo "kubectl is required." >&2; exit 1; }
kubectl -n comfyui get deployment comfyui -o name >/dev/null
echo "Create a Hugging Face Read token at https://huggingface.co/settings/tokens."
echo "The account must already have access to the gated model."
if ! IFS= read -r -s -p 'Hugging Face Read-Token (hidden): ' hf_token; then
  printf '\nToken input cancelled.\n' >&2
  exit 1
fi
printf '\n' >&2
if [[ ! "$hf_token" =~ ^hf_[a-zA-Z0-9]+$ ]]; then
  unset hf_token
  echo "Invalid token: enter only the Hugging Face token starting with hf_." >&2
  exit 2
fi

umask 077
secret_dir=$(mktemp -d)
trap 'rm -rf -- "$secret_dir"; unset hf_token' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s' "$hf_token" > "$secret_dir/token"
unset hf_token
if ! kubectl -n comfyui create secret generic comfyui-huggingface \
    --from-file="token=$secret_dir/token" --dry-run=client -o yaml 2>/dev/null \
    | kubectl -n comfyui apply --server-side --field-manager=hf-token-setup -f - >/dev/null 2>&1; then
  echo "HF Secret setup failed. Check namespace, Secret permissions and field-manager conflicts." >&2
  exit 1
fi
rm -rf -- "$secret_dir"
trap - EXIT INT TERM
echo "HF token saved in Secret comfyui-huggingface; local model API token unchanged."
echo "Restarting ComfyUI; active generations/transfers may be interrupted."
kubectl -n comfyui rollout restart deployment/comfyui
echo "ComfyUI must use the updated image and deployment with HF_TOKEN support."
