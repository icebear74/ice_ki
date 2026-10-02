#!/bin/sh
set -eu
mkdir -p /data/models /data/input /data/output /data/temp /data/user /data/custom_nodes /data/cache
exec python /opt/ComfyUI/main.py "$@"
