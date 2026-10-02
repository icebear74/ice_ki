#!/bin/sh
set -eu
mkdir -p /data/models /data/input /data/output /data/temp /data/user /data/custom_nodes /data/cache
mkdir -p /data/user/__manager
if [ ! -e /data/user/__manager/config.ini ]; then
    cp /opt/manager-config.ini /data/user/__manager/config.ini
fi
exec python /opt/ComfyUI/main.py "$@"
