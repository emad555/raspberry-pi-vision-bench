#!/usr/bin/env bash
# Compatibility entry point; installs camera, web, and RTSP system dependencies.
set -euo pipefail
exec "$(dirname "$0")/install-system-deps.sh" "$@"
