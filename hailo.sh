#!/usr/bin/env bash
set -euo pipefail
if ! hailortcli scan; then
  echo 'Hailo unavailable. Fix the driver/device before using hardware acceleration.' >&2
  exit 1
fi
exec rpicam-hello -t 0 --post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_inference.json --lores-width 640 --lores-height 640 "$@"
