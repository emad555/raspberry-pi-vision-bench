#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3-venv python3-opencv python3-numpy python3-yaml \
  python3-picamera2 python3-gi gir1.2-gst-rtsp-server-1.0 \
  gir1.2-gst-plugins-base-1.0 gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly v4l-utils
