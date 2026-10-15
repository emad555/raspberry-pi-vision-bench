# Raspberry Pi Vision Bench

Run and compare seven object detectors on a Raspberry Pi 5, using the CPU with **no Hailo accelerator required**. Watch completed detection frames in your browser, switch models live, and compare measured frame rates.

## Features

- USB camera capture through native GStreamer MJPEG, with OpenCV fallback; CSI cameras use Picamera2.
- MobileNet SSD, MobileNetV2/V3 SSDLite, SqueezeNet SSD, and YOLO11 Nano/Small/Medium.
- Browser model selector, confidence slider, class list, readable labels, and FPS counters.
- A separate inference process and a latest-frame queue to avoid accumulating stale camera frames.
- Optional H.264 RTSP output and camera-free image benchmarking.

## Requirements

Tested on Raspberry Pi 5 with Raspberry Pi OS Bookworm **64-bit**, Python 3.11, and a USB MJPEG camera. A CSI camera can also be selected. Cooling and a suitable power supply help sustain performance.

## Install

```bash
git clone https://github.com/emad555/raspberry-pi-vision-bench.git
cd raspberry-pi-vision-bench
./install-system-deps.sh
./setup.sh
```

The system-dependency script uses `sudo`. Setup creates a local virtual environment with access to Raspberry Pi OS camera/OpenCV packages and downloads four checksum-verified SSD models. YOLO models are optional and are prepared separately below. Model weights, virtual environments, recordings, and logs are excluded from Git.

## Run the web app

```bash
./run.sh --source 0 --stream http --host 0.0.0.0 --port 8080
```

Open `http://<PI_IP>:8080` on a device connected to the same network. Find the Pi's address with `hostname -I`. On the Pi itself, use `http://localhost:8080`. Keep the terminal running; press **Ctrl+C** to stop.

Select a model in the dropdown to change the running detector. Missing models are disabled until their assets are installed. Settings are shared by all viewers. The first selection loads and verifies the model; subsequent selections reuse it. A failed load keeps the previous detector and reports the error.

The feed shows **only frames that completed detection**. Slower models therefore produce visibly slower video. USB camera JPEGs are forwarded without re-encoding; boxes are drawn on the web page using the corresponding latest result. The browser polls detection/status data five times per second, so overlay updates can lag the MJPEG image. The raw `/stream` endpoint does not contain embedded boxes for this USB path.

The HTTP/RTSP servers do not require authentication. Bind to localhost for local-only use, or use a trusted LAN; do not expose them directly to the internet.

### Other inputs and modes

```bash
./run.sh --source pi                         # CSI camera, local web preview
./run.sh --source /dev/video0                # USB device path
./run.sh --source 0 --capture-backend opencv # fallback for non-MJPEG USB cameras
./run.sh --source 0 --stream none            # detection without streaming
./run.sh --source 0 --backend yolo-small     # start with a prepared YOLO model
./run.sh --help
```

USB cameras must support the requested MJPEG mode for GStreamer capture. Defaults are 640×480 at a requested 30 FPS; actual capture rate depends on the camera and its exposure settings. Changing camera resolution does not change the network input size.

## Models

| CLI backend | Model | Classes | Network input | Tested weight size |
|---|---|---:|---:|---:|
| `ssdlite-v2` | MobileNetV2 SSDLite (default) | 20 VOC | 300×300 | 6.4 MiB |
| `ssd` | MobileNet SSD | 20 VOC | 300×300 | 11.1 MiB |
| `ssdlite-v3` | MobileNetV3 SSDLite | 20 VOC | 300×300 | 9.2 MiB |
| `squeezenet` | SqueezeNet SSD | 20 VOC | 300×300 | 10.6 MiB |
| `yolo` | YOLO11 Nano | 80 COCO | 640×640 | 10.1 MiB |
| `yolo-small` | YOLO11 Small | 80 COCO | 640×640 | 36.1 MiB |
| `yolo-medium` | YOLO11 Medium | 80 COCO | 640×640 | 76.7 MiB |

Weight size is disk space, not runtime RAM. Model labels and availability come from one shared registry. View the active model's class list in the web app.

### Prepare YOLO models

Detection itself does not need PyTorch. Exporting YOLO models does, so use a separate environment with enough disk space and RAM:

```bash
python3 -m venv --system-site-packages .venv-export
.venv-export/bin/python -m pip install -r requirements-export.txt
.venv-export/bin/python export_yolo_models.py --sizes n s m
```

Restart the web app after adding assets. To export only Small, use `--sizes s`. The converter downloads official Ultralytics v8.3.0 checkpoints and writes NCNN models into `models/`. The PNNX wheel is pinned to **20260526**, matching the tested NCNN runtime; a newer auto-downloaded converter previously produced incompatible graphs. Existing assets are never required for a fresh export.

See [model provenance and download details](models/README.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## FPS: what is measured?

- **Latest completed frame FPS:** inverse elapsed monotonic time between consecutive completed results, including input waits and processing/output preparation. The first result has no interval. The value is calculated for every result; the page displays the latest sample it polls.
- **Detection FPS:** unique frames completed during a rolling three-second window, divided by elapsed window time.
- **Video output FPS:** completed frames prepared for streaming over the same window. Network/client playback can be slower.
- **Camera FPS:** successful source reads, separately counted. Camera frames may be dropped before inference when a detector is slower.
- **Mean inference ms:** preprocessing, neural-network execution, and detection decoding. It excludes input JPEG decoding, model loading, and network transfer.
- **Capture → detection ms:** elapsed time from camera-read completion to receipt of the result by the web process; this excludes sensor exposure and browser rendering.

There is no frame duplication to force video to the configured camera rate. Pauses remain in elapsed time. Rolling rates decay to zero when new results stop. The number `1000 / inference_ms` is an image-inference speed estimate, not the live-camera FPS.

### Example measurements

These are historical live measurements on one Pi 5, with two inference threads. They are observations, not guaranteed rates or controlled accuracy tests.

| Model | Received completed-frame video FPS |
|---|---:|
| MobileNetV2 SSDLite | ~15 (camera-limited to ~15 in that run) |
| YOLO11 Nano | ~11 |
| YOLO11 Small | ~4.5 |
| YOLO11 Medium | ~1.7 |

The camera delivered ~30 FPS in a separate raw V4L2 test. Camera-only, image-only, and full live-stream tests measure different workloads. Lighting/exposure, temperature, scene content, settings, and other processes affect results. The current counters are the best guide for your setup.

## RTSP

```bash
./run.sh --source 0 --stream rtsp
```

Open `rtsp://<PI_IP>:8554/detect` in VLC, or:

```bash
ffplay -rtsp_transport tcp rtsp://<PI_IP>:8554/detect
```

RTSP uses server-side annotations and a CPU H.264 encoder. It runs the synchronous detection pipeline and does not expose the browser model selector. `--rtsp-host`, `--rtsp-port`, `--rtsp-path`, `--bitrate`, and `--stream-fps` configure it. Advertised stream FPS does not guarantee detection throughput.

## Benchmark and develop

Use your own image, without opening a camera:

```bash
./run.sh --benchmark /path/to/image.jpg --backend ssdlite-v2 --runs 100
.venv/bin/python benchmark_models.py --image /path/to/image.jpg
.venv/bin/python -m unittest discover -s tests -v
```

Benchmark results exclude camera capture and delivery. The single-model benchmark warms up before timing; it saves an annotated `benchmark.jpg`. The comparison script skips models that have not been downloaded.

For source checks:

```bash
.venv/bin/python -m pip install ruff==0.14.1
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

See [architecture and API](docs/architecture.md) for implementation details.

## Troubleshooting

- **Camera busy:** stop the existing runner before starting another one. Each runner owns the camera.
- **No camera:** check `rpicam-hello --list-cameras` for CSI cameras or `v4l2-ctl --list-devices` for USB devices. Power off before adjusting a CSI ribbon.
- **No MJPEG frames:** choose supported `--width`/`--height`, or try `--capture-backend opencv`.
- **Missing model:** rerun `./setup.sh` for SSD assets, or prepare the YOLO export environment above.
- **Browser cannot connect:** check the Pi's current IP, use `--host 0.0.0.0`, and connect to the same network.
- **Small/Medium is slow:** this is CPU inference. Choose MobileNetV2 SSDLite or YOLO Nano and compare actual completed-frame FPS.

## License

Project source is licensed under **AGPL-3.0**; see [LICENSE](LICENSE). Downloaded third-party weights retain their upstream terms. No model binaries, local credentials, camera recordings, or private network addresses are published in this repository.
