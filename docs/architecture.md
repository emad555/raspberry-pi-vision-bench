# Architecture and local API

`detect.py` parses CLI options and selects HTTP, RTSP, or image-benchmark mode. The shell entry point selects the local runtime virtual environment.

| Module | Responsibility |
|---|---|
| `model_registry.py` | Shared model IDs, labels, classes, asset paths, and availability |
| `camera.py` / `gst_camera.py` | Latest-frame capture and native USB MJPEG transport |
| `detectors.py` / `light_detectors.py` | NCNN model construction, preprocessing, detection decoding |
| `model_control.py` | Detector-process model cache and safe switching |
| `async_web.py` | Camera/web parent process, inference child, bounded input/results queues |
| `web_preview.py` / `web/` | MJPEG endpoint, control API, and browser overlay |
| `runtime_metrics.py` | Monotonic frame counts, timing samples, and rolling rates |
| `rtsp_output.py` | Shared GStreamer H.264 RTSP server |

HTTP mode uses multiprocessing `spawn` so NCNN inference cannot hold up the web process through Python's GIL. A waiting input is replaced with the newest frame. Completed results include the sampled JPEG, frame ID, detections, completion interval, and model state. Only completed results are published. The same captured frame ID cannot be counted twice as processed.

## Endpoints

- `GET /` serves the browser UI.
- `GET /stream` serves multipart MJPEG. USB/GStreamer JPEG frames have browser-side boxes; CSI/OpenCV frames receive server-side annotations.
- `GET /api/status` returns model availability, current/requested model, class labels, detections, and FPS counters.
- `POST /api/model` accepts `application/json`, such as `{"model":"yolo-small","confidence":0.35}`. Valid confidence values are 0.25–0.95. A valid request returns 202; invalid input returns 400. Model changes are applied between inferences.

Model settings are shared by all clients. First-time loads are warmed up and verified before replacing the active detector. Ordinary load errors keep the previous model. A fatal native-library crash is detected when the inference process exits and stops the runner.

## Timing details

The three-second counters count events in `(now - window, now]` using monotonic time. During startup, elapsed uptime is the denominator until the window is full. Camera capture and processed-frame publication are independent counters. `frame_fps` uses consecutive worker completion timestamps and includes an intervening model-load pause.

`capture_to_output_ms` is retained as a compatibility API field: in HTTP mode it means capture-read completion to detection-result receipt, not browser-display latency. `inference_ms` excludes JPEG decode and model switching. Status polling happens five times per second; not every per-frame sample can be rendered in the UI when detection is faster than polling.
