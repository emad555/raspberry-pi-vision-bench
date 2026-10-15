#!/usr/bin/env python3
"""MobileNet SSD / YOLO NCNN detection with latest-frame capture and RTSP output."""

import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np

from camera import Camera
from detectors import create_detector
from model_registry import LABELS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="pi, USB index (0), or video path")
    parser.add_argument("--capture-backend", choices=["gstreamer", "opencv"], default="gstreamer")
    parser.add_argument("--backend", choices=list(LABELS), default="ssdlite-v2")
    parser.add_argument("--model", type=Path, help="Override model directory")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-preview", action="store_true", help="Disable all streaming")
    parser.add_argument("--stream", choices=["rtsp", "http", "none"], default="http")
    parser.add_argument("--rtsp-host", default="0.0.0.0")
    parser.add_argument("--rtsp-port", type=int, default=8554)
    parser.add_argument("--rtsp-path", default="/detect")
    parser.add_argument("--stream-fps", type=int, default=30)
    parser.add_argument("--bitrate", type=int, default=1500, help="H.264 bitrate in kbit/s")
    parser.add_argument(
        "--benchmark", type=Path, help="Run inference on an image instead of opening a camera"
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark.jpg")
    args = parser.parse_args()
    if args.threads < 1 or args.runs < 1 or not 0 < args.confidence < 1:
        parser.error("threads/runs must be positive; confidence must be between 0 and 1")
    if args.width < 2 or args.height < 2 or args.width % 2 or args.height % 2:
        parser.error("width and height must be positive even numbers")
    if not 1 <= args.port <= 65535:
        parser.error("HTTP port must be between 1 and 65535")
    if args.stream_fps < 1 or args.bitrate < 1 or not 1 <= args.rtsp_port <= 65535:
        parser.error("Invalid RTSP fps, bitrate or port")
    if not args.rtsp_path.startswith("/") or any(c.isspace() for c in args.rtsp_path):
        parser.error("RTSP path must start with / and contain no whitespace")
    cv2.setNumThreads(1)
    if args.stream == "http" and not args.no_preview and not args.benchmark:
        from async_web import run

        run(args)
        return

    def factory(backend):
        return create_detector(backend, args.threads, args.confidence, args.model)

    detector = factory(args.backend)
    from model_control import ModelControl
    from runtime_metrics import RuntimeMetrics

    metrics = RuntimeMetrics()
    control = ModelControl(args.backend, detector, factory, args.confidence, metrics)
    camera = preview = None
    from annotation import annotate

    try:
        if args.benchmark:
            frame = cv2.imread(str(args.benchmark))
            if frame is None:
                raise ValueError(f"Cannot read image: {args.benchmark}")
            for _ in range(3):
                detector.predict(frame)
            timings = []
            for _ in range(args.runs):
                start = time.perf_counter()
                detections = detector.predict(frame)
                timings.append(time.perf_counter() - start)
            ms = float(np.mean(timings) * 1000)
            annotate(frame, detections, f"{ms:.1f} ms / {1000 / ms:.1f} inference FPS")
            if not cv2.imwrite(str(args.output), frame):
                raise RuntimeError("Cannot save annotated image")
            print(
                json.dumps(
                    {
                        "backend": args.backend,
                        "threads": args.threads,
                        "mean_ms": ms,
                        "inference_fps": 1000 / ms,
                        "runs": args.runs,
                        "detections": detections,
                        "output": str(args.output),
                    },
                    indent=2,
                )
            )
            return
        camera = Camera(args.source, args.width, args.height, metrics, args.capture_backend)
        if not args.no_preview and args.stream == "rtsp":
            from rtsp_output import RTSPOutput

            preview = RTSPOutput(
                args.rtsp_host,
                args.rtsp_port,
                args.rtsp_path,
                args.width,
                args.height,
                args.stream_fps,
                args.bitrate,
            )
            print(
                f"RTSP: rtsp://{args.rtsp_host}:{args.rtsp_port}{args.rtsp_path} (replace 0.0.0.0 with the Pi IP)",
                flush=True,
            )
        report = time.perf_counter()
        while True:
            packet = camera.read()
            if packet is None:
                time.sleep(0.002)
                continue
            frame, frame_id, captured_at = packet
            detector = control.apply(frame)
            start = time.perf_counter()
            detections = detector.predict(frame)
            ms = (time.perf_counter() - start) * 1000
            control.update(ms, len(detections))
            if preview:
                annotate(frame, detections, f"{control.active} | inference {ms:.0f} ms")
                published = preview.publish(frame)
            else:
                published = False
            metrics.complete(frame_id, captured_at, ms, published=bool(published))
            elapsed = time.perf_counter() - report
            if elapsed >= 2:
                measured = metrics.snapshot()
                print(
                    f"Camera {measured['camera_fps']:.1f} FPS | processed {measured['pipeline_fps']:.1f} FPS | output {measured['preview_fps']:.1f} FPS | mean inference {measured['inference_ms']:.1f} ms",
                    flush=True,
                )
                report = time.perf_counter()
    finally:
        if camera:
            camera.close()
        if preview:
            preview.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        raise SystemExit(f"Error: {exc}")
