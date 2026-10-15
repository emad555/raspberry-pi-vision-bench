"""Camera/preview process with a separate detector process to avoid NCNN GIL stalls."""

import multiprocessing as mp
from queue import Empty, Full
import threading
import time
from model_registry import LABELS, model_catalog, model_classes


def inference_worker(frames, results, commands, stopped, backend, threads, confidence, override):
    from detectors import create_detector
    from model_control import ModelControl
    from runtime_metrics import RuntimeMetrics
    import cv2
    import numpy as np
    from annotation import annotate

    cv2.setNumThreads(1)

    def factory(name):
        return create_detector(name, threads, confidence, override if name == backend else None)

    previous_finished = None
    try:
        control = ModelControl(backend, factory(backend), factory, confidence, RuntimeMetrics())
        while not stopped.is_set():
            try:
                frame, frame_id, stamp = frames.get(timeout=0.2)
            except Empty:
                continue
            while True:
                try:
                    name, threshold = commands.get_nowait()
                    control.request(name, threshold)
                except Empty:
                    break
            jpeg = frame if isinstance(frame, bytes) else None
            if isinstance(frame, bytes):
                frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("Cannot decode camera JPEG")
            detector = control.apply(frame)
            start_inference = time.monotonic()
            detections = detector.predict(frame)
            ms = (time.monotonic() - start_inference) * 1000
            state = control.status()
            if jpeg is None:
                annotated = frame.copy()
                annotate(annotated, detections, control.active)
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                jpeg = encoded.tobytes()
            finished = time.monotonic()
            frame_fps = (
                1 / (finished - previous_finished) if previous_finished is not None else None
            )
            previous_finished = finished
            payload = {
                "jpeg": jpeg,
                "frame_fps": frame_fps,
                "frame_id": frame_id,
                "captured_at": stamp,
                "ms": ms,
                "detections": detections,
                "state": {k: state[k] for k in ["active", "confidence", "classes", "error"]},
                "finished_at": finished,
            }
            while not stopped.is_set():
                try:
                    results.put(payload, timeout=0.2)
                    break
                except Full:
                    continue
    except Exception as exc:
        results.put({"fatal": str(exc)})


class Control:
    def __init__(self, args, commands, metrics):
        self.commands, self.metrics = commands, metrics
        self.browser_overlay = args.capture_backend == "gstreamer" and (
            args.source.isdigit() or args.source.startswith("/dev/video")
        )
        self.width, self.height = args.width, args.height
        self.lock = threading.Lock()
        self.active = self.requested = args.backend
        self.confidence = args.confidence
        self.requested_confidence = args.confidence
        self.error = None
        self.objects = 0
        self.ready = False
        self.finished_at = None
        self.captured_at = None
        self.frame_fps = None
        self.detections = []
        self.classes = model_classes(args.backend)

    def request(self, backend, confidence):
        if backend not in LABELS:
            raise ValueError("Unknown model")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.25 <= confidence <= 0.95
        ):
            raise ValueError("Confidence must be between 0.25 and 0.95")
        with self.lock:
            self.requested = backend
            self.requested_confidence = confidence
            self.detections = []
            self.error = None
            self.ready = False
            self.commands.put((backend, confidence))
        return self.status()

    def result(self, payload):
        if "fatal" in payload:
            raise RuntimeError("Detector process: " + payload["fatal"])
        self.metrics.complete(payload["frame_id"], payload["captured_at"], payload["ms"])
        with self.lock:
            self.frame_fps = payload["frame_fps"]
            self.active = payload["state"]["active"]
            self.confidence = payload["state"]["confidence"]
            self.classes = payload["state"]["classes"]
            self.error = payload["state"]["error"]
            self.finished_at = payload["finished_at"]
            self.captured_at = payload["captured_at"]
            self.ready = (
                self.active == self.requested and self.confidence == self.requested_confidence
            ) or bool(self.error)
            self.detections = payload["detections"] if self.ready else []
            self.objects = len(self.detections)

    def status(self):
        with self.lock:
            age = (time.monotonic() - self.captured_at) * 1000 if self.captured_at else None
            measured = self.metrics.snapshot()
            return {
                "models": model_catalog(),
                "active": self.active,
                "requested": self.requested,
                "switching": not self.ready,
                "confidence": self.confidence,
                "classes": self.classes,
                "objects": self.objects,
                "error": self.error,
                "frame_fps": round(self.frame_fps, 2) if self.frame_fps is not None else None,
                "video_mode": "completed_detections",
                "browser_overlay": self.browser_overlay,
                "frame_size": [self.width, self.height],
                "detections": self.detections,
                "async_detection": True,
                "detection_age_ms": round(age, 1) if age is not None else None,
                **measured,
            }


def run(args):
    from camera import Camera
    from runtime_metrics import RuntimeMetrics
    from web_preview import Preview

    ctx = mp.get_context("spawn")
    frames = ctx.Queue(maxsize=1)
    results = ctx.Queue(maxsize=4)
    commands = ctx.Queue()
    stopped = ctx.Event()
    metrics = RuntimeMetrics()
    control = Control(args, commands, metrics)
    worker = ctx.Process(
        target=inference_worker,
        args=(
            frames,
            results,
            commands,
            stopped,
            args.backend,
            args.threads,
            args.confidence,
            args.model,
        ),
        daemon=True,
    )
    camera = preview = None
    worker.start()
    try:
        camera = Camera(
            args.source,
            args.width,
            args.height,
            metrics,
            args.capture_backend,
            raw_jpeg=control.browser_overlay,
        )
        preview = Preview(args.host, args.port, control)
        print(f"Preview: http://{args.host}:{args.port} | completed detection frames", flush=True)
        report = time.monotonic()
        while True:
            while True:
                try:
                    payload = results.get_nowait()
                    control.result(payload)
                    if preview.publish_jpeg(payload["jpeg"]):
                        metrics.publish()
                except Empty:
                    break
            if not worker.is_alive():
                raise RuntimeError("Detector process stopped")
            packet = camera.read()
            if packet is None:
                time.sleep(0.001)
                continue
            frame, frame_id, stamp = packet
            inference_packet = (
                frame if isinstance(frame, bytes) else frame.copy(),
                frame_id,
                stamp,
            )
            try:
                frames.put_nowait(inference_packet)
            except Full:
                # Replace the waiting input so slow models start with the newest image.
                try:
                    frames.get_nowait()
                except Empty:
                    pass
                try:
                    frames.put_nowait(inference_packet)
                except Full:
                    pass
            if time.monotonic() - report >= 3:
                s = metrics.snapshot()
                print(
                    f"Camera {s['camera_fps']:.1f} FPS | detection {s['pipeline_fps']:.1f} FPS | preview {s['preview_fps']:.1f} FPS",
                    flush=True,
                )
                report = time.monotonic()
    finally:
        stopped.set()
        if camera:
            camera.close()
        if preview:
            preview.close()
        worker.join(timeout=3)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
        for queue in [frames, results, commands]:
            queue.cancel_join_thread()
            queue.close()
