"""Queue model changes for the inference thread; HTTP never touches NCNN."""

import threading
import time

from model_registry import LABELS, VOC, model_catalog


class ModelControl:
    def __init__(self, backend, detector, factory, confidence, metrics):
        self.metrics = metrics
        self.lock = threading.Lock()
        self.factory = factory
        self.cache = {backend: detector}
        self.detector = detector
        self.active = backend
        self.requested = backend
        self.confidence = confidence
        self.revision = 0
        self.applied = 0
        self.error = None
        self.switching = False
        self.ms = 0
        self.objects = 0
        self.updated = time.monotonic()

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
            self.confidence = confidence
            self.revision += 1
            self.error = None
        return self.status()

    def apply(self, frame):
        with self.lock:
            revision, backend, confidence = self.revision, self.requested, self.confidence
            if revision == self.applied:
                return self.detector
            self.switching = True
        try:
            candidate = self.cache.get(backend)
            if candidate is None:
                candidate = self.factory(backend)
                candidate.confidence = confidence
                candidate.predict(frame)  # Verify a newly loaded model before swapping.
                self.cache[backend] = candidate
            candidate.confidence = confidence
            with self.lock:
                self.detector = candidate
                self.active = backend
                self.error = None
        except Exception as exc:
            with self.lock:
                self.error = f"Could not switch model: {exc}"
        finally:
            with self.lock:
                self.applied = revision
                self.switching = False
        return self.detector

    def update(self, ms, objects):
        with self.lock:
            self.ms, self.objects = ms, objects
            self.updated = time.monotonic()

    def status(self):
        with self.lock:
            detector = self.detector
            names = getattr(detector, "names", None)
            if names is None:
                classes = VOC[1:]
            else:
                classes = list(names.values()) if isinstance(names, dict) else list(names)
            return {
                "models": model_catalog(),
                "active": self.active,
                "requested": self.requested,
                "switching": self.switching or self.applied != self.revision,
                "confidence": detector.confidence,
                "classes": classes,
                "inference_ms": round(self.ms, 1),
                "objects": self.objects,
                "error": self.error,
                "stale": time.monotonic() - self.updated > 5,
                **self.metrics.snapshot(),
            }
