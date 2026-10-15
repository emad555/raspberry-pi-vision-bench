"""Actual frame counts over a rolling monotonic-time window, never 1 / inference time."""

from collections import deque
import threading
import time


class RuntimeMetrics:
    def __init__(self, window=3.0, clock=time.monotonic):
        self.clock, self.window = clock, window
        self.started = clock()
        self.lock = threading.Lock()
        self.events = {name: deque() for name in ["captured", "processed", "published"]}
        self.totals = {name: 0 for name in self.events}
        self.samples = deque()
        self.dropped = 0
        self.last_processed_id = 0

    def capture(self, replaced=False):
        now = self.clock()
        with self.lock:
            self.events["captured"].append(now)
            self.totals["captured"] += 1
            self.dropped += int(replaced)
            return self.totals["captured"], now

    def complete(self, frame_id, captured_at, inference_ms, published=False):
        now = self.clock()
        with self.lock:
            if frame_id <= self.last_processed_id:
                raise ValueError("A captured frame must be processed only once")
            self.last_processed_id = frame_id
            self.events["processed"].append(now)
            self.totals["processed"] += 1
            if published:
                self.events["published"].append(now)
                self.totals["published"] += 1
            self.samples.append((now, inference_ms, (now - captured_at) * 1000))

    def publish(self):
        with self.lock:
            self.events["published"].append(self.clock())
            self.totals["published"] += 1

    def snapshot(self):
        now = self.clock()
        with self.lock:
            cutoff = now - self.window
            for queue in self.events.values():
                while queue and queue[0] <= cutoff:
                    queue.popleft()
            while self.samples and self.samples[0][0] <= cutoff:
                self.samples.popleft()
            seconds = min(self.window, max(0, now - self.started))
            rates = {
                name: len(queue) / seconds if seconds > 0 else 0
                for name, queue in self.events.items()
            }
            mean_ms = sum(s[1] for s in self.samples) / len(self.samples) if self.samples else None
            latency = sum(s[2] for s in self.samples) / len(self.samples) if self.samples else None
            return {
                "camera_fps": round(rates["captured"], 2),
                "pipeline_fps": round(rates["processed"], 2),
                "preview_fps": round(rates["published"], 2),
                "inference_ms": round(mean_ms, 2) if mean_ms is not None else None,
                "capture_to_output_ms": round(latency, 2) if latency is not None else None,
                "captured_frames": self.totals["captured"],
                "processed_frames": self.totals["processed"],
                "published_frames": self.totals["published"],
                "dropped_capture_frames": self.dropped,
                "fps_window_seconds": round(seconds, 3),
                "uptime_seconds": round(now - self.started, 3),
                "stale": not self.samples or now - self.samples[-1][0] > 2,
            }
