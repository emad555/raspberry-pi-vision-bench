from queue import Queue
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from async_web import Control
from model_control import ModelControl
from model_registry import model_catalog
from runtime_metrics import RuntimeMetrics


class FakeDetector:
    confidence = 0.35
    names = {0: "person"}

    def predict(self, frame):
        return []


class ModelControlTests(unittest.TestCase):
    def test_failed_model_load_keeps_current_detector(self):
        original = FakeDetector()

        def fail(backend):
            raise RuntimeError("missing weights")

        control = ModelControl("ssdlite-v2", original, fail, 0.35, RuntimeMetrics())
        control.request("yolo-small", 0.5)
        self.assertIs(control.apply(None), original)
        self.assertEqual(control.status()["active"], "ssdlite-v2")
        self.assertIn("missing weights", control.status()["error"])

    def test_cached_model_is_reused(self):
        loaded = []

        def factory(backend):
            loaded.append(backend)
            return FakeDetector()

        control = ModelControl("ssdlite-v2", FakeDetector(), factory, 0.35, RuntimeMetrics())
        control.request("ssd", 0.5)
        first = control.apply(None)
        control.request("ssdlite-v2", 0.35)
        control.apply(None)
        control.request("ssd", 0.6)
        second = control.apply(None)
        self.assertIs(first, second)
        self.assertEqual(loaded, ["ssd"])
        self.assertEqual(second.confidence, 0.6)

    def test_missing_assets_are_disabled_in_catalog(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("model_registry.MODELS_DIR", Path(folder)):
                self.assertFalse(any(item["available"] for item in model_catalog()))

    def test_invalid_controls_are_rejected_before_queueing(self):
        args = SimpleNamespace(
            capture_backend="gstreamer",
            source="0",
            width=640,
            height=480,
            backend="ssdlite-v2",
            confidence=0.35,
        )
        commands = Queue()
        control = Control(args, commands, RuntimeMetrics())
        for value in [True, float("nan"), 0, 0.99]:
            with self.assertRaises(ValueError):
                control.request("ssd", value)
        with self.assertRaises(ValueError):
            control.request("missing", 0.35)
        self.assertTrue(commands.empty())
