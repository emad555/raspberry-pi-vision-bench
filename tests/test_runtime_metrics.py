import unittest
from runtime_metrics import RuntimeMetrics


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.metrics = RuntimeMetrics(clock=self.clock)

    def test_counts_frames_not_inverse_inference_time(self):
        for i in range(90):
            self.clock.now = (i + 1) / 30
            frame_id, stamp = self.metrics.capture()
            self.metrics.complete(frame_id, stamp, 1, published=True)
        result = self.metrics.snapshot()
        self.assertEqual(result["camera_fps"], 30)
        self.assertEqual(result["pipeline_fps"], 30)
        self.assertEqual(result["preview_fps"], 30)
        self.assertEqual(result["inference_ms"], 1)
        self.assertEqual(result["processed_frames"], 90)

    def test_capture_drops_and_publish_failures(self):
        self.clock.now = 1
        self.metrics.capture()
        frame_id, stamp = self.metrics.capture(replaced=True)
        self.metrics.complete(frame_id, stamp, 20, published=False)
        result = self.metrics.snapshot()
        self.assertEqual(result["camera_fps"], 2)
        self.assertEqual(result["pipeline_fps"], 1)
        self.assertEqual(result["preview_fps"], 0)
        self.assertEqual(result["dropped_capture_frames"], 1)
        with self.assertRaises(ValueError):
            self.metrics.complete(frame_id, stamp, 20)

    def test_idle_and_switch_pauses_reduce_rate_and_expire(self):
        self.clock.now = 1
        frame_id, stamp = self.metrics.capture()
        self.metrics.complete(frame_id, stamp, 20, True)
        self.clock.now = 3
        self.assertAlmostEqual(self.metrics.snapshot()["pipeline_fps"], 0.33, places=2)
        self.clock.now = 4
        result = self.metrics.snapshot()
        self.assertEqual(result["pipeline_fps"], 0)
        self.assertIsNone(result["inference_ms"])
        self.assertTrue(result["stale"])
        self.assertEqual(result["processed_frames"], 1)

    def test_video_and_detection_counts_are_independent(self):
        for i in range(90):
            self.clock.now = (i + 1) / 30
            frame_id, stamp = self.metrics.capture()
            self.metrics.publish()
            if i % 3 == 2:
                self.metrics.complete(frame_id, stamp, 90)
        result = self.metrics.snapshot()
        self.assertEqual(result["camera_fps"], 30)
        self.assertEqual(result["preview_fps"], 30)
        self.assertEqual(result["pipeline_fps"], 10)
        self.assertEqual(result["published_frames"], 90)
        self.assertEqual(result["processed_frames"], 30)

    def test_means_and_latency(self):
        for timestamp, ms in [(1, 10), (2, 30)]:
            self.clock.now = timestamp
            frame_id, stamp = self.metrics.capture()
            self.clock.now += 0.1
            self.metrics.complete(frame_id, stamp, ms)
        result = self.metrics.snapshot()
        self.assertEqual(result["inference_ms"], 20)
        self.assertEqual(result["capture_to_output_ms"], 100)


if __name__ == "__main__":
    unittest.main()
