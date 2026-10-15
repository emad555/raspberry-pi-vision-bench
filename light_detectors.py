"""NCNN MobileNet SSD backends; preprocessing follows Tencent NCNN examples."""

import ncnn
import numpy as np

from model_registry import VOC


class SSDDetector:
    def __init__(self, model, threads, confidence, variant):
        self.variant = variant
        self.confidence = confidence
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        # MobileNetV2 export has a training-only Silence layer with no outputs.
        self.noop = None
        if variant == "ssdlite-v2":

            class Noop(ncnn.Layer):
                pass

            self.noop = Noop
            self.net.register_custom_layer("Silence", Noop, lambda layer: None)
        if self.net.load_param(str(model / "model.param")) or self.net.load_model(
            str(model / "model.bin")
        ):
            raise RuntimeError(f"Cannot load {variant} model")

    def predict(self, frame):
        h, w = frame.shape[:2]
        v3 = self.variant == "ssdlite-v3"
        pixel = ncnn.Mat.PixelType.PIXEL_BGR2RGB if v3 else ncnn.Mat.PixelType.PIXEL_BGR
        mat = ncnn.Mat.from_pixels_resize(frame, pixel, w, h, 300, 300)
        if self.variant == "squeezenet":
            mat.substract_mean_normalize([104.0, 117.0, 123.0], [])
        else:
            mat.substract_mean_normalize(
                [123.675, 116.28, 103.53] if v3 else [127.5] * 3,
                [1.0] * 3 if v3 else [1 / 127.5] * 3,
            )
        with self.net.create_extractor() as ex:
            if ex.input("input" if v3 else "data", mat):
                raise RuntimeError("SSD input failed")
            status, out = ex.extract("detection_out")
            if status:
                raise RuntimeError(f"SSD extraction failed: {status}")
            # NCNN DetectionOutput returns an empty Mat when no objects pass NMS.
            # Its Python NumPy conversion can segfault on that empty buffer.
            if out.empty():
                return []
            rows = np.array(out).copy()
        detections = []
        for row in rows:
            cls, score = int(row[0]), float(row[1])
            if cls <= 0 or cls >= len(VOC) or score < self.confidence:
                continue
            coords = np.clip(row[2:6], 0, 1) * np.array([w, h, w, h])
            detections.append(
                {"label": VOC[cls], "confidence": score, "box": coords.astype(int).tolist()}
            )
        return detections
