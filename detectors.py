"""NCNN YOLO detection and shared backend construction."""

from pathlib import Path
import cv2
import ncnn
import numpy as np
import yaml
from model_registry import MODELS_DIR, YOLO_MODELS


class YOLODetector:
    def __init__(self, model, threads, confidence):
        self.net = ncnn.Net()
        self.net.opt.num_threads = threads
        self.net.opt.use_vulkan_compute = False
        if self.net.load_param(str(model / "model.ncnn.param")) or self.net.load_model(
            str(model / "model.ncnn.bin")
        ):
            raise RuntimeError("Cannot load NCNN model")
        metadata = yaml.safe_load((model / "metadata.yaml").read_text())
        self.size = int(metadata["imgsz"][0])
        if metadata.get("task") != "detect":
            raise ValueError("This runner requires a YOLO detection NCNN export")
        self.names = metadata["names"]
        self.confidence = confidence

    def predict(self, frame):
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        nw, nh = round(w * scale), round(h * scale)
        left, top = (self.size - nw) // 2, (self.size - nh) // 2
        resized = cv2.resize(frame, (nw, nh))
        padded = cv2.copyMakeBorder(
            resized,
            top,
            self.size - nh - top,
            left,
            self.size - nw - left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        mat = ncnn.Mat.from_pixels(padded, ncnn.Mat.PixelType.PIXEL_BGR2RGB, self.size, self.size)
        mat.substract_mean_normalize([], [1 / 255.0] * 3)
        with self.net.create_extractor() as extractor:
            extractor.input("in0", mat)
            status, output = extractor.extract("out0")
            if status:
                raise RuntimeError(f"NCNN extraction failed: {status}")
            predictions = np.array(output).copy()
        classes_count = len(self.names)
        if predictions.shape[0] == classes_count + 4:
            predictions = predictions.T
        if predictions.ndim != 2 or predictions.shape[1] != classes_count + 4:
            raise ValueError(f"Unsupported YOLO output shape: {predictions.shape}")
        ids = predictions[:, 4:].argmax(axis=1)
        scores = predictions[np.arange(len(predictions)), ids + 4]
        keep = scores >= self.confidence
        predictions, ids, scores = predictions[keep], ids[keep], scores[keep]
        boxes = predictions[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 0] = (boxes[:, 0] - left) / scale
        boxes[:, 1] = (boxes[:, 1] - top) / scale
        boxes[:, 2:] /= scale
        selected = []
        for cls in np.unique(ids):
            indices = np.where(ids == cls)[0]
            retained = cv2.dnn.NMSBoxes(
                boxes[indices].tolist(), scores[indices].tolist(), self.confidence, 0.45
            )
            selected.extend(indices[np.asarray(retained, dtype=int).reshape(-1)].tolist())
        detections = []
        for i in selected:
            x, y, bw, bh = boxes[i]
            box = [
                int(np.clip(x, 0, w)),
                int(np.clip(y, 0, h)),
                int(np.clip(x + bw, 0, w)),
                int(np.clip(y + bh, 0, h)),
            ]
            label = self.names[int(ids[i])]
            detections.append({"label": label, "confidence": float(scores[i]), "box": box})
        return detections


def create_detector(backend, threads=2, confidence=0.35, model_dir=None):
    """Build a detector without importing training/export dependencies."""
    from model_registry import MODELS

    folder = (
        Path(model_dir)
        if model_dir
        else MODELS_DIR / (YOLO_MODELS[backend] if backend in YOLO_MODELS else MODELS[backend])
    )
    required = (
        ["model.ncnn.param", "model.ncnn.bin", "metadata.yaml"]
        if backend in YOLO_MODELS
        else ["model.param", "model.bin"]
    )
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing model assets in {folder}: {', '.join(missing)}. See README.md for download/export instructions."
        )
    if backend in YOLO_MODELS:
        return YOLODetector(
            Path(model_dir) if model_dir else MODELS_DIR / YOLO_MODELS[backend], threads, confidence
        )
    from light_detectors import SSDDetector

    return SSDDetector(
        Path(model_dir) if model_dir else MODELS_DIR / MODELS[backend], threads, confidence, backend
    )
