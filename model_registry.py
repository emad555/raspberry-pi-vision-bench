"""Model names, labels, and asset availability (no inference dependencies)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
VOC = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
MODELS = {
    "ssd": "mobilenet-ssd",
    "ssdlite-v2": "mobilenetv2_ssdlite_voc",
    "ssdlite-v3": "mobilenetv3_ssdlite_voc",
    "squeezenet": "squeezenet-ssd",
}

LABELS = {
    "ssdlite-v2": "MobileNetV2 SSDLite",
    "ssd": "MobileNet SSD",
    "ssdlite-v3": "MobileNetV3 SSDLite",
    "squeezenet": "SqueezeNet SSD",
    "yolo": "YOLO11 Nano",
    "yolo-small": "YOLO11 Small",
    "yolo-medium": "YOLO11 Medium",
}
YOLO_MODELS = {"yolo": "yolo11n", "yolo-small": "yolo11s", "yolo-medium": "yolo11m"}
COCO = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def model_classes(backend):
    return list(COCO if backend in YOLO_MODELS else VOC[1:])


def model_catalog():
    catalog = []
    for key, label in LABELS.items():
        folder = MODELS_DIR / (YOLO_MODELS[key] if key in YOLO_MODELS else MODELS[key])
        required = (
            ["model.ncnn.param", "model.ncnn.bin", "metadata.yaml"]
            if key in YOLO_MODELS
            else ["model.param", "model.bin"]
        )
        catalog.append(
            {
                "id": key,
                "name": label,
                "class_count": len(model_classes(key)),
                "available": all((folder / name).is_file() for name in required),
            }
        )
    return catalog
