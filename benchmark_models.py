"""Compare installed models on one image; excludes camera and web delivery."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark-results.json"))
    args = parser.parse_args()
    if args.runs < 1 or args.threads < 1:
        parser.error("runs and threads must be positive")
    import cv2
    from detectors import create_detector
    from model_registry import model_catalog

    cv2.setNumThreads(1)
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    results = []
    for entry in model_catalog():
        if not entry["available"]:
            continue
        detector = create_detector(entry["id"], args.threads)
        for _ in range(3):
            detector.predict(image)
        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            detections = detector.predict(image)
            timings.append(time.perf_counter() - start)
        mean = sum(timings) / len(timings)
        row = {
            "backend": entry["id"],
            "runs": args.runs,
            "threads": args.threads,
            "mean_ms": mean * 1000,
            "image_inference_fps": 1 / mean,
            "detections": detections,
        }
        results.append(row)
        print(f"{entry['name']}: {1 / mean:.2f} image inference FPS", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
