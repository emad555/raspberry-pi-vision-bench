"""Export official YOLO11 detection models to the pinned NCNN runtime."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

from model_registry import MODELS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=["n", "s", "m"], default=["n", "s", "m"])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    # Export-only dependencies never get imported by the detection app.
    import pnnx
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    converter = Path(pnnx.__file__).parent / "pnnx"
    if not converter.is_file():
        raise RuntimeError(
            "PNNX binary missing. Install requirements-export.txt in an export venv."
        )
    with tempfile.TemporaryDirectory(prefix="yolo-export-") as directory:
        work = Path(directory)
        for size in args.sizes:
            name = f"yolo11{size}"
            checkpoint = work / f"{name}.pt"
            url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}.pt"
            urllib.request.urlretrieve(url, checkpoint)
            model = YOLO(str(checkpoint))
            torchscript = Path(model.export(format="torchscript", imgsz=640, device="cpu", batch=1))
            export = work / f"{name}_ncnn"
            export.mkdir()
            # Invoke the pinned wheel binary, avoiding Ultralytics' latest-PNNX auto-download.
            subprocess.run(
                [
                    str(converter),
                    str(torchscript),
                    f"ncnnparam={export}/model.ncnn.param",
                    f"ncnnbin={export}/model.ncnn.bin",
                    "inputshape=[1,3,640,640]",
                    "fp16=0",
                ],
                cwd=work,
                check=True,
            )
            import yaml

            metadata = {
                "description": f"Ultralytics {name} COCO detection model",
                "task": "detect",
                "imgsz": [640, 640],
                "stride": 32,
                "names": model.names,
                "license": "AGPL-3.0",
                "source": url,
                "runtime": "ncnn==1.0.20260526",
                "converter": "pnnx==20260526",
            }
            (export / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
            destination = MODELS_DIR / name
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ["model.ncnn.param", "model.ncnn.bin", "metadata.yaml"]:
                shutil.copy2(export / filename, destination / filename)
            print(f"Installed {destination}", flush=True)


if __name__ == "__main__":
    main()
