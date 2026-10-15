"""Download pinned SSD weights and verify SHA-256 before replacing any asset."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

from model_registry import ROOT


def install_asset(entry, root=ROOT):
    """Install one asset atomically; reject traversal and checksum mismatches."""
    root = Path(root).resolve()
    destination = (root / entry["path"]).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Model destination must stay inside the project")
    if (
        destination.is_file()
        and hashlib.sha256(destination.read_bytes()).hexdigest() == entry["sha256"]
    ):
        return False
    if not entry["url"].startswith("https://"):
        raise ValueError("Asset URL must use HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        digest = hashlib.sha256()
        request = urllib.request.Request(
            entry["url"], headers={"User-Agent": "raspberry-pi-vision-bench"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    stream.write(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch for {destination.name}; existing file preserved")
        temporary.replace(destination)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "models/ssd-manifest.json")
    args = parser.parse_args()
    for entry in json.loads(args.manifest.read_text()):
        if install_asset(entry):
            print(f"Downloaded {entry['path']}")
    print("SSD model assets verified. YOLO models are optional: see README.md.")


if __name__ == "__main__":
    main()
