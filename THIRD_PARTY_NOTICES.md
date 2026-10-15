# Third-party notices

This repository contains application source and asset provenance; downloaded model binaries are not committed.

- **NCNN** is developed by Tencent and contributors under the BSD 3-Clause license: https://github.com/Tencent/ncnn/blob/master/LICENSE.txt. SSD preprocessing/output layouts follow the NCNN reference examples. Their copyright and license notice is reproduced in `docs/NCNN-LICENSE.txt`.
- **Ultralytics YOLO11** model weights and export tooling use upstream licensing described at https://www.ultralytics.com/license. The optional exporter downloads official release checkpoints. See each generated model's metadata.
- **SSD models** originate from the projects linked in `models/README.md`; converted assets are retrieved from pinned revisions of https://github.com/nihui/ncnn-assets. Downloaded weights keep their original terms.
- **OpenCV**, **GStreamer**, **Picamera2**, **PyTorch**, and other installed dependencies retain their own licenses. They are installed dependencies, not vendored source.

The project license applies to this repository's application code and does not replace third-party model or dependency terms.
