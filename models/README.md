# Model assets

Weights are downloaded locally and are excluded from Git.

- `python download_models.py` installs the four SSD variants from immutable upstream URLs in `ssd-manifest.json`, with SHA-256 verification and atomic replacement.
- `python export_yolo_models.py` installs optional YOLO11 Nano, Small, and Medium exports. See the root README for the export environment.
- The web selector disables models whose required files are missing.
- `yolo-sm-manifest.json` records the original tested Small/Medium export hashes and weight origins. Re-exported files may have different hashes across export environments.

SSD origins: [MobileNet SSD](https://github.com/chuanqi305/MobileNet-SSD), [MobileNetV2 SSDLite](https://github.com/chuanqi305/MobileNetv2-SSDLite), [MobileNetV3 SSD](https://github.com/ujsyehao/mobilenetv3-ssd), and [SqueezeNet SSD](https://github.com/chuanqi305/SqueezeNet-SSD). NCNN conversions come from [nihui/ncnn-assets](https://github.com/nihui/ncnn-assets).

YOLO weights: [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0). YOLO model metadata identifies AGPL-3.0; see [Ultralytics licensing](https://www.ultralytics.com/license). Model weights keep their upstream terms; this repository does not relicense them.
