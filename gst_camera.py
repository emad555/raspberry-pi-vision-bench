"""USB MJPEG capture and decoding in GStreamer's native worker threads."""

import re
import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo  # noqa: E402 - require_version must precede GI imports


class GStreamerCapture:
    def __init__(self, source, width, height, fps=30, raw_jpeg=False):
        Gst.init(None)
        device = f"/dev/video{source}" if source.isdigit() else source
        if not re.fullmatch(r"/dev/video\d+", device):
            raise ValueError("Expected a USB /dev/videoN device")
        self.raw_jpeg = raw_jpeg
        decode = "" if raw_jpeg else "! jpegdec ! videoconvert ! video/x-raw,format=BGR "
        self.pipeline = Gst.parse_launch(
            f"v4l2src device={device} do-timestamp=true "
            f"! image/jpeg,width={width},height={height},framerate={fps}/1 "
            f"{decode}! appsink name=frames sync=false max-buffers=1 drop=true"
        )
        self.sink = self.pipeline.get_by_name("frames")
        self.bus = self.pipeline.get_bus()
        self.closed = False
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            message = self.bus.pop_filtered(Gst.MessageType.ERROR)
            detail = message.parse_error()[0].message if message else "Camera unavailable"
            self.release()
            raise RuntimeError(f"Cannot start USB capture: {detail}")

    def read(self):
        sample = self.sink.emit("try-pull-sample", 2 * Gst.SECOND)
        if sample is None:
            if self.closed:
                return False, None
            message = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if message and message.type == Gst.MessageType.ERROR:
                error, detail = message.parse_error()
                raise RuntimeError(f"USB capture: {error.message}")
            raise RuntimeError(
                "No USB camera frames; check camera connection and MJPEG resolution support"
            )
        if self.raw_jpeg:
            buffer = sample.get_buffer()
            return True, buffer.extract_dup(0, buffer.get_size())
        info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("Cannot map camera frame")
        try:
            frame = np.ndarray(
                (info.height, info.width, 3),
                dtype=np.uint8,
                buffer=mapped.data,
                offset=info.offset[0],
                strides=(info.stride[0], 3, 1),
            ).copy()
        finally:
            buffer.unmap(mapped)
        return True, frame

    def release(self):
        if not self.closed:
            self.closed = True
            self.pipeline.set_state(Gst.State.NULL)
