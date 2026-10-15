"""Shared H.264 RTSP output. Retains only the latest annotated frame."""

import threading
import time
import cv2
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib  # noqa: E402 - require_version first


class RTSPOutput:
    def __init__(self, host, port, path, width, height, fps, bitrate):
        Gst.init(None)
        for element in ["appsrc", "videoconvert", "x264enc", "h264parse", "rtph264pay"]:
            if Gst.ElementFactory.find(element) is None:
                raise RuntimeError(f"Missing GStreamer element: {element}. Run setup-rtsp.sh")
        self.width, self.height = width, height
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.closed = False
        self.loop = GLib.MainLoop()
        self.server = GstRtspServer.RTSPServer()
        self.server.set_address(host)
        self.server.set_service(str(port))
        self.factory = GstRtspServer.RTSPMediaFactory()
        self.factory.set_shared(True)
        self.factory.set_launch(
            f"( appsrc name=frames is-live=true format=time do-timestamp=false "
            f"block=false max-buffers=2 leaky-type=downstream "
            f'caps="video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1" '
            f"! queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream "
            f"! videoconvert ! video/x-raw,format=I420 "
            f"! x264enc tune=zerolatency speed-preset=ultrafast bitrate={bitrate} "
            f"key-int-max={fps} threads=1 bframes=0 "
            f"! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
        )
        self.factory.connect("media-configure", self.configure)
        self.server.get_mount_points().add_factory(path, self.factory)
        self.source_id = self.server.attach(None)
        if not self.source_id:
            raise RuntimeError(f"Cannot bind RTSP server on {host}:{port}")
        self.thread = threading.Thread(target=self.loop.run, daemon=True)
        self.thread.start()

    def configure(self, factory, media):
        source = media.get_element().get_by_name("frames")
        session = {"last": -1, "stopped": False, "start": None}

        def stop(_media):
            with self.condition:
                session["stopped"] = True
                self.condition.notify_all()

        def need_data(appsrc, length):
            with self.condition:
                self.condition.wait_for(
                    lambda: self.closed
                    or session["stopped"]
                    or (self.latest is not None and self.sequence != session["last"])
                )
                if self.closed or session["stopped"]:
                    appsrc.emit("end-of-stream")
                    return
                data = self.latest
                session["last"] = self.sequence
            buffer = Gst.Buffer.new_allocate(None, len(data), None)
            buffer.fill(0, data)
            now = time.monotonic_ns()
            if session["start"] is None:
                session["start"] = now
            buffer.pts = now - session["start"]
            buffer.dts = buffer.pts
            appsrc.emit("push-buffer", buffer)

        media.connect("unprepared", stop)
        source.connect("need-data", need_data)

    def publish(self, frame):
        if frame.shape[:2] != (self.height, self.width):
            frame = cv2.resize(frame, (self.width, self.height))
        data = frame.tobytes()
        with self.condition:
            self.latest = data
            self.sequence += 1
            self.condition.notify_all()
        return True

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        GLib.source_remove(self.source_id)
        self.server.client_filter(lambda server, client: GstRtspServer.RTSPFilterResult.REMOVE)
        self.loop.quit()
        self.thread.join(timeout=3)
