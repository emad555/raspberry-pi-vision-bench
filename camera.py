"""Latest-frame camera capture for USB, CSI, and video files."""

import threading
import cv2


class Camera:
    def __init__(self, source, width, height, metrics, capture_backend="gstreamer", raw_jpeg=False):
        self.metrics = metrics
        self.pi = None
        self.cap = None
        self.frame = None
        self.error = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        if source == "pi":
            from picamera2 import Picamera2

            if not Picamera2.global_camera_info():
                raise RuntimeError(
                    "No Pi camera detected. Check the CSI cable with power off, or use --source 0 for a USB camera."
                )
            self.pi = Picamera2()
            self.pi.configure(
                self.pi.create_video_configuration(
                    main={"size": (width, height), "format": "RGB888"}, buffer_count=4
                )
            )
            self.pi.start()
        else:
            device = int(source) if source.isdigit() else source
            usb = source.isdigit() or source.startswith("/dev/video")
            self.gst_capture = usb and capture_backend == "gstreamer"
            if self.gst_capture:
                from gst_camera import GStreamerCapture

                self.cap = GStreamerCapture(source, width, height, raw_jpeg=raw_jpeg)
            else:
                self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2 if usb else cv2.CAP_ANY)
                if usb:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if usb:
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    self.cap.release()
                    raise RuntimeError(f"Cannot open camera/video: {source}")
        self.thread = threading.Thread(target=self.capture, daemon=True)
        self.thread.start()

    def capture(self):
        try:
            while not self.stop.is_set():
                if self.pi:
                    frame = self.pi.capture_array()
                else:
                    ok, frame = self.cap.read()
                    if not ok:
                        raise RuntimeError("Camera disconnected or video ended")
                with self.lock:
                    frame_id, captured_at = self.metrics.capture(replaced=self.frame is not None)
                    self.frame = (
                        frame if isinstance(frame, bytes) else frame.copy(),
                        frame_id,
                        captured_at,
                    )
        except Exception as exc:
            self.error = exc

    def read(self):
        with self.lock:
            frame, self.frame = self.frame, None
        if frame is None and self.error:
            raise self.error
        return frame

    def close(self):
        self.stop.set()
        if getattr(self, "gst_capture", False):
            self.cap.release()  # NULL state unblocks a waiting appsink read.
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            # Never release a native capture while another thread is inside read().
            return
        if self.pi:
            self.pi.stop()
            self.pi.close()
        if self.cap:
            self.cap.release()
