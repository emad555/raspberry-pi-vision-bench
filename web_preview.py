"""Browser MJPEG stream and model-control API."""

import json
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Preview:
    def __init__(self, host, port, control):
        self.jpeg = None
        self.condition = threading.Condition()
        self.closed = False
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def respond(self, status, body, mime="application/json"):
                if mime == "application/json":
                    body = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                assets = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                }
                if self.path in assets:
                    filename, mime = assets[self.path]
                    self.respond(
                        200,
                        (Path(__file__).parent / "web" / filename).read_bytes(),
                        mime,
                    )
                elif self.path == "/api/status":
                    self.respond(200, control.status())
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        last = None
                        while not preview.closed:
                            with preview.condition:
                                preview.condition.wait_for(
                                    lambda: preview.jpeg is not last or preview.closed, timeout=2
                                )
                                jpeg = preview.jpeg
                            if jpeg is None or jpeg is last:
                                continue
                            last = jpeg
                            self.wfile.write(
                                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                + str(len(jpeg)).encode()
                                + b"\r\n\r\n"
                                + jpeg
                                + b"\r\n"
                            )
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.respond(404, {"error": "Not found"})

            def do_POST(self):
                if self.path != "/api/model":
                    self.respond(404, {"error": "Not found"})
                    return
                # JSON-only same-origin requests; no permissive CORS.
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self.respond(415, {"error": "Use application/json"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 2048:
                        raise ValueError("Invalid request size")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("Expected a JSON object")
                    result = control.request(
                        body.get("model"), body.get("confidence", control.status()["confidence"])
                    )
                    self.respond(202, result)
                except (ValueError, TypeError) as exc:
                    self.respond(400, {"error": str(exc)})

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish_jpeg(self, data):
        with self.condition:
            self.jpeg = data
            self.condition.notify_all()
        return True

    def publish(self, frame):
        import cv2

        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.condition:
                self.jpeg = jpeg.tobytes()
                self.condition.notify_all()
            return True
        return False

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
