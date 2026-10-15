import json
import unittest
import urllib.error
import urllib.request

from web_preview import Preview


class FakeControl:
    def status(self):
        return {"confidence": 0.35, "active": "ssd"}

    def request(self, backend, confidence):
        return {"requested": backend, "confidence": confidence}


class WebTests(unittest.TestCase):
    def setUp(self):
        self.preview = Preview("127.0.0.1", 0, FakeControl())
        self.base = f"http://127.0.0.1:{self.preview.server.server_port}"

    def tearDown(self):
        self.preview.close()

    def test_static_routes_and_status(self):
        for path, mime in [
            ("/", "text/html"),
            ("/app.js", "text/javascript"),
            ("/styles.css", "text/css"),
        ]:
            with urllib.request.urlopen(self.base + path, timeout=2) as response:
                self.assertEqual(response.status, 200)
                self.assertIn(mime, response.headers["Content-Type"])
                self.assertTrue(response.read())
        with urllib.request.urlopen(self.base + "/api/status") as response:
            self.assertEqual(json.load(response)["active"], "ssd")
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.base + "/../LICENSE")
        self.assertEqual(error.exception.code, 404)

    def test_json_model_request_and_rejected_body(self):
        request = urllib.request.Request(
            self.base + "/api/model",
            data=b'{"model":"ssd","confidence":0.5}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 202)
            self.assertEqual(json.load(response)["confidence"], 0.5)
        request = urllib.request.Request(
            self.base + "/api/model",
            data=b"[]",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 400)

    def test_mjpeg_publishes_one_complete_payload(self):
        jpeg = b"\xff\xd8test-payload\xff\xd9"
        self.preview.publish_jpeg(jpeg)
        with urllib.request.urlopen(self.base + "/stream", timeout=2) as response:
            self.assertEqual(response.readline(), b"--frame\r\n")
            headers = {}
            while True:
                line = response.readline()
                if line == b"\r\n":
                    break
                key, value = line.decode().split(":", 1)
                headers[key.lower()] = value.strip()
            self.assertEqual(headers["content-type"], "image/jpeg")
            self.assertEqual(response.read(int(headers["content-length"])), jpeg)
