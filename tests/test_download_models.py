import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download_models import install_asset


class DownloadTests(unittest.TestCase):
    def entry(self, data=b"valid model"):
        return {
            "path": "models/test/model.bin",
            "url": "https://example.com/model.bin",
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def test_existing_valid_asset_skips_network(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "models/test/model.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"valid model")
            with patch("urllib.request.urlopen") as request:
                self.assertFalse(install_asset(self.entry(), folder))
                request.assert_not_called()

    def test_corrupt_download_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "models/test/model.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"old model")
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"corrupt")):
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    install_asset(self.entry(), folder)
            self.assertEqual(path.read_bytes(), b"old model")
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_verified_download_installs_atomically(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"valid model")):
                self.assertTrue(install_asset(self.entry(), folder))
            self.assertEqual((Path(folder) / self.entry()["path"]).read_bytes(), b"valid model")

    def test_destination_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as folder:
            entry = self.entry()
            entry["path"] = "../outside.bin"
            with self.assertRaises(ValueError):
                install_asset(entry, folder)
