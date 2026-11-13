r"""Offline tests: .\.venv\Scripts\python.exe -B scripts/test_download_pilot_videos.py"""

from contextlib import redirect_stdout
from email.message import Message
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import cv2
import numpy as np
import pandas as pd

import download_pilot_videos as pilot


class PilotDownloadTests(unittest.TestCase):
    def test_selection_limits_duplicates_and_split_preference(self):
        rows = []
        for word_index, word in enumerate(pilot.WORDS):
            for number, split in enumerate(["test", "train", "val", "train", "train"]):
                rows.append(dict(gloss=word, video_id=f"{word_index}{number:03d}",
                                 split=split, url="https://example.org/video.mp4"))
        selected = pilot.select_pilot(pd.DataFrame(rows + rows))
        self.assertEqual(len(selected), 30)
        self.assertEqual(len({row["video_id"] for row in selected}), 30)
        self.assertTrue(all(row["split"] == "train" for row in selected))

    def test_source_restrictions(self):
        for url in ("https://www.youtube.com/watch?v=test", "https://example.org/a.swf",
                    "file:///video.mp4", ""):
            self.assertEqual(pilot.source_problem(url)[0], "unsupported_source")
        self.assertEqual(pilot.source_problem(
            "https://example.org/protected_media/a.mp4")[0], "restricted_source")
        self.assertIsNone(pilot.source_problem("https://example.org/a.MP4?version=1"))

    def test_default_dry_run_has_no_network_or_video_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_dir = root / "dataset/metadata"
            metadata_dir.mkdir(parents=True)
            pd.DataFrame([dict(gloss="book", video_id="001", split="train",
                               url="https://example.org/a.mp4")]).to_csv(
                                   metadata_dir / "wlasl_instances.csv", index=False)
            with patch.object(pilot, "PROJECT_ROOT", root), patch(
                "sys.argv", ["download_pilot_videos.py"]
            ), patch.object(pilot, "build_opener", side_effect=AssertionError("Network forbidden")), patch(
                "socket.socket", side_effect=AssertionError("Network forbidden")
            ), redirect_stdout(StringIO()):
                self.assertEqual(pilot.main(), 0)
            report = pd.read_csv(metadata_dir / "pilot_download_report.csv", dtype=str)
            self.assertEqual(report.iloc[0]["status"], "dry_run")
            self.assertEqual(report.iloc[0]["video_id"], "001")
            self.assertFalse((root / "dataset/raw_videos").exists())

    def response(self, body):
        response = MagicMock()
        stream = BytesIO(body)
        response.read.side_effect = stream.read
        response.headers = Message()
        response.headers["Content-Type"] = "video/mp4"
        response.headers["Content-Length"] = str(len(body))
        response.__enter__.return_value = response
        return response

    def test_publish_cleanup_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "001.mp4"
            opener = MagicMock()
            opener.open.return_value = self.response(b"test video bytes")
            with patch.object(pilot, "build_opener", return_value=opener), patch.object(
                pilot, "valid_video", return_value=True
            ):
                self.assertEqual(pilot.download_video(
                    "https://example.org/a.mp4", destination)[0], "downloaded")
                self.assertEqual(pilot.download_video(
                    "https://example.org/a.mp4", destination)[0], "already_valid")
            self.assertEqual(opener.open.call_count, 1)
            self.assertEqual(destination.read_bytes(), b"test video bytes")
            self.assertEqual(list(Path(directory).iterdir()), [destination])

    def test_rejection_retries_and_interruption_cleanup(self):
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "001.mp4"
            opener = MagicMock()
            with patch.object(pilot, "build_opener", return_value=opener), patch.object(
                pilot.time, "sleep"
            ), patch.object(pilot, "valid_video", return_value=False):
                opener.open.return_value = self.response(b"not a video")
                self.assertEqual(pilot.download_video(
                    "https://example.org/a.mp4", destination)[0], "rejected")
                self.assertFalse(list(Path(directory).iterdir()))
                opener.open.reset_mock()
                opener.open.side_effect = HTTPError("url", 403, "Forbidden", {}, None)
                self.assertEqual(pilot.download_video(
                    "https://example.org/a.mp4", destination)[0], "access_refused")
                self.assertEqual(opener.open.call_count, 1)
                opener.open.reset_mock()
                opener.open.side_effect = TimeoutError("timed out")
                self.assertEqual(pilot.download_video(
                    "https://example.org/a.mp4", destination)[0], "failed")
                self.assertEqual(opener.open.call_count, 2)
                opener.open.side_effect = KeyboardInterrupt
                with self.assertRaises(KeyboardInterrupt):
                    pilot.download_video("https://example.org/a.mp4", destination)
                self.assertFalse(list(Path(directory).iterdir()))

    def test_opencv_validation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "valid.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 32))
            self.assertTrue(writer.isOpened())
            try:
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            finally:
                writer.release()
            self.assertTrue(pilot.valid_video(path))
            path.write_bytes(b"")
            self.assertFalse(pilot.valid_video(path))


if __name__ == "__main__":
    unittest.main()
