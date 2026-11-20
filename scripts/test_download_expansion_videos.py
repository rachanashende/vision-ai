"""Offline checks: python -B scripts/test_download_expansion_videos.py."""

from contextlib import redirect_stdout
from email.message import Message
from io import BytesIO, StringIO
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import cv2
import numpy as np
import pandas as pd

import download_expansion_videos as downloader


def candidate(identifier="001", split="val", priority="1"):
    return dict(gloss="book", video_id=identifier, split=split, signer_id="007", source="test",
                url=f"https://example.org/{identifier}.mp4", priority=priority,
                recommended_for_initial_pilot="True", download_status="candidate_not_downloaded",
                source_support="supported_direct_mp4")


def response(body, content_type="video/mp4", length=None):
    result = MagicMock()
    result.status = 200
    result.headers = Message()
    result.headers["Content-Type"] = content_type
    result.headers["Content-Length"] = str(len(body) if length is None else length)
    result.__enter__.return_value = result
    result.read.side_effect = BytesIO(body).read
    return result


class ExpansionDownloadTests(unittest.TestCase):
    def video_bytes(self, directory):
        video = Path(directory) / "fixture.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
        self.assertTrue(writer.isOpened())
        try:
            for _ in range(3):
                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
        finally:
            writer.release()
        return video.read_bytes()

    def test_dry_run_no_network_and_append_only_history(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "history.csv"
            with patch.object(downloader, "build_opener", side_effect=AssertionError("Network forbidden")), patch(
                "socket.socket", side_effect=AssertionError("Network forbidden")
            ), redirect_stdout(StringIO()):
                downloader.run_batch([candidate()], [], root / "raw", report)
                original = report.read_bytes()
                downloader.run_batch([candidate()], downloader.read_history(report), root / "raw", report)
            self.assertTrue(report.read_bytes().startswith(original))
            self.assertEqual(len(downloader.read_history(report)), 2)
            self.assertFalse((root / "raw").exists())

    def test_split_filter_and_held_out_order(self):
        plan = pd.DataFrame([candidate("001", "train", "1"), candidate("002", "test", "3"),
                             candidate("003", "val", "2")])
        self.assertEqual([row["video_id"] for row in downloader.select_candidates(plan, "all")], ["003", "002", "001"])
        self.assertEqual([row["split"] for row in downloader.select_candidates(plan, "test")], ["test"])
        self.assertTrue(all(row["signer_id"] == "007" for row in downloader.select_candidates(plan, "all")))

    def test_existing_valid_file_is_never_overwritten(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.video_bytes(root)
            destination = root / "raw/book/001.mp4"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(data)
            with patch.object(downloader, "build_opener", side_effect=AssertionError("Network forbidden")), redirect_stdout(StringIO()):
                downloader.run_batch([candidate()], [], root / "raw", root / "report.csv", download=True)
            self.assertEqual(destination.read_bytes(), data)
            self.assertEqual(downloader.read_history(root / "report.csv")[0]["status"], "already_valid")

    def test_invalid_and_incomplete_downloads_are_removed(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "raw/001.mp4"
            for body, content_type, length in ((b"<html>error</html>", "text/html", None),
                                                (b"<html>error</html>", "application/octet-stream", None),
                                                (b"not video", "video/mp4", None),
                                                (b"short", "video/mp4", 100)):
                with self.subTest(body=body, content_type=content_type), patch.object(
                    downloader, "build_opener"
                ) as opener:
                    opener.return_value.open.return_value = response(body, content_type, length)
                    result, _ = downloader.transfer_once(candidate()["url"], destination, 7)
                    self.assertNotEqual(result["status"], "downloaded")
                    opener.return_value.open.assert_called_once()
                    self.assertEqual(opener.return_value.open.call_args.kwargs["timeout"], 7)
                    self.assertFalse(destination.exists())
                    self.assertFalse(list(destination.parent.glob("*.part")))

    def test_actual_video_bytes_validate_from_part_and_publish(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = self.video_bytes(root)
            destination = root / "raw/001.mp4"
            with patch.object(downloader, "build_opener") as opener:
                opener.return_value.open.return_value = response(data)
                result, retryable = downloader.transfer_once(candidate()["url"], destination, 10)
            self.assertEqual(result["status"], "downloaded")
            self.assertFalse(retryable)
            self.assertEqual(destination.read_bytes(), data)
            self.assertTrue(downloader.validate_video(destination)[0])
            self.assertFalse(list(destination.parent.glob("*.part")))

    def test_404_resume_and_transient_failure_do_not_stop_next_candidate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.csv"
            with patch.object(downloader, "build_opener") as opener, patch.object(
                downloader.time, "sleep"
            ), redirect_stdout(StringIO()):
                opener.return_value.open.side_effect = HTTPError("url", 404, "Not found", {}, None)
                downloader.run_batch([candidate()], [], root / "raw", report, download=True)
                self.assertEqual(opener.return_value.open.call_count, 1)
                history = downloader.read_history(report)
                opener.return_value.open.reset_mock()
                downloader.run_batch([candidate()], history, root / "raw", report, download=True)
                opener.return_value.open.assert_not_called()
                opener.return_value.open.side_effect = URLError(socket.gaierror("DNS lookup failed"))
                downloader.run_batch([candidate("002"), candidate("003")], history,
                                      root / "raw", report, download=True, retries=1)
                self.assertEqual(opener.return_value.open.call_count, 4)
            rows = downloader.read_history(report)
            self.assertEqual([row["attempt"] for row in rows if row["video_id"] == "002"], ["1", "2"])
            self.assertTrue(any(row["video_id"] == "003" for row in rows))
            self.assertTrue(all(row["error_type"] == "dns_error" for row in rows if row["video_id"] in ("002", "003")))


if __name__ == "__main__":
    unittest.main()
