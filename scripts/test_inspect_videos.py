r"""Run offline: .\.venv\Scripts\python.exe -B scripts/test_inspect_videos.py"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd

import inspect_videos as inspector


class VideoInspectionTests(unittest.TestCase):
    def test_real_video_and_unchanged_bytes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "001.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3):
                    writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            finally:
                writer.release()
            original = path.read_bytes()
            result = inspector.inspect_video(path)
            self.assertTrue(result["valid"])
            self.assertEqual(result["decoded_frame_count"], 3)
            self.assertEqual((result["width"], result["height"]), (32, 24))
            self.assertAlmostEqual(result["duration_seconds"], 0.3)
            self.assertTrue(result["unusually_short"])
            self.assertEqual(path.read_bytes(), original)

    def test_bad_fps_and_partial_decoding(self):
        for fps in (0, float("nan"), float("inf")):
            capture = MagicMock()
            capture.isOpened.return_value = True
            properties = {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: 2,
                          cv2.CAP_PROP_FRAME_WIDTH: 32, cv2.CAP_PROP_FRAME_HEIGHT: 24}
            capture.get.side_effect = properties.get
            capture.read.side_effect = [(True, np.zeros((24, 32, 3))), (False, None)]
            with patch.object(inspector.cv2, "VideoCapture", return_value=capture):
                result = inspector.inspect_video(Path("test.mp4"))
            self.assertIsNone(result["duration_seconds"])
            self.assertTrue(result["frames_decodable"])
            self.assertFalse(result["valid"])
            self.assertIn("possible truncation", result["errors"])
            capture.release.assert_called_once()

    def test_unopenable_video(self):
        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch.object(inspector.cv2, "VideoCapture", return_value=capture):
            result = inspector.inspect_video(Path("test.mp4"))
        self.assertFalse(result["valid"])
        self.assertFalse(result["frames_decodable"])
        self.assertIn("could not open", result["errors"])
        capture.release.assert_called_once()

    def test_metadata_and_empty_report(self):
        metadata = pd.DataFrame([
            dict(gloss="book", video_id="001", frame_start=1, frame_end=-1,
                 split="train", signer_id=5),
        ])
        matched, status = inspector.match_metadata(metadata, "book", "001")
        self.assertEqual(status, "matched")
        self.assertEqual(matched["split"], "train")
        self.assertEqual(inspector.match_metadata(metadata, "yes", "001")[1], "missing")
        duplicate = pd.concat([metadata, metadata])
        self.assertEqual(inspector.match_metadata(duplicate, "book", "001")[1], "ambiguous")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(inspector, "PROJECT_ROOT", root), patch.object(
                inspector, "load_metadata", return_value=(metadata, "test")
            ), redirect_stdout(StringIO()):
                self.assertEqual(inspector.main(), 0)
            report = pd.read_csv(root / "dataset/metadata/video_inspection_report.csv")
            self.assertTrue(report.empty)
            self.assertEqual(report.columns.tolist(), inspector.REPORT_COLUMNS)


if __name__ == "__main__":
    unittest.main()
