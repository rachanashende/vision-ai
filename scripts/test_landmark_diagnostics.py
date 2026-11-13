r"""Offline checks: python -B scripts/test_landmark_diagnostics.py

Inference is mocked because no local Holistic model asset is available.
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd

import test_landmark_extraction as diagnostic


class LandmarkDiagnosticTests(unittest.TestCase):
    def result(self):
        landmark = SimpleNamespace(x=0.25, y=0.5, z=-0.1, visibility=0.9, presence=0.8)
        return SimpleNamespace(left_hand_landmarks=[landmark] * 21,
                               right_hand_landmarks=[], pose_landmarks=[landmark] * 33)

    def test_missing_and_failed_inference_are_distinct(self):
        rows = diagnostic.diagnostic_rows(self.result(), 0, 0)
        absent = next(row for row in rows if row["component"] == "right_hand")
        self.assertFalse(absent["present"])
        self.assertIsNone(absent["x"])
        self.assertEqual(rows[0]["x"], 0.25)
        failed = diagnostic.diagnostic_rows(None, 1, 33, "Inference failed")
        self.assertTrue(all(row["present"] is None for row in failed))
        self.assertTrue(all(row["processing_status"] == "error" for row in failed))
        bad_result = self.result()
        bad_result.pose_landmarks = bad_result.pose_landmarks[:10]
        with self.assertRaisesRegex(ValueError, "Unexpected pose schema"):
            diagnostic.diagnostic_rows(bad_result, 0, 0)

    def test_missing_model_is_explicitly_blocked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "dataset/raw_videos/book/001.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            with patch.object(diagnostic, "PROJECT_ROOT", root), patch(
                "sys.argv", ["test_landmark_extraction.py"]
            ), patch.object(diagnostic.cv2, "VideoCapture") as capture, redirect_stdout(
                StringIO()
            ), redirect_stderr(StringIO()):
                self.assertEqual(diagnostic.main(), 1)
                capture.assert_not_called()
            report = pd.read_csv(root / "results/evaluation/001_landmark_diagnostic.csv")
            self.assertEqual(report.iloc[0]["processing_status"], "blocked")
            self.assertIn("model is missing", report.iloc[0]["error"])
            self.assertTrue(report["present"].isna().all())

    def test_video_rgb_timestamps_csv_and_annotation_with_mock_inference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "dataset/raw_videos/book/001.mp4"
            video.parent.mkdir(parents=True)
            model = root / "models/holistic_landmarker.task"
            model.parent.mkdir()
            model.touch()  # No inference: the factory below is mocked.
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(3):
                    writer.write(np.full((24, 32, 3), (0, 0, 255), dtype=np.uint8))
            finally:
                writer.release()
            original = video.read_bytes()
            detector = MagicMock()
            detector.__enter__.return_value = detector
            detector.detect_for_video.return_value = self.result()
            with patch.object(diagnostic, "PROJECT_ROOT", root), patch(
                "sys.argv", ["test_landmark_extraction.py"]
            ), patch.object(diagnostic.mp.tasks.vision.HolisticLandmarker,
                            "create_from_options", return_value=detector), patch.object(
                diagnostic.mp, "Image"
            ) as image, redirect_stdout(StringIO()):
                self.assertEqual(diagnostic.main(), 0)
            self.assertEqual([call.args[1] for call in detector.detect_for_video.call_args_list],
                             [0, 100, 200])
            rgb = image.call_args.kwargs["data"]
            self.assertGreater(int(rgb[0, 0, 0]), 200)
            self.assertLess(int(rgb[0, 0, 2]), 20)
            report = pd.read_csv(root / "results/evaluation/001_landmark_diagnostic.csv")
            self.assertEqual(report["frame_index"].nunique(), 3)
            self.assertTrue(report["processing_status"].eq("ok").all())
            self.assertIsNotNone(cv2.imread(str(root / "results/graphs/001_landmark_sample.png")))
            self.assertEqual(video.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
