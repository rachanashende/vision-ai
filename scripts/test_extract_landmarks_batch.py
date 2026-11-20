r"""Offline: python -B scripts/test_extract_landmarks_batch.py

Model inference is mocked; local OpenCV videos exercise decoding and timing.
"""

import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd

import extract_landmarks_batch as batch


def sample_result():
    point = SimpleNamespace(x=0.25, y=0.5, z=-0.1)
    return SimpleNamespace(left_hand_landmarks=[], right_hand_landmarks=[point] * 21,
                           pose_landmarks=[point] * 33)


def sample_metadata():
    return dict(gloss="book", video_id="001", split="val", signer_id="007",
                frame_start="3", frame_end="-1", trimming_status="uncertain; source boundaries not applied",
                metadata_source="WLASL.json", source_video="001.mp4", mediapipe_version="1.2.2")


def sample_payload():
    packed = batch.pack_result(sample_result())
    payload = {key: np.asarray([value, value]) for key, value in packed.items()}
    payload.update(frame_indices=np.arange(2, dtype=np.int64),
                   timestamps_ms=np.asarray([0, 100], dtype=np.int64),
                   inference_ok=np.asarray([True, True]), fps=np.asarray(10.0),
                   timestamp_source=np.asarray("test"), schema_version=np.asarray(batch.SCHEMA_VERSION))
    payload.update({key: np.asarray(value) for key, value in sample_metadata().items()})
    return payload


class BatchExtractionTests(unittest.TestCase):
    def test_shapes_missing_landmarks_and_schema_rejection(self):
        packed = batch.pack_result(sample_result())
        self.assertEqual(packed["left_hand_landmarks"].shape, (21, 3))
        self.assertEqual(packed["pose_landmarks"].shape, (33, 3))
        self.assertEqual(packed["pose_landmarks"].dtype, np.float32)
        self.assertFalse(packed["left_hand_present"])
        self.assertTrue(packed["right_hand_present"])
        self.assertTrue((packed["left_hand_landmarks"] == 0).all())
        self.assertTrue(all(not value for key, value in batch.pack_result(None).items()
                            if key.endswith("_present")))
        invalid = sample_result()
        invalid.pose_landmarks = invalid.pose_landmarks[:10]
        with self.assertRaises(ValueError):
            batch.pack_result(invalid)

    def test_save_metadata_validation_and_overwrite_protection(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "book/001.npz"
            batch.save_output(path, sample_payload())
            self.assertEqual(batch.validate_output(path, sample_metadata()), (True, ""))
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(saved["video_id"].item(), "001")
                self.assertEqual(saved["signer_id"].item(), "007")
                self.assertEqual(saved["split"].item(), "val")
                self.assertEqual(saved["frame_start"].item(), "3")
                self.assertEqual(saved["frame_end"].item(), "-1")
                self.assertEqual(saved["right_hand_landmarks"].shape, (2, 21, 3))
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                batch.save_output(path, sample_payload())
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(path.parent.glob("tmp*.npz")))
            different = {**sample_metadata(), "split": "test"}
            self.assertFalse(batch.validate_output(path, different)[0])
            invalid = sample_payload()
            invalid["left_hand_landmarks"][0, 0, 0] = 1
            with self.assertRaises(ValueError):
                batch.save_output(path, invalid, overwrite=True)
            self.assertEqual(path.read_bytes(), original)
            path.write_bytes(b"broken archive")
            self.assertFalse(batch.validate_output(path)[0])

    def test_every_frame_inference_errors_rgb_and_fresh_detector(self):
        with TemporaryDirectory() as directory:
            video = Path(directory) / "001.mp4"
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
            detector.detect_for_video.side_effect = [sample_result(), RuntimeError("test error"), sample_result()]
            with patch.object(batch.mp.tasks.vision.HolisticLandmarker, "create_from_options",
                              return_value=detector) as factory, patch.object(batch.mp, "Image") as image:
                payload, metrics = batch.extract_video(video, Path("model.task"))
                self.assertEqual(metrics["status"], "processed_with_errors")
                self.assertEqual(metrics["decoded_frame_count"], 3)
                self.assertEqual(metrics["inference_error_frames"], 1)
                self.assertEqual(payload["inference_ok"].tolist(), [True, False, True])
                self.assertEqual(payload["right_hand_present"].tolist(), [True, False, True])
                self.assertTrue((payload["right_hand_landmarks"][1] == 0).all())
                self.assertEqual(payload["timestamps_ms"].tolist(), [0, 100, 200])
                rgb = image.call_args.kwargs["data"]
                self.assertGreater(int(rgb[0, 0, 0]), 200)
                self.assertLess(int(rgb[0, 0, 2]), 20)
                detector.detect_for_video.side_effect = None
                detector.detect_for_video.return_value = sample_result()
                second, _ = batch.extract_video(video, Path("model.task"))
                self.assertEqual(second["timestamps_ms"][0], 0)
                self.assertEqual(factory.call_count, 2)
            self.assertEqual(video.read_bytes(), original)

    def test_default_dry_run_no_detector_no_processed_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "dataset/raw_videos/book/001.mp4"
            video.parent.mkdir(parents=True)
            video.touch()
            metadata = pd.DataFrame([sample_metadata()])
            with patch.object(batch, "PROJECT_ROOT", root), patch.object(
                batch, "load_metadata", return_value=(metadata, "test.json")
            ), patch.object(batch, "extract_video", side_effect=AssertionError("Inference forbidden")), patch(
                "sys.argv", ["extract_landmarks_batch.py"]
            ), redirect_stdout(StringIO()):
                self.assertEqual(batch.main(), 0)
            self.assertFalse((root / "dataset/processed").exists())
            report = pd.read_csv(root / "dataset/metadata/landmark_quality_report.csv")
            self.assertEqual(report.iloc[0]["status"], "dry_run")

    def test_limit_must_be_positive(self):
        self.assertEqual(batch.positive_limit("3"), 3)
        with self.assertRaises(argparse.ArgumentTypeError):
            batch.positive_limit("0")


if __name__ == "__main__":
    unittest.main()
