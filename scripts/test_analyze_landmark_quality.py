"""Offline checks: python -B scripts/test_analyze_landmark_quality.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

import analyze_landmark_quality as analysis


def sample_arrays():
    arrays = {key: np.asarray("test") for key in analysis.TEXT_FIELDS}
    arrays.update(schema_version=np.asarray(analysis.SCHEMA_VERSION), gloss=np.asarray("book"),
                  video_id=np.asarray("001"), split=np.asarray("val"), signer_id=np.asarray("007"),
                  frame_indices=np.arange(4, dtype=np.int64), timestamps_ms=np.arange(4, dtype=np.int64) * 100,
                  fps=np.asarray(10.0, dtype=np.float64), inference_ok=np.asarray([True, True, True, False]))
    for name, count in analysis.GROUPS.items():
        mask = np.asarray([True, False, True, False]) if name == "left_hand" else np.asarray([False, True, True, False])
        arrays[f"{name}_present"] = mask
        coordinates = np.zeros((4, count, 3), dtype=np.float32)
        coordinates[mask] = 0.25
        arrays[f"{name}_landmarks"] = coordinates
    return arrays


class LandmarkAnalysisTests(unittest.TestCase):
    def test_rates_exclude_failed_inference_from_absence(self):
        arrays = sample_arrays()
        self.assertEqual(analysis.validate_arrays(arrays, Path("001.npz"), "book"), [])
        statistics = analysis.detection_statistics(arrays)
        self.assertAlmostEqual(statistics["either_hand_detection_rate"], 1.0)
        self.assertAlmostEqual(statistics["both_hands_detection_rate"], 1 / 3)
        self.assertAlmostEqual(statistics["either_hand_all_frames_rate"], 0.75)
        self.assertEqual(statistics["failed_inference_frames"], 1)
        self.assertEqual(statistics["left_hand_absent_on_success_frames"], 1)

    def test_bad_shapes_masks_nan_and_metadata(self):
        mutations = [
            ("pose_landmarks", np.zeros((3, 33, 3), dtype=np.float32)),
            ("left_hand_present", np.asarray([1, 0, 1, 0])),
            ("right_hand_landmarks", np.full((4, 21, 3), np.nan, dtype=np.float32)),
            ("split", np.asarray("unknown")),
            ("signer_id", np.asarray("")),
            ("schema_version", np.asarray("future-schema")),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                arrays = sample_arrays()
                arrays[key] = value
                self.assertTrue(analysis.validate_arrays(arrays, Path("001.npz"), "book"))
        arrays = sample_arrays()
        arrays["pose_present"][-1] = True
        self.assertTrue(any("failed inference" in error for error in
                            analysis.validate_arrays(arrays, Path("001.npz"), "book")))

    def test_archive_metadata_report_comparison_and_source_preservation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "book/001.npz"
            path.parent.mkdir()
            np.savez_compressed(path, **sample_arrays())
            original = path.read_bytes()
            source = pd.DataFrame([dict(gloss="book", video_id="001", split="val", signer_id="007",
                                        decoded_frame_count="4", left_hand_detection_rate="0.5", status="processed_with_errors")])
            row = analysis.analyze_archive(path, root, source)
            self.assertEqual(row["validation_status"], "valid")
            self.assertEqual(row["signer_id"], "007")
            self.assertEqual(row["split"], "val")
            self.assertTrue(row["needs_manual_review"])
            self.assertFalse(row["source_report_notes"])
            self.assertEqual(path.read_bytes(), original)
            source.loc[0, "split"] = "test"
            mismatch = analysis.analyze_archive(path, root, source)
            self.assertIn("split differs", mismatch["source_report_notes"])
            broken = root / "book/broken.npz"
            broken.write_bytes(b"not an archive")
            invalid = analysis.analyze_archive(broken, root, source)
            self.assertEqual(invalid["validation_status"], "invalid")
            self.assertTrue(invalid["needs_manual_review"])
            self.assertNotIn("either_hand_detection_rate", invalid)


if __name__ == "__main__":
    unittest.main()
