"""Run: python -B scripts/test_review_flagged_clips.py (no model or inference)."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import pandas as pd

from review_flagged_clips import RATE_COLUMNS, make_tile, read_aligned_frames, select_flagged


class FlaggedReviewTests(unittest.TestCase):
    def test_exact_flags_and_identifiers(self):
        rows = []
        for identifier, flag in (("001", "True"), ("002", "False"), ("003", "true")):
            rows.append(dict(gloss="book", video_id=identifier, needs_manual_review=flag,
                             split="train", signer_id="007", **{key: "0.4" for key in RATE_COLUMNS}))
        report = pd.DataFrame(rows)
        self.assertEqual(select_flagged(report)["video_id"].tolist(), ["001", "003"])
        report.loc[0, "needs_manual_review"] = "maybe"
        with self.assertRaises(ValueError):
            select_flagged(report)

    def test_source_indices_and_count_mismatch(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "001.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(11):
                    writer.write(np.full((24, 32, 3), index * 15, dtype=np.uint8))
            finally:
                writer.release()
            original = path.read_bytes()
            arrays = dict(frame_indices=np.arange(11), fps=np.asarray(10.0))
            samples, _ = read_aligned_frames(path, arrays)
            self.assertEqual([index for _, index, _ in samples], [0, 2, 4, 6, 8, 10])
            self.assertLess(samples[0][2].mean(), samples[-1][2].mean())
            arrays["frame_indices"] = np.arange(10)
            with self.assertRaisesRegex(ValueError, "Frame-count mismatch"):
                read_aligned_frames(path, arrays)
            self.assertEqual(path.read_bytes(), original)

    def test_missing_groups_and_inference_failure_draw_no_zero_landmarks(self):
        arrays = dict(inference_ok=np.asarray([True, False]))
        for group, count in (("left_hand", 21), ("right_hand", 21), ("pose", 33)):
            arrays[f"{group}_landmarks"] = np.zeros((2, count, 3), dtype=np.float32)
            arrays[f"{group}_present"] = np.asarray([False, False])
        frame = np.full((360, 640, 3), 100, dtype=np.uint8)
        for slot in (0, 1):
            tile = make_tile(frame, arrays, slot, slot, "book", "001")
            self.assertEqual(tile.shape, (460, 640, 3))
            np.testing.assert_array_equal(tile[:360], frame)


if __name__ == "__main__":
    unittest.main()
