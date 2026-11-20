"""Offline checks: python -B scripts/test_inspect_sign_boundaries.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from inspect_sign_boundaries import (boundary_evidence, check_outputs, inspect_video,
                                     match_video_metadata, sample_indices, validate_proposed_boundaries)


class SignBoundaryTests(unittest.TestCase):
    def test_metadata_matches_video_id_and_rejects_ambiguity(self):
        rows = [dict(gloss="book", video_id="001", frame_start=1, frame_end=-1)]
        self.assertEqual(match_video_metadata(rows, "001", "book"), rows[0])
        for metadata, identifier, gloss in ((rows, "1", "book"), (rows, "001", "drink"),
                                            (rows + rows, "001", "book")):
            with self.assertRaises(ValueError):
                match_video_metadata(metadata, identifier, gloss)

    def test_boundary_ranges_and_no_automatic_trim_claim(self):
        validate_proposed_boundaries("", "", 26)
        validate_proposed_boundaries("0", "25", 26, manually_verified=True)
        for start, end in (("0", "26"), ("8", "2"), ("", "2"), ("-1", "3")):
            with self.assertRaises(ValueError):
                validate_proposed_boundaries(start, end, 26, manually_verified=True)
        with self.assertRaises(ValueError):
            validate_proposed_boundaries("0", "25", 26)
        self.assertEqual(boundary_evidence(dict(frame_start=1, frame_end=-1), 26, 24)[0], "unknown")
        status, evidence = boundary_evidence(dict(frame_start=40, frame_end=100, fps=25), 26, 24)
        self.assertEqual(status, "needs_review")
        self.assertIn("not proof", evidence)

    def test_source_frame_indexing_and_video_preservation(self):
        self.assertEqual(sample_indices(23), list(range(0, 23, 2)))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "001.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                for index in range(23):
                    writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
            finally:
                writer.release()
            original = path.read_bytes()
            count, fps, indices, samples, _ = inspect_video(path)
            self.assertEqual(count, 23)
            self.assertEqual(indices[0], 0)
            self.assertEqual(indices[-1], 22)
            self.assertAlmostEqual(fps, 10)
            self.assertLess(samples[0][0].mean(), samples[22][0].mean())
            self.assertEqual(path.read_bytes(), original)

    def test_existing_outputs_are_preserved(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            path.write_bytes(b"manual decision must survive")
            with self.assertRaises(FileExistsError):
                check_outputs([path], overwrite=False)
            check_outputs([path], overwrite=True)
            self.assertEqual(path.read_bytes(), b"manual decision must survive")


if __name__ == "__main__":
    unittest.main()
