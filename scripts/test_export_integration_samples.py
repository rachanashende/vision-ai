"""Offline: python -B scripts/test_export_integration_samples.py. No inference."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import export_integration_samples as exporter
from test_analyze_landmark_quality import sample_arrays


class IntegrationExportTests(unittest.TestCase):
    def test_shape_finite_and_short_clip_right_padding(self):
        arrays = sample_arrays()
        output, slots = exporter.convert_sequence(arrays)
        self.assertEqual(output.shape, (60, 149))
        self.assertEqual(output.dtype, np.dtype("<f4"))
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(output.flags.c_contiguous)
        np.testing.assert_array_equal(slots[:4], np.arange(4))
        self.assertTrue((slots[4:] == -1).all())
        self.assertTrue((output[4:] == 0).all())
        self.assertTrue((output[:4, 148] == 1).all())

    def test_exact_feature_offsets_and_no_coordinate_clipping(self):
        arrays = sample_arrays()
        arrays["inference_ok"][:] = True
        for group, count, offset in (("left_hand", 21, 1000), ("right_hand", 21, 2000), ("pose", 33, 3000)):
            arrays[f"{group}_present"][:] = True
            arrays[f"{group}_landmarks"][:] = np.arange(count * 3, dtype=np.float32).reshape(count, 3) + offset
        features = exporter.frame_features(arrays)
        np.testing.assert_array_equal(features[0, :63], arrays["left_hand_landmarks"][0].reshape(-1))
        np.testing.assert_array_equal(features[0, 63:126], arrays["right_hand_landmarks"][0].reshape(-1))
        np.testing.assert_array_equal(features[0, 126:144], arrays["pose_landmarks"][0, [11, 12, 13, 14, 15, 16]].reshape(-1))
        np.testing.assert_array_equal(features[:, 144:], np.ones((4, 5)))

    def test_missing_hand_failed_inference_and_padding_distinct(self):
        arrays = sample_arrays()
        arrays["left_hand_present"][:] = False
        arrays["left_hand_landmarks"][:] = 0
        output, _ = exporter.convert_sequence(arrays)
        self.assertTrue((output[:, :63] == 0).all())
        self.assertTrue((output[:, 144] == 0).all())
        self.assertEqual((output[1, 145], output[1, 147], output[1, 148]), (1, 1, 1))
        self.assertTrue((output[3, :148] == 0).all())
        self.assertEqual(output[3, 148], 1)  # Failed inference is still a real frame.
        self.assertTrue((output[4] == 0).all())

    def test_deterministic_time_selection_ties_and_single_frame(self):
        # Target k=1 is exactly timestamp 10; slots at 9 and 11 tie -> earlier.
        times = np.asarray([0, 9, 11] + list(range(20, 591, 10)), dtype=np.int64)
        first = exporter.resample_slots(times)
        self.assertEqual(first[1], 1)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], len(times) - 1)
        np.testing.assert_array_equal(first, exporter.resample_slots(times))
        self.assertTrue((np.diff(first) >= 0).all())
        one = exporter.resample_slots(np.asarray([0], dtype=np.int64))
        self.assertEqual(one[0], 0)
        self.assertTrue((one[1:] == -1).all())
        with self.assertRaises(ValueError):
            exporter.resample_slots(np.asarray([0, 0], dtype=np.int64))

    def test_real_format_archives_roundtrip_and_source_preservation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "dataset/processed/book"
            folder.mkdir(parents=True)
            pairs, metadata, protected = [], [], {}
            for number in range(5):
                identifier = f"{number:03d}"
                arrays = sample_arrays()
                arrays.update(video_id=np.asarray(identifier), frame_start=np.asarray("1"), frame_end=np.asarray("-1"))
                path = folder / f"{identifier}.npz"
                np.savez_compressed(path, **arrays)
                protected[path] = path.read_bytes()
                pairs.append(("book", identifier))
                metadata.append({key: arrays[key].item() for key in
                                 ("gloss", "video_id", "split", "signer_id", "frame_start", "frame_end")})
            destination = root / "exports"
            with patch.object(exporter, "load_metadata", return_value=(pd.DataFrame(metadata), "test")), redirect_stdout(StringIO()):
                manifest = exporter.export_samples(root, destination, pairs)
                self.assertEqual(exporter.verify_samples(root, destination), 5)
                with self.assertRaises(FileExistsError):
                    exporter.export_samples(root, destination, pairs)
            self.assertEqual(manifest["status"], "PROVISIONAL")
            self.assertEqual(manifest["samples"][0]["split"], "val")
            self.assertEqual(manifest["samples"][0]["signer_id"], "007")
            for path, original in protected.items():
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
