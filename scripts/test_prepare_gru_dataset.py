"""Offline checks against real archives: python -B scripts/test_prepare_gru_dataset.py."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import prepare_gru_dataset as preparation


class GruPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = preparation.PROJECT_ROOT
        cls.protected = {path: path.read_bytes() for path in (cls.root / "dataset/processed").rglob("*.npz")}
        with redirect_stdout(StringIO()):
            cls.arrays, cls.records = preparation.collect_dataset(cls.root)

    def test_exact_parity_with_ten_real_integration_samples(self):
        with redirect_stdout(StringIO()):
            self.assertEqual(preparation.check_integration_parity(self.root, self.arrays, self.records), 10)

    def test_original_split_identity_and_single_class_mapping(self):
        labels = preparation.labels_document()
        self.assertEqual(labels["status"], "PROVISIONAL")
        self.assertEqual(labels["index_to_class"], preparation.integration.LABELS)
        self.assertEqual(set(labels["class_to_index"].values()), set(range(10)))
        self.assertEqual(len({row["video_id"] for row in self.records}), len(self.records))
        for row in self.records:
            with np.load(self.root / row["source_npz"], allow_pickle=False) as source:
                for key in ("gloss", "video_id", "split", "signer_id"):
                    self.assertEqual(row[key], source[key].item())
            target = self.arrays[f"y_{row['split']}"][row["index_in_split"]]
            self.assertEqual(target, labels["class_to_index"][row["gloss"]])
        self.assertEqual(sum(len(self.arrays[f"X_{split}"]) for split in preparation.SPLITS), len(self.records))

    def test_shapes_flags_finite_values_and_invalid_missing_hand(self):
        for split in preparation.SPLITS:
            x = self.arrays[f"X_{split}"]
            self.assertEqual(x.shape[1:], (60, 149))
            self.assertEqual(x.dtype, np.float32)
            self.assertTrue(np.isfinite(x).all())
            for sequence in x:
                preparation.validate_features(sequence)
        invalid = self.arrays["X_train"][0].copy()
        invalid[0, 144] = 0
        invalid[0, 0] = 1
        with self.assertRaises(ValueError):
            preparation.validate_features(invalid)

    def test_determinism_and_source_preservation(self):
        with redirect_stdout(StringIO()):
            repeated, records = preparation.collect_dataset(self.root)
        self.assertEqual(records, self.records)
        for key, array in self.arrays.items():
            np.testing.assert_array_equal(array, repeated[key])
        for path, original in self.protected.items():
            self.assertEqual(path.read_bytes(), original)

    def test_output_preservation_and_read_only_verification(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = preparation.write_outputs(root, self.arrays, self.records)
            summary = root / "results/evaluation/gru_dataset_preparation_summary.md"
            protected = {path: path.read_bytes() for path in list(output.iterdir()) + [summary]}
            with self.assertRaises(FileExistsError):
                preparation.write_outputs(root, self.arrays, self.records)
            with redirect_stdout(StringIO()):
                preparation.verify_dataset(root, self.arrays, self.records)
            for path, original in protected.items():
                self.assertEqual(path.read_bytes(), original)
            preparation.write_outputs(root, self.arrays, self.records, overwrite=True)
            for path, original in protected.items():
                self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
