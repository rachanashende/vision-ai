"""Offline checks: python -B scripts/test_train_gru_baseline.py. No training."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

import train_gru_baseline as baseline


class GruBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arrays, cls.labels, cls.manifest = baseline.load_inputs(baseline.PROJECT_ROOT)

    def test_input_shapes_dtypes_values_and_labels(self):
        x, y = self.arrays["X_train"], self.arrays["y_train"]
        baseline.validate_split(x, y, 17)
        for bad_x, bad_y in ((x[:, :, :148], y), (x.astype(np.float64), y), (x, y.astype(np.float32))):
            with self.assertRaises(ValueError):
                baseline.validate_split(bad_x, bad_y, 17)
        invalid = x.copy()
        invalid[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            baseline.validate_split(invalid, y, 17)
        invalid_y = y.copy()
        invalid_y[0] = 10
        with self.assertRaises(ValueError):
            baseline.validate_split(x, invalid_y, 17)

    def test_label_mapping_conflict_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "dataset/model_ready/PROVISIONAL_v1"
            shutil.copytree(baseline.PROJECT_ROOT / "dataset/model_ready/PROVISIONAL_v1", destination)
            labels = json.loads((destination / "labels.json").read_text())
            labels["class_to_index"]["book"] = 9
            (destination / "labels.json").write_text(json.dumps(labels))
            with self.assertRaises(ValueError):
                baseline.load_inputs(root)

    def test_prediction_probabilities_and_model_reload(self):
        tf.keras.utils.set_random_seed(123)
        model = baseline.build_model()
        probabilities = model(self.arrays["X_train"][:2], training=False).numpy()
        baseline.validate_probabilities(probabilities, 2)
        self.assertEqual(probabilities.shape, (2, 10))
        self.assertLess(model.count_params(), 20000)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "unit_test_only.keras"
            model.save(path)
            reloaded = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(reloaded(self.arrays["X_train"][:2], training=False).numpy(), probabilities, atol=1e-6)
        with self.assertRaises(ValueError):
            baseline.validate_probabilities(np.zeros((2, 10)), 2)

    def test_default_does_not_train_and_protects_existing_outputs(self):
        with patch("sys.argv", ["train_gru_baseline.py"]), patch.object(
            baseline, "train_baseline", side_effect=AssertionError("Training forbidden")
        ) as training, redirect_stdout(StringIO()):
            self.assertEqual(baseline.main(), 0)
            training.assert_not_called()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model, _ = baseline.output_paths(root)
            model.parent.mkdir(parents=True)
            model.write_bytes(b"existing model")
            with self.assertRaises(FileExistsError):
                baseline.check_outputs(root, False)
            self.assertEqual(model.read_bytes(), b"existing model")

    def test_original_prepared_files_preserved(self):
        folder = baseline.PROJECT_ROOT / "dataset/model_ready/PROVISIONAL_v1"
        original = {path: path.read_bytes() for path in folder.iterdir() if path.is_file()}
        baseline.load_inputs(baseline.PROJECT_ROOT)
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
