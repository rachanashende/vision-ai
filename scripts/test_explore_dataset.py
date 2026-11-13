"""Run parser checks without pandas: python -B scripts/test_explore_dataset.py."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from explore_dataset import find_metadata, flatten_entries, summarize_instances


class ExploreDatasetTests(unittest.TestCase):
    def test_missing_fields_and_malformed_records(self):
        diagnostics = StringIO()
        with redirect_stderr(diagnostics), redirect_stdout(StringIO()):
            rows, glosses = flatten_entries([
                {"gloss": "book", "note": "extra", "instances": [
                    {"video_id": "001", "signer_id": 3, "bbox": [1, 2, 3, 4]},
                    {"signer_id": [], "split": {}},
                    None,
                ]},
                {"gloss": "empty", "instances": []},
                {"gloss": "missing"},
                {"instances": []},
                None,
            ])
        self.assertEqual(glosses, ["book", "empty", "missing"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["video_id"], "001")
        self.assertEqual(rows[0]["signer_id"], "3")
        self.assertEqual(rows[0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(rows[0]["entry_note"], "extra")
        self.assertIsNone(rows[0]["frame_start"])
        self.assertIsNone(rows[1]["signer_id"])
        self.assertIn("entry[0].instances[1]: invalid signer_id", diagnostics.getvalue())
        self.assertIn("missing or invalid gloss", diagnostics.getvalue())
        with self.assertRaisesRegex(ValueError, "must be a list"):
            flatten_entries({})

    def test_discovery_prefers_start_kit(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            with self.assertRaises(FileNotFoundError):
                find_metadata(root)
            (root / "WLASL_v9.json").touch()
            preferred = root / "start_kit" / "nested" / "WLASL_v0.3.json"
            preferred.parent.mkdir(parents=True)
            preferred.touch()
            with redirect_stdout(StringIO()):
                self.assertEqual(find_metadata(root), preferred)

    def test_summary_counts_instances_signers_and_splits(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is unavailable in this interpreter")
        instances = pd.DataFrame([
            {"gloss": "book", "signer_id": "1", "split": "train"},
            {"gloss": "book", "signer_id": "1", "split": "val"},
            {"gloss": "book", "signer_id": "2", "split": "test"},
            {"gloss": "book", "signer_id": None, "split": None},
        ])
        summary = summarize_instances(instances, ["book", "empty"]).set_index("gloss")
        self.assertEqual(summary.loc["book"].to_dict(), {
            "instance_count": 4, "signer_count": 2,
            "train_count": 1, "val_count": 1, "test_count": 1,
            "other_or_missing_split_count": 1,
        })
        self.assertTrue(summary.loc["empty"].eq(0).all())


if __name__ == "__main__":
    unittest.main()
