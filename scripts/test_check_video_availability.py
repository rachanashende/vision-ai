r"""Run: .\.venv\Scripts\python.exe -B scripts/test_check_video_availability.py"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from check_video_availability import check_availability


class VideoAvailabilityTests(unittest.TestCase):
    def test_local_matching_and_missing_fields(self):
        with TemporaryDirectory() as directory:
            dataset_dir = Path(directory)
            nested = dataset_dir / "videos" / "nested"
            nested.mkdir(parents=True)
            (nested / "001.MP4").touch()
            (nested / "002.txt").touch()
            metadata = pd.DataFrame({
                "gloss": ["book", "drink", "yes", "book", "unselected"],
                "video_id": ["001", "002", "", "001", "001"],
                "split": ["train", "val", "test", "test", "train"],
            })
            report = check_availability(metadata, dataset_dir)
            self.assertEqual(report["video_id"].tolist(), ["001", "002", "", "001"])
            self.assertEqual(report["split"].tolist(), ["train", "val", "test", "test"])
            self.assertEqual(report["local_file_status"].tolist(),
                             ["found", "missing", "missing_video_id", "found"])
            self.assertTrue(report["remote_status"].eq("not_checked").all())
            self.assertTrue(report["signer_id"].eq("").all())
            self.assertTrue(report["url"].eq("").all())
            self.assertEqual(Path(report.iloc[0]["local_file_path"]),
                             (nested / "001.MP4").resolve())
            empty = check_availability(metadata.iloc[0:0], dataset_dir)
            self.assertTrue(empty.empty)
            with self.assertRaisesRegex(ValueError, "gloss"):
                check_availability(pd.DataFrame(), dataset_dir)


if __name__ == "__main__":
    unittest.main()
