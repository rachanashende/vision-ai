"""Offline checks: python -B scripts/test_plan_dataset_expansion.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np
import pandas as pd

from plan_dataset_expansion import build_plan, check_outputs, inventory_local, reconcile_metadata


def row(identifier, split="train", signer="1"):
    return dict(gloss="book", video_id=identifier, split=split, signer_id=signer,
                url=f"https://example.org/{identifier}.mp4", source="test", frame_start="1", frame_end="-1", fps="25")


class ExpansionPlannerTests(unittest.TestCase):
    def test_unique_ids_and_split_preservation(self):
        rows = [row("001"), row("002", "val"), row("003", "test")]
        reconciled = reconcile_metadata(rows, pd.DataFrame(rows))
        self.assertEqual([item["split"] for item in reconciled], ["train", "val", "test"])
        with self.assertRaises(ValueError):
            reconcile_metadata(rows + [rows[0]], pd.DataFrame(rows))
        conflicting = pd.DataFrame(rows)
        conflicting.loc[1, "split"] = "train"
        with self.assertRaisesRegex(ValueError, "conflict"):
            reconcile_metadata(rows, conflicting)

    def test_existing_file_detection_and_preservation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "book/001.MP4"
            video.parent.mkdir()
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
            self.assertTrue(writer.isOpened())
            try:
                writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
            finally:
                writer.release()
            original = video.read_bytes()
            inventory = inventory_local([row("001"), row("002")], root)
            self.assertEqual(inventory["001"]["local_status"], "valid")
            self.assertEqual(inventory["002"]["local_status"], "missing")
            self.assertEqual(video.read_bytes(), original)

    def test_prioritizes_held_out_splits_and_new_signers(self):
        rows = [row("001", "train"), row("002", "val"), row("003", "test"),
                row("004", "val", "2"), row("005", "val", "1")]
        inventory = {item["video_id"]: dict(local_status="valid" if item["video_id"] == "005" else "missing",
                                           local_path="", local_errors="") for item in rows}
        plan = build_plan(rows, inventory, pd.DataFrame(), Path("raw_videos"))
        candidates = plan.loc[plan["local_status"].eq("missing")]
        identifiers = candidates["video_id"].tolist()
        self.assertLess(identifiers.index("002"), identifiers.index("001"))
        self.assertLess(identifiers.index("003"), identifiers.index("001"))
        self.assertLess(identifiers.index("004"), identifiers.index("002"))
        self.assertFalse(plan["video_id"].duplicated().any())
        self.assertEqual(dict(zip(plan["video_id"], plan["split"])),
                         {item["video_id"]: item["split"] for item in rows})
        self.assertTrue(plan.loc[plan["video_id"].eq("005"), "priority"].eq("").all())
        self.assertTrue(plan["remote_status"].eq("not_checked").all())

    def test_output_protection(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            path.write_bytes(b"existing plan")
            with self.assertRaises(FileExistsError):
                check_outputs([path], False)
            check_outputs([path], True)
            self.assertEqual(path.read_bytes(), b"existing plan")


if __name__ == "__main__":
    unittest.main()
