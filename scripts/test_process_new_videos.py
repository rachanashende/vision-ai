"""Offline checks: python -B scripts/test_process_new_videos.py."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import process_new_videos as workflow
from test_extract_landmarks_batch import sample_payload


class NewVideoTests(unittest.TestCase):
    def setup_inputs(self, root):
        target = ("drink", "17713", "test")
        gloss, identifier, split = target
        video = root / "dataset/raw_videos" / gloss / f"{identifier}.mp4"
        video.parent.mkdir(parents=True)
        video.touch()
        model = root / "models/holistic_landmarker.task"
        model.parent.mkdir()
        model.touch()
        record = dict(gloss=gloss, video_id=identifier, split=split, signer_id="007", frame_start="1", frame_end="-1")
        return target, pd.DataFrame([record]), pd.DataFrame([{**record, "status": "downloaded"}])

    def test_split_preservation_and_no_duplicate_extraction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target, metadata, downloads = self.setup_inputs(root)
            payload = sample_payload()
            inspected = dict(valid=True, fps=10.0, total_frame_count=2, decoded_frame_count=2, errors="")
            metrics = dict(decoded_frame_count=2, inference_error_frames=0, status="processed", errors="")
            with patch.object(workflow, "inspect_video", return_value=inspected), patch.object(
                workflow.batch, "extract_video", return_value=(payload, metrics)
            ) as extraction:
                first = workflow.process_clip(root, target, metadata, "source", downloads, extract=True)
                self.assertEqual(first["extraction_status"], "processed")
                output = Path(first["npz_path"])
                original = output.read_bytes()
                second = workflow.process_clip(root, target, metadata, "source", downloads, extract=True)
                self.assertEqual(second["extraction_status"], "skipped_existing_valid")
                self.assertEqual(extraction.call_count, 1)
                self.assertEqual(output.read_bytes(), original)
                with workflow.np.load(output, allow_pickle=False) as saved:
                    self.assertEqual(saved["split"].item(), "test")
                    self.assertEqual(saved["signer_id"].item(), "007")
                    self.assertEqual(saved["video_id"].item(), "17713")

    def test_failed_validation_blocks_inference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target, metadata, downloads = self.setup_inputs(root)
            inspected = dict(valid=False, fps=None, total_frame_count=0, decoded_frame_count=0, errors="No frames")
            with patch.object(workflow, "inspect_video", return_value=inspected), patch.object(
                workflow.batch, "extract_video", side_effect=AssertionError("Inference forbidden")
            ) as extraction:
                row = workflow.process_clip(root, target, metadata, "source", downloads, extract=True)
                self.assertFalse(row["video_valid"])
                extraction.assert_not_called()
                metadata.loc[0, "split"] = "train"
                row = workflow.process_clip(root, target, metadata, "source", downloads, extract=True)
                self.assertFalse(row["metadata_valid"])
                extraction.assert_not_called()

    def test_report_integrity_and_existing_report_preservation(self):
        targets = [("drink", "17713", "test"), ("computer", "12312", "val")]
        rows = [dict(gloss=gloss, video_id=identifier, split=split) for gloss, identifier, split in targets]
        workflow.validate_report(rows, targets)
        with self.assertRaises(ValueError):
            workflow.validate_report(rows[:-1] + [rows[0]], targets)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "existing.csv"
            path.write_bytes(b"previous report")
            with patch("sys.argv", ["process_new_videos.py", "--report", str(path)]), patch.object(
                workflow, "load_metadata", side_effect=AssertionError("Should stop before processing")
            ):
                self.assertEqual(workflow.main(), 1)
            self.assertEqual(path.read_bytes(), b"previous report")

    def test_dynamic_explicit_ids_and_auto_missing_selection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target, metadata, downloads = self.setup_inputs(root)
            second = {**metadata.iloc[0].to_dict(), "video_id": "17714", "split": "val"}
            metadata = pd.concat([metadata, pd.DataFrame([second])], ignore_index=True)
            downloads = pd.concat([downloads, pd.DataFrame([{**second, "status": "downloaded"}]), downloads], ignore_index=True)
            video = root / "dataset/raw_videos/drink/17714.mp4"
            video.touch()
            selected = workflow.select_targets(root, metadata, downloads, ["17714", "17713"])
            self.assertEqual(selected, [("drink", "17714", "val"), target])
            self.assertEqual(workflow.select_targets(root, metadata, downloads), [target, ("drink", "17714", "val")])
            payload = sample_payload()
            payload.update({key: workflow.np.asarray(str(metadata.iloc[0][key])) for key in
                            ("gloss", "video_id", "split", "signer_id", "frame_start", "frame_end")})
            output = root / "dataset/processed/drink/17713.npz"
            workflow.batch.save_output(output, payload)
            self.assertEqual(workflow.select_targets(root, metadata, downloads), [("drink", "17714", "val")])
            original = output.read_bytes()
            workflow.select_targets(root, metadata, downloads, ["17713"])
            self.assertEqual(output.read_bytes(), original)
            with self.assertRaises(ValueError):
                workflow.select_targets(root, metadata, downloads, ["17713", "17713"])
            with self.assertRaises(ValueError):
                workflow.select_targets(root, metadata, downloads, ["99999"])


if __name__ == "__main__":
    unittest.main()
