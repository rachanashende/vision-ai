r"""Validate and optionally extract explicitly selected or newly downloaded clips.

PowerShell, from the project root:
    python scripts/process_new_videos.py --video-ids 11311 63226 --report dataset/metadata/preflight.csv
    python scripts/process_new_videos.py --video-ids 11311 63226 --extract --report dataset/metadata/extraction.csv
    python scripts/process_new_videos.py --auto-missing

Default is full video validation without inference. --extract reuses the existing
batch extractor, validator, and atomic NPZ writer. Existing NPZs are never replaced:
valid ones skip inference; invalid ones are preserved and reported for manual repair.
Without explicit IDs, selection uses successful expansion downloads whose local
MP4 exists and whose NPZ is missing or invalid. No inference occurs in selection.
Reports are never overwritten: the default filename includes a UTC timestamp,
or choose a new --report path for a later invocation. No shared quality
or inspection report is generated. Source boundaries remain unapplied/uncertain.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

import extract_landmarks_batch as batch
from inspect_videos import inspect_video, load_metadata, match_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["video_id", "gloss", "split", "signer_id", "metadata_valid", "video_valid",
           "npz_valid", "fps", "reported_frame_count", "decoded_frame_count",
           "frames_processed", "npz_frame_count", "extraction_status", "inference_error_frames",
           "video_path", "npz_path", "errors"]


def select_targets(root, metadata, download_report, video_ids=None):
    """Resolve IDs from WLASL; auto mode skips valid saved identity metadata."""
    if video_ids is not None:
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("Duplicate requested video IDs.")
        identifiers = video_ids
    else:
        identifiers = sorted(set(download_report.loc[
            download_report["status"].isin(["downloaded", "already_valid"]), "video_id"
        ]))
    targets = []
    for video_id in identifiers:
        if not re.fullmatch(r"[0-9]+", video_id):
            raise ValueError(f"Invalid video_id: {video_id!r}")
        matches = metadata.loc[metadata["video_id"].eq(video_id)]
        if len(matches) != 1:
            raise ValueError(f"video_id {video_id} has {len(matches)} WLASL matches; selection is ambiguous or unknown.")
        row = matches.iloc[0]
        gloss, split = row["gloss"], row["split"]
        if not gloss or gloss in (".", "..") or any(character in gloss for character in "/\\:"):
            raise ValueError(f"Invalid WLASL gloss for {video_id}.")
        if split not in ("train", "val", "test") or not str(row["signer_id"]):
            raise ValueError(f"Missing/invalid WLASL split or signer for {video_id}.")
        if video_ids is None:
            video = root / "dataset/raw_videos" / gloss / f"{video_id}.mp4"
            if not video.is_file():
                continue
            output = root / "dataset/processed" / gloss / f"{video_id}.npz"
            valid, _ = batch.validate_output(output)
            if valid:
                with np.load(output, allow_pickle=False) as archive:
                    consistent = all(archive[key].item() == str(row[key]) for key in
                                     ("gloss", "video_id", "split", "signer_id", "frame_start", "frame_end"))
                if consistent:
                    continue
        targets.append((gloss, video_id, split))
    return targets


def process_clip(root, target, metadata, metadata_source, download_report, extract=False):
    gloss, video_id, split = target
    video = root / "dataset/raw_videos" / gloss / f"{video_id}.mp4"
    output = root / "dataset/processed" / gloss / f"{video_id}.npz"
    row = dict(video_id=video_id, gloss=gloss, split=split, signer_id="", metadata_valid=False,
               video_valid=None, npz_valid=None, frames_processed=0, npz_frame_count=0,
               extraction_status="validation_failed", inference_error_frames=0,
               video_path=str(video), npz_path=str(output), errors="")
    try:
        matched, status = match_metadata(metadata, gloss, video_id)
        if status in ("missing", "ambiguous") or matched.get("split") != split or not str(matched.get("signer_id", "")):
            raise ValueError("WLASL identity/split/signer does not match the requested clip.")
        by_id = metadata.loc[metadata["video_id"].eq(video_id)]
        if len(by_id) != 1:
            raise ValueError("video_id is not unique in WLASL metadata.")
        row["signer_id"] = str(matched["signer_id"])
        successful = download_report.loc[download_report["video_id"].eq(video_id)
                                          & download_report["status"].isin(["downloaded", "already_valid"])]
        if successful.empty or any(
            record[key] != row[key] for record in successful.to_dict("records")
            for key in ("gloss", "split", "signer_id")
        ):
            raise ValueError("Successful expansion-download history missing or conflicts with WLASL identity metadata.")
        row["metadata_valid"] = True
        if not video.is_file():
            row["video_valid"] = False
            raise FileNotFoundError(f"Source MP4 missing: {video}")
        inspected = inspect_video(video)
        row.update(fps=inspected["fps"], reported_frame_count=inspected["total_frame_count"],
                   decoded_frame_count=inspected["decoded_frame_count"])
        row["video_valid"] = bool(inspected["valid"] and inspected["fps"] is not None
                                  and inspected["fps"] > 0 and inspected["total_frame_count"] is not None
                                  and inspected["total_frame_count"] > 0)
        if not row["video_valid"]:
            raise ValueError(inspected["errors"] or "Positive FPS/frame count and decoded frames are required.")
        saved_metadata = {key: str(matched.get(key, ""))
                          for key in ("gloss", "video_id", "split", "signer_id", "frame_start", "frame_end")}
        saved_metadata.update(trimming_status="uncertain; source boundaries not applied",
                              source_video=str(video.resolve()), metadata_source=metadata_source,
                              mediapipe_version=batch.mp.__version__)
        if output.exists():
            valid, error = batch.validate_output(output, saved_metadata)
            if valid:
                with np.load(output, allow_pickle=False) as archive:
                    row["npz_frame_count"] = len(archive["frame_indices"])
                    valid = (row["npz_frame_count"] == row["decoded_frame_count"]
                             and np.isclose(archive["fps"].item(), row["fps"], rtol=1e-4, atol=1e-6))
                    row["inference_error_frames"] = int((~archive["inference_ok"]).sum())
                    if not valid:
                        error = "Saved NPZ frame count/FPS differs from the validated local video."
            row["npz_valid"] = bool(valid)
            row["extraction_status"] = "skipped_existing_valid" if valid else "existing_npz_invalid"
            row["errors"] = "" if valid else error + " Existing NPZ preserved; inspect or move it manually before retrying."
            return row
        if not extract:
            row["extraction_status"] = "validated_not_extracted"
            return row
        model = root / "models/holistic_landmarker.task"
        if not model.is_file():
            raise FileNotFoundError(f"Missing existing Holistic model: {model}")
        row["extraction_status"] = "extraction_failed"
        payload, metrics = batch.extract_video(video, model)
        row.update(frames_processed=metrics["decoded_frame_count"],
                   inference_error_frames=metrics["inference_error_frames"])
        if payload is None:
            raise ValueError(metrics["errors"])
        if len(payload["frame_indices"]) != row["decoded_frame_count"]:
            raise ValueError("Extraction frame count differs from preflight decoding; output not saved.")
        payload.update({key: np.asarray(value) for key, value in saved_metadata.items()})
        batch.save_output(output, payload, overwrite=False)
        valid, error = batch.validate_output(output, saved_metadata)
        row.update(npz_valid=valid, npz_frame_count=len(payload["frame_indices"]),
                   extraction_status=metrics["status"], errors="; ".join(filter(None, [metrics["errors"], error])))
    except Exception as error:
        # Keep a report row for each requested clip and continue the independent batch.
        row["errors"] = str(error)
    return row


def validate_report(rows, targets):
    expected = set(targets)
    identities = [(row["gloss"], row["video_id"], row["split"]) for row in rows]
    if len(rows) != len(targets) or set(identities) != expected or len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError("Report must contain exactly the selected identities and original splits, without duplicates.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="Run extraction only after video/metadata validation.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--video-ids", nargs="+", help="Only these original WLASL IDs, preserving leading zeros.")
    selection.add_argument("--auto-missing", action="store_true", help="Select local successful downloads without valid NPZs (default).")
    parser.add_argument("--report", type=Path, help="New CSV path; default is a timestamped file under dataset/metadata.")
    args = parser.parse_args()
    report_path = (args.report.resolve() if args.report else PROJECT_ROOT / "dataset/metadata" /
                   f"new_video_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}.csv")
    try:
        if report_path.exists():
            raise FileExistsError(f"Report already exists: {report_path}. Preserve it and choose a new --report path.")
        metadata, source = load_metadata(PROJECT_ROOT)
        downloads = pd.read_csv(PROJECT_ROOT / "dataset/metadata/expansion_download_report.csv", dtype=str, keep_default_na=False)
        if not {"video_id", "gloss", "split", "signer_id", "status"}.issubset(downloads.columns):
            raise ValueError("Expansion download report is missing required columns.")
        targets = select_targets(PROJECT_ROOT, metadata, downloads, args.video_ids)
        rows = []
        for target in targets:
            row = process_clip(PROJECT_ROOT, target, metadata, source, downloads, args.extract)
            rows.append(row)
            print(f"{row['gloss']}/{row['video_id']} [{row['split']}]: {row['extraction_status']}"
                  f"; video_valid={row['video_valid']}; npz_valid={row['npz_valid']}"
                  + (f" - {row['errors']}" if row["errors"] else ""))
        validate_report(rows, targets)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8", newline="") as stream:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(stream, index=False)
        print(f"Report: {report_path}")
        print("frames_processed counts this run's inference frames; npz_frame_count includes skipped cached outputs.")
        print(f"Valid videos: {sum(row['video_valid'] is True for row in rows)}; "
              f"extracted archives: {sum(row['extraction_status'] in ('processed', 'processed_with_errors') for row in rows)}; "
              f"processed frames: {sum(row['frames_processed'] for row in rows)}; "
              f"inference errors: {sum(row['inference_error_frames'] for row in rows)}")
        return 1 if any(row["errors"] for row in rows) else 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
