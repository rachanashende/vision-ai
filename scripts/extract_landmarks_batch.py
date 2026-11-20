r"""Extract variable-length Holistic diagnostics; default is dry-run.

PowerShell from the project root, with the existing environment active:
    python scripts/extract_landmarks_batch.py --dry-run
    python scripts/extract_landmarks_batch.py --extract --limit 1
    python scripts/extract_landmarks_batch.py --extract

See results/evaluation/landmark_batch_schema.md. No downloads, training,
trimming, resampling, or final GRU feature-vector concatenation is performed.
"""

import argparse
import math
from pathlib import Path
import sys
import tempfile
from zipfile import BadZipFile

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from inspect_videos import load_metadata, match_metadata
from test_landmark_extraction import GROUPS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "asl-holistic-xyz-v1"
METADATA_KEYS = ["gloss", "video_id", "split", "signer_id", "frame_start", "frame_end",
                 "trimming_status", "metadata_source", "source_video", "mediapipe_version"]
REPORT_COLUMNS = [
    "gloss", "video_id", "split", "signer_id", "source_video", "output", "status",
    "reported_frame_count", "decoded_frame_count", "successful_inference_frames",
    "inference_error_frames", "left_hand_detection_rate", "right_hand_detection_rate",
    "pose_detection_rate", "frame_start", "frame_end", "trimming_status", "errors",
]


def pack_result(result):
    """Missing groups are zeros with False masks; reject unexpected model schemas."""
    packed = {}
    for group, count in GROUPS.items():
        landmarks = getattr(result, f"{group}_landmarks") if result is not None else []
        if len(landmarks) not in (0, count):
            raise ValueError(f"Unexpected {group} landmark count: {len(landmarks)}")
        coordinates = np.zeros((count, 3), dtype=np.float32)
        if landmarks:
            coordinates[:] = [[point.x, point.y, point.z] for point in landmarks]
            if not np.isfinite(coordinates).all():
                raise ValueError(f"Non-finite {group} coordinates returned.")
        packed[f"{group}_landmarks"] = coordinates
        packed[f"{group}_present"] = bool(landmarks)
    return packed


def validate_output(path, expected=None):
    """Load without pickle and check shapes, masks, timing, and metadata."""
    try:
        with np.load(path, allow_pickle=False) as saved:
            if saved["schema_version"].item() != SCHEMA_VERSION:
                raise ValueError("Schema version differs.")
            fps = saved["fps"].item()
            if not math.isfinite(fps) or fps <= 0 or saved["timestamp_source"].shape != ():
                raise ValueError("Invalid timing metadata.")
            for key in METADATA_KEYS:
                if saved[key].shape != () or saved[key].dtype.kind != "U":
                    raise ValueError(f"Invalid metadata field: {key}")
                if expected is not None and saved[key].item() != str(expected[key]):
                    raise ValueError(f"Metadata differs: {key}")
            if saved["split"].item() not in ("train", "val", "test") or not saved["signer_id"].item():
                raise ValueError("Missing split or signer metadata.")
            indices = saved["frame_indices"]
            count = len(indices)
            if count == 0 or indices.dtype != np.int64 or not np.array_equal(indices, np.arange(count)):
                raise ValueError("Invalid frame indices.")
            timestamps = saved["timestamps_ms"]
            if timestamps.shape != (count,) or timestamps.dtype != np.int64 or not np.all(np.diff(timestamps) > 0):
                raise ValueError("Invalid timestamps.")
            inference_ok = saved["inference_ok"]
            if inference_ok.shape != (count,) or inference_ok.dtype != np.bool_ or not inference_ok.any():
                raise ValueError("Invalid inference mask.")
            for group, landmarks in GROUPS.items():
                coordinates = saved[f"{group}_landmarks"]
                mask = saved[f"{group}_present"]
                if coordinates.shape != (count, landmarks, 3) or coordinates.dtype != np.float32:
                    raise ValueError(f"Invalid {group} coordinate shape/type.")
                if mask.shape != (count,) or mask.dtype != np.bool_ or np.any(mask & ~inference_ok):
                    raise ValueError(f"Invalid {group} presence mask.")
                if not np.isfinite(coordinates).all() or np.any(coordinates[~mask] != 0):
                    raise ValueError(f"Invalid {group} coordinates or missing-group padding.")
        return True, ""
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, TypeError, IndexError) as error:
        return False, str(error)


def save_output(destination, payload, overwrite=False):
    """Validate a temporary archive, then publish it without partial outputs."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
        valid, error = validate_output(temporary)
        if not valid:
            raise ValueError(f"Output validation failed: {error}")
        if overwrite:
            temporary.replace(destination)
        else:
            # Publish atomically without overwriting a concurrently created output.
            destination.hardlink_to(temporary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def extract_video(video_path, model_path):
    """Use a new detector per video, so tracking and timestamps never leak."""
    frames = []
    timestamps = []
    inference_ok = []
    errors = []
    metrics = dict(reported_frame_count=None, decoded_frame_count=0,
                   successful_inference_frames=0, inference_error_frames=0)
    capture = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid FPS; cannot derive VIDEO timestamps safely.")
        reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if math.isfinite(reported) and reported > 0:
            metrics["reported_frame_count"] = int(reported)
        vision = mp.tasks.vision
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
        )
        with vision.HolisticLandmarker.create_from_options(options) as detector:
            while True:
                decoded, frame = capture.read()
                if not decoded:
                    break
                index = metrics["decoded_frame_count"]
                metrics["decoded_frame_count"] += 1
                timestamp = max(timestamps[-1] + 1 if timestamps else 0, round(index * 1000 / fps))
                timestamps.append(timestamp)
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(image, timestamp)
                    frames.append(pack_result(result))
                    inference_ok.append(True)
                    metrics["successful_inference_frames"] += 1
                except (AttributeError, cv2.error, RuntimeError, ValueError) as error:
                    frames.append(pack_result(None))
                    inference_ok.append(False)
                    metrics["inference_error_frames"] += 1
                    errors.append(f"Frame {index}: {error}")
        if not frames:
            raise ValueError("No frames could be decoded.")
        if metrics["reported_frame_count"] and len(frames) < metrics["reported_frame_count"]:
            raise ValueError("Decoding ended before the reported frame count; invalid or inaccurate container count. No output saved.")
        if not any(inference_ok):
            raise ValueError("Inference failed on every frame. No output saved.")
        payload = {key: np.asarray([frame[key] for frame in frames]) for key in frames[0]}
        payload.update(frame_indices=np.arange(len(frames), dtype=np.int64),
                       timestamps_ms=np.asarray(timestamps, dtype=np.int64),
                       inference_ok=np.asarray(inference_ok, dtype=np.bool_),
                       fps=np.asarray(fps, dtype=np.float64),
                       timestamp_source=np.asarray("frame_index/reported_fps; rounded and strictly increasing"),
                       schema_version=np.asarray(SCHEMA_VERSION))
        for group in GROUPS:
            metrics[f"{group}_detection_rate"] = float(payload[f"{group}_present"].mean())
        metrics["status"] = "processed_with_errors" if errors else "processed"
        metrics["errors"] = "; ".join(errors)
        return payload, metrics
    except (AttributeError, OSError, cv2.error, RuntimeError, ValueError) as error:
        metrics.update(status="failed", errors="; ".join(errors + [str(error)]))
        return None, metrics
    finally:
        if capture is not None:
            capture.release()


def positive_limit(value):
    count = int(value)
    if count <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer.")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Plan only (default); no inference or NPZ writes.")
    mode.add_argument("--extract", action="store_true", help="Actually run extraction and save NPZ files.")
    parser.add_argument("--limit", type=positive_limit, help="Consider only the first N MP4 paths in sorted order.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing NPZ files in extraction mode.")
    args = parser.parse_args()
    raw_dir = PROJECT_ROOT / "dataset/raw_videos"
    model_path = PROJECT_ROOT / "models/holistic_landmarker.task"
    report_path = PROJECT_ROOT / "dataset/metadata/landmark_quality_report.csv"
    try:
        metadata, metadata_source = load_metadata(PROJECT_ROOT)
        videos = sorted(path for path in raw_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() == ".mp4")
        if args.limit:
            videos = videos[:args.limit]
        if args.extract and not model_path.is_file():
            raise FileNotFoundError(f"Missing local model: {model_path}. Nothing downloaded.")
        print("EXTRACTION MODE" if args.extract else "DRY RUN: no inference or NPZ writes")
        report = []
        destinations = set()
        for video in videos:
            parts = video.relative_to(raw_dir).parts
            gloss = parts[0] if len(parts) > 1 else ""
            row = dict(gloss=gloss, video_id=video.stem, source_video=str(video.resolve()),
                       status="failed", errors="", trimming_status="uncertain; source boundaries not applied")
            try:
                matched, metadata_status = match_metadata(metadata, gloss, video.stem)
                if metadata_status in ("missing", "ambiguous"):
                    raise ValueError(f"WLASL metadata is {metadata_status}; split/signer cannot be preserved reliably.")
                gloss = gloss or matched["gloss"]
                if not gloss or gloss in (".", "..") or any(character in gloss for character in "/\\:"):
                    raise ValueError("Invalid gloss for output directory.")
                row.update({key: str(matched.get(key, ""))
                            for key in ("split", "signer_id", "frame_start", "frame_end")})
                row["gloss"] = gloss
                if row["split"] not in ("train", "val", "test") or not row["signer_id"]:
                    raise ValueError("Missing/unknown WLASL split or signer; no output saved.")
                destination = PROJECT_ROOT / "dataset/processed" / gloss / f"{video.stem}.npz"
                row["output"] = str(destination)
                if destination in destinations:
                    raise ValueError("Multiple source paths map to the same output; duplicate skipped.")
                destinations.add(destination)
                saved_metadata = {key: row[key] for key in
                                  ("gloss", "video_id", "split", "signer_id", "frame_start", "frame_end",
                                   "trimming_status", "source_video")}
                saved_metadata.update(metadata_source=metadata_source, mediapipe_version=mp.__version__)
                if destination.exists() and not args.overwrite:
                    valid, error = validate_output(destination, saved_metadata)
                    row.update(status="skipped_valid" if valid else "existing_invalid",
                               errors="" if valid else f"{error}; use --overwrite to replace.")
                elif not args.extract:
                    row["status"] = "dry_run"
                else:
                    payload, metrics = extract_video(video, model_path)
                    row.update(metrics)
                    if payload is not None:
                        payload.update({key: np.asarray(str(value)) for key, value in saved_metadata.items()})
                        save_output(destination, payload, overwrite=args.overwrite)
            except (OSError, ValueError, KeyError) as error:
                row.update(status="failed", errors=str(error))
            report.append(row)
            print(f"{row['gloss']}/{row['video_id']}: {row['status']}"
                  + (f" - {row['errors']}" if row["errors"] else ""))
            report_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(report, columns=REPORT_COLUMNS).to_csv(report_path, index=False)
        if not videos:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=REPORT_COLUMNS).to_csv(report_path, index=False)
            print(f"No MP4 files found under {raw_dir}.")
        print(f"Quality report (this invocation): {report_path}")
        return 1 if any(row["status"] in ("failed", "existing_invalid", "processed_with_errors") for row in report) else 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
