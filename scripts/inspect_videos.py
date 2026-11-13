r"""Inspect local MP4 files without changing them or contacting any URLs.

PowerShell, from the project root:
    .\.venv\Scripts\python.exe scripts/inspect_videos.py

Uses existing OpenCV and pandas. Reads every frame sequentially. A video is
valid if at least one frame decodes and no decoding/count error is detected.
Unusually short means a measured duration below 1 second. Unknown FPS gives
an unknown duration, not a zero duration. WLASL boundaries are never applied.
Overwrites dataset/metadata/video_inspection_report.csv, even with no videos.
"""

import json
import math
from pathlib import Path
import sys

import cv2
import pandas as pd

from explore_dataset import find_metadata, flatten_entries


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHORT_SECONDS = 1.0
METADATA_COLUMNS = ["gloss", "video_id", "frame_start", "frame_end", "split", "signer_id"]
REPORT_COLUMNS = [
    "gloss", "video_id", "path", "width", "height", "fps", "total_frame_count",
    "decoded_frame_count", "duration_seconds", "frames_decodable", "valid",
    "unusually_short", "frame_start", "frame_end", "split", "signer_id",
    "metadata_status", "metadata_source", "trimming_status", "errors", "warnings",
]


def load_metadata(project_root):
    """Prefer original WLASL JSON; fall back to the exploration CSV if needed."""
    try:
        source = find_metadata(project_root / "dataset/WLASL")
        with source.open(encoding="utf-8-sig") as stream:
            rows, _ = flatten_entries(json.load(stream))
        if not rows:
            raise ValueError("Original metadata contains no usable instances.")
        metadata = pd.DataFrame(rows)
    except (OSError, ValueError) as error:
        print(f"WARNING: Original WLASL metadata unavailable: {error}", file=sys.stderr)
        source = project_root / "dataset/metadata/wlasl_instances.csv"
        try:
            metadata = pd.read_csv(source, dtype=str, keep_default_na=False)
        except (OSError, ValueError) as error:
            print(f"WARNING: Metadata CSV unavailable: {error}", file=sys.stderr)
            return pd.DataFrame(columns=METADATA_COLUMNS), ""
    metadata = metadata.reindex(columns=METADATA_COLUMNS).fillna("")
    return metadata.drop_duplicates(), str(source)


def match_metadata(metadata, gloss, video_id):
    matches = metadata.loc[metadata["video_id"].eq(video_id)]
    if gloss:
        matches = matches.loc[matches["gloss"].eq(gloss)]
    if matches.empty:
        return {}, "missing"
    if len(matches) > 1:
        return {}, "ambiguous"
    row = matches.iloc[0].to_dict()
    status = "incomplete" if any(row[column] == "" for column in METADATA_COLUMNS) else "matched"
    return row, status


def positive_property(capture, property_id):
    value = capture.get(property_id)
    return value if math.isfinite(value) and value > 0 else None


def inspect_video(path):
    """Record reported properties separately from the number of decoded frames."""
    result = dict(width=None, height=None, fps=None, total_frame_count=None,
                  decoded_frame_count=0, duration_seconds=None,
                  frames_decodable=False, valid=False, unusually_short=None)
    errors = []
    warnings = []
    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            errors.append("OpenCV could not open the video.")
        else:
            for name, property_id in (
                ("width", cv2.CAP_PROP_FRAME_WIDTH),
                ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                ("fps", cv2.CAP_PROP_FPS),
                ("total_frame_count", cv2.CAP_PROP_FRAME_COUNT),
            ):
                value = positive_property(capture, property_id)
                result[name] = value if name == "fps" or value is None else int(value)
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame is None or frame.size == 0:
                    errors.append("Decoder returned an empty frame.")
                    break
                result["decoded_frame_count"] += 1
            result["frames_decodable"] = result["decoded_frame_count"] > 0
            if not result["frames_decodable"]:
                errors.append("No frames could be decoded.")
            reported = result["total_frame_count"]
            if reported is not None and result["decoded_frame_count"] < reported:
                errors.append(
                    f"Decoded {result['decoded_frame_count']} of {reported} reported frames; "
                    "possible truncation, decode failure, or inaccurate container frame count."
                )
            if result["fps"] is None:
                warnings.append("FPS is zero, missing, or non-finite; duration is unknown.")
            elif result["frames_decodable"]:
                result["duration_seconds"] = result["decoded_frame_count"] / result["fps"]
                result["unusually_short"] = result["duration_seconds"] < SHORT_SECONDS
            if reported is None:
                warnings.append("Container frame count unavailable; see decoded_frame_count.")
            result["valid"] = result["frames_decodable"] and not errors
    except (cv2.error, OSError, ValueError) as error:
        errors.append(str(error))
        result["frames_decodable"] = result["decoded_frame_count"] > 0
        result["valid"] = False
    finally:
        if capture is not None:
            capture.release()
    result["errors"] = "; ".join(errors)
    result["warnings"] = "; ".join(warnings)
    return result


def main():
    raw_dir = PROJECT_ROOT / "dataset/raw_videos"
    report_path = PROJECT_ROOT / "dataset/metadata/video_inspection_report.csv"
    try:
        metadata, metadata_source = load_metadata(PROJECT_ROOT)
        videos = sorted(path for path in raw_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() == ".mp4")
        if not raw_dir.is_dir():
            print(f"WARNING: Raw video directory does not exist: {raw_dir}")
        rows = []
        for path in videos:
            relative_path = path.relative_to(raw_dir)
            gloss = relative_path.parts[0] if len(relative_path.parts) > 1 else ""
            video_id = path.stem
            matched, metadata_status = match_metadata(metadata, gloss, video_id)
            row = dict(gloss=gloss or matched.get("gloss", ""), video_id=video_id,
                       path=str(path.resolve()), **inspect_video(path))
            row.update({key: matched.get(key, "")
                        for key in ("frame_start", "frame_end", "split", "signer_id")})
            row.update(metadata_status=metadata_status, metadata_source=metadata_source,
                       trimming_status="uncertain: source timing and trimming history unverified; boundaries not applied")
            rows.append(row)
            print(f"{row['gloss']}/{video_id}: {'valid' if row['valid'] else 'invalid'}")
        report = pd.DataFrame(rows, columns=REPORT_COLUMNS)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
        valid = sum(row["valid"] for row in rows)
        short = sum(row["unusually_short"] is True for row in rows)
        missing = sum(row["metadata_status"] != "matched" for row in rows)
        print(f"\nVideos inspected: {len(rows)}")
        print(f"Valid: {valid}; invalid: {len(rows) - valid}")
        print(f"Unusually short (< {SHORT_SECONDS:g} second): {short}")
        print(f"Missing, incomplete, or ambiguous metadata: {missing}")
        print("Short-video and metadata counts can overlap valid/invalid counts.")
        print("Duration uses decoded frames / reported FPS; unknown duration is left blank.")
        print("Trimming status is uncertain for every video; WLASL boundaries were not applied.")
        print(f"Report saved to: {report_path}")
        return 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
