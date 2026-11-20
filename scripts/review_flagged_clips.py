r"""Overlay saved NPZ landmarks for exactly the clips flagged in the analysis CSV.

PowerShell from the project root:
    python scripts/review_flagged_clips.py
    python scripts/review_flagged_clips.py --overwrite

No MediaPipe import or inference. Uses asl-holistic-xyz-v1 and its original
zero-based source frame indices. Six samples span the saved sequence, including
both endpoints. Very short clips repeat frames with an explicit warning.
No WLASL frame_start/frame_end offsets, mirroring, or resampling are applied.
Left/right are saved MediaPipe anatomical labels, not screen-side labels.
Alignment checks compare decoded frame counts, source indices, and FPS; they
cannot prove unchanged video content because NPZs contain no source checksum.
"""

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

from analyze_landmark_quality import RATE_NAMES, validate_arrays


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATE_COLUMNS = [f"{name}_detection_rate" for name in RATE_NAMES]
OUTPUT_COLUMNS = ["gloss", "video_id", "split", "signer_id"] + RATE_COLUMNS + [
    "review_reasons", "failed_inference_frames", "npz_path", "video_path",
    "contact_sheet_path", "sampled_frame_indices", "status", "alignment_notes",
    "errors", "manual_review_decision",
]
COLORS = {"left_hand": (80, 255, 80), "right_hand": (255, 80, 255), "pose": (0, 220, 255)}


def select_flagged(report):
    required = {"needs_manual_review", "gloss", "video_id", "split", "signer_id", *RATE_COLUMNS}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"Analysis report is missing columns: {', '.join(sorted(missing))}")
    flags = report["needs_manual_review"].str.strip().str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError("needs_manual_review must contain only True or False; refusing to guess flags.")
    flagged = report.loc[flags.eq("true")].copy()
    if flagged.duplicated(["gloss", "video_id"]).any():
        raise ValueError("Duplicate flagged gloss/video_id identities; resolve report ambiguity first.")
    for row in flagged.to_dict("records"):
        for key in ("gloss", "video_id"):
            value = row[key]
            if not value or value in (".", "..") or any(character in value for character in "/\\:"):
                raise ValueError(f"Unsafe or missing {key}: {value!r}")
    return flagged


def read_aligned_frames(video_path, arrays):
    """Decode sequentially and select by saved SOURCE indices, never row guesses."""
    indices = arrays["frame_indices"]
    slots = np.rint(np.linspace(0, len(indices) - 1, 6)).astype(int)
    source_indices = [int(indices[slot]) for slot in slots]
    wanted = set(source_indices)
    selected = {}
    notes = []
    if len(wanted) < 6:
        notes.append("Fewer than six saved frames; repeated samples are explicitly labelled.")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open original video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            notes.append("Original video FPS unavailable; FPS alignment could not be checked.")
        elif not np.isclose(fps, arrays["fps"].item(), rtol=1e-4, atol=1e-6):
            raise ValueError(f"FPS mismatch: original={fps:g}, saved={arrays['fps'].item():g}; refusing overlay.")
        decoded_count = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if decoded_count in wanted:
                selected[decoded_count] = frame
            decoded_count += 1
    finally:
        capture.release()
    if decoded_count != len(indices):
        raise ValueError(f"Frame-count mismatch: original decoded {decoded_count}, NPZ has {len(indices)}; refusing overlay.")
    missing = wanted - selected.keys()
    if missing:
        raise ValueError(f"Saved source frame indices not decoded: {sorted(missing)}")
    notes.append("Saved source frame indices and decoded frame count match; no trim offsets applied.")
    return [(int(slot), index, selected[index]) for slot, index in zip(slots, source_indices)], notes


def make_tile(frame, arrays, slot, source_index, gloss, video_id, repeated=False):
    # Letterbox rather than stretch source frames with different aspect ratios.
    scale = min(640 / frame.shape[1], 360 / frame.shape[0])
    width, height = max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    inference_ok = bool(arrays["inference_ok"][slot])
    for group in ("pose", "left_hand", "right_hand"):
        if not inference_ok or not arrays[f"{group}_present"][slot]:
            continue
        for x, y, _ in arrays[f"{group}_landmarks"][slot]:
            if 0 <= x < 1 and 0 <= y < 1:
                cv2.circle(image, (round(float(x) * (width - 1)), round(float(y) * (height - 1))),
                           3, COLORS[group], -1, cv2.LINE_AA)
    tile = np.full((460, 640, 3), 24, dtype=np.uint8)
    top, left = (360 - height) // 2, (640 - width) // 2
    tile[top:top + height, left:left + width] = image
    cv2.putText(tile, f"{gloss}/{video_id} | source frame {source_index}" + (" (repeat)" if repeated else ""),
                (10, 381), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    for offset, group, label in ((403, "left_hand", "MP LEFT"), (425, "right_hand", "MP RIGHT"),
                                 (447, "pose", "POSE")):
        status = ("detected" if arrays[f"{group}_present"][slot] else "not detected") if inference_ok else "UNKNOWN (inference failed)"
        cv2.putText(tile, f"{label}: {status}", (10, offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, COLORS[group], 1, cv2.LINE_AA)
    return tile


def build_sheet(video_path, npz_path, gloss, video_id, report_row):
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    errors = validate_arrays(arrays, npz_path, gloss)
    if errors:
        raise ValueError("NPZ validation failed: " + "; ".join(errors))
    for key in ("gloss", "video_id", "split", "signer_id"):
        if arrays[key].item() != report_row[key]:
            raise ValueError(f"Analysis report / NPZ metadata mismatch: {key}")
    samples, notes = read_aligned_frames(video_path, arrays)
    tiles = []
    seen = set()
    for slot, source_index, frame in samples:
        tiles.append(make_tile(frame, arrays, slot, source_index, gloss, video_id, source_index in seen))
        seen.add(source_index)
    grid = cv2.vconcat([cv2.hconcat(tiles[:3]), cv2.hconcat(tiles[3:])])
    header = np.full((100, grid.shape[1], 3), 24, dtype=np.uint8)
    lines = [f"FLAGGED VISUAL REVIEW: {gloss}/{video_id} - SAVED landmarks only; no inference",
             "Source orientation; no horizontal flip. LEFT/RIGHT = saved MediaPipe anatomical labels.",
             "Green: left hand | Magenta: right hand | Yellow: pose | Decisions remain manual"]
    for index, line in enumerate(lines):
        cv2.putText(header, line, (18, 29 + index * 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.70 if index == 0 else 0.63, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([header, grid]), [index for _, index, _ in samples], notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace generated sheets and review CSV, including any manual decisions in it.")
    args = parser.parse_args()
    input_path = PROJECT_ROOT / "dataset/metadata/landmark_analysis_report.csv"
    output_path = PROJECT_ROOT / "dataset/metadata/flagged_visual_review.csv"
    sheet_dir = PROJECT_ROOT / "results/graphs/flagged_review"
    try:
        report = pd.read_csv(input_path, dtype=str, keep_default_na=False)
        flagged = select_flagged(report)
        records = flagged.to_dict("records")
        sheet_paths = [sheet_dir / f"{row['gloss']}_{row['video_id']}_review.png" for row in records]
        if not args.overwrite:
            for path in [output_path] + sheet_paths:
                if path.exists():
                    raise FileExistsError(f"Output already exists: {path}. Use --overwrite only if replacement is intended.")
        rows = []
        for source, sheet_path in zip(records, sheet_paths):
            gloss, video_id = source["gloss"], source["video_id"]
            video_path = PROJECT_ROOT / "dataset/raw_videos" / gloss / f"{video_id}.mp4"
            npz_path = PROJECT_ROOT / "dataset/processed" / gloss / f"{video_id}.npz"
            row = {key: source.get(key, "") for key in OUTPUT_COLUMNS}
            row.update(npz_path=str(npz_path), video_path=str(video_path), contact_sheet_path="",
                       status="failed", errors="", alignment_notes="", manual_review_decision="")
            try:
                sheet, samples, notes = build_sheet(video_path, npz_path, gloss, video_id, source)
                success, encoded = cv2.imencode(".png", sheet)
                if not success:
                    raise OSError("OpenCV could not encode contact sheet.")
                sheet_dir.mkdir(parents=True, exist_ok=True)
                with sheet_path.open("wb" if args.overwrite else "xb") as stream:
                    stream.write(encoded.tobytes())
                row.update(contact_sheet_path=str(sheet_path), status="generated",
                           sampled_frame_indices=";".join(map(str, samples)), alignment_notes="; ".join(notes))
            except Exception as error:
                # Each flagged clip still gets a review row when its data is missing or corrupt.
                row["errors"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            print(f"{gloss}/{video_id}: {row['status']}" + (f" - {row['errors']}" if row["errors"] else ""))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as stream:
            pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(stream, index=False)
        generated = sum(row["status"] == "generated" for row in rows)
        print(f"\nExactly {len(records)} clips flagged; {generated} sheets generated; {len(rows) - generated} failures.")
        print(f"Review CSV: {output_path}\nContact sheets: {sheet_dir}")
        print("manual_review_decision remains blank. No clips rejected or deleted; no inference performed.")
        return 0 if generated == len(records) else 1
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
