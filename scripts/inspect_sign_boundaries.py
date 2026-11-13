r"""Inspect local sign timing without inference, cropping, or automatic boundaries.

PowerShell: python scripts/inspect_sign_boundaries.py
Use --overwrite only to replace this workflow's CSV, summary, and contact sheets;
doing so resets any manual edits to those generated outputs.
WLASL boundaries are 1-based source annotations; proposed boundaries, when
manually verified later, must be zero-based inclusive LOCAL decoded indices.
The script always leaves proposals blank. Twelve chronological samples include
both endpoints. Source orientation is preserved, with no WLASL offsets applied.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

from analyze_landmark_quality import validate_arrays
from explore_dataset import find_metadata, flatten_entries


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["gloss", "video_id", "split", "signer_id", "video_path", "npz_path",
           "metadata_frame_start", "metadata_frame_end", "metadata_fps", "decoded_frame_count",
           "fps", "duration_seconds", "trim_status", "trim_evidence", "proposed_sign_start",
           "proposed_sign_end", "review_status", "reviewer_notes", "manual_keep_decision",
           "manual_boundary_verification", "contact_sheet_path", "sampled_frame_indices",
           "timestamp_source", "mask_alignment", "processing_status", "errors"]


def match_video_metadata(metadata, video_id, gloss):
    matches = [row for row in metadata if str(row.get("video_id", "")) == video_id]
    if len(matches) != 1:
        raise ValueError(f"video_id {video_id}: expected one WLASL instance; found {len(matches)}.")
    if matches[0]["gloss"] != gloss:
        raise ValueError(f"video_id {video_id}: directory gloss differs from WLASL gloss.")
    return matches[0]


def validate_proposed_boundaries(start, end, frame_count, manually_verified=False):
    """Validate later manual proposals; blank pairs are intentionally accepted."""
    if start == "" and end == "":
        return
    if not manually_verified:
        raise ValueError("Nonblank proposals require explicit manual verification.")
    if not str(start).isdigit() or not str(end).isdigit():
        raise ValueError("Both local boundaries must be nonnegative integer frame indices.")
    if not 0 <= int(start) <= int(end) < frame_count:
        raise ValueError("Proposed local boundaries must satisfy 0 <= start <= end < decoded frame count.")


def check_outputs(paths, overwrite):
    if not overwrite:
        for path in paths:
            if path.exists():
                raise FileExistsError(f"Output exists: {path}. Preserve it or explicitly use --overwrite.")


def sample_indices(frame_count):
    if frame_count < 1:
        raise ValueError("No decoded frames available.")
    return np.rint(np.linspace(0, frame_count - 1, 12)).astype(int).tolist()


def inspect_video(path):
    """Count actual decoded frames first, then sample original frames sequentially."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else None
        count = 0
        while True:
            success, _ = capture.read()
            if not success:
                break
            count += 1
    finally:
        capture.release()
    indices = sample_indices(count)
    capture = cv2.VideoCapture(str(path))
    samples = {}
    estimated = False
    try:
        for index in range(count):
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Second decode failed at source frame {index}.")
            if index not in indices:
                continue
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(timestamp) or timestamp < 0 or (index > 0 and timestamp <= 0):
                timestamp = index * 1000 / fps if fps else None
                estimated = True
            samples[index] = (frame, timestamp)
    finally:
        capture.release()
    return count, fps, indices, samples, "source position with FPS fallback" if estimated else "OpenCV source position"


def boundary_evidence(metadata, count, fps):
    try:
        start, end = int(metadata["frame_start"]), int(metadata["frame_end"])
        if start < 1 or (end != -1 and end < start):
            raise ValueError("Invalid 1-based source boundary ordering.")
    except (KeyError, TypeError, ValueError) as error:
        return "needs_review", f"Missing/invalid WLASL boundaries: {error}"
    if start == 1 and end == -1:
        return "unknown", "WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset."
    notes = "Nontrivial source annotations exist; local/source alignment and prior trimming are unverified."
    metadata_fps = metadata.get("fps")
    if fps and isinstance(metadata_fps, (int, float)) and math.isfinite(metadata_fps) and metadata_fps > 0:
        duration = count / fps
        if (start - 1) / metadata_fps >= duration or (end != -1 and end / metadata_fps > duration + max(1 / fps, 1 / metadata_fps)):
            notes += " Source boundary time exceeds local duration: possible prior trimming, different source, or timing mismatch; not proof of trimming."
    return "needs_review", notes


def load_masks(path, gloss, video_id, count, fps, metadata):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    errors = validate_arrays(arrays, path, gloss)
    if errors:
        raise ValueError("; ".join(errors))
    if arrays["video_id"].item() != video_id or len(arrays["frame_indices"]) != count:
        raise ValueError("NPZ identity/frame count differs from decoded video.")
    for key in ("split", "signer_id"):
        if arrays[key].item() != str(metadata.get(key, "")):
            raise ValueError(f"NPZ / WLASL {key} mismatch.")
    if fps and not np.isclose(arrays["fps"].item(), fps, rtol=1e-4, atol=1e-6):
        raise ValueError("NPZ / local FPS mismatch.")
    return {int(index): (bool(arrays["inference_ok"][slot]), bool(arrays["left_hand_present"][slot]),
                         bool(arrays["right_hand_present"][slot]))
            for slot, index in enumerate(arrays["frame_indices"])}


def make_sheet(gloss, video_id, indices, samples, masks):
    tiles = []
    seen = set()
    for index in indices:
        frame, timestamp = samples[index]
        scale = min(480 / frame.shape[1], 270 / frame.shape[0])
        width, height = max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        tile = np.full((345, 480, 3), 24, dtype=np.uint8)
        y, x = (270 - height) // 2, (480 - width) // 2
        tile[y:y + height, x:x + width] = image
        time_label = f"{timestamp / 1000:.3f}s" if timestamp is not None else "time unknown"
        repeat = " (repeat)" if index in seen else ""
        state = masks.get(index) if masks is not None else None
        hand_labels = ["unknown", "unknown"]
        if state and state[0]:
            hand_labels = ["detected" if present else "not detected" for present in state[1:]]
        elif state:
            hand_labels = ["unknown: inference failed"] * 2
        lines = [(f"{gloss}/{video_id} | frame {index}{repeat} | {time_label}", (255, 255, 255)),
                 (f"MP LEFT: {hand_labels[0]}", (80, 255, 80)),
                 (f"MP RIGHT: {hand_labels[1]}", (255, 80, 255))]
        for line_index, (text, color) in enumerate(lines):
            cv2.putText(tile, text, (8, 290 + line_index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, color, 1, cv2.LINE_AA)
        tiles.append(tile)
        seen.add(index)
    grid = cv2.vconcat([cv2.hconcat(tiles[offset:offset + 4]) for offset in (0, 4, 8)])
    header = np.full((80, grid.shape[1], 3), 24, dtype=np.uint8)
    for line, text in enumerate((f"SIGN BOUNDARY REVIEW: {gloss}/{video_id} - chronological source frames",
                                 "Source orientation; no flip or trim offsets. MP left/right = anatomical labels.",
                                 "Saved presence masks only; no inference. Presence does not establish sign boundaries.")):
        cv2.putText(header, text, (15, 23 + line * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.vconcat([header, grid])


def write_summary(path, rows, metadata_path, overwrite):
    lines = ["# Sign-boundary inspection", "", f"Inspected {len(rows)} local videos using `{metadata_path}`.",
             "All clips require manual boundary review. Proposed start/end values remain blank.", "",
             "WLASL frame_start/frame_end are 1-based source annotations; -1 means source EOF.",
             "Local review proposals must use zero-based inclusive decoded frame indices.",
             "Metadata FPS can differ from local FPS; source frame numbers cannot be copied blindly.",
             "No clip is labelled already_trimmed or untrimmed without verified provenance.",
             "unknown means insufficient evidence; needs_review flags missing data or a mismatch.", "",
             "Landmark presence alone cannot identify sign boundaries: hands may be detected at rest,",
             "absent during signing, or unobserved because inference failed. Rest/transition frames",
             "can coexist with a pre-cut clip. Short duration or an annotation outside local duration",
             "is ambiguous evidence, not proof of prior trimming. Sampled frames cannot verify sign",
             "correctness or every-frame tracking. Full temporal review is still required.", "",
             "Saved masks are aligned by source frame index, decoded length, identity, and FPS.",
             "No content checksum exists in NPZs, so unchanged video content cannot be proven.",
             "Timestamps use OpenCV source positions, with labelled FPS estimates when unavailable.",
             "Twelve evenly spaced samples include endpoints; very short videos repeat labelled frames.", "",
             "The five provisional_keep decisions are retained as context, not boundary verification.",
             "Original videos, NPZs, metadata, and manual decisions are unchanged. No inference,",
             "cropping, automatic proposals, trimming, or training was performed.", "",
             "| Clip | Decoded frames | Local FPS | WLASL start/end | Trim status | Inspection |",
             "| --- | ---: | ---: | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['gloss']}/{row['video_id']} | {row.get('decoded_frame_count', '')} | "
                     f"{row.get('fps', '')} | {row.get('metadata_frame_start', '')}/{row.get('metadata_frame_end', '')} | "
                     f"{row['trim_status']} | {row['processing_status']} |")
    lines += ["", "## Ambiguities and errors", ""]
    for row in rows:
        lines.append(f"- {row['gloss']}/{row['video_id']}: {row.get('trim_evidence', '')} {row.get('errors', '')}")
    lines += ["", "Manual workflow: inspect the sheet, then watch the entire source clip around candidate",
              "onset/offset frames. Record only manually verified local boundaries and the reviewer’s",
              "evidence in the CSV. Check 0 <= start <= end < decoded_frame_count using",
              "validate_proposed_boundaries(..., manually_verified=True). Do not run --overwrite",
              "after recording decisions unless intentional replacement of generated reviews is desired.", ""]
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Replace generated outputs, including manual edits to the review CSV.")
    args = parser.parse_args()
    csv_path = PROJECT_ROOT / "dataset/metadata/sign_boundary_review.csv"
    summary_path = PROJECT_ROOT / "results/evaluation/sign_boundary_inspection_summary.md"
    sheet_dir = PROJECT_ROOT / "results/graphs/sign_boundary_review"
    try:
        raw_dir = PROJECT_ROOT / "dataset/raw_videos"
        videos = sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4")
        sheets = [sheet_dir / f"{path.relative_to(raw_dir).parts[0]}_{path.stem}_boundaries.png" for path in videos]
        if len(set(sheets)) != len(sheets):
            raise ValueError("Duplicate local video identities would share an output; resolve ambiguity first.")
        check_outputs([csv_path, summary_path] + sheets, args.overwrite)
        metadata_path = PROJECT_ROOT / "dataset/WLASL/start_kit/WLASL_v0.3.json"
        if not metadata_path.is_file():
            metadata_path = find_metadata(PROJECT_ROOT / "dataset/WLASL")
        metadata, _ = flatten_entries(json.loads(metadata_path.read_text(encoding="utf-8-sig")))
        decisions = pd.read_csv(PROJECT_ROOT / "dataset/metadata/manual_review_decisions.csv", dtype=str, keep_default_na=False)
        if decisions.duplicated(["gloss", "video_id"]).any():
            raise ValueError("Duplicate manual review identities; existing decisions were not changed.")
        rows = []
        for video, sheet_path in zip(videos, sheets):
            gloss, video_id = video.relative_to(raw_dir).parts[0], video.stem
            npz_path = PROJECT_ROOT / "dataset/processed" / gloss / f"{video_id}.npz"
            row = dict(gloss=gloss, video_id=video_id, video_path=str(video), npz_path=str(npz_path),
                       trim_status="needs_review", proposed_sign_start="", proposed_sign_end="",
                       review_status="pending_manual_boundary_review", reviewer_notes="",
                       processing_status="failed", errors="", contact_sheet_path="")
            try:
                matched = match_video_metadata(metadata, video_id, gloss)
                row.update(split=matched.get("split", ""), signer_id=matched.get("signer_id", ""),
                           metadata_frame_start=matched.get("frame_start", ""), metadata_frame_end=matched.get("frame_end", ""),
                           metadata_fps=matched.get("fps", ""))
                manual = decisions.loc[decisions["gloss"].eq(gloss) & decisions["video_id"].eq(video_id)]
                if len(manual) == 1:
                    row.update(manual_keep_decision=manual.iloc[0]["decision"],
                               manual_boundary_verification=manual.iloc[0]["sign_boundary_verification"])
                count, fps, indices, samples, timing = inspect_video(video)
                row.update(decoded_frame_count=count, fps=fps, duration_seconds=count / fps if fps else None,
                           sampled_frame_indices=";".join(map(str, indices)), timestamp_source=timing)
                row["trim_status"], row["trim_evidence"] = boundary_evidence(matched, count, fps)
                try:
                    masks = load_masks(npz_path, gloss, video_id, count, fps, matched)
                    row["mask_alignment"] = "matched source frame indices; length and identity checked"
                except Exception as error:
                    masks = None
                    row.update(mask_alignment="unavailable_or_mismatched", trim_status="needs_review", errors=str(error))
                if fps is None or count < 9:
                    row["trim_status"] = "needs_review"
                validate_proposed_boundaries("", "", count)
                sheet = make_sheet(gloss, video_id, indices, samples, masks)
                success, encoded = cv2.imencode(".png", sheet)
                if not success:
                    raise OSError("Contact sheet encoding failed.")
                sheet_dir.mkdir(parents=True, exist_ok=True)
                with sheet_path.open("wb" if args.overwrite else "xb") as stream:
                    stream.write(encoded.tobytes())
                row.update(contact_sheet_path=str(sheet_path), processing_status="inspected" if masks is not None else "inspected_with_warnings")
            except Exception as error:
                row.update(trim_status="needs_review", errors=str(error))
            rows.append(row)
            print(f"{gloss}/{video_id}: {row['processing_status']}; trim={row['trim_status']}; manual boundaries pending")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w" if args.overwrite else "x", encoding="utf-8", newline="") as stream:
            pd.DataFrame(rows, columns=COLUMNS).to_csv(stream, index=False)
        write_summary(summary_path, rows, metadata_path, args.overwrite)
        inspected = sum(row["processing_status"].startswith("inspected") for row in rows)
        print(f"\n{inspected}/{len(rows)} clips inspected. All {len(rows)} require manual sign-boundary review.")
        print("Pending: " + ", ".join(f"{row['gloss']}/{row['video_id']}" for row in rows))
        print(f"Review CSV: {csv_path}\nSummary: {summary_path}\nSheets: {sheet_dir}")
        return 0 if all(row["processing_status"] == "inspected" for row in rows) else 1
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
