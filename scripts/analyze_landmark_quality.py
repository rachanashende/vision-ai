r"""Analyze existing landmark NPZs without inference or changes to source data.

PowerShell from the project root:
    python scripts/analyze_landmark_quality.py

Inspected schema: results/evaluation/landmark_batch_schema.md,
asl-holistic-xyz-v1. Array names/shapes below follow that version explicitly;
unknown versions are rejected. No pickle loading is allowed.

Detection rates use successful inference frames; *_all_frames_rate uses all
frames, matching the extraction report. Missing groups on failed inference
frames are UNKNOWN, not absent. Review heuristics: either-hand rate < 0.50,
pose rate < 0.80, any inference failures, invalid archives, or source-report
inconsistencies. A single missing hand alone is not a reason to reject a sign.
These are diagnostic thresholds, not automatic exclusion rules.

Writes new landmark_analysis_report.csv and landmark_quality_summary.png.
Refuses to overwrite either output if it already exists. Original archives,
metadata reports, scripts, and the environment are never modified.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "asl-holistic-xyz-v1"
GROUPS = {"left_hand": 21, "right_hand": 21, "pose": 33}
RATE_NAMES = ["left_hand", "right_hand", "either_hand", "both_hands", "pose"]
TEXT_FIELDS = ["schema_version", "gloss", "video_id", "split", "signer_id", "frame_start",
               "frame_end", "trimming_status", "metadata_source", "source_video",
               "mediapipe_version", "timestamp_source"]
BASE_COLUMNS = ["npz_path", "gloss", "video_id", "split", "signer_id", "schema_version",
                "validation_status", "frame_count", "successful_inference_frames",
                "failed_inference_frames", "inference_failure_rate", "needs_manual_review",
                "review_reasons", "validation_errors", "extraction_report_status",
                "extraction_errors", "source_report_notes"]
OUTPUT_COLUMNS = BASE_COLUMNS + [f"{name}_detection_rate" for name in RATE_NAMES] + [
    f"{name}_all_frames_rate" for name in RATE_NAMES
] + [f"{name}_absent_on_success_frames" for name in GROUPS]


def validate_arrays(arrays, path, expected_gloss=""):
    """Return all detected problems before any statistics are trusted."""
    errors = []
    for key in TEXT_FIELDS:
        value = arrays.get(key)
        if value is None or value.shape != () or value.dtype.kind != "U":
            errors.append(f"Missing/invalid scalar Unicode metadata: {key}")
    if errors:
        return errors
    if arrays["schema_version"].item() != SCHEMA_VERSION:
        errors.append("Unsupported schema version.")
    for key in ("gloss", "video_id", "signer_id"):
        if not arrays[key].item().strip():
            errors.append(f"Empty required metadata: {key}")
    if arrays["split"].item() not in ("train", "val", "test"):
        errors.append("Invalid split; expected train, val, or test.")
    if arrays["video_id"].item() != path.stem:
        errors.append("video_id does not match the archive filename.")
    if expected_gloss and arrays["gloss"].item() != expected_gloss:
        errors.append("gloss does not match the processed class directory.")
    for key, value in arrays.items():
        if value.dtype.kind in "biufc" and not np.isfinite(value).all():
            errors.append(f"Non-finite numerical values: {key}")
    indices = arrays.get("frame_indices")
    if indices is None or indices.ndim != 1 or indices.dtype != np.int64 or len(indices) == 0:
        return errors + ["Missing/invalid nonempty int64 frame_indices."]
    count = len(indices)
    if not np.array_equal(indices, np.arange(count)):
        errors.append("Frame indices must preserve every decoded frame in order, starting at zero.")
    timestamps = arrays.get("timestamps_ms")
    if timestamps is None or timestamps.shape != (count,) or timestamps.dtype != np.int64:
        errors.append("Timestamp shape/type does not match frame count.")
    elif timestamps[0] < 0 or not np.all(np.diff(timestamps) > 0):
        errors.append("Timestamps must be nonnegative and strictly increasing.")
    fps = arrays.get("fps")
    if fps is None or fps.shape != () or fps.dtype != np.float64 or not np.isfinite(fps).all() or fps.item() <= 0:
        errors.append("Missing/invalid positive float64 FPS.")
    inference = arrays.get("inference_ok")
    inference_valid = inference is not None and inference.shape == (count,) and inference.dtype == np.bool_
    if not inference_valid:
        errors.append("Inference mask must be Boolean and match frame count.")
    elif not inference.any():
        errors.append("No successful inference frames.")
    for group, landmarks in GROUPS.items():
        coordinates = arrays.get(f"{group}_landmarks")
        mask = arrays.get(f"{group}_present")
        coordinates_valid = (coordinates is not None and coordinates.shape == (count, landmarks, 3)
                             and coordinates.dtype == np.float32)
        mask_valid = mask is not None and mask.shape == (count,) and mask.dtype == np.bool_
        if not coordinates_valid:
            errors.append(f"{group}: expected float32 coordinates shaped ({count}, {landmarks}, 3).")
        if not mask_valid:
            errors.append(f"{group}: expected Boolean presence mask shaped ({count},).")
        if mask_valid and inference_valid and np.any(mask & ~inference):
            errors.append(f"{group}: present=True on failed inference frames.")
        if mask_valid and coordinates_valid and np.any(coordinates[~mask] != 0):
            errors.append(f"{group}: absent/failed group coordinates are not zero-filled.")
    return errors


def detection_statistics(arrays):
    inference = arrays["inference_ok"]
    successful = int(inference.sum())
    count = len(inference)
    masks = {name: arrays[f"{name}_present"] for name in GROUPS}
    masks["either_hand"] = masks["left_hand"] | masks["right_hand"]
    masks["both_hands"] = masks["left_hand"] & masks["right_hand"]
    statistics = dict(frame_count=count, successful_inference_frames=successful,
                      failed_inference_frames=count - successful,
                      inference_failure_rate=(count - successful) / count)
    for name, mask in masks.items():
        statistics[f"{name}_detection_rate"] = float(mask[inference].mean()) if successful else np.nan
        statistics[f"{name}_all_frames_rate"] = float(mask.mean())
    for name in GROUPS:
        statistics[f"{name}_absent_on_success_frames"] = int((inference & ~masks[name]).sum())
    return statistics


def compare_extraction_report(row, source_report):
    """The extraction CSV may contain only a limited invocation; absence is a note."""
    if not {"gloss", "video_id"}.issubset(source_report.columns):
        return ["Extraction report unavailable or missing identity columns."], []
    matches = source_report.loc[source_report["gloss"].eq(row["gloss"])
                                & source_report["video_id"].eq(row["video_id"])]
    if matches.empty:
        return ["No extraction report row; the report may cover a limited invocation."], []
    if len(matches) != 1:
        return [], ["Ambiguous duplicate extraction report rows."]
    source = matches.iloc[0]
    row["extraction_report_status"] = source.get("status", "")
    row["extraction_errors"] = source.get("errors", "")
    mismatches = []
    for key in ("split", "signer_id"):
        if source.get(key, "") and source[key] != row.get(key):
            mismatches.append(f"Extraction report {key} differs from NPZ metadata.")
    if row["validation_status"] == "valid":
        comparisons = {"decoded_frame_count": row["frame_count"],
                       "successful_inference_frames": row["successful_inference_frames"],
                       "inference_error_frames": row["failed_inference_frames"]}
        comparisons.update({f"{group}_detection_rate": row[f"{group}_all_frames_rate"] for group in GROUPS})
        for key, value in comparisons.items():
            if source.get(key, "") != "":
                try:
                    if not np.isclose(float(source[key]), value, rtol=1e-6, atol=1e-8):
                        mismatches.append(f"Extraction report {key} differs from NPZ calculation.")
                except ValueError:
                    mismatches.append(f"Extraction report {key} is not numeric.")
    return [], mismatches


def analyze_archive(path, processed_dir, source_report):
    parts = path.relative_to(processed_dir).parts
    gloss = parts[0] if len(parts) > 1 else ""
    row = dict(npz_path=str(path.resolve()), gloss=gloss, video_id=path.stem,
               validation_status="invalid", validation_errors="")
    arrays = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            # Read every member so corrupt or unreadable optional arrays are also reported.
            arrays = {key: archive[key] for key in archive.files}
        errors = validate_arrays(arrays, path, gloss)
        for key in ("gloss", "video_id", "split", "signer_id", "schema_version"):
            value = arrays.get(key)
            if value is not None and value.shape == () and value.dtype.kind == "U":
                row[key] = value.item()
        if not errors:
            row["validation_status"] = "valid"
            row.update(detection_statistics(arrays))
    except Exception as error:
        # Treat each archive as an independent input; one corrupt file must not end the batch.
        errors = [f"Could not read/validate archive: {type(error).__name__}: {error}"]
    row["validation_errors"] = "; ".join(errors)
    reasons = ["Archive validation failed."] if errors else []
    if row["validation_status"] == "valid":
        if row["either_hand_detection_rate"] < 0.50:
            reasons.append("Either-hand detection below 50% of successful inference frames.")
        if row["pose_detection_rate"] < 0.80:
            reasons.append("Pose detection below 80% of successful inference frames.")
        if row["failed_inference_frames"]:
            reasons.append("Inference failed on one or more frames; absence is unknown there.")
    notes, mismatches = compare_extraction_report(row, source_report)
    row["source_report_notes"] = "; ".join(notes + mismatches)
    reasons.extend(mismatches)
    row["needs_manual_review"] = bool(reasons)
    row["review_reasons"] = "; ".join(reasons)
    return row


def make_summary(report):
    rows = []
    for gloss, group in report.groupby("gloss", dropna=False, sort=True):
        valid = group.loc[group["validation_status"].eq("valid")]
        row = dict(gloss=gloss or "(unknown)", videos=len(group), valid=len(valid),
                   invalid=len(group) - len(valid), review=int(group["needs_manual_review"].sum()))
        for name in RATE_NAMES:
            row[name] = valid[f"{name}_detection_rate"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def draw_summary(summary, output_path):
    figure, (counts, rates) = plt.subplots(1, 2, figsize=(14, max(4, 0.48 * len(summary) + 2)))
    if summary.empty:
        for axis in (counts, rates):
            axis.text(0.5, 0.5, "No NPZ files found", ha="center", va="center")
            axis.set_axis_off()
    else:
        positions = np.arange(len(summary))
        counts.barh(positions, summary["valid"], label="Valid", color="#337b88")
        counts.barh(positions, summary["invalid"], left=summary["valid"], label="Invalid", color="#bd594b")
        counts.set_yticks(positions, summary["gloss"])
        counts.invert_yaxis()
        counts.set_xlabel("Number of NPZ files")
        counts.set_title("Archives per class (review flags do not delete data)")
        counts.legend()
        matrix = summary[RATE_NAMES].to_numpy(dtype=float)
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#dddddd")
        plot = rates.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, aspect="auto", cmap=cmap)
        rates.set_xticks(np.arange(5), ["Left", "Right", "Either", "Both", "Pose"])
        rates.set_yticks(positions, summary["gloss"])
        rates.set_title("Mean per-video detection rates\nValid archives; successful inference frames only")
        for y in range(len(summary)):
            for x in range(5):
                value = matrix[y, x]
                rates.text(x, y, f"{value:.0%}" if np.isfinite(value) else "N/A", ha="center", va="center",
                           color="white" if np.isfinite(value) and value > 0.6 else "black")
        figure.colorbar(plot, ax=rates, label="Detection fraction")
    figure.tight_layout()
    with output_path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=150)
    plt.close(figure)


def main():
    processed_dir = PROJECT_ROOT / "dataset/processed"
    source_path = PROJECT_ROOT / "dataset/metadata/landmark_quality_report.csv"
    report_path = PROJECT_ROOT / "dataset/metadata/landmark_analysis_report.csv"
    chart_path = PROJECT_ROOT / "results/graphs/landmark_quality_summary.png"
    try:
        for path in (report_path, chart_path):
            if path.exists():
                raise FileExistsError(f"Output already exists: {path}. Move it aside before rerunning; existing files are preserved.")
        try:
            source_report = pd.read_csv(source_path, dtype=str, keep_default_na=False)
        except (OSError, ValueError) as error:
            print(f"WARNING: Could not read extraction report: {error}", file=sys.stderr)
            source_report = pd.DataFrame()
        paths = sorted(path for path in processed_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".npz")
        rows = [analyze_archive(path, processed_dir, source_report) for path in paths]
        report = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
        summary = make_summary(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("x", encoding="utf-8", newline="") as stream:
            report.to_csv(stream, index=False)
        draw_summary(summary, chart_path)
        print("Per-class counts and mean per-video detection rates (0-1, successful inference frames):")
        print(summary.round(3).to_string(index=False) if not summary.empty else "No NPZ files found.")
        print(f"\nFiles: {len(report)}; valid: {sum(row['validation_status'] == 'valid' for row in rows)}; "
              f"manual review: {sum(row['needs_manual_review'] for row in rows)}")
        print("Review thresholds: either hand < 50%, pose < 80%; inference/validation/report inconsistencies also flagged.")
        print("Failed inference is unknown, not confirmed landmark absence. No files were deleted or excluded.")
        print(f"Report: {report_path}\nChart: {chart_path}")
        return 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
