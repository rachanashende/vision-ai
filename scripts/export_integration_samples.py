r"""PROVISIONAL real NPZ -> (60,149) float32 integration samples; no inference/model.

Generate: python -B scripts/export_integration_samples.py
Verify:   python -B scripts/export_integration_samples.py --verify
Use --output-dir for a fresh destination; existing export directories are preserved.
Contract: results/evaluation/ml_extension_contract.md. No source files are written.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from analyze_landmark_quality import validate_arrays
from inspect_videos import load_metadata, match_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_VERSION = "asl-extension-149-v0.1-PROVISIONAL"
POSE_INDICES = [11, 12, 13, 14, 15, 16]
LABELS = ["book", "drink", "computer", "before", "chair", "go", "clothes", "who", "help", "yes"]
SAMPLES = [("book", "07068"), ("drink", "17713"), ("computer", "12312"),
           ("before", "05728"), ("chair", "09851"), ("go", "24946"),
           ("clothes", "11311"), ("who", "63229"), ("help", "27209"), ("yes", "64287")]
MASK_COLUMNS = {"left_hand_present": 144, "right_hand_present": 145, "pose_present": 146,
                "inference_ok": 147, "is_real_frame": 148}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    errors = validate_arrays(arrays, path, path.parent.name)
    if errors:
        raise ValueError(f"{path}: {'; '.join(errors)}")
    return arrays


def frame_features(arrays):
    """144 coordinates + 5 explicit flags, in the contract's exact order."""
    count = len(arrays["frame_indices"])
    features = np.zeros((count, 149), dtype="<f4")
    inference = arrays["inference_ok"]
    for group, start, end, indices in (
        ("left_hand", 0, 63, list(range(21))),
        ("right_hand", 63, 126, list(range(21))),
        ("pose", 126, 144, POSE_INDICES),
    ):
        mask = inference & arrays[f"{group}_present"]
        coordinates = arrays[f"{group}_landmarks"][:, indices, :]
        features[mask, start:end] = coordinates[mask].reshape((-1, end - start))
        features[:, MASK_COLUMNS[f"{group}_present"]] = mask
    features[:, 147] = inference
    features[:, 148] = 1
    if not np.isfinite(features).all():
        raise ValueError("Non-finite feature values; refusing export.")
    return features


def resample_slots(timestamps):
    """Nearest source timestamp, earlier tie; short sequences retain timing + pad."""
    times = np.asarray(timestamps)
    if times.ndim != 1 or times.dtype != np.int64 or len(times) == 0:
        raise ValueError("Expected nonempty int64 timestamps.")
    if times[0] < 0 or not np.all(np.diff(times) > 0):
        raise ValueError("Timestamps must be nonnegative and strictly increasing.")
    if int(times[-1]) > (2**53 - 1) // 59:
        raise ValueError("Timestamp range exceeds exact JavaScript integer arithmetic for this algorithm.")
    if len(times) < 60:
        return np.concatenate([np.arange(len(times), dtype=np.int64),
                               np.full(60 - len(times), -1, dtype=np.int64)])
    # Integer target numerators avoid Python/JavaScript rounding differences.
    return np.asarray([np.argmin(np.abs(59 * times - (59 * times[0] + index * (times[-1] - times[0]))))
                       for index in range(60)], dtype=np.int64)


def convert_sequence(arrays):
    features = frame_features(arrays)
    slots = resample_slots(arrays["timestamps_ms"])
    result = np.zeros((60, 149), dtype="<f4")
    real = slots >= 0
    result[real] = features[slots[real]]
    return result, slots


def export_samples(root, output_dir, samples=SAMPLES):
    """Validate all inputs first, create a new directory, and write real arrays."""
    if output_dir.exists():
        raise FileExistsError(f"Export directory exists: {output_dir}. Verify it or choose a new --output-dir.")
    if not 5 <= len(samples) <= 10 or len({identifier for _, identifier in samples}) != len(samples):
        raise ValueError("Export must contain 5-10 distinct real video IDs.")
    metadata, _ = load_metadata(root)
    records, outputs = [], []
    for gloss, video_id in samples:
        path = root / "dataset/processed" / gloss / f"{video_id}.npz"
        digest = sha256(path)
        arrays = load_archive(path)
        matched, status = match_metadata(metadata, gloss, video_id)
        if status in ("missing", "ambiguous") or any(str(matched.get(key, "")) != arrays[key].item()
                                                      for key in ("gloss", "video_id", "split", "signer_id")):
            raise ValueError(f"WLASL metadata conflict for {gloss}/{video_id}.")
        features, slots = convert_sequence(arrays)
        filename = f"PROVISIONAL_{gloss}_{video_id}.npy"
        real = slots >= 0
        source_indices = np.full(60, -1, dtype=np.int64)
        source_times = np.full(60, -1, dtype=np.int64)
        source_indices[real] = arrays["frame_indices"][slots[real]]
        source_times[real] = arrays["timestamps_ms"][slots[real]]
        records.append(dict(status="PROVISIONAL", array_file=filename, source_npz=str(path.relative_to(root)),
                            source_sha256=digest, raw_schema=arrays["schema_version"].item(),
                            gloss=gloss, video_id=video_id, split=arrays["split"].item(),
                            signer_id=arrays["signer_id"].item(), proposed_label_index=LABELS.index(gloss),
                            original_frame_count=len(arrays["frame_indices"]), source_fps=arrays["fps"].item(),
                            source_timestamp_convention=arrays["timestamp_source"].item(),
                            source_frame_start=arrays["frame_start"].item(), source_frame_end=arrays["frame_end"].item(),
                            trimming_status=arrays["trimming_status"].item(), boundaries_verified=False,
                            sampling_mode="nearest_time_60" if len(arrays["frame_indices"]) >= 60 else "original_then_right_pad",
                            source_slots=slots.tolist(), source_frame_indices=source_indices.tolist(),
                            source_timestamps_ms=source_times.tolist(), padding_frames=int((~real).sum()),
                            frame_valid=((features[:, 148] == 1) & (features[:, 147] == 1)
                                         & ((features[:, 144] == 1) | (features[:, 145] == 1))).tolist()))
        outputs.append(features)
        if sha256(path) != digest:
            raise ValueError(f"Source changed while exporting: {path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    for record, array in zip(records, outputs):
        destination = output_dir / record["array_file"]
        with destination.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
        record["array_sha256"] = sha256(destination)
    manifest = dict(status="PROVISIONAL", contract_version=CONTRACT_VERSION, shape=[60, 149],
                    dtype="little-endian float32", order="C / time-major", pose_indices=POSE_INDICES,
                    mask_columns=MASK_COLUMNS, proposed_labels=LABELS, samples=records)
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def verify_samples(root, output_dir):
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "PROVISIONAL" or manifest["contract_version"] != CONTRACT_VERSION
        or manifest["shape"] != [60, 149] or manifest["pose_indices"] != POSE_INDICES
        or manifest["mask_columns"] != MASK_COLUMNS or manifest["proposed_labels"] != LABELS):
        raise ValueError("Manifest contract differs from the exporter.")
    identifiers = set()
    if not 5 <= len(manifest["samples"]) <= 10:
        raise ValueError("Expected 5-10 real samples.")
    for record in manifest["samples"]:
        if record["video_id"] in identifiers:
            raise ValueError("Duplicate sample video ID.")
        identifiers.add(record["video_id"])
        source = (root / record["source_npz"]).resolve()
        destination = (output_dir / record["array_file"]).resolve()
        if not source.is_relative_to((root / "dataset/processed").resolve()) or not destination.is_relative_to(output_dir.resolve()):
            raise ValueError("Manifest paths must remain inside the source/export directories.")
        if sha256(source) != record["source_sha256"] or sha256(destination) != record["array_sha256"]:
            raise ValueError("Source or sample checksum differs from the manifest.")
        arrays = load_archive(source)
        expected, slots = convert_sequence(arrays)
        saved = np.load(destination, allow_pickle=False)
        if saved.shape != (60, 149) or saved.dtype != np.dtype("<f4") or not saved.flags.c_contiguous or not np.isfinite(saved).all():
            raise ValueError("Invalid sample shape, dtype or values.")
        np.testing.assert_array_equal(saved, expected)
        if slots.tolist() != record["source_slots"]:
            raise ValueError("Resampling provenance differs.")
        real = slots >= 0
        indices = np.full(60, -1, dtype=np.int64)
        times = np.full(60, -1, dtype=np.int64)
        indices[real] = arrays["frame_indices"][slots[real]]
        times[real] = arrays["timestamps_ms"][slots[real]]
        valid_frames = ((saved[:, 148] == 1) & (saved[:, 147] == 1)
                        & ((saved[:, 144] == 1) | (saved[:, 145] == 1)))
        if (record["source_frame_indices"] != indices.tolist() or record["source_timestamps_ms"] != times.tolist()
            or record["frame_valid"] != valid_frames.tolist() or record["padding_frames"] != int((~real).sum())
            or record["original_frame_count"] != len(arrays["frame_indices"])
            or record["proposed_label_index"] != LABELS.index(record["gloss"])):
            raise ValueError("Manifest timing, masks, or label proposal differs.")
        for key in ("gloss", "video_id", "split", "signer_id"):
            if record[key] != arrays[key].item():
                raise ValueError(f"Sample metadata differs: {key}")
        print(f"PASS: {record['gloss']}/{record['video_id']} [{record['split']}] (60,149) float32; padding={record['padding_frames']}")
    return len(manifest["samples"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify without writing; recompute arrays from original NPZs.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/integration_samples/PROVISIONAL_v1")
    args = parser.parse_args()
    try:
        output_dir = args.output_dir.resolve()
        if not args.verify:
            export_samples(PROJECT_ROOT, output_dir)
        count = verify_samples(PROJECT_ROOT, output_dir)
        print(f"PROVISIONAL: {count} real sequences verified. No model, inference, or source edits. Output: {output_dir}")
        return 0
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
