r"""Prepare PROVISIONAL GRU arrays using the existing integration converter.

PowerShell from the project root:
    python -B scripts/prepare_gru_dataset.py
    python -B scripts/prepare_gru_dataset.py --verify

--overwrite explicitly replaces only this script's six arrays, labels, manifest,
and preparation summary, after staging/validation. Source NPZs, raw videos,
reviews, integration fixtures, and the feature contract are never modified.
Invalid archives, duplicate IDs, or WLASL metadata conflicts abort preparation.
No inference, training, downloads, TensorFlow import, or TF.js export occurs.
"""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

import export_integration_samples as integration
from inspect_videos import load_metadata, match_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
ARRAY_FILES = [f"{prefix}_{split}.npy" for split in SPLITS for prefix in ("X", "y")]
DATA_FILES = ARRAY_FILES + ["labels.json", "manifest.json"]
DATASET_VERSION = "gru-dataset-PROVISIONAL_v1"


def labels_document():
    return dict(status="PROVISIONAL", contract_version=integration.CONTRACT_VERSION,
                class_to_index={label: index for index, label in enumerate(integration.LABELS)},
                index_to_class=list(integration.LABELS), num_classes=len(integration.LABELS),
                note="Proposed vocabulary/output order; Member 2/3 confirmation pending. No model trained.")


def validate_features(features):
    if features.shape != (60, 149) or features.dtype != np.dtype("<f4") or not np.isfinite(features).all():
        raise ValueError("Expected finite (60,149) float32 features.")
    flags = features[:, 144:149]
    if not np.isin(flags, [0, 1]).all():
        raise ValueError("Presence/inference/real-frame flags must be binary.")
    real = features[:, 148] == 1
    inference = features[:, 147] == 1
    if np.any(features[~real] != 0) or np.any(np.diff(real.astype(int)) > 0):
        raise ValueError("Padding must be zero-filled and occur only at the end.")
    if np.any(flags[:, :3][~inference] != 0):
        raise ValueError("Failed inference cannot mark groups present.")
    for start, end, key in ((0, 63, "left_hand_present"), (63, 126, "right_hand_present"), (126, 144, "pose_present")):
        absent = features[:, integration.MASK_COLUMNS[key]] == 0
        if np.any(features[absent, start:end] != 0):
            raise ValueError(f"Absent {key} coordinates must be zero-filled.")


def collect_dataset(root):
    paths = sorted((root / "dataset/processed").rglob("*.npz"))
    if not paths:
        raise ValueError("No NPZ archives found under dataset/processed.")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Duplicate video IDs in source archive paths; no outputs written.")
    metadata, _ = load_metadata(root)
    labels = labels_document()["class_to_index"]
    grouped = {split: [] for split in SPLITS}
    targets = {split: [] for split in SPLITS}
    records = []
    for path in paths:
        digest = integration.sha256(path)
        source = integration.load_archive(path)
        identity = {key: source[key].item() for key in ("gloss", "video_id", "split", "signer_id")}
        gloss, split = identity["gloss"], identity["split"]
        if gloss not in labels or split not in SPLITS:
            raise ValueError(f"Unknown gloss/split in {path}.")
        matched, status = match_metadata(metadata, gloss, identity["video_id"])
        if status in ("missing", "ambiguous") or len(metadata.loc[metadata["video_id"].eq(identity["video_id"])]) != 1:
            raise ValueError(f"Missing/ambiguous WLASL identity: {path}")
        for key in ("gloss", "video_id", "split", "signer_id", "frame_start", "frame_end"):
            if str(matched.get(key, "")) != source[key].item():
                raise ValueError(f"WLASL/source metadata mismatch for {path}: {key}")
        # One authoritative implementation for feature packing and temporal sampling.
        features, slots = integration.convert_sequence(source)
        validate_features(features)
        repeated, repeated_slots = integration.convert_sequence(source)
        np.testing.assert_array_equal(features, repeated)
        np.testing.assert_array_equal(slots, repeated_slots)
        index = len(grouped[split])
        grouped[split].append(features)
        targets[split].append(labels[gloss])
        records.append(dict(**identity, index_in_split=index, label_index=labels[gloss],
                            source_npz=path.relative_to(root).as_posix(), source_sha256=digest,
                            raw_schema=source["schema_version"].item(), original_frame_count=len(source["frame_indices"]),
                            source_frame_start=source["frame_start"].item(), source_frame_end=source["frame_end"].item(),
                            trimming_status=source["trimming_status"].item(), boundaries_verified=False,
                            source_slots=slots.tolist(), padding_frames=int((slots < 0).sum()),
                            source_inference_error_frames=int((~source["inference_ok"]).sum())))
        if integration.sha256(path) != digest:
            raise ValueError(f"Source changed during preparation: {path}")
    arrays = {}
    for split in SPLITS:
        arrays[f"X_{split}"] = np.stack(grouped[split]) if grouped[split] else np.empty((0, 60, 149), dtype="<f4")
        arrays[f"y_{split}"] = np.asarray(targets[split], dtype=np.int64)
    return arrays, records


def check_integration_parity(root, arrays, records):
    folder = root / "results/integration_samples/PROVISIONAL_v1"
    integration.verify_samples(root, folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    by_id = {row["video_id"]: row for row in records}
    for sample in manifest["samples"]:
        row = by_id[sample["video_id"]]
        if any(row[key] != sample[key] for key in ("gloss", "split", "signer_id")):
            raise ValueError("Integration sample identity differs from dataset.")
        saved = np.load(folder / sample["array_file"], allow_pickle=False)
        np.testing.assert_array_equal(arrays[f"X_{row['split']}"][row["index_in_split"]], saved)
    return len(manifest["samples"])


def make_manifest(records, files):
    signers = {split: {row["signer_id"] for row in records if row["split"] == split} for split in SPLITS}
    overlaps = {f"{left}_{right}": sorted(signers[left] & signers[right]) for left, right in combinations(SPLITS, 2)}
    counts = {gloss: {split: sum(row["gloss"] == gloss and row["split"] == split for row in records)
                      for split in SPLITS} for gloss in integration.LABELS}
    return dict(status="PROVISIONAL", dataset_version=DATASET_VERSION, contract_version=integration.CONTRACT_VERSION,
                preprocessor_sha256=integration.sha256(Path(integration.__file__)),
                feature_contract_sha256=integration.sha256(PROJECT_ROOT / "results/evaluation/ml_extension_contract.md"),
                labels=labels_document(), order="Source NPZ paths sorted; original split retained; no shuffle or augmentation",
                per_class_counts=counts, signer_overlap=overlaps, files=files, samples=records)


def summary_text(manifest):
    lines = ["# PROVISIONAL GRU dataset preparation", "", f"Dataset: `{DATASET_VERSION}`; contract: `{integration.CONTRACT_VERSION}`.",
             "Feature packing and timestamp resampling call export_integration_samples.convert_sequence directly.",
             "Arrays were recomputed twice for determinism and compared exactly with all ten real integration fixtures.",
             "No model was trained or exported. Member 2/3 JavaScript parity and final contract confirmation remain pending.", "",
             "| Class index | Gloss | Train | Val | Test |", "| ---: | --- | ---: | ---: | ---: |"]
    for index, gloss in enumerate(integration.LABELS):
        counts = manifest["per_class_counts"][gloss]
        lines.append(f"| {index} | {gloss} | {counts['train']} | {counts['val']} | {counts['test']} |")
    lines += ["", "## Saved array schema", "",
              "`X_<split>.npy`: (N,60,149), float32; `y_<split>.npy`: (N,), int64 sparse class indices.",
              "Use labels.json for the one shared PROVISIONAL class mapping, including classes absent in a split.",
              "Manifest samples map each split row to original gloss, video ID, signer, source checksum and sampling slots.",
              "All original splits and signer IDs are preserved. There is no split mixing, reassignment or shuffling.",
              "Mask columns 144-148 retain left/right/pose/inference/real-frame flags. Zero coordinates alone do not mean missing.",
              "Failed inference, absent hands and right-padding remain distinguishable. No frame is dropped for lacking hands.",
              "Long clips use timestamp-nearest 60-frame selection; short clips retain their frames then right-pad zeros.",
              "No extra normalization, cropping, WLASL frame offsets, or 60-frame interpolation was introduced.", "",
              "## Training limitations and leakage risks", ""]
    for split in ("val", "test"):
        missing = [gloss for gloss in integration.LABELS if manifest["per_class_counts"][gloss][split] == 0]
        lines.append(f"- Missing {split} classes: {', '.join(missing) if missing else 'none'}.")
    for pair, signers in manifest["signer_overlap"].items():
        lines.append(f"- Cross-split signer overlap {pair}: {', '.join(signers) if signers else 'none observed'}.")
    lines += ["- Shared signers can inflate apparent generalization. Original WLASL splits are not claimed signer-disjoint; do not silently reassign them.",
              f"- Only {len(manifest['samples'])} current clips: sparse training coverage creates severe overfitting and unstable evaluation risks, not measured model results.",
              "- Validation/test class gaps prevent complete ten-class evaluation; samples are not independent evidence of reliable real-world recognition.",
              "- Sign boundaries, source trimming and sign correctness remain unverified. Provisional_keep is not boundary verification; reviews were not changed.",
              "- Presence masks do not guarantee landmark visibility or accuracy. Coordinate z references differ between hands and pose.",
              "- Timestamp estimates assume constant reported FPS. Long-clip compression and short-clip padding yield different duration distributions.",
              "- No idle/background class, confidence threshold, or browser preprocessing parity has been finalized.",
              "- Integration fixtures include held-out examples: use them for engineering parity, not classifier tuning or training leakage.", "",
              "Generation preserves outputs unless --overwrite is explicitly supplied. That flag replaces only this dataset's eight files and this summary; unrelated files are kept.",
              "Verification uses --verify and writes nothing. Sources, reviews, raw videos, integration samples and the feature contract are never edited.", ""]
    return "\n".join(lines)


def write_outputs(root, arrays, records, overwrite=False):
    output = root / "dataset/model_ready/PROVISIONAL_v1"
    summary = root / "results/evaluation/gru_dataset_preparation_summary.md"
    if not overwrite and (output.exists() or summary.exists()):
        raise FileExistsError("Model-ready output or summary exists; use --verify, or explicitly --overwrite to regenerate.")
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".gru-prepare-") as temporary:
        staging = Path(temporary)
        files = {}
        for name in ARRAY_FILES:
            array = arrays[name[:-4]]
            with (staging / name).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            np.testing.assert_array_equal(np.load(staging / name, allow_pickle=False), array)
            files[name] = dict(shape=list(array.shape), dtype=str(array.dtype), sha256=integration.sha256(staging / name))
        manifest = make_manifest(records, files)
        for name, content in (("labels.json", labels_document()), ("manifest.json", manifest)):
            (staging / name).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
        (staging / summary.name).write_text(summary_text(manifest), encoding="utf-8")
        output.mkdir(exist_ok=True)
        for name in DATA_FILES:
            if overwrite:
                (staging / name).replace(output / name)
            else:
                (output / name).hardlink_to(staging / name)
        if overwrite:
            (staging / summary.name).replace(summary)
        else:
            summary.hardlink_to(staging / summary.name)
    return output


def verify_dataset(root, arrays, records):
    output = root / "dataset/model_ready/PROVISIONAL_v1"
    labels = json.loads((output / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if labels != labels_document():
        raise ValueError("Saved label mapping differs from the provisional integration mapping.")
    files = {}
    for split in SPLITS:
        x = np.load(output / f"X_{split}.npy", allow_pickle=False)
        y = np.load(output / f"y_{split}.npy", allow_pickle=False)
        count = sum(row["split"] == split for row in records)
        if x.shape != (count, 60, 149) or x.dtype != np.float32 or y.shape != (count,) or y.dtype != np.int64:
            raise ValueError(f"Invalid saved {split} array shape/dtype.")
        if np.any(y < 0) or np.any(y >= len(integration.LABELS)):
            raise ValueError("Label index outside the shared mapping.")
        for sequence in x:
            validate_features(sequence)
        for prefix, array in (("X", x), ("y", y)):
            name = f"{prefix}_{split}.npy"
            np.testing.assert_array_equal(array, arrays[f"{prefix}_{split}"])
            files[name] = dict(shape=list(array.shape), dtype=str(array.dtype), sha256=integration.sha256(output / name))
        print(f"PASS: X_{split}{x.shape} {x.dtype}; y_{split}{y.shape} {y.dtype}")
    if manifest != make_manifest(records, files):
        raise ValueError("Manifest/source identity, split, signer, checksums or provenance differ.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="Recompute and verify existing outputs without writing.")
    mode.add_argument("--overwrite", action="store_true", help="Explicitly replace this prepared dataset and summary only.")
    args = parser.parse_args()
    try:
        arrays, records = collect_dataset(PROJECT_ROOT)
        parity = check_integration_parity(PROJECT_ROOT, arrays, records)
        if not args.verify:
            write_outputs(PROJECT_ROOT, arrays, records, args.overwrite)
        verify_dataset(PROJECT_ROOT, arrays, records)
        print(f"PROVISIONAL: {len(records)} clips; exact parity with {parity} integration samples. No training performed.")
        return 0
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
