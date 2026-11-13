r"""Train an explicitly requested EXPERIMENTAL GRU, never by default.

PowerShell:
    python -B scripts/train_gru_baseline.py
    python -B scripts/train_gru_baseline.py --train --epochs 30 --batch-size 4 --seed 42

Default validates inputs only. --overwrite explicitly replaces this experiment's
generated model/reports; source data and the provisional contract are read-only.
The user-authorized experimental baseline does not finalize browser preprocessing.
Test probabilities are computed once after fitting; no test-driven tuning occurs.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import sys
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

import prepare_gru_dataset as preparation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"train": 17, "val": 5, "test": 6}
MODEL_NAME = "gru_baseline_EXPERIMENTAL.keras"
RESULT_FILES = ["training_config.json", "history.json", "history.csv", "predictions.csv",
                "confusion_matrix_test.csv", "confusion_matrix_test.png", "evaluation_summary.json",
                "evaluation_summary.md", "model_summary.txt"]


def load_inputs(root):
    folder = root / "dataset/model_ready/PROVISIONAL_v1"
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if labels != preparation.labels_document() or manifest["labels"] != labels:
        raise ValueError("Class mapping differs from the single provisional dataset mapping.")
    arrays = {}
    for split, count in COUNTS.items():
        x = np.load(folder / f"X_{split}.npy", allow_pickle=False)
        y = np.load(folder / f"y_{split}.npy", allow_pickle=False)
        validate_split(x, y, count)
        for prefix, array in (("X", x), ("y", y)):
            name = f"{prefix}_{split}.npy"
            if preparation.integration.sha256(folder / name) != manifest["files"][name]["sha256"]:
                raise ValueError(f"Prepared data checksum mismatch: {name}")
            arrays[f"{prefix}_{split}"] = array
    records = manifest["samples"]
    if len(records) != sum(COUNTS.values()) or len({row["video_id"] for row in records}) != len(records):
        raise ValueError("Dataset manifest has incorrect sample count or duplicate video IDs.")
    for split, count in COUNTS.items():
        selected = sorted([row for row in records if row["split"] == split], key=lambda row: row["index_in_split"])
        if [row["index_in_split"] for row in selected] != list(range(count)):
            raise ValueError("Manifest split row indices are invalid.")
        for row in selected:
            if (not row["signer_id"] or row["label_index"] != labels["class_to_index"][row["gloss"]]
                or arrays[f"y_{split}"][row["index_in_split"]] != row["label_index"]):
                raise ValueError("Manifest labels/signer metadata disagree with prepared targets.")
    return arrays, labels, manifest


def validate_split(x, y, count):
    if x.shape != (count, 60, 149) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("Expected finite float32 X with the prepared (N,60,149) shape.")
    if y.shape != (count,) or y.dtype != np.int64 or np.any(y < 0) or np.any(y >= 10):
        raise ValueError("Expected int64 targets in the shared 0-9 class mapping.")
    for sequence in x:
        preparation.validate_features(sequence)


def validate_probabilities(probabilities, count):
    if probabilities.shape != (count, 10) or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite (N,10) predictions.")
    if np.any(probabilities < 0) or np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid softmax probability values or row sums.")


def set_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    notes = []
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError) as error:
        notes.append(f"Could not enable deterministic ops: {error}")
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError as error:
        notes.append(f"Thread configuration already initialized: {error}")
    return notes


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(60, 149)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.GRU(16),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ], name="experimental_asl_gru")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def output_paths(root):
    return root / "models/trained/PROVISIONAL_v1" / MODEL_NAME, root / "results/evaluation/gru_baseline"


def check_outputs(root, overwrite):
    model, results = output_paths(root)
    if not overwrite and (model.parent.exists() or results.exists()):
        raise FileExistsError("Model/results directory exists; preserve it or explicitly supply --overwrite.")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_baseline(root, arrays, labels, manifest, epochs=30, batch_size=4, seed=42, patience=5, overwrite=False):
    check_outputs(root, overwrite)
    notes = set_reproducibility(seed)
    model = build_model()
    config = dict(status="EXPERIMENTAL / PROVISIONAL", created_utc=datetime.now(timezone.utc).isoformat(),
                  tensorflow_version=tf.__version__, keras_version=tf.keras.__version__, python_version=platform.python_version(),
                  platform=platform.platform(), devices=[str(device) for device in tf.config.list_physical_devices()],
                  seed=seed, max_epochs=epochs, batch_size=batch_size, patience=patience, learning_rate=0.001,
                  shuffle=True, early_stopping_monitor="val_loss", min_delta=0.0001, restore_best_weights=True,
                  architecture="Masking(0) -> GRU(16) -> Dropout(0.2) -> Dense(10, softmax)",
                  parameters=model.count_params(), input_shape=[60, 149], class_to_index=labels["class_to_index"],
                  index_to_class=labels["index_to_class"], contract_version=labels["contract_version"],
                  reproducibility_notes=notes + ["Bitwise reproducibility is not guaranteed across TF versions, devices, or numerical kernels."],
                  dataset_files=manifest["files"], labels_sha256=preparation.integration.sha256(root / "dataset/model_ready/PROVISIONAL_v1/labels.json"),
                  test_selection_policy="Test split excluded from fitting/stopping; one post-training probability pass.")
    stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, min_delta=0.0001,
                                                restore_best_weights=True)
    history = model.fit(arrays["X_train"], arrays["y_train"], validation_data=(arrays["X_val"], arrays["y_val"]),
                        epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[stopping], verbose=2)
    saved_history = {key: [float(value) for value in values] for key, values in history.history.items()}
    # Match EarlyStopping's min_delta rule to identify the restored checkpoint.
    best_loss, best_epoch = float("inf"), 0
    for index, value in enumerate(saved_history["val_loss"]):
        if value < best_loss - 0.0001:
            best_loss, best_epoch = value, index + 1
    metrics, predictions = {}, []
    probabilities_by_split = {}
    for split in COUNTS:
        # Exactly one test forward pass here; metrics and confusion reuse its values.
        probabilities = model(arrays[f"X_{split}"], training=False).numpy()
        validate_probabilities(probabilities, COUNTS[split])
        probabilities_by_split[split] = probabilities
        truth = arrays[f"y_{split}"]
        chosen = probabilities.argmax(axis=1)
        metrics[split] = dict(status="EXPLORATORY", clips=len(truth), accuracy=float(np.mean(chosen == truth)),
                              crossentropy=float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-7, 1)).mean()),
                              correct=int(np.sum(chosen == truth)))
        records = sorted([row for row in manifest["samples"] if row["split"] == split], key=lambda row: row["index_in_split"])
        for index, row in enumerate(records):
            record = {key: row[key] for key in ("gloss", "video_id", "split", "signer_id", "index_in_split")}
            record.update(true_index=int(truth[index]), predicted_index=int(chosen[index]),
                          predicted_gloss=labels["index_to_class"][chosen[index]], confidence=float(probabilities[index].max()))
            record.update({f"probability_{label}": float(probabilities[index, label]) for label in range(10)})
            predictions.append(record)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (arrays["y_test"], probabilities_by_split["test"].argmax(axis=1)), 1)
    evaluation = dict(status="EXPLORATORY / NOT WEBCAM-VALIDATED", metrics=metrics,
                      epochs_run=len(saved_history["loss"]), restored_epoch=best_epoch,
                      class_counts=manifest["per_class_counts"], signer_overlap=manifest["signer_overlap"],
                      missing_classes={split: [label for label in labels["index_to_class"] if manifest["per_class_counts"][label][split] == 0]
                                       for split in ("val", "test")})
    model_path, results = output_paths(root)
    model_path.parent.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=model_path.parent.parent, prefix=".gru-training-") as temporary:
        staging = Path(temporary)
        model.save(staging / MODEL_NAME)
        reloaded = tf.keras.models.load_model(staging / MODEL_NAME, compile=False)
        reload_prediction = reloaded(arrays["X_train"][:1], training=False).numpy()
        validate_probabilities(reload_prediction, 1)
        np.testing.assert_allclose(reload_prediction, probabilities_by_split["train"][:1], atol=1e-6)
        evaluation["reload_check"] = "passed on one training sample; no additional test evaluation"
        for filename, content in (("training_config.json", config), ("history.json", saved_history),
                                  ("evaluation_summary.json", evaluation)):
            (staging / filename).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
        write_csv(staging / "history.csv", [dict(epoch=index + 1, **{key: values[index] for key, values in saved_history.items()})
                                            for index in range(evaluation["epochs_run"])], ["epoch"] + list(saved_history))
        write_csv(staging / "predictions.csv", predictions, list(predictions[0]))
        write_csv(staging / "confusion_matrix_test.csv", [dict(actual_gloss=label, **{name: int(confusion[index, j])
                                                                                 for j, name in enumerate(labels["index_to_class"])})
                                                          for index, label in enumerate(labels["index_to_class"])],
                  ["actual_gloss"] + labels["index_to_class"])
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(9, 8))
        plot = axis.imshow(confusion, cmap="Blues", vmin=0)
        axis.set_xticks(range(10), labels["index_to_class"], rotation=45, ha="right")
        axis.set_yticks(range(10), labels["index_to_class"])
        axis.set(xlabel="Predicted class", ylabel="Actual class", title="EXPLORATORY test confusion matrix (6 clips)")
        for i in range(10):
            for j in range(10):
                axis.text(j, i, str(confusion[i, j]), ha="center", va="center")
        figure.colorbar(plot, ax=axis)
        figure.tight_layout()
        figure.savefig(staging / "confusion_matrix_test.png", dpi=150)
        plt.close(figure)
        summary = []
        model.summary(print_fn=lambda text, **kwargs: summary.append(text))
        (staging / "model_summary.txt").write_text("\n".join(summary), encoding="utf-8")
        lines = ["# EXPERIMENTAL PROVISIONAL GRU baseline", "", f"TensorFlow {tf.__version__}; seed {seed}; parameters {model.count_params()}.",
                 f"Epochs run: {evaluation['epochs_run']}; restored validation-selected epoch: {best_epoch}.",
                 "Test data was excluded from fitting, early stopping, and hyperparameter selection.",
                 "Test metrics/confusion/predictions reuse one post-training forward pass. All metrics are exploratory.", "",
                 "| Split | Clips | Correct | Accuracy | Cross-entropy |", "| --- | ---: | ---: | ---: | ---: |"]
        for split, values in metrics.items():
            lines.append(f"| {split} | {values['clips']} | {values['correct']} | {values['accuracy']:.6f} | {values['crossentropy']:.6f} |")
        lines += ["", "| Gloss | Train | Val | Test |", "| --- | ---: | ---: | ---: |"]
        for label, counts in manifest["per_class_counts"].items():
            lines.append(f"| {label} | {counts['train']} | {counts['val']} | {counts['test']} |")
        lines += ["", "## Limitations", "", "Only 17 training clips (1-3/class), five validation clips, and six test clips.",
                  "Validation accuracy changes by 20 percentage points per example; early stopping is a noisy heuristic."]
        for split, missing in evaluation["missing_classes"].items():
            lines.append(f"Missing {split} classes: {', '.join(missing)}.")
        lines += [f"Observed signer overlaps: {manifest['signer_overlap']}.",
                  "Shared signers can inflate apparent generalization. No original split or signer assignments were changed.",
                  "Severe overfitting risk; these metrics cannot establish robust ten-class recognition.",
                  "Boundaries, trimming and sign correctness remain unverified. Input schema, live cadence and browser parity remain provisional.",
                  "Masking ignores all-zero padding only; real failed-inference frames retain their real-frame flag and are not silently dropped.",
                  "Softmax confidence is uncalibrated; there is no idle/background class or webcam demonstration yet.",
                  "No webcam application, TensorFlow.js export, dependency installation, or source-data edits occurred.",
                  "Reproducibility is limited to seeded deterministic operations where supported; cross-platform bitwise equality is not guaranteed.", ""]
        (staging / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        for filename, destination in [(MODEL_NAME, model_path)] + [(name, results / name) for name in RESULT_FILES]:
            if overwrite:
                (staging / filename).replace(destination)
            else:
                destination.hardlink_to(staging / filename)
    print(json.dumps(evaluation, indent=2))
    print(f"Model: {model_path}\nReports: {results}")
    return evaluation


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Explicitly train the real experimental baseline.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace generated model/reports only.")
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=positive_int, default=5)
    args = parser.parse_args()
    try:
        arrays, labels, manifest = load_inputs(PROJECT_ROOT)
        for split in COUNTS:
            print(f"Validated {split}: {arrays[f'X_{split}'].shape}, {arrays[f'y_{split}'].shape}")
        if args.train:
            train_baseline(PROJECT_ROOT, arrays, labels, manifest, args.epochs, args.batch_size, args.seed, args.patience, args.overwrite)
        else:
            print("Inspection only. No model trained or outputs written; --train is required.")
        return 0
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
