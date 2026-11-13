r"""Load the saved baseline and predict ONE training sample, never a webcam.

PowerShell: python -B scripts/smoke_test_gru_baseline.py
--sample-index selects a prepared training row (default 0). --report optionally
creates a new JSON file; existing reports are never overwritten.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

from train_gru_baseline import PROJECT_ROOT, load_inputs, output_paths, validate_probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        if args.report and args.report.exists():
            raise FileExistsError("Smoke-test report exists; choose a new path.")
        arrays, labels, manifest = load_inputs(PROJECT_ROOT)
        if not 0 <= args.sample_index < len(arrays["X_train"]):
            raise ValueError("Training sample index is out of range.")
        model_path, results = output_paths(PROJECT_ROOT)
        config = json.loads((results / "training_config.json").read_text(encoding="utf-8"))
        if config["class_to_index"] != labels["class_to_index"] or config["contract_version"] != labels["contract_version"]:
            raise ValueError("Model label/feature configuration differs from the prepared dataset.")
        model = tf.keras.models.load_model(model_path, compile=False)
        if tuple(model.input_shape[1:]) != (60, 149) or model.output_shape[-1] != 10:
            raise ValueError("Unexpected saved model input/output shape.")
        probabilities = model(arrays["X_train"][args.sample_index:args.sample_index + 1], training=False).numpy()
        validate_probabilities(probabilities, 1)
        selected = int(probabilities[0].argmax())
        record = next(row for row in manifest["samples"] if row["split"] == "train" and row["index_in_split"] == args.sample_index)
        result = dict(status="SMOKE TEST ONLY / NOT WEBCAM-VALIDATED", model=str(model_path),
                      split="train", video_id=record["video_id"], true_gloss=record["gloss"],
                      predicted_gloss=labels["index_to_class"][selected], predicted_index=selected,
                      confidence=float(probabilities[0, selected]), output_shape=list(probabilities.shape),
                      probabilities=probabilities[0].tolist())
        print(json.dumps(result, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
        return 0
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
