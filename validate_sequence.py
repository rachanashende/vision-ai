"""Validate a landmark sequence.

Usage:
    python validate_sequence.py                # checks a dummy zero sequence
    python validate_sequence.py path/to/seq.npy  # checks a real (60, 149) sequence
"""
import sys

import numpy as np

import config
from validation import check_sequence


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        sequence = np.load(argv[1])
        print("Loaded:", argv[1])
    else:
        print("No file given - using dummy data (real landmarks not validated yet).")
        sequence = np.zeros((config.FRAMES, config.FEATURES), dtype=np.float32)

    print("Shape:", sequence.shape)
    print("Data type:", sequence.dtype)
    if sequence.dtype.kind in "fc":
        print("Contains NaN:", bool(np.isnan(sequence).any()))
        print("Contains infinity:", bool(np.isinf(sequence).any()))

    report = check_sequence(sequence)
    if report.ok:
        print("Validation passed!")
        return 0
    for err in report.errors:
        print("FAILED:", err)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
