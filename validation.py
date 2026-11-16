"""Landmark validation (Member 3, Weeks 2-3).

Two levels:
  * sanitize_frame()  - one frame of landmarks coming from the extractor.
  * check_sequence()  - the final (FRAMES, FEATURES) array handed to the model.

Rules marked PROVISIONAL in config.py still need confirmation from Member 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class FrameResult:
    ok: bool
    frame: np.ndarray | None = None  # sanitized float32 copy when ok
    bad_values: int = 0              # NaN/inf values that were replaced
    reason: str = ""                 # why the frame was rejected


def sanitize_frame(
    frame,
    features: int = config.FEATURES,
    max_bad_ratio: float = config.MAX_BAD_VALUE_RATIO,
    fill_value: float = config.FILL_VALUE,
) -> FrameResult:
    """Validate one frame and replace stray NaN/inf values.

    A frame is rejected if it is not a numeric 1-D array of `features` values,
    or if more than `max_bad_ratio` of its values are NaN/inf (for example a
    frame where the extractor found nothing). Otherwise the bad values are
    replaced by `fill_value`.
    """
    try:
        arr = np.asarray(frame, dtype=np.float32)
    except (TypeError, ValueError):
        return FrameResult(False, reason="frame is not numeric")

    if arr.ndim != 1:
        return FrameResult(False, reason=f"expected a 1-D frame, got {arr.ndim}-D")
    if arr.shape[0] != features:
        return FrameResult(
            False, reason=f"expected {features} features, got {arr.shape[0]}"
        )

    bad = ~np.isfinite(arr)
    n_bad = int(bad.sum())
    if n_bad / features > max_bad_ratio:
        return FrameResult(
            False,
            bad_values=n_bad,
            reason=f"{n_bad}/{features} values are NaN/inf (limit {max_bad_ratio:.0%})",
        )

    clean = arr.copy()
    if n_bad:
        clean[bad] = fill_value
    return FrameResult(True, frame=clean, bad_values=n_bad)


@dataclass(frozen=True)
class SequenceReport:
    ok: bool
    errors: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok


def check_sequence(
    sequence,
    frames: int = config.FRAMES,
    features: int = config.FEATURES,
) -> SequenceReport:
    """Check that a model-ready sequence has the agreed shape, dtype and values."""
    if not isinstance(sequence, np.ndarray):
        return SequenceReport(False, ("sequence is not a numpy array",))

    errors: list[str] = []
    if sequence.ndim != 2:
        errors.append(f"expected 2-D array, got {sequence.ndim}-D")
    elif sequence.shape != (frames, features):
        errors.append(f"expected shape {(frames, features)}, got {sequence.shape}")
    if sequence.dtype != np.float32:
        errors.append(f"expected dtype float32, got {sequence.dtype}")
    if sequence.dtype.kind in "fc" and not np.isfinite(sequence).all():
        errors.append("sequence contains NaN or infinity")
    return SequenceReport(not errors, tuple(errors))


def assert_valid_sequence(sequence, **kwargs) -> None:
    report = check_sequence(sequence, **kwargs)
    if not report.ok:
        raise ValueError("invalid sequence: " + "; ".join(report.errors))
