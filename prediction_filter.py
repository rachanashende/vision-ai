"""Prediction mapping and filtering (Member 3, Weeks 4-5).

Model output  ->  Prediction {label, confidence, timestamp}  ->  caption text.

The prediction message format is PROVISIONAL (see INTERFACES.md):
    {"label": "hello", "confidence": 0.93, "timestamp": 12.35}
`timestamp` is seconds on the same clock as the landmark frames.

Filtering rules (all thresholds live in config.py and are meant to be tuned):
  1. A prediction below CONFIDENCE_THRESHOLD, or one of the `ignore_labels`
     (e.g. a "no sign" class), or a malformed one, breaks the current streak.
  2. A caption is emitted once the same label has been predicted
     STABILITY_COUNT times in a row.
  3. A held sign is captioned once: the streak must break (label changes, low
     confidence, or no prediction) before the same label can be captioned again.
  4. Even then, the caption most recently shown is not shown again sooner than
     REPEAT_COOLDOWN_SECONDS after it was emitted. (A different caption in
     between resets this, so "a b a" is captioned as three captions.)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import config


class PredictionError(ValueError):
    """Raised for malformed model output or prediction messages."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    timestamp: float

    def to_message(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


def parse_prediction(raw) -> Prediction:
    """Validate a prediction message (dict) and return a Prediction."""
    if not isinstance(raw, dict):
        raise PredictionError("prediction must be an object")
    for key in ("label", "confidence", "timestamp"):
        if key not in raw:
            raise PredictionError(f"missing field: {key}")

    label = raw["label"]
    if not isinstance(label, str) or not label.strip():
        raise PredictionError("label must be a non-empty string")

    try:
        confidence = float(raw["confidence"])
        timestamp = float(raw["timestamp"])
    except (TypeError, ValueError):
        raise PredictionError("confidence and timestamp must be numbers") from None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise PredictionError("confidence must be a finite number in [0, 1]")
    if not math.isfinite(timestamp):
        raise PredictionError("timestamp must be finite")
    return Prediction(label.strip(), confidence, timestamp)


class LabelMap:
    """Maps model output indices to readable labels."""

    def __init__(self, labels: Sequence[str]) -> None:
        labels = list(labels)
        if not labels:
            raise ValueError("label list is empty")
        if any(not isinstance(l, str) or not l.strip() for l in labels):
            raise ValueError("labels must be non-empty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        self.labels = labels

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LabelMap":
        """Load a JSON list of labels, ordered by model output index."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("label file must contain a JSON list")
        return cls(data)

    def __len__(self) -> int:
        return len(self.labels)

    def label_for(self, index: int) -> str:
        if not 0 <= index < len(self.labels):
            raise PredictionError(f"class index {index} out of range")
        return self.labels[index]


def probs_to_prediction(probs, label_map: LabelMap, timestamp: float) -> Prediction:
    """Turn a model's probability vector into a Prediction (argmax).

    Expects softmax probabilities. Raw logits are rejected on purpose so a
    mismatch with the model owner is noticed early rather than silently
    producing nonsense confidences.
    """
    try:
        p = np.asarray(probs, dtype=np.float64)
    except (TypeError, ValueError):
        raise PredictionError("model output is not numeric") from None
    if p.ndim != 1 or p.shape[0] != len(label_map):
        raise PredictionError(
            f"expected {len(label_map)} class scores, got shape {p.shape}"
        )
    if not np.isfinite(p).all():
        raise PredictionError("model output contains NaN or infinity")
    if (p < 0).any() or (p > 1 + 1e-6).any() or abs(p.sum() - 1.0) > 1e-3:
        raise PredictionError("model output is not a probability distribution")

    i = int(np.argmax(p))
    return parse_prediction(
        {
            "label": label_map.label_for(i),
            "confidence": min(float(p[i]), 1.0),
            "timestamp": timestamp,
        }
    )


class PredictionFilter:
    """Turns a stream of predictions into stable, non-repeating captions."""

    def __init__(
        self,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        stability_count: int = config.STABILITY_COUNT,
        repeat_cooldown: float = config.REPEAT_COOLDOWN_SECONDS,
        ignore_labels: Iterable[str] = (),
    ) -> None:
        if stability_count < 1:
            raise ValueError("stability_count must be >= 1")
        self.confidence_threshold = confidence_threshold
        self.stability_count = stability_count
        self.repeat_cooldown = repeat_cooldown
        self.ignore_labels = frozenset(ignore_labels)
        self.reset()

    def reset(self) -> None:
        self._candidate: str | None = None
        self._count = 0
        self._emitted_this_streak = False
        self._last_label: str | None = None
        self._last_emit_ts: float | None = None
        self.invalid_count = 0  # malformed inputs seen

    def _break_streak(self) -> None:
        self._candidate = None
        self._count = 0
        self._emitted_this_streak = False

    def update(self, prediction) -> str | None:
        """Feed one prediction (Prediction, dict, or None).

        Returns the caption text to display, or None if nothing new should be
        shown. Malformed input never raises; it breaks the streak and is counted.
        """
        if prediction is None:
            self._break_streak()
            return None
        if not isinstance(prediction, Prediction):
            try:
                prediction = parse_prediction(prediction)
            except PredictionError:
                self.invalid_count += 1
                self._break_streak()
                return None

        if (
            prediction.confidence < self.confidence_threshold
            or prediction.label in self.ignore_labels
        ):
            self._break_streak()
            return None

        if prediction.label == self._candidate:
            self._count += 1
        else:
            self._candidate = prediction.label
            self._count = 1
            self._emitted_this_streak = False

        if self._count < self.stability_count or self._emitted_this_streak:
            return None

        self._emitted_this_streak = True
        if (
            prediction.label == self._last_label
            and self._last_emit_ts is not None
            and prediction.timestamp - self._last_emit_ts < self.repeat_cooldown
        ):
            return None

        self._last_label = prediction.label
        self._last_emit_ts = prediction.timestamp
        return prediction.label
