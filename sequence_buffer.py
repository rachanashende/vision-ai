"""Rolling landmark buffer and resampling (Member 3, Week 3).

Frames arrive at an irregular rate (the camera/MediaPipe may run at ~15 FPS or
slower). The buffer keeps the last WINDOW_SECONDS of valid frames and, on
request, resamples them to exactly FRAMES frames for the model.

RESAMPLING STRATEGY (PROVISIONAL - agree with Member 1, who must train the
model on data prepared the same way):
  * Target times are FRAMES evenly spaced points from (latest - window) to
    latest, inclusive.
  * For each target time the NEAREST buffered frame is used (ties go to the
    earlier frame). No interpolation, so landmark values are never blended.
  * Targets earlier than the first buffered frame reuse the first frame.

The JavaScript port (js/sequence_buffer.js) must follow exactly the same rules;
tools/make_parity_fixtures.py + js/parity.test.js check that.
"""
from __future__ import annotations

from collections import deque

import numpy as np

import config
from validation import sanitize_frame


def resample_nearest(
    timestamps: np.ndarray, frames: np.ndarray, t_end: float, window: float, n_out: int
) -> np.ndarray:
    """Pick `n_out` frames by nearest timestamp over [t_end - window, t_end]."""
    n = len(timestamps)
    step = window / (n_out - 1)
    targets = (t_end - window) + np.arange(n_out, dtype=np.float64) * step
    targets[-1] = t_end

    if n == 1:
        idx = np.zeros(n_out, dtype=np.int64)
    else:
        right = np.clip(np.searchsorted(timestamps, targets, side="left"), 1, n - 1)
        left = right - 1
        left_closer = (targets - timestamps[left]) <= (timestamps[right] - targets)
        idx = np.where(left_closer, left, right)
    return frames[idx]


class RollingBuffer:
    def __init__(
        self,
        window_seconds: float = config.WINDOW_SECONDS,
        frames: int = config.FRAMES,
        features: int = config.FEATURES,
        min_fill_ratio: float = config.MIN_FILL_RATIO,
        max_bad_ratio: float = config.MAX_BAD_VALUE_RATIO,
        fill_value: float = config.FILL_VALUE,
    ) -> None:
        if frames < 2:
            raise ValueError("frames must be at least 2")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window = float(window_seconds)
        self.frames = frames
        self.features = features
        self.min_fill_ratio = min_fill_ratio
        self.max_bad_ratio = max_bad_ratio
        self.fill_value = fill_value
        self._items: deque[tuple[float, np.ndarray]] = deque()
        self.rejected = 0  # frames dropped as invalid or out of order

    # -- input ----------------------------------------------------------------
    def push(self, frame, timestamp: float) -> bool:
        """Add one frame. Returns False (and counts it) if it was rejected."""
        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            self.rejected += 1
            return False
        if not np.isfinite(ts) or (self._items and ts < self._items[-1][0]):
            self.rejected += 1
            return False

        result = sanitize_frame(
            frame, self.features, self.max_bad_ratio, self.fill_value
        )
        if not result.ok:
            self.rejected += 1
            return False

        self._items.append((ts, result.frame))
        cutoff = ts - self.window
        while self._items and self._items[0][0] < cutoff:
            self._items.popleft()
        return True

    def clear(self) -> None:
        self._items.clear()

    # -- state ----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    @property
    def span(self) -> float:
        """Seconds covered by the buffered frames."""
        if len(self._items) < 2:
            return 0.0
        return self._items[-1][0] - self._items[0][0]

    def is_ready(self) -> bool:
        return len(self._items) >= 2 and self.span >= self.min_fill_ratio * self.window

    # -- output ---------------------------------------------------------------
    def get_sequence(self) -> np.ndarray | None:
        """Return a (frames, features) float32 array, or None if not ready."""
        if not self.is_ready():
            return None
        ts = np.fromiter((t for t, _ in self._items), dtype=np.float64)
        data = np.stack([f for _, f in self._items]).astype(np.float32, copy=False)
        out = resample_nearest(ts, data, float(ts[-1]), self.window, self.frames)
        return np.ascontiguousarray(out, dtype=np.float32)
