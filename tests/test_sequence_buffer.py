import numpy as np
import pytest

import config
from sequence_buffer import RollingBuffer
from validation import check_sequence

F = config.FEATURES


def frame(value):
    """A frame whose every feature equals `value`, so we can see which was picked."""
    return np.full(F, value, dtype=np.float32)


def fill(buf, n, fps=15.0, start=0.0):
    for i in range(n):
        assert buf.push(frame(i), start + i / fps)


def test_not_ready_when_empty_or_too_short():
    buf = RollingBuffer()
    assert buf.get_sequence() is None
    buf.push(frame(0), 0.0)
    assert buf.get_sequence() is None  # one frame
    fill(buf := RollingBuffer(), 10)  # 10 frames @15fps = 0.6s < 2s
    assert not buf.is_ready() and buf.get_sequence() is None


def test_full_window_gives_valid_sequence():
    buf = RollingBuffer()
    fill(buf, 61)  # 4.0 s at 15 fps
    seq = buf.get_sequence()
    assert seq is not None and check_sequence(seq).ok
    assert seq.shape == (config.FRAMES, F)


def test_old_frames_are_evicted():
    buf = RollingBuffer()
    fill(buf, 200)  # ~13 s of data
    assert buf.span <= config.WINDOW_SECONDS + 1e-9
    assert len(buf) <= 61 + 1


def test_oversampling_slow_camera_repeats_frames():
    buf = RollingBuffer()
    fill(buf, 21, fps=5.0)  # 4 s at 5 fps -> only 21 distinct frames
    seq = buf.get_sequence()
    assert seq.shape == (60, F)
    assert len(np.unique(seq[:, 0])) <= 21


def test_downsampling_fast_camera_keeps_order_and_endpoints():
    buf = RollingBuffer()
    fill(buf, 121, fps=30.0)  # 4 s at 30 fps
    seq = buf.get_sequence()
    col = seq[:, 0]
    assert np.all(np.diff(col) >= 0)  # time order preserved
    assert col[0] == 0 and col[-1] == 120  # first and latest frame included


def test_partially_filled_window_reuses_first_frame_at_start():
    buf = RollingBuffer()
    fill(buf, 31, fps=15.0)  # 2.0 s = exactly min_fill_ratio
    seq = buf.get_sequence()
    assert seq is not None
    assert seq[0, 0] == 0 and seq[-1, 0] == 30


def test_tie_goes_to_earlier_frame():
    buf = RollingBuffer(window_seconds=1.0, frames=3, min_fill_ratio=0.0)
    buf.push(frame(0), 0.0)
    buf.push(frame(1), 1.0)
    # targets 0, 0.5, 1.0 -> 0.5 is equidistant -> earlier frame (0)
    assert buf.get_sequence()[:, 0].tolist() == [0, 0, 1]


def test_invalid_frames_are_rejected_and_counted():
    buf = RollingBuffer()
    assert not buf.push(np.zeros(F - 1), 0.0)
    assert not buf.push(np.full(F, np.nan), 0.1)
    assert not buf.push(None, 0.2)
    assert buf.rejected == 3 and len(buf) == 0


def test_bad_timestamps_rejected():
    buf = RollingBuffer()
    assert buf.push(frame(0), 1.0)
    assert not buf.push(frame(1), 0.5)  # goes backwards
    assert not buf.push(frame(1), float("nan"))
    assert not buf.push(frame(1), None)
    assert buf.push(frame(2), 1.0)  # equal timestamp is allowed
    assert buf.rejected == 3


def test_stray_nans_are_filled():
    buf = RollingBuffer(min_fill_ratio=0.0)
    f = frame(1)
    f[5] = np.nan
    assert buf.push(f, 0.0) and buf.push(frame(1), 1.0)
    assert np.isfinite(buf.get_sequence()).all()


def test_gap_in_stream_still_produces_sequence():
    buf = RollingBuffer()
    fill(buf, 20, fps=15.0)  # 0 .. 1.27 s
    fill(buf, 20, fps=15.0, start=3.0)  # then a gap until 3.0 .. 4.27 s
    seq = buf.get_sequence()
    assert seq is not None and check_sequence(seq).ok


def test_clear_resets():
    buf = RollingBuffer()
    fill(buf, 61)
    buf.clear()
    assert len(buf) == 0 and buf.get_sequence() is None


def test_constructor_validation():
    with pytest.raises(ValueError):
        RollingBuffer(frames=1)
    with pytest.raises(ValueError):
        RollingBuffer(window_seconds=0)
