import numpy as np
import pytest

import config
from validation import assert_valid_sequence, check_sequence, sanitize_frame

F = config.FEATURES


def good_frame():
    return np.linspace(0, 1, F, dtype=np.float32)


def test_valid_frame_passes_and_is_float32_copy():
    src = good_frame()
    r = sanitize_frame(src)
    assert r.ok and r.bad_values == 0
    assert r.frame.dtype == np.float32 and r.frame.shape == (F,)
    r.frame[0] = 99
    assert src[0] == 0  # original untouched


def test_list_input_is_accepted():
    assert sanitize_frame(good_frame().tolist()).ok


@pytest.mark.parametrize("bad", [None, "abc", [[1, 2], [3]], object()])
def test_non_numeric_or_ragged_rejected(bad):
    assert not sanitize_frame(bad).ok


def test_wrong_length_and_ndim_rejected():
    assert not sanitize_frame(np.zeros(F - 1)).ok
    assert not sanitize_frame(np.zeros(F + 1)).ok
    assert not sanitize_frame(np.zeros((F, 1))).ok
    assert not sanitize_frame(np.float32(1.0)).ok


def test_few_nans_are_replaced():
    f = good_frame()
    f[3] = np.nan
    f[10] = np.inf
    f[11] = -np.inf
    r = sanitize_frame(f)
    assert r.ok and r.bad_values == 3
    assert np.isfinite(r.frame).all()
    assert r.frame[3] == config.FILL_VALUE


def test_mostly_nan_frame_rejected():
    f = np.full(F, np.nan, dtype=np.float32)
    r = sanitize_frame(f)
    assert not r.ok and "NaN" in r.reason


def test_bad_ratio_boundary():
    limit = int(F * config.MAX_BAD_VALUE_RATIO)
    f = good_frame()
    f[:limit] = np.nan
    assert sanitize_frame(f).ok
    f[: limit + 1] = np.nan
    assert not sanitize_frame(f).ok


def test_check_sequence_ok():
    seq = np.zeros((config.FRAMES, F), dtype=np.float32)
    assert check_sequence(seq).ok
    assert_valid_sequence(seq)


def test_check_sequence_reports_all_problems():
    seq = np.zeros((10, F), dtype=np.float64)
    seq[0, 0] = np.nan
    rep = check_sequence(seq)
    assert not rep.ok
    assert len(rep.errors) == 3  # shape, dtype, non-finite
    with pytest.raises(ValueError):
        assert_valid_sequence(seq)


def test_check_sequence_rejects_non_array():
    assert not check_sequence([[0.0] * F] * config.FRAMES).ok
    assert not check_sequence(np.zeros(F, dtype=np.float32)).ok
