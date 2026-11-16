import json

import numpy as np
import pytest

from prediction_filter import (
    LabelMap,
    Prediction,
    PredictionError,
    PredictionFilter,
    parse_prediction,
    probs_to_prediction,
)


def P(label, conf=0.9, t=0.0):
    return Prediction(label, conf, t)


def run(flt, preds):
    return [flt.update(p) for p in preds]


# ---- parse_prediction -------------------------------------------------------
def test_parse_valid():
    p = parse_prediction({"label": " hello ", "confidence": "0.8", "timestamp": 1})
    assert p == Prediction("hello", 0.8, 1.0)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"label": "a", "confidence": 0.5},
        {"label": "", "confidence": 0.5, "timestamp": 0},
        {"label": 3, "confidence": 0.5, "timestamp": 0},
        {"label": "a", "confidence": 1.5, "timestamp": 0},
        {"label": "a", "confidence": -0.1, "timestamp": 0},
        {"label": "a", "confidence": float("nan"), "timestamp": 0},
        {"label": "a", "confidence": "high", "timestamp": 0},
        {"label": "a", "confidence": 0.5, "timestamp": float("inf")},
    ],
)
def test_parse_rejects_malformed(raw):
    with pytest.raises(PredictionError):
        parse_prediction(raw)


# ---- LabelMap / probs_to_prediction -----------------------------------------
def test_label_map_validation():
    with pytest.raises(ValueError):
        LabelMap([])
    with pytest.raises(ValueError):
        LabelMap(["a", "a"])
    with pytest.raises(ValueError):
        LabelMap(["a", ""])
    with pytest.raises(PredictionError):
        LabelMap(["a"]).label_for(1)


def test_label_map_from_json(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps(["hello", "thanks"]))
    assert LabelMap.from_json_file(f).label_for(1) == "thanks"
    f.write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError):
        LabelMap.from_json_file(f)


def test_probs_to_prediction():
    lm = LabelMap(["hello", "thanks", "please"])
    p = probs_to_prediction(np.array([0.1, 0.7, 0.2], dtype=np.float32), lm, 4.2)
    assert p.label == "thanks" and p.confidence == pytest.approx(0.7) and p.timestamp == 4.2


@pytest.mark.parametrize(
    "probs",
    [[0.5, 0.5], [0.1, 0.2, np.nan], [2.0, -1.0, 0.0], [3.0, 1.0, -2.0], "abc", [[0.3, 0.3, 0.4]]],
)
def test_probs_rejected(probs):
    with pytest.raises(PredictionError):
        probs_to_prediction(probs, LabelMap(["a", "b", "c"]), 0.0)


# ---- PredictionFilter -------------------------------------------------------
def test_needs_three_consecutive_matches():
    f = PredictionFilter()
    out = run(f, [P("hello", t=0), P("hello", t=0.35), P("hello", t=0.7)])
    assert out == [None, None, "hello"]


def test_label_change_resets_streak():
    f = PredictionFilter()
    out = run(f, [P("a"), P("a"), P("b"), P("b"), P("a")])
    assert out == [None] * 5


def test_low_confidence_breaks_streak():
    f = PredictionFilter()
    out = run(f, [P("a"), P("a"), P("a", 0.5), P("a")])
    assert out == [None] * 4


def test_threshold_boundary_is_inclusive():
    f = PredictionFilter(confidence_threshold=0.7)
    assert run(f, [P("a", 0.7)] * 3)[-1] == "a"


def test_held_sign_is_captioned_once():
    f = PredictionFilter()
    out = run(f, [P("hello", t=i * 0.35) for i in range(20)])
    assert out.count("hello") == 1


def test_repeat_after_gap_needs_cooldown():
    f = PredictionFilter(repeat_cooldown=2.0)
    first = [P("hello", t=0.0), P("hello", t=0.3), P("hello", t=0.6)]
    gap = [P("hello", 0.1, t=0.9)]  # low confidence breaks the streak
    soon = [P("hello", t=1.2), P("hello", t=1.5), P("hello", t=1.8)]  # <2s after 0.6
    later = [P("hello", t=3.0), P("hello", t=3.3), P("hello", t=3.6)]  # >2s after 0.6
    out = run(f, first + gap + soon + gap + later)
    assert out.count("hello") == 2
    assert out[2] == "hello" and out[-1] == "hello"


def test_different_labels_back_to_back():
    f = PredictionFilter()
    out = run(f, [P("a")] * 3 + [P("b")] * 3)
    assert out == [None, None, "a", None, None, "b"]


def test_cooldown_compares_against_most_recent_caption_only():
    # The cooldown suppresses repeating the caption that was just shown. A
    # different caption in between makes the next "a" a new caption.
    f = PredictionFilter(repeat_cooldown=10.0)
    out = run(f, [P("a", t=0)] * 3 + [P("b", t=0.1)] * 3 + [P("a", t=0.2)] * 3)
    assert [x for x in out if x] == ["a", "b", "a"]


def test_none_and_malformed_inputs_break_streak_without_raising():
    f = PredictionFilter()
    out = run(f, [P("a"), P("a"), None, P("a"), P("a"), {"label": "a"}, P("a")])
    assert out == [None] * 7
    assert f.invalid_count == 1


def test_dict_input_is_accepted():
    f = PredictionFilter(stability_count=1)
    assert f.update({"label": "yes", "confidence": 0.95, "timestamp": 0}) == "yes"


def test_ignore_labels():
    f = PredictionFilter(ignore_labels={"<none>"})
    assert run(f, [P("<none>", 0.99)] * 5) == [None] * 5


def test_noisy_stream_produces_no_false_captions():
    rng = np.random.default_rng(0)
    labels = ["a", "b", "c", "d"]
    f = PredictionFilter()
    out = [
        f.update(P(labels[rng.integers(4)], float(rng.uniform(0.3, 0.69)), i * 0.35))
        for i in range(500)
    ]
    assert not any(out)  # nothing reaches the threshold


def test_reset_clears_state():
    f = PredictionFilter()
    run(f, [P("a")] * 3)
    f.reset()
    assert run(f, [P("a")] * 3)[-1] == "a"


def test_invalid_config():
    with pytest.raises(ValueError):
        PredictionFilter(stability_count=0)
