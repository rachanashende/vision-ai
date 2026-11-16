"""Generate js/fixtures/parity.json from the Python implementation.

The JavaScript tests (js/parity.test.js) replay the same inputs and require the
same outputs, which is how we check Python and JS preprocessing stay identical.
Re-run this whenever the buffer or filter logic changes:

    python tools/make_parity_fixtures.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from prediction_filter import PredictionFilter  # noqa: E402
from sequence_buffer import RollingBuffer  # noqa: E402


def num(x):
    """JSON-safe number (NaN -> None; the JS test turns null back into NaN)."""
    return None if isinstance(x, float) and np.isnan(x) else x


def buffer_case(name, rng, n_pushes, frames, features, window, fps_range, nan_rate,
                invalid_rate, **kwargs):
    buf = RollingBuffer(window_seconds=window, frames=frames, features=features, **kwargs)
    pushes, accepted = [], []
    t = 0.0
    for _ in range(n_pushes):
        t += float(rng.uniform(1 / fps_range[1], 1 / fps_range[0]))
        ts = t
        roll = rng.random()
        if roll < invalid_rate / 2:
            frame = [float(v) for v in rng.integers(0, 1000, features - 1)]  # wrong length
        else:
            frame = [float(v) for v in rng.integers(0, 10000, features)]
            for j in range(features):
                if rng.random() < nan_rate:
                    frame[j] = float("nan")
            if roll > 1 - invalid_rate / 2:
                ts = t - 1.0  # goes backwards in time
        ok = buf.push(frame, ts)
        pushes.append({"t": ts, "frame": [num(v) for v in frame]})
        accepted.append(ok)

    seq = buf.get_sequence()
    return {
        "name": name,
        "params": {"windowSeconds": window, "frames": frames, "features": features, **{
            {"min_fill_ratio": "minFillRatio", "max_bad_ratio": "maxBadRatio"}[k]: v
            for k, v in kwargs.items()
        }},
        "pushes": pushes,
        "accepted": accepted,
        "rejected": buf.rejected,
        "length": len(buf),
        "ready": buf.is_ready(),
        "sequence": None if seq is None else [float(v) for v in seq.reshape(-1)],
    }


def filter_case(name, rng, n, **kwargs):
    js_kwargs = {
        "confidence_threshold": "confidenceThreshold",
        "stability_count": "stabilityCount",
        "repeat_cooldown": "repeatCooldown",
        "ignore_labels": "ignoreLabels",
    }
    flt = PredictionFilter(**kwargs)
    labels = ["hello", "thanks", "please", "<none>"]
    confs = [0.4, 0.65, 0.7, 0.85, 0.95]
    conf_p = [0.1, 0.1, 0.15, 0.25, 0.4]
    t, inputs, outputs = 0.0, [], []
    run_label, run_left = "hello", 0
    for _ in range(n):
        t += float(rng.uniform(0.2, 0.5))
        roll = rng.random()
        if roll < 0.03:
            raw = None
        elif roll < 0.06:
            raw = {"label": "hello"}  # malformed
        else:
            # Sign-like runs: the same label for 1-8 predictions, then a new one,
            # so real streaks (and held signs, repeats, cooldowns) occur.
            if run_left == 0:
                run_label = labels[int(rng.integers(0, len(labels)))]
                run_left = int(rng.integers(1, 9))
            run_left -= 1
            conf = float(rng.choice(confs, p=conf_p))
            raw = {"label": run_label, "confidence": conf, "timestamp": t}
        inputs.append(raw)
        outputs.append(flt.update(raw))
    return {
        "name": name,
        "params": {js_kwargs[k]: (sorted(v) if k == "ignore_labels" else v)
                   for k, v in kwargs.items()},
        "inputs": inputs,
        "outputs": outputs,
        "invalidCount": flt.invalid_count,
    }


def main():
    rng = np.random.default_rng(1234)
    buffers = [
        # Real config: 60 frames x 149 features, ~15 fps irregular, some NaNs/invalid frames
        buffer_case("default-15fps", rng, 120, config.FRAMES, config.FEATURES,
                    config.WINDOW_SECONDS, (12, 18), 0.02, 0.1),
        # Slow camera (oversampling)
        buffer_case("slow-6fps", rng, 60, config.FRAMES, config.FEATURES,
                    config.WINDOW_SECONDS, (4, 8), 0.0, 0.0),
        # Fast camera (downsampling)
        buffer_case("fast-30fps", rng, 300, config.FRAMES, config.FEATURES,
                    config.WINDOW_SECONDS, (25, 35), 0.01, 0.05),
        # Small custom shape, partial fill
        buffer_case("small-partial", rng, 14, 10, 8, 2.0, (6, 10), 0.1, 0.1,
                    min_fill_ratio=0.0),
        # Not ready -> sequence is null
        buffer_case("not-ready", rng, 5, config.FRAMES, config.FEATURES,
                    config.WINDOW_SECONDS, (12, 18), 0.0, 0.0),
    ]
    filters = [
        filter_case("defaults", rng, 400),
        filter_case("strict", rng, 400, confidence_threshold=0.9, stability_count=4),
        filter_case("ignore-and-cooldown", rng, 400, stability_count=2, repeat_cooldown=5.0,
                    ignore_labels={"<none>"}),
    ]

    out = ROOT / "js" / "fixtures" / "parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"buffers": buffers, "filters": filters}), encoding="utf-8")
    n_emit = sum(sum(1 for o in f["outputs"] if o) for f in filters)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB); "
          f"{len(buffers)} buffer cases, {len(filters)} filter cases, {n_emit} captions")
    for b in buffers:
        print(f"  buffer {b['name']}: ready={b['ready']} rejected={b['rejected']} len={b['length']}")


if __name__ == "__main__":
    main()
