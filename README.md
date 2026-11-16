# Vision AI - Member 3 pipeline

Real-time American Sign Language captioning for video conferencing using
MediaPipe and deep learning. This repository holds **Member 3's part**:
landmark validation, the rolling sequence buffer, and prediction filtering,
implemented in both Python (training/offline checks) and JavaScript (the
Chrome extension) with a test that keeps the two identical.

Dataset, landmark extraction and model training (Member 1) and the Chrome
extension UI (Member 2) are separate. Assumptions about their interfaces are
listed in [INTERFACES.md](INTERFACES.md) and are **provisional until confirmed**.

## What is here

| Piece | Python | JavaScript | Purpose |
|---|---|---|---|
| Settings | `config.py` | `js/config.js` | Frame/feature counts, window, thresholds (keep in sync) |
| Frame + sequence validation | `validation.py` | `sanitizeFrame` in `js/sequence_buffer.js` | Reject bad frames, fill stray NaN/inf, check final `(60, 149)` float32 shape |
| Rolling buffer | `sequence_buffer.py` | `js/sequence_buffer.js` | Keep last 4 s of frames, resample to 60 frames |
| Prediction filter | `prediction_filter.py` | `js/prediction_filter.js` | Index -> label, confidence threshold, 3-in-a-row stability, duplicate suppression |
| Parity check | `tools/make_parity_fixtures.py` | `js/parity.test.js` | Same inputs must give the same outputs in both languages |
| CLI check | `validate_sequence.py` | | Validate a saved `.npy` sequence |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Node 18+ is needed only for the JavaScript tests (no npm packages required).

## Run the tests

```bash
python -m pytest                   # Python: validation, buffer, filter
python tools/make_parity_fixtures.py   # regenerate fixtures after changing logic
cd js && node --test               # JavaScript unit tests + Python/JS parity
```

## Usage

### Python

```python
from sequence_buffer import RollingBuffer
from prediction_filter import LabelMap, PredictionFilter, probs_to_prediction

buf = RollingBuffer()
labels = LabelMap.from_json_file("labels.json")
flt = PredictionFilter(ignore_labels={"<none>"})

# for every landmark frame (149 numbers) and its timestamp in seconds:
buf.push(frame, t)

# about every 350 ms:
seq = buf.get_sequence()                     # (60, 149) float32, or None if not ready
if seq is not None:
    probs = model.predict(seq[None])[0]      # softmax vector
    pred = probs_to_prediction(probs, labels, t)
    caption = flt.update(pred)               # str to display, or None
```

### JavaScript (extension)

```js
import { RollingBuffer } from './sequence_buffer.js';
import { LabelMap, PredictionFilter, probsToPrediction } from './prediction_filter.js';

const buf = new RollingBuffer();
const labels = new LabelMap(await (await fetch('labels.json')).json());
const flt = new PredictionFilter({ ignoreLabels: ['<none>'] });

buf.push(frame, performance.now() / 1000);          // per landmark frame

// about every 350 ms:
const seq = buf.getSequence();                       // Float32Array(60*149) or null
if (seq) {
  const probs = model.predict(tf.tensor(seq, [1, 60, 149])).dataSync();
  const caption = flt.update(probsToPrediction(probs, labels, performance.now() / 1000));
  if (caption) showCaption(caption);
}
```

## Behaviour summary

- **Frames:** wrong shape or mostly NaN/inf -> rejected and counted (`buf.rejected`); a few bad values -> replaced by `0.0`; timestamps must not go backwards.
- **Sequence:** last 4 s, 60 evenly spaced times, nearest frame (ties -> earlier), available once 2 s are buffered.
- **Captions:** confidence >= 0.70, same label 3 times in a row, one caption per held sign, same caption not repeated within 2 s. Bad input never throws; it just breaks the streak.

## Status

| Area | State |
|---|---|
| Validation, buffer, filter (Python) | Done, 61 tests passing |
| JavaScript port + Python/JS parity | Done, 17 tests passing |
| Real landmark data | **Not validated yet** - waiting on Member 1 |
| Real model input/output | **Untested** - waiting on Member 1 (tested with mock probabilities only) |
| Extension integration, latency, live testing | Not started - needs Member 2 |

Tuning of thresholds and the resampling rule must be revisited once real data
and a real model exist. Performance targets (about 15 FPS, inference roughly
every 350 ms) are goals to measure, not results achieved.
