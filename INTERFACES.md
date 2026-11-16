# Interfaces and open decisions (Member 3)

This is the contract between the pieces of the pipeline. Everything marked
**PROVISIONAL** is an assumption made so Member 3's code could be built and
tested early. Each one is a single value in `config.py` / `js/config.js`, so
changing it later is cheap, but **Members 1 and 2 must confirm or correct it**.

```
webcam -> MediaPipe landmarks -> [validate + rolling buffer] -> model -> [prediction filter] -> caption UI
 (Member 2)   (Member 1)              Member 3 (this repo)     (Member 1)    Member 3          (Member 2)
```

## 1. Landmark frame (Member 1 -> Member 3)

| Item | Current assumption | Status |
|---|---|---|
| One frame | 1-D array of **149** numbers | PROVISIONAL |
| Type | JS: `Array`/`Float32Array` of numbers; Python: float32 | PROVISIONAL |
| Timestamp | seconds (float) on a monotonic clock, e.g. `performance.now()/1000` | PROVISIONAL |
| Missing landmarks | frame still delivered, missing values are `NaN` (or `0`) | **needs decision** |
| Feature order, normalization, coordinate convention | unknown - not used by Member 3 code | **needs Member 1** |

What Member 3 does with a frame (`validation.sanitize_frame`):
- wrong length / not numeric / not 1-D -> **rejected**
- more than 50% of values NaN/inf -> **rejected** (treated as "nothing detected")
- a few NaN/inf values -> replaced with `0.0`
- timestamps that go backwards -> **rejected**

> If Member 1 encodes "hand not visible" as zeros rather than NaN, those frames
> are accepted as-is. If the model was trained with a different missing-value
> convention than `0.0`, change `FILL_VALUE` so training and live inference match.

## 2. Model input (Member 3 -> Member 1's model)

- Shape `(1, 60, 149)`, float32 (PROVISIONAL). Python gives `(60, 149)`; JS gives a
  flat `Float32Array(60*149)` in row-major (frame-major) order, for
  `tf.tensor(seq, [1, 60, 149])`.
- Built from the last **4 seconds** of frames, resampled to 60 frames by picking,
  for each of 60 evenly spaced times, the **nearest** frame (ties -> earlier; no
  interpolation). Times before the first buffered frame reuse the first frame.
- A sequence is only produced once the buffer spans at least **2 s** (50% of the window).

**Critical:** Member 1 must prepare *training* sequences with the same
resampling rule (or tell us a different one and we change `resample_nearest`).
If training clips are resampled differently from live data, accuracy drops
even though nothing "errors".

## 3. Model output (Member 1 -> Member 3)

- Softmax probability vector, length = number of classes (PROVISIONAL).
  Raw logits are rejected on purpose so a mismatch is noticed.
- Label list: JSON array of strings ordered by output index, e.g. `labels.json`
  (`LabelMap.from_json_file` / `new LabelMap(array)`). Needs delivering with the model.
- Optional: a "no sign / idle" class. If one exists, pass its name in
  `ignore_labels` / `ignoreLabels` so it never becomes a caption.

## 4. Prediction message (Member 3 -> Member 2's UI)

```json
{ "label": "hello", "confidence": 0.93, "timestamp": 12.35 }
```

Member 3's `PredictionFilter.update()` takes that message (or `null` when
inference failed) and returns either a caption string to display or `null`
("show nothing new"). Member 2 only needs to append non-null results to the
caption bar. Filtering rules and defaults:

| Setting | Default | Meaning |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.70 | below this a prediction is ignored and breaks the streak |
| `STABILITY_COUNT` | 3 | same label this many times in a row before it is shown |
| `REPEAT_COOLDOWN_SECONDS` | 2.0 | the caption just shown is not shown again sooner than this |

A held sign produces **one** caption. Defaults are starting points to be tuned
with real model output in Weeks 7-9.

## 5. Where code runs in the extension (Member 2 + Member 3)

Not decided yet. Suggested (Chrome MV3): camera + MediaPipe + buffer + TF.js in an
**offscreen document** (service workers cannot access the camera or DOM), sending
only the `{label, confidence, timestamp}` message or the final caption text to the
content script that draws the overlay. The JS modules in `js/` are plain ES
modules with no dependencies, so they can live in whichever component is chosen.

## Questions to send

**Member 1**
1. Is 149 features per frame final? What is the exact feature order and are values normalized (relative to what)?
2. How is a missing/undetected hand represented (NaN, zeros, something else)?
3. How will training clips be resampled to 60 frames? (We propose nearest-frame over 4 s - see section 2.)
4. Output: softmax over N classes? Is there an idle/"no sign" class? Where will `labels.json` come from?
5. Is the fingerspelling classifier a second model or part of the main one? (Affects the model-output contract.)

**Member 2**
1. Which component runs MediaPipe and TF.js (offscreen document?) and how are messages passed?
2. What timestamp is attached to each landmark frame, and what frame rate do you expect on the test laptop?
3. What should the UI do on `null` (nothing to show) and when the buffer is not ready yet (e.g. "waiting for signer")?
