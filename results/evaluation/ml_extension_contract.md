# PROVISIONAL ML-to-extension input contract

Version: **asl-extension-149-v0.1-PROVISIONAL**. Integration testing only.
Member 2 and Member 3 must confirm matching JavaScript behavior before adoption.
No trained model or final GRU architecture is defined by this document.

## Observed existing data, before this proposal

Inspected `extract_landmarks_batch.py`, `landmark_batch_schema.md`, metadata and
five real archives: before/05728 (26 frames, train), book/07068 (68, train),
drink/17713 (91, test), who/63229 (47, val), yes/64287 (68, val).
All use `asl-holistic-xyz-v1`. Raw arrays are float32 hands `(T,21,3)` each and
pose `(T,33,3)`, with Boolean `(T,)` group masks and `inference_ok`.
The archives are variable length, not pre-existing 60-frame/149-feature inputs.
Raw coordinate total is 225/frame before masks. Whole missing groups are zeros;
real zero coordinates are possible, so missingness must never be inferred from xyz.
Frame indices are zero-based source indices; timestamps are rounded constant-FPS
estimates from the extractor, forced strictly increasing. They are not original
variable-frame-rate presentation timestamps. Trimming is uncertain, including
the five provisionally retained clips. No WLASL boundary offsets are applied.

## Newly proposed layout: exactly 149 features

149 is supported as **144 coordinates + 5 explicit masks**, not 149 coordinates.
It retains all finger points and six arm/shoulder pose landmarks. Other pose,
face and world-coordinate features and confidence scores are omitted. This
subset is a provisional engineering choice, not a demonstrated optimal model input.

All offsets below are zero-based; ranges are inclusive. Each point uses xyz.

| Features | Contents | Exact formula |
| --- | --- | --- |
| 0–62 | Anatomical LEFT hand, MediaPipe indices 0–20 | point i starts at `3*i` |
| 63–125 | Anatomical RIGHT hand, indices 0–20 | point i starts at `63+3*i` |
| 126–128 | Pose 11: left shoulder | xyz |
| 129–131 | Pose 12: right shoulder | xyz |
| 132–134 | Pose 13: left elbow | xyz |
| 135–137 | Pose 14: right elbow | xyz |
| 138–140 | Pose 15: left wrist | xyz |
| 141–143 | Pose 16: right wrist | xyz |
| 144 | Left-hand group present | float32 0 or 1 |
| 145 | Right-hand group present | float32 0 or 1 |
| 146 | Pose group present | float32 0 or 1 |
| 147 | Inference succeeded (`inference_ok`) | float32 0 or 1 |
| 148 | Real selected source frame (`is_real_frame`) | float32 0 or 1; 0 on padding |

Hand indices: 0 wrist; 1–4 thumb; 5–8 index; 9–12 middle; 13–16 ring; 17–20 little.
Pose 15/16 are body-model wrists; hand point 0 is the hand-model wrist. Both are
retained intentionally and are not assumed identical. The pose mask is a group
return mask; it does not establish confidence/visibility of the six individual points.

## Coordinates and normalization

Input frames used for extraction were BGR-to-RGB converted with no horizontal flip.
JavaScript must use the same RGB orientation (CSS mirroring must not alter features).
Left/right always means the model's anatomical labels, not screen sides. Whether
a source recording was itself mirrored is unknown; do not automatically swap hands.

Normalization is **identity on the saved normalized coordinates**: copy float32 xyz
without wrist centering, shoulder scaling, clipping, z standardization or pixel
conversion. x/y are normalized by image width/height, top-left origin, x rightward,
y downward. Predictions outside [0,1] are retained. These values are dimensionless.
Hand z is wrist-relative; pose z is hip-midpoint-relative; smaller is toward the
camera. z is not metric and is not comparable as shared depth across groups.
This preserves hand position in the image but provides no translation/scale
invariance. Any future body-relative normalization needs a new agreed version.

## Missing values and frame validity

Absent group: its coordinate block is exactly zero and its presence bit is 0.
Other present groups remain available. Failed inference: all coordinate blocks
and presence flags are zero, feature 147 is 0, feature 148 is 1. This is UNKNOWN
observation, not confirmed absence. Padding is an all-zero 149-vector, including
feature 148=0. Never carry a previous hand forward or invent landmarks.

Derived diagnostic `frame_valid = is_real_frame && inference_ok && (left_present || right_present)`.
Pose is optional for this diagnostic flag. It is NOT a training-loss mask or an
instruction to remove frames. Keep invalid frames and their position in the buffer.
An inference-successful frame without hands has 147=1, 148=1, and frame_valid=false.
All arrays must be finite; reject corrupt input instead of replacing NaN/Inf with zeros.

## Deterministic variable-length to 60-frame conversion

Output is **(60,149), little-endian float32, C-contiguous/time-major**. Feature f
of step k is at flattened offset `149*k+f`. Future batch shape is `(B,60,149)`;
this does not imply a model currently exists. The `.npy` files have NPY headers;
Member 3 must parse them with an NPY-capable loader before using Float32Array.

Require a nonempty sequence with strictly increasing nonnegative integer saved
timestamps, and matching frame indices/masks. Invalid source schemas are rejected.

- **T >= 60:** choose 60 target times spanning saved timestamp 0 through timestamp
  T-1, including endpoints. For target k (0..59), use integer numerator
  `q = 59*t[0] + k*(t[T-1]-t[0])`. Choose source slot i minimizing
  `abs(59*t[i]-q)`. An exact tie chooses the EARLIER source slot. No interpolation.
  Repeated slots are allowed when timestamps have large gaps; selected source
  timestamps may therefore repeat. Output slots stay chronological.
- **T < 60:** retain every source frame exactly once in original order in slots
  0..T-1, then right-pad with all-zero vectors through slot 59. Do not stretch,
  repeat the last pose, or extrapolate timestamps. For T=1 there is one real frame
  and 59 pads. Empty inputs are rejected.
- Selected source indices/timestamps and source slots are recorded in the manifest;
  padding uses -1 for all three. Do not feed these provenance fields as extra features.
  Long sequences span different durations; short sequences retain original cadence.
  This is an offline fixture policy, not a settled real-time sampling cadence.
- Integer arithmetic must be exact: require `59*max(t) <= 2^53-1` in both languages.
  Do not use Python/JS rounding of fractional targets. JavaScript can implement
  the search with a loop that replaces the best slot only when distance is smaller.

Short-clip padding and long-clip temporal compression change the timing distribution;
the team must agree on live buffer cadence, rolling-window duration, readiness,
warm-up padding and inference scheduling before training. Do not claim 60 frames
means a fixed number of seconds. No sign boundaries have been verified or cropped.

## Minimal JavaScript frame packer

This adapter input is NEW and explicit: flat arrays of `{x,y,z}` points, matching
the original MediaPipe ordering; absent groups are null/empty. Member 2 must map
the actual browser API result to this adapter, including anatomical hand labels.
Do not silently accept nested/multi-person arrays or another landmark schema.

```javascript
function packFrame({leftHand, rightHand, pose, inferenceOk}, isReal = true) {
  const out = new Float32Array(149);
  if (!isReal) return out;
  out[148] = 1;
  if (!inferenceOk) return out;
  out[147] = 1;
  const groups = [
    [leftHand, 21, Array.from({length: 21}, (_, i) => i), 0, 144],
    [rightHand, 21, Array.from({length: 21}, (_, i) => i), 63, 145],
    [pose, 33, [11,12,13,14,15,16], 126, 146]
  ];
  for (const [points, count, indices, start, mask] of groups) {
    if (points == null || points.length === 0) continue;
    if (points.length !== count) throw new Error("Landmark schema mismatch");
    out[mask] = 1;
    indices.forEach((index, slot) => {
      const p = points[index];
      [p.x, p.y, p.z].forEach((value, axis) => {
        if (!Number.isFinite(value) || Math.abs(value) > 3.4028234663852886e38)
          throw new Error("Non-finite/overflowing coordinate");
        out[start + 3*slot + axis] = value;
      });
    });
  }
  return out;
}
```

## Proposed label order and team decisions

Proposed indices, not a trained model's class mapping:
`0 book, 1 drink, 2 computer, 3 before, 4 chair, 5 go, 6 clothes, 7 who, 8 help, 9 yes`.
There is no unknown/background class yet. Original sample splits and signer IDs
remain in the manifest. Validation/test fixtures are for integration checks only;
do not pool them into training or tune classifier decisions on them.

Still open: Member 2/3 parity for all offsets and float32 conversion; browser
Holistic availability and adapter shape; mirrored camera behavior; pose subset;
confidence filtering; missing-hand/frame readiness policy; live cadence and window
duration; padding versus another policy; person normalization; sign-boundary and
sign-correctness review; final labels/background class; trained model input/output.
Provisional fixture export does not settle any of those model-design questions.
