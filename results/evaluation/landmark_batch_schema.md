# Batch landmark schema: asl-holistic-xyz-v1

Each `dataset/processed/<gloss>/<video_id>.npz` contains one video's decoded
frames in their original order. Load with `numpy.load(path, allow_pickle=False)`.
There are no Python objects or pickle payloads. `T` is variable; no 60-frame
resampling, trimming, mirroring, or final GRU vector concatenation is performed.

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `left_hand_landmarks` | `(T, 21, 3)`, float32 | MediaPipe anatomical left hand, index order, xyz |
| `right_hand_landmarks` | `(T, 21, 3)`, float32 | MediaPipe anatomical right hand, index order, xyz |
| `pose_landmarks` | `(T, 33, 3)`, float32 | All pose points, including upper body, index order, xyz |
| `left_hand_present` | `(T,)`, bool | Left-hand group returned coordinates |
| `right_hand_present` | `(T,)`, bool | Right-hand group returned coordinates |
| `pose_present` | `(T,)`, bool | Pose group returned coordinates |
| `inference_ok` | `(T,)`, bool | Combined Holistic inference and schema checks succeeded |
| `frame_indices` | `(T,)`, int64 | Original zero-based decoded frame indices |
| `timestamps_ms` | `(T,)`, int64 | VIDEO-mode timestamp for each frame |
| `fps` | scalar float64 | OpenCV-reported positive finite FPS |
| `timestamp_source` | scalar Unicode | Timing derivation |
| `schema_version` | scalar Unicode | `asl-holistic-xyz-v1` |
| `gloss`, `video_id`, `split`, `signer_id` | scalar Unicode | Original metadata identity; IDs retain leading zeros |
| `frame_start`, `frame_end` | scalar Unicode | Source WLASL boundaries, including `-1`; blank if unavailable |
| `trimming_status` | scalar Unicode | `uncertain; source boundaries not applied` |
| `metadata_source`, `source_video`, `mediapipe_version` | scalar Unicode | Provenance |

Coordinate order is always **x, y, z**. x/y are normalized by original image
width/height, with origin at top left, x rightward, y downward. Predictions are
not clipped to [0, 1]. Input is converted from BGR to RGB without horizontal
flipping. MediaPipe anatomical left/right labels are retained; these are not
screen-side labels and cannot establish whether a source recording was mirrored.

Normalized z is relative depth, smaller toward the camera. Hand depth is
wrist-relative; pose depth is hip-midpoint-relative. These values are not metric
and are not a shared depth coordinate system across groups. World coordinates,
face landmarks, visibility and presence confidence scores are not exported in
this schema. A group mask indicates returned coordinates, not visibility or
confidence of each individual landmark.

Hand order: 0 wrist; 1–4 thumb; 5–8 index; 9–12 middle; 13–16 ring; 17–20 little.
Pose order: 0 nose; 1–10 eyes/ears/mouth; 11/12 shoulders; 13/14 elbows;
15/16 wrists; 17/18 little fingers; 19/20 index fingers; 21/22 thumbs;
23/24 hips; 25/26 knees; 27/28 ankles; 29/30 heels; 31/32 foot tips.
In these pose pairs the first index is left. The upper-body training subset is
not finalized; all 33 pose landmarks are retained.

Missing groups are exactly zero-filled with a False group mask. If inference
fails, all group coordinates are zero, all group masks are False, and
`inference_ok=False`; this must not be interpreted as a confirmed absence.
Successful inference with no hands has `inference_ok=True` and False hand masks.
Coordinates equal to zero can be real, so always use the masks.

Timing follows the existing extraction diagnostic: round(frame_index * 1000 /
reported_fps), then increase to previous_timestamp + 1 if rounding would cause
a duplicate. These are constant-FPS estimates, not recovered variable-frame-rate
presentation timestamps. Each video starts with a fresh Holistic VIDEO detector.
Invalid FPS, no decoded frames, early decoding relative to the container frame
count, and total inference failure produce no NPZ. Isolated inference errors
are retained with masks and `processed_with_errors` status.

Source frame boundaries are recorded but never applied: source FPS and prior
trimming are unverified. Split and signer fields are preserved. Missing or
ambiguous identity metadata, unknown splits, or missing signer IDs block that
video instead of inventing assignments. No train/val/test pooling is performed;
downstream loaders must filter using the saved split.

The quality CSV is replaced with the current invocation's results, including
dry-run and limited runs. Detection rates are fractions from 0 to 1 over **all
decoded frames**, including inference failures; inference failure counts are
reported separately. Rates/counts are blank for dry-run and skipped outputs
because those videos were not inspected again. `--limit N` considers the first
N sorted MP4 paths, including paths later skipped or rejected. Defaults do no
inference or NPZ writing; actual extraction requires `--extract`. Existing outputs
are checked for schema/metadata consistency and preserved; replacement requires
`--overwrite`. Temporary archives are validated before atomic publication.

Agree on any future normalization, handedness correction, upper-body subset,
confidence filtering, temporal sampling, or concatenation with the extension
developer first. Changing those choices requires an explicit schema version.
