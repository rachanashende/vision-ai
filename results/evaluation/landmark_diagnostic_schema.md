Diagnostic only: MediaPipe Tasks Holistic hands + pose on one local MP4.

Run from the project root:
    python scripts/test_landmark_extraction.py
Optional existing assets:
    python scripts/test_landmark_extraction.py --video dataset/raw_videos/book/07068.mp4 --model models/holistic_landmarker.task

Requires existing mediapipe, OpenCV, pandas, and a compatible local Holistic
.task model. Never downloads assets. Missing API/model produces a blocked CSV,
not fabricated missing detections. No legacy or hand-only fallback is used.

Diagnostic schema (not a GRU feature vector): one row per frame/component/index,
including missing landmarks. Left/right hands each use MediaPipe's 21-point
ordering; pose uses its 33-point ordering, including upper-body points. RGB
input is not mirrored. x/y are image-width/height-normalized coordinates, with
origin at top left, x rightward and y downward; predictions may be out of bounds.
z is relative depth, smaller toward the camera: hands use wrist-relative depth,
pose uses hip-midpoint-relative depth. z is not metric or shared across groups.
Visibility/presence are optional model scores, not detection booleans. Missing
coordinates/scores stay blank; no zero filling, scaling, trimming, or resampling.
World coordinates and face landmarks are deliberately not exported. Agree on
ordering, subset, mirroring, normalization, missing masks, and temporal sampling
with the extension developer before building the shared GRU preprocessing.


CSV details:
- frame_index starts at 0; landmark_index follows MediaPipe ordering.
- present means a coordinate was returned, not that it is visible/confident.
- component is left_hand, right_hand, or pose; pipeline denotes a run-level error.
- A successful inference with no detections has present=False and blank coordinates.
- An inference error has present blank (unknown), never False (absent).
- Hand indices: 0 wrist; 1-4 thumb; 5-8 index; 9-12 middle; 13-16 ring; 17-20 little.
- Pose: 0 nose; 1-10 eyes/ears/mouth; 11/12 shoulders; 13/14 elbows;
  15/16 wrists; 17/18 little fingers; 19/20 index fingers; 21/22 thumbs;
  23/24 hips; 25/26 knees; 27/28 ankles; 29/30 heels; 31/32 foot tips.
  In left/right pose pairs, the first index is left. All 33 are diagnostic output;
  no upper-body subset has been finalized for training.
- VIDEO timestamps use frame_index / reported FPS, rounded to milliseconds and
  forced strictly increasing. Invalid FPS blocks inference rather than inventing timing.
- Detection rates use successful inference frames as denominator. Errors are
  counted separately; group presence means at least one landmark was returned.
- Annotations are colored dots: left hand green, right hand red, pose cyan.
  They are drawn on a copy of the original BGR frame. No video is changed.
- Existing diagnostic outputs with the same video stem are overwritten on rerun.

References:
https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
