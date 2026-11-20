r"""Diagnostic only: MediaPipe Tasks Holistic hands + pose on one local MP4.

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
"""

import argparse
import math
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"left_hand": 21, "right_hand": 21, "pose": 33}
COLUMNS = ["video", "mediapipe_version", "api", "model", "frame_index", "timestamp_ms",
           "component", "landmark_index", "present", "x", "y", "z", "visibility",
           "presence_score", "processing_status", "error"]
SCHEMA = __doc__ + """

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
"""


def diagnostic_rows(result, frame_index, timestamp_ms, error=""):
    """Export expected landmark slots, keeping absent and unprocessed distinct."""
    rows = []
    for component, count in GROUPS.items():
        landmarks = getattr(result, f"{component}_landmarks") if result is not None else []
        if len(landmarks) not in (0, count):
            raise ValueError(f"Unexpected {component} schema: {len(landmarks)} landmarks, expected {count}.")
        for index in range(count):
            landmark = landmarks[index] if index < len(landmarks) else None
            row = dict(frame_index=frame_index, timestamp_ms=timestamp_ms,
                       component=component, landmark_index=index,
                       present=None if error else landmark is not None,
                       processing_status="error" if error else "ok", error=error)
            for name in ("x", "y", "z", "visibility", "presence"):
                row["presence_score" if name == "presence" else name] = (
                    getattr(landmark, name, None) if landmark is not None else None
                )
            rows.append(row)
    return rows


def annotate(frame, result):
    image = frame.copy()
    height, width = image.shape[:2]
    for component, color in (("left_hand", (0, 255, 0)),
                             ("right_hand", (0, 0, 255)), ("pose", (255, 255, 0))):
        for landmark in getattr(result, f"{component}_landmarks", []):
            if math.isfinite(landmark.x) and math.isfinite(landmark.y):
                if 0 <= landmark.x < 1 and 0 <= landmark.y < 1:
                    cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)),
                               3, color, -1)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="One existing MP4 under dataset/raw_videos.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/holistic_landmarker.task",
                        help="Path to an existing compatible Holistic .task model; never downloaded.")
    args = parser.parse_args()
    raw_dir = (PROJECT_ROOT / "dataset/raw_videos").resolve()
    evaluation_dir = PROJECT_ROOT / "results/evaluation"
    graph_dir = PROJECT_ROOT / "results/graphs"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "landmark_diagnostic_schema.md").write_text(SCHEMA, encoding="utf-8")
    vision = getattr(getattr(mp, "tasks", None), "vision", None)
    print(f"Python: {sys.executable}\nMediaPipe: {mp.__version__}")
    print(f"mp.solutions available: {hasattr(mp, 'solutions')}")
    print("Available Tasks landmarkers:", [name for name in (
        "HolisticLandmarker", "HandLandmarker", "PoseLandmarker") if hasattr(vision, name)])

    rows = []
    errors = []
    frames_read = 0
    successful = 0
    detections = {component: 0 for component in GROUPS}
    capture = None
    video = None
    annotated = False
    try:
        videos = sorted(path for path in raw_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() == ".mp4")
        video = args.video.resolve() if args.video else (videos[0] if videos else None)
        if video is None:
            raise FileNotFoundError(f"No existing MP4 found under {raw_dir}.")
        if not video.is_relative_to(raw_dir) or not video.is_file() or video.suffix.lower() != ".mp4":
            raise ValueError("--video must be an existing MP4 under dataset/raw_videos.")
        print(f"Video: {video}\nModel: {args.model.resolve()}")
        if not all(hasattr(vision, name) for name in (
            "HolisticLandmarker", "HolisticLandmarkerOptions", "RunningMode"
        )) or not hasattr(getattr(mp, "tasks", None), "BaseOptions"):
            raise RuntimeError("Installed MediaPipe lacks the required Tasks Holistic API for combined hands + pose. No feature-format fallback was used.")
        if not args.model.is_file():
            raise FileNotFoundError(
                f"Required local Holistic model is missing: {args.model.resolve()}. "
                "This API needs a compatible .task model. Supply an existing asset with --model. "
                "No model was downloaded; hands and pose were not evaluated."
            )
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the selected video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS is invalid; VIDEO-mode timestamps cannot be derived safely.")
        reported_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve())),
            running_mode=vision.RunningMode.VIDEO,
        )
        previous_timestamp = -1
        with vision.HolisticLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                index = frames_read
                frames_read += 1
                timestamp = max(previous_timestamp + 1, round(index * 1000 / fps))
                previous_timestamp = timestamp
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(image, timestamp)
                    frame_rows = diagnostic_rows(result, index, timestamp)
                except (AttributeError, cv2.error, RuntimeError, ValueError) as error:
                    message = f"Frame {index}: {error}"
                    errors.append(message)
                    rows.extend(diagnostic_rows(None, index, timestamp, message))
                    continue
                rows.extend(frame_rows)
                successful += 1
                present = {component: bool(getattr(result, f"{component}_landmarks"))
                           for component in GROUPS}
                for component in GROUPS:
                    detections[component] += present[component]
                if not annotated and any(present.values()):
                    graph_dir.mkdir(parents=True, exist_ok=True)
                    sample_path = graph_dir / f"{video.stem}_landmark_sample.png"
                    if cv2.imwrite(str(sample_path), annotate(frame, result)):
                        annotated = True
                        print(f"Annotated sample: {sample_path}")
                    else:
                        errors.append(f"Could not write annotated sample: {sample_path}")
        if frames_read == 0:
            raise RuntimeError("No video frames could be decoded.")
        if math.isfinite(reported_frames) and frames_read < reported_frames:
            errors.append(f"Read {frames_read} of {reported_frames:g} reported frames; possible decode failure or inaccurate container count.")
    except (OSError, cv2.error, RuntimeError, ValueError) as error:
        errors.append(str(error))
        rows.append(dict(component="pipeline", processing_status="blocked" if not frames_read else "error",
                         error=str(error)))
    finally:
        if capture is not None:
            capture.release()

    report = pd.DataFrame(rows)
    report["video"] = str(video) if video else ""
    report["mediapipe_version"] = mp.__version__
    report["api"] = "mp.tasks.vision.HolisticLandmarker"
    report["model"] = str(args.model.resolve())
    report_path = evaluation_dir / f"{video.stem if video else 'no_video'}_landmark_diagnostic.csv"
    report.reindex(columns=COLUMNS).to_csv(report_path, index=False)
    print(f"Total decoded frames processed: {frames_read}; successful inference frames: {successful}")
    for component, count in detections.items():
        rate = f"{100 * count / successful:.1f}% ({count}/{successful})" if successful else "N/A (no successful inference)"
        print(f"{component} detection rate: {rate}")
    print(f"Processing errors: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not annotated:
        print("No annotated sample created: no successful landmark detections were available.")
    print(f"Diagnostic CSV: {report_path}")
    print(f"Schema: {evaluation_dir / 'landmark_diagnostic_schema.md'}")
    print("Diagnostic schema only; no final GRU feature vector is defined.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
