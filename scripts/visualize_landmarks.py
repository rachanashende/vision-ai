r"""Create a 2-by-3 contact sheet using actual MediaPipe Tasks Holistic inference.

Run from the project root: python scripts/visualize_landmarks.py
Uses only the existing model and video; never downloads or changes them.
Frames include both endpoints and are evenly spaced by zero-based frame index.
VIDEO timestamps come from OpenCV's source position in milliseconds. If those
are unavailable, frame_index / FPS is used with a warning (constant-FPS estimate).
Inference runs on the six sampled frames in chronological order, not every frame.
RGB inference input and the display are unmirrored. Left/right mean MediaPipe's
anatomical labels for the subject, not the left/right side of the contact sheet.
Pose drawing includes indices 0-24 (head, arms, torso, and hip anchors); low-score
pose points are omitted from drawing. This display subset does not change the
diagnostic schema or define the GRU feature format.
"""

import math
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np

from test_landmark_extraction import diagnostic_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEFT_COLOR = (80, 255, 80)       # BGR: green
RIGHT_COLOR = (255, 80, 255)     # BGR: magenta
POSE_COLOR = (0, 220, 255)       # BGR: yellow


def sample_indices(frame_count):
    if frame_count < 6:
        raise ValueError("At least six frames are required for six distinct samples.")
    return [round(index * (frame_count - 1) / 5) for index in range(6)]


def read_samples(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open {video_path}")
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or count <= 0:
            raise ValueError("Video has no usable reported frame count.")
        indices = sample_indices(int(count))
        fps = capture.get(cv2.CAP_PROP_FPS)
        samples = []
        previous_timestamp = -1
        for index in range(indices[-1] + 1):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"Could not decode frame {index}; contact sheet not written.")
            if index not in indices:
                continue
            source_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(source_ms) or source_ms < 0 or (index > 0 and source_ms <= 0):
                if not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Source timestamps and FPS are unavailable.")
                source_ms = index * 1000 / fps
                print(f"WARNING: Frame {index}: timestamp estimated from frame index/FPS.")
            timestamp = round(source_ms)
            if timestamp <= previous_timestamp:
                raise ValueError("Sample timestamps are not strictly increasing; refusing to invent timing.")
            samples.append((index, timestamp, frame))
            previous_timestamp = timestamp
        return samples
    finally:
        capture.release()


def draw_group(image, landmarks, connections, color, pose=False):
    """Empty landmark lists are normal when a hand is outside the image."""
    height, width = image.shape[:2]
    points = {}
    for index, landmark in enumerate(landmarks):
        if pose and index > 24:
            continue
        if pose and any(getattr(landmark, score, None) is not None
                        and getattr(landmark, score) < 0.5 for score in ("visibility", "presence")):
            continue
        if math.isfinite(landmark.x) and math.isfinite(landmark.y):
            if 0 <= landmark.x < 1 and 0 <= landmark.y < 1:
                points[index] = (round(landmark.x * (width - 1)), round(landmark.y * (height - 1)))
    for connection in connections:
        if connection.start in points and connection.end in points:
            cv2.line(image, points[connection.start], points[connection.end], color, 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)


def make_tile(frame, result, index, timestamp, vision):
    # Annotate a display copy; original-size RGB frames are used for inference.
    width = 640
    height = round(frame.shape[0] * width / frame.shape[1])
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    draw_group(image, result.pose_landmarks, vision.PoseLandmarksConnections.POSE_LANDMARKS,
               POSE_COLOR, pose=True)
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    draw_group(image, result.left_hand_landmarks, connections, LEFT_COLOR)
    draw_group(image, result.right_hand_landmarks, connections, RIGHT_COLOR)
    tile = cv2.copyMakeBorder(image, 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    left = bool(result.left_hand_landmarks)
    right = bool(result.right_hand_landmarks)
    cv2.putText(tile, f"Frame {index} | source timestamp {timestamp} ms", (12, height + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tile, f"MP anatomical LEFT: {'detected' if left else 'not detected'}", (12, height + 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, LEFT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(tile, f"MP anatomical RIGHT: {'detected' if right else 'not detected'}", (12, height + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, RIGHT_COLOR, 1, cv2.LINE_AA)
    return tile


def main():
    video_path = PROJECT_ROOT / "dataset/raw_videos/before/05728.mp4"
    model_path = PROJECT_ROOT / "models/holistic_landmarker.task"
    output_path = PROJECT_ROOT / "results/graphs/05728_landmark_contact_sheet.png"
    try:
        for path in (video_path, model_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required local file missing: {path}. No downloads attempted.")
        vision = mp.tasks.vision
        print(f"MediaPipe {mp.__version__}: Tasks HolisticLandmarker, VIDEO mode")
        print("Unmirrored input/display; LEFT and RIGHT are MediaPipe anatomical labels.")
        samples = read_samples(video_path)
        options = vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
        )
        tiles = []
        with vision.HolisticLandmarker.create_from_options(options) as detector:
            for index, timestamp, frame in samples:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(image, timestamp)
                diagnostic_rows(result, index, timestamp)  # Validate the existing hands + pose schema.
                print(f"Frame {index:2d} | {timestamp:4d} ms | "
                      f"MP left={bool(result.left_hand_landmarks)} | "
                      f"MP right={bool(result.right_hand_landmarks)} | "
                      f"pose={bool(result.pose_landmarks)}")
                tiles.append(make_tile(frame, result, index, timestamp, vision))
        grid = cv2.vconcat([cv2.hconcat(tiles[:3]), cv2.hconcat(tiles[3:])])
        header = np.full((100, grid.shape[1], 3), 24, dtype=np.uint8)
        cv2.putText(header, "BEFORE / 05728 - MediaPipe Holistic: six evenly spaced frames", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(header, "Unmirrored display. LEFT / RIGHT = subject's anatomical side, as labeled by MediaPipe.",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        for text, position, color in (("LEFT hand: green", (20, 87), LEFT_COLOR),
                                      ("RIGHT hand: magenta", (340, 87), RIGHT_COLOR),
                                      ("Upper-body pose: yellow", (720, 87), POSE_COLOR)):
            cv2.putText(header, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), cv2.vconcat([header, grid])):
            raise OSError(f"Could not write {output_path}")
        print(f"Saved contact sheet: {output_path}")
        return 0
    except (AttributeError, OSError, cv2.error, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
