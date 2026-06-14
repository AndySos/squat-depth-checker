"""Video pose extraction with MediaPipe Pose Landmarker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.request import urlretrieve

import numpy as np


POSE_LANDMARKER_FULL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)

LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

LOWER_BODY_JOINTS = (
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
)


@dataclass(frozen=True)
class LandmarkFrame:
    """Pose landmarks for one decoded video frame.

    Coordinates are normalized image coordinates from MediaPipe. Shape is
    `(33, 3)` for x, y, z. Visibility and presence are shape `(33,)`.
    """

    frame_index: int
    timestamp_ms: int
    landmarks: np.ndarray
    visibility: np.ndarray
    presence: np.ndarray


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int


def ensure_pose_model(model_path: str | Path = "models/pose_landmarker_full.task") -> Path:
    """Download the default MediaPipe pose model if it is not present."""

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(POSE_LANDMARKER_FULL_URL, path)
    return path


def read_video_metadata(video_path: str | Path) -> VideoMetadata:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return VideoMetadata(fps=fps, frame_count=frame_count, width=width, height=height)
    finally:
        capture.release()


def iter_video_frames(
    video_path: str | Path,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield `(frame_index, timestamp_ms, bgr_frame)` from a video file."""

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    emitted = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                yield frame_index, timestamp_ms, frame
                emitted += 1
                if max_frames is not None and emitted >= max_frames:
                    break
            frame_index += 1
    finally:
        capture.release()


def run_pose_landmarker(
    video_path: str | Path,
    model_path: str | Path | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> list[LandmarkFrame]:
    """Run MediaPipe Pose Landmarker in VIDEO mode over a video."""

    cv2 = _import_cv2()
    mp = _import_mediapipe()
    model = ensure_pose_model(model_path or "models/pose_landmarker_full.task")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    frames: list[LandmarkFrame] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        for frame_index, timestamp_ms, bgr_frame in iter_video_frames(
            video_path,
            frame_stride=frame_stride,
            max_frames=max_frames,
        ):
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frames.append(_result_to_frame(result, frame_index, timestamp_ms))
    return frames


def _result_to_frame(result, frame_index: int, timestamp_ms: int) -> LandmarkFrame:
    if not result.pose_landmarks:
        return LandmarkFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            landmarks=np.full((33, 3), np.nan, dtype=float),
            visibility=np.zeros(33, dtype=float),
            presence=np.zeros(33, dtype=float),
        )

    landmarks = result.pose_landmarks[0]
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=float)
    visibility = np.array([getattr(lm, "visibility", 0.0) for lm in landmarks], dtype=float)
    presence = np.array([getattr(lm, "presence", 0.0) for lm in landmarks], dtype=float)
    return LandmarkFrame(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        landmarks=coords,
        visibility=visibility,
        presence=presence,
    )


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for video IO. Install opencv-python.") from exc
    return cv2


def _import_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError("MediaPipe is required for pose extraction. Install mediapipe.") from exc
    return mp
