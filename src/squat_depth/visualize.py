"""Visualization helpers for squat-depth results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .depth import DepthResult, SIDES
from .pose import iter_video_frames
from .temporal import CleanedPose


def save_annotated_bottom_frame(
    video_path: str | Path,
    cleaned: CleanedPose,
    result: DepthResult,
    output_path: str | Path = "outputs/bottom_frame.jpg",
) -> Path:
    """Save the selected bottom frame with landmark and label overlays."""

    cv2 = _import_cv2()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected = None
    for frame_index, _timestamp_ms, frame in iter_video_frames(video_path):
        if frame_index == result.bottom_frame_index:
            selected = frame
            break
    if selected is None:
        raise ValueError(f"Could not find frame {result.bottom_frame_index} in video")

    package_index = int(np.argmin(np.abs(cleaned.frame_indices - result.bottom_frame_index)))
    annotated = draw_result(selected, cleaned.landmarks[package_index], result)
    cv2.imwrite(str(output), annotated)
    return output


def draw_result(frame: np.ndarray, landmarks: np.ndarray, result: DepthResult) -> np.ndarray:
    cv2 = _import_cv2()
    image = frame.copy()
    height, width = image.shape[:2]
    joints = SIDES[result.side]
    hip = _to_pixel(landmarks[joints["hip"]], width, height)
    knee = _to_pixel(landmarks[joints["knee"]], width, height)
    ankle = _to_pixel(landmarks[joints["ankle"]], width, height)

    color = (0, 180, 0) if result.label == "to_depth" else (0, 0, 220)
    if result.label == "uncertain":
        color = (0, 165, 255)

    for point in (hip, knee, ankle):
        if point is not None:
            cv2.circle(image, point, 7, color, -1)
    if hip is not None and knee is not None:
        cv2.line(image, hip, knee, color, 3)
        cv2.line(image, (0, knee[1]), (width, knee[1]), (255, 180, 0), 2)
    if knee is not None and ankle is not None:
        cv2.line(image, knee, ankle, color, 3)

    label = f"{result.label} | {result.side} | margin={result.hip_knee_margin:.3f}"
    cv2.rectangle(image, (12, 12), (min(width - 12, 720), 64), (0, 0, 0), -1)
    cv2.putText(image, label, (24, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return image


def _to_pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
    if np.isnan(point[:2]).any():
        return None
    return int(round(point[0] * width)), int(round(point[1] * height))


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for visualization. Install opencv-python.") from exc
    return cv2
