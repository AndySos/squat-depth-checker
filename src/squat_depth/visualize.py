"""Visualization helpers for squat-depth results."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .depth import DepthResult, FrameDepthTrace, SIDES, frame_depth_trace
from .pose import iter_video_frames
from .reps import RepSegment
from .temporal import CleanedPose
from .constraints import ConstraintReport


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


def save_annotated_video(
    video_path: str | Path,
    cleaned: CleanedPose,
    result: DepthResult,
    constraints: ConstraintReport | None = None,
    rep_segments: list[RepSegment] | None = None,
    rep_results: list[DepthResult] | None = None,
    output_path: str | Path = "outputs/annotated_depth.mp4",
    fps: float | None = None,
) -> Path:
    """Save a video with per-frame depth labels and pose overlays."""

    cv2 = _import_cv2()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    trace = frame_depth_trace(cleaned, constraints=constraints, side=result.side)
    writer = None
    try:
        for frame_index, _timestamp_ms, frame in iter_video_frames(video_path):
            package_index = int(np.argmin(np.abs(cleaned.frame_indices - frame_index)))
            annotated = draw_frame_depth(
                frame,
                cleaned.landmarks[package_index],
                trace,
                package_index,
                overall_result=result,
                rep_info=_rep_info_for_index(package_index, rep_segments, rep_results),
            )
            if writer is None:
                height, width = annotated.shape[:2]
                inferred_fps = fps or _read_fps(video_path) or 30.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output), fourcc, inferred_fps, (width, height))
                if not writer.isOpened():
                    raise ValueError(f"Could not create annotated video: {output}")
            writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()

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


def draw_frame_depth(
    frame: np.ndarray,
    landmarks: np.ndarray,
    trace: FrameDepthTrace,
    trace_index: int,
    overall_result: DepthResult | None = None,
    rep_info: tuple[RepSegment, DepthResult] | None = None,
) -> np.ndarray:
    cv2 = _import_cv2()
    image = frame.copy()
    height, width = image.shape[:2]
    joints = SIDES[trace.side]
    hip = _to_pixel(landmarks[joints["hip"]], width, height)
    knee = _to_pixel(landmarks[joints["knee"]], width, height)
    ankle = _to_pixel(landmarks[joints["ankle"]], width, height)

    label = trace.labels[trace_index]
    margin = trace.margins[trace_index]
    color = _label_color(label)

    for point in (hip, knee, ankle):
        if point is not None:
            cv2.circle(image, point, 7, color, -1)
    if hip is not None and knee is not None:
        cv2.line(image, hip, knee, color, 3)
        cv2.line(image, (0, knee[1]), (width, knee[1]), (255, 180, 0), 2)
    if knee is not None and ankle is not None:
        cv2.line(image, knee, ankle, color, 3)

    frame_text = f"frame: {label} | margin={margin:.3f}"
    if rep_info is not None:
        segment, rep_result = rep_info
        overall_text = f"rep {segment.rep_index}: {rep_result.label} | bottom={rep_result.bottom_timestamp_ms / 1000:.2f}s"
    else:
        overall_text = f"overall: {overall_result.label} | bottom={overall_result.bottom_timestamp_ms / 1000:.2f}s" if overall_result else ""
    cv2.rectangle(image, (12, 12), (min(width - 12, 860), 88), (0, 0, 0), -1)
    cv2.putText(image, frame_text, (24, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if overall_text:
        cv2.putText(image, overall_text, (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    return image


def _rep_info_for_index(
    trace_index: int,
    rep_segments: list[RepSegment] | None,
    rep_results: list[DepthResult] | None,
) -> tuple[RepSegment, DepthResult] | None:
    if not rep_segments or not rep_results:
        return None
    for segment, result in zip(rep_segments, rep_results):
        if segment.start_index <= trace_index <= segment.end_index:
            return segment, result
    return None


def _label_color(label: str) -> tuple[int, int, int]:
    if label == "to_depth":
        return (0, 180, 0)
    if label == "uncertain":
        return (0, 165, 255)
    return (0, 0, 220)


def _to_pixel(point: np.ndarray, width: int, height: int) -> tuple[int, int] | None:
    if np.isnan(point[:2]).any():
        return None
    return int(round(point[0] * width)), int(round(point[1] * height))


def _read_fps(video_path: str | Path) -> float:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for visualization. Install opencv-python.") from exc
    return cv2
