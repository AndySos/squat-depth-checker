"""End-to-end first-run squat-depth pipeline."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .constraints import evaluate_constraints
from .depth import analyze_depth
from .pose import run_pose_landmarker
from .temporal import clean_trajectory
from .visualize import save_annotated_bottom_frame, save_annotated_video


def analyze_video(
    video_path: str | Path,
    output_dir: str | Path = "outputs",
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """Run pose extraction, cleaning, constraints, depth, and visualization."""

    output = Path(output_dir)
    frames = run_pose_landmarker(video_path, frame_stride=frame_stride, max_frames=max_frames)
    cleaned = clean_trajectory(frames)
    constraint_report = evaluate_constraints(cleaned)
    result = analyze_depth(cleaned, constraint_report)
    annotated_path = save_annotated_bottom_frame(
        video_path,
        cleaned,
        result,
        output_path=output / "bottom_frame.jpg",
    )
    annotated_video_path = save_annotated_video(
        video_path,
        cleaned,
        result,
        constraints=constraint_report,
        output_path=output / "annotated_depth.mp4",
    )

    return {
        "result": asdict(result),
        "annotated_bottom_frame": str(annotated_path),
        "annotated_video": str(annotated_video_path),
        "frame_count": cleaned.frame_count,
        "constraint_flagged_frames": int(constraint_report.frame_flags.sum()),
        "jump_flagged_frames": int(cleaned.jump_flags.any(axis=1).sum()),
        "low_confidence_frames": int(cleaned.low_confidence.any(axis=1).sum()),
        "cleaned": cleaned,
        "constraints": constraint_report,
    }
