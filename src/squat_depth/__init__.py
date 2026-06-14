"""Squat depth checker MVP package."""

from .depth import DepthResult, FrameDepthTrace, analyze_depth, frame_depth_trace
from .pose import LandmarkFrame
from .reps import RepSegment, analyze_reps, segment_reps

__all__ = [
    "DepthResult",
    "FrameDepthTrace",
    "LandmarkFrame",
    "RepSegment",
    "analyze_depth",
    "analyze_reps",
    "frame_depth_trace",
    "segment_reps",
]
